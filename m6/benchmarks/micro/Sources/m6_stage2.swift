import Accelerate
import Foundation
import Metal
import MetalPerformanceShadersGraph

// =============================================================================
// paper_m6 Stage 2 — FFT landscape on M6: size x memory-level, all backends.
//
//   m6-landscape [csv-path]
//
// Batch is chosen per N so the in+out footprint (2 * B * N * 8 bytes) lands in
// one memory level: L2 (1 MiB), SLC (8 MiB), DRAM (256 MiB). B >= 1 always,
// so the footprint of large N at the small targets is larger than the target
// (recorded in the CSV as the actual footprint).
//
// Backends: ours (repo Metal kernels, every variant that exists for that N),
// MPSGraph fastFourierTransform, vDSP_fft_zop (1 thread, UI QoS -> Super core,
// which in practice runs on the shared SME unit; see Stage 1). MLX is timed by
// the python script paper_m6/src/mlx_landscape.py.
//
// GPU: HOT = K back-to-back executions in one command buffer (>= 64 MiB of
// traffic), per-execution time; COLD = 512 MiB flush CB then one timed CB.
// Every GPU kernel is validated against vDSP (rel L2) before timing.
// FLOPs = 5 N log2 N per FFT.
// =============================================================================

private struct Variant {
    let name: String
    let n: Int
    /// encode one full FFT of `batch` transforms: in -> out (tmp available)
    let encode: (MTLComputeCommandEncoder, MTLBuffer, MTLBuffer, MTLBuffer, Int) -> Void
}

private func tgDispatch(_ p: MTLComputePipelineState, tgs: Int, threads: Int)
    -> (MTLComputeCommandEncoder, MTLBuffer, MTLBuffer) -> Void {
    return { e, a, b in
        e.setComputePipelineState(p)
        e.setBuffer(a, offset: 0, index: 0); e.setBuffer(b, offset: 0, index: 1)
        e.dispatchThreadgroups(MTLSize(width: tgs, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
    }
}

private final class S2 {
    let dev = MTLCreateSystemDefaultDevice()!
    lazy var queue = dev.makeCommandQueue()!
    var variants: [Variant] = []
    let flushP: MTLComputePipelineState
    let flushBuf: MTLBuffer
    let flushBytes = 512 << 20

    init() {
        let o = MTLCompileOptions(); o.fastMathEnabled = true
        let dx = try! dev.makeLibrary(source: loadCombinedSource(), options: o)
        let msSrc = try! String(contentsOfFile: try! FFTDeviceSource.path("src/metal/fft_multisize.metal"), encoding: .utf8)
        let ms = try! dev.makeLibrary(source: msSrc, options: o)
        let fl = try! dev.makeLibrary(source: """
            #include <metal_stdlib>
            using namespace metal;
            kernel void flush_write(device float4* d [[buffer(0)]], constant uint& n [[buffer(1)]],
                                    uint g [[thread_position_in_grid]], uint s [[threads_per_grid]]) {
                for (uint i = g; i < n; i += s) d[i] = float4(float(i)); }
            """, options: o)
        let d = dev
        func P(_ l: MTLLibrary, _ n: String) -> MTLComputePipelineState {
            try! d.makeComputePipelineState(function: l.makeFunction(name: n)!)
        }
        flushP = P(fl, "flush_write")
        flushBuf = dev.makeBuffer(length: flushBytes, options: .storageModePrivate)!

        // single-pass, one FFT per threadgroup: (lib, kernel, N, threads/TG)
        let single: [(MTLLibrary, String, Int, Int)] = [
            (ms, "fft_32_stockham", 32, 8), (ms, "fft_64_stockham", 64, 16),
            (ms, "fft_128_stockham", 128, 32), (ms, "fft_256_stockham", 256, 64),
            (ms, "fft_512_stockham", 512, 128), (ms, "fft_1024_stockham", 1024, 256),
            (ms, "fft_2048_stockham", 2048, 512), (ms, "fft_4096_stockham", 4096, 1024),
            (dx, "fft64_dx_r8", 64, 8), (dx, "fft256_dx_r4", 256, 64),
            (dx, "fft1024_dx_r4", 1024, 256),
            (dx, "fft4096_dx", 4096, 512), (dx, "fft4096_dx_r4", 4096, 1024),
            (dx, "fft4096_dx_r16", 4096, 256),
        ]
        for (lib, k, n, t) in single {
            let p = P(lib, k)
            variants.append(Variant(name: k, n: n) { e, a, b, _, bs in tgDispatch(p, tgs: bs, threads: t)(e, a, b) })
        }
        // four-step N=16384 (dx, batched packing): A: 4 TGs/FFT, B: 8 TGs/FFT, 512 thr
        let pA = P(dx, "fourstep16384_A_b"), pB = P(dx, "fourstep16384_B_b")
        variants.append(Variant(name: "fourstep16384_b", n: 16384) { e, a, b, tmp, bs in
            tgDispatch(pA, tgs: 4 * bs, threads: 512)(e, a, tmp)
            e.memoryBarrier(scope: .buffers)
            tgDispatch(pB, tgs: 8 * bs, threads: 512)(e, tmp, b)
        })
    }

    func flush() {
        let cb = queue.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
        e.setComputePipelineState(flushP); e.setBuffer(flushBuf, offset: 0, index: 0)
        var n = UInt32(flushBytes / 16); e.setBytes(&n, length: 4, index: 1)
        e.dispatchThreads(MTLSize(width: 12 * 1024 * 32, height: 1, depth: 1),
                          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    }
}

func m6Med(_ x: [Double]) -> Double { let s = x.sorted(); return s[s.count / 2] }
func m6Q13(_ x: [Double]) -> (Double, Double) { let s = x.sorted(); return (s[s.count / 4], s[3 * s.count / 4]) }

/// vDSP reference (forward, unnormalised) for interleaved float2 batch.
func vdspRef(_ x: [Float], n: Int, batch: Int) -> [Float] {
    let l2 = vDSP_Length(Int(log2(Double(n))))
    let s = vDSP_create_fftsetup(l2, FFTRadix(kFFTRadix2))!
    defer { vDSP_destroy_fftsetup(s) }
    var re = [Float](repeating: 0, count: n), im = re, ore = re, oim = re
    var out = [Float](repeating: 0, count: 2 * n * batch)
    for b in 0..<batch {
        for i in 0..<n { re[i] = x[2 * (b * n + i)]; im[i] = x[2 * (b * n + i) + 1] }
        re.withUnsafeMutableBufferPointer { r in im.withUnsafeMutableBufferPointer { i in
        ore.withUnsafeMutableBufferPointer { orr in oim.withUnsafeMutableBufferPointer { oi in
            var si = DSPSplitComplex(realp: r.baseAddress!, imagp: i.baseAddress!)
            var so = DSPSplitComplex(realp: orr.baseAddress!, imagp: oi.baseAddress!)
            vDSP_fft_zop(s, &si, 1, &so, 1, l2, FFTDirection(kFFTDirection_Forward))
        }}}}
        for i in 0..<n { out[2 * (b * n + i)] = ore[i]; out[2 * (b * n + i) + 1] = oim[i] }
    }
    return out
}
func relL2(_ a: UnsafePointer<Float>, _ ref: [Float]) -> Double {
    var num = 0.0, den = 0.0
    for i in 0..<ref.count { let d = Double(a[i] - ref[i]); num += d * d; den += Double(ref[i]) * Double(ref[i]) }
    return (num / max(den, 1e-30)).squareRoot()
}

// ---- MPSGraph FFT ------------------------------------------------------------
@available(macOS 14.0, *)
final class MPSFFT {
    let graph = MPSGraph()
    let input: MPSGraphTensor
    let output: MPSGraphTensor
    let exe: MPSGraphExecutable
    init(dev: MTLDevice, n: Int, batch: Int, dtype: MPSDataType = .complexFloat32) {
        let shape: [NSNumber] = [NSNumber(value: batch), NSNumber(value: n)]
        input = graph.placeholder(shape: shape, dataType: dtype, name: nil)
        let d = MPSGraphFFTDescriptor(); d.inverse = false; d.scalingMode = .none
        output = graph.fastFourierTransform(input, axes: [1], descriptor: d, name: nil)
        let gd = MPSGraphDevice(mtlDevice: dev)
        exe = graph.compile(with: gd, feeds: [input: MPSGraphShapedType(shape: shape, dataType: dtype)],
                            targetTensors: [output], targetOperations: nil, compilationDescriptor: nil)
    }
}

// -----------------------------------------------------------------------------
@available(macOS 14.0, *)
func runM6Landscape() {
    let s = S2()
    let csvPath = CommandLine.arguments.count >= 3 ? CommandLine.arguments[2] : "m6_landscape.csv"
    var csv = "backend,variant,N,level,batch,footprint_MiB,mode,gflops_med,gflops_q1,gflops_q3,us_per_fft,gbps,relL2\n"
    let levels: [(String, Int)] = [("L2", 1 << 20), ("SLC", 8 << 20), ("DRAM", 256 << 20)]
    let sizes = (5...20).map { 1 << $0 }
    let maxBytes = 256 << 20
    let bufIn = s.dev.makeBuffer(length: maxBytes / 2 + (1 << 24), options: .storageModeShared)!
    let bufOut = s.dev.makeBuffer(length: maxBytes / 2 + (1 << 24), options: .storageModeShared)!
    let bufTmp = s.dev.makeBuffer(length: maxBytes / 2 + (1 << 24), options: .storageModeShared)!
    let inP = bufIn.contents().bindMemory(to: Float.self, capacity: bufIn.length / 4)
    for i in 0..<(bufIn.length / 4) { inP[i] = Float.random(in: -1...1) }

    print("=" * 110)
    print("m6-landscape  Device: \(s.dev.name)   (GF = 5N log2N / t;  HOT/COLD as in m6-batch)")
    print("=" * 110)
    print(String(format: "%-8@ %-22@ %7@ %-4@ %6@ %8@ | %9@ %9@ %8@ | %9@ | %8@",
                 "backend", "variant", "N", "lvl", "batch", "MiB", "HOT GF", "GB/s", "us/FFT", "COLD GF", "relL2"))
    print("-" * 110)

    func emit(_ backend: String, _ vname: String, _ n: Int, _ lvl: String, _ bs: Int, _ mode: String,
              _ gfs: [Double], _ tMs: Double, _ err: Double) {
        let (a, b) = m6Q13(gfs)
        let fp = Double(2 * bs * n * 8)
        csv += String(format: "%@,%@,%d,%@,%d,%.3f,%@,%.2f,%.2f,%.2f,%.4f,%.2f,%.2e\n",
                      backend, vname, n, lvl, bs, fp / 1048576, mode, m6Med(gfs), a, b,
                      tMs * 1e3 / Double(bs), fp / (tMs * 1e-3) / 1e9, err)
    }

    for n in sizes {
        let flops = 5.0 * Double(n) * log2(Double(n))
        for (lvl, target) in levels {
            let bs = max(1, target / (2 * n * 8))
            let fpBytes = 2 * bs * n * 8
            if fpBytes > maxBytes + (1 << 25) { continue }
            func gf(_ ms: Double) -> Double { flops * Double(bs) / (ms * 1e-3) / 1e9 }

            // reference for validation on the first min(bs,4) transforms
            let vb = min(bs, 4)
            let ref = vdspRef(Array(UnsafeBufferPointer(start: inP, count: 2 * n * vb)), n: n, batch: vb)

            // ---- ours
            for v in s.variants where v.n == n {
                func run(_ reps: Int) -> Double {
                    let cb = s.queue.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
                    for _ in 0..<reps { v.encode(e, bufIn, bufOut, bufTmp, bs) }
                    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
                    return (cb.gpuEndTime - cb.gpuStartTime) * 1e3
                }
                _ = run(1)
                let err = relL2(bufOut.contents().bindMemory(to: Float.self, capacity: ref.count), ref)
                let K = max(1, (64 << 20) / fpBytes)
                for _ in 0..<2 { _ = run(K); s.flush(); _ = run(1) }
                var hot: [Double] = [], cold: [Double] = []
                for _ in 0..<11 { hot.append(run(K) / Double(K)); s.flush(); cold.append(run(1)) }
                emit("ours", v.name, n, lvl, bs, "hot", hot.map(gf), m6Med(hot), err)
                emit("ours", v.name, n, lvl, bs, "cold", cold.map(gf), m6Med(cold), err)
                print(String(format: "%-8@ %-22@ %7d %-4@ %6d %8.2f | %9.1f %9.1f %8.3f | %9.1f | %8.1e%@",
                             "ours", v.name, n, lvl, bs, Double(fpBytes) / 1048576,
                             gf(m6Med(hot)), Double(fpBytes) / (m6Med(hot) * 1e-3) / 1e9, m6Med(hot) * 1e3 / Double(bs),
                             gf(m6Med(cold)), err, err > 1e-4 ? "  FAIL" : ""))
            }

            // ---- MPSGraph
            autoreleasepool {
                let m = MPSFFT(dev: s.dev, n: n, batch: bs)
                let shape: [NSNumber] = [NSNumber(value: bs), NSNumber(value: n)]
                let tdIn = MPSGraphTensorData(bufIn, shape: shape, dataType: .complexFloat32)
                let tdOut = MPSGraphTensorData(bufOut, shape: shape, dataType: .complexFloat32)
                let ed = MPSGraphExecutableExecutionDescriptor()
                func run(_ reps: Int) -> Double {
                    let cb = MPSCommandBuffer(from: s.queue)
                    for _ in 0..<reps {
                        _ = m.exe.encode(to: cb, inputs: [tdIn], results: [tdOut], executionDescriptor: ed)
                    }
                    cb.commit(); cb.waitUntilCompleted()
                    return (cb.commandBuffer.gpuEndTime - cb.commandBuffer.gpuStartTime) * 1e3
                }
                _ = run(1)
                let err = relL2(bufOut.contents().bindMemory(to: Float.self, capacity: ref.count), ref)
                let K = max(1, (64 << 20) / fpBytes)
                for _ in 0..<2 { _ = run(K); s.flush(); _ = run(1) }
                var hot: [Double] = [], cold: [Double] = []
                for _ in 0..<11 { hot.append(run(K) / Double(K)); s.flush(); cold.append(run(1)) }
                emit("mpsgraph", "fastFourierTransform", n, lvl, bs, "hot", hot.map(gf), m6Med(hot), err)
                emit("mpsgraph", "fastFourierTransform", n, lvl, bs, "cold", cold.map(gf), m6Med(cold), err)
                print(String(format: "%-8@ %-22@ %7d %-4@ %6d %8.2f | %9.1f %9.1f %8.3f | %9.1f | %8.1e%@",
                             "mpsgraph", "fastFourierTransform", n, lvl, bs, Double(fpBytes) / 1048576,
                             gf(m6Med(hot)), Double(fpBytes) / (m6Med(hot) * 1e-3) / 1e9, m6Med(hot) * 1e3 / Double(bs),
                             gf(m6Med(cold)), err, err > 1e-4 ? "  FAIL" : ""))
            }

            // ---- vDSP, 1 thread (split complex; working set = footprint)
            do {
                let l2 = vDSP_Length(Int(log2(Double(n))))
                let setup = vDSP_create_fftsetup(l2, FFTRadix(kFFTRadix2))!
                let cnt = n * bs
                let ri = UnsafeMutablePointer<Float>.allocate(capacity: cnt), ii = UnsafeMutablePointer<Float>.allocate(capacity: cnt)
                let ro = UnsafeMutablePointer<Float>.allocate(capacity: cnt), io = UnsafeMutablePointer<Float>.allocate(capacity: cnt)
                for i in 0..<cnt { ri[i] = inP[2 * i]; ii[i] = inP[2 * i + 1] }
                func once() -> Double {
                    let t0 = DispatchTime.now().uptimeNanoseconds
                    for b in 0..<bs {
                        var si = DSPSplitComplex(realp: ri + b * n, imagp: ii + b * n)
                        var so = DSPSplitComplex(realp: ro + b * n, imagp: io + b * n)
                        vDSP_fft_zop(setup, &si, 1, &so, 1, l2, FFTDirection(kFFTDirection_Forward))
                    }
                    return Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
                }
                // repeat so each sample is >= ~20 ms
                let one = max(once(), 1e-4)
                let R = max(1, Int(20.0 / one))
                var t: [Double] = []
                for _ in 0..<2 { for _ in 0..<R { _ = once() } }
                for _ in 0..<11 { var acc = 0.0; for _ in 0..<R { acc += once() }; t.append(acc / Double(R)) }
                emit("vdsp", "fft_zop_1thr", n, lvl, bs, "hot", t.map(gf), m6Med(t), 0)
                print(String(format: "%-8@ %-22@ %7d %-4@ %6d %8.2f | %9.1f %9.1f %8.3f | %9@ |",
                             "vdsp", "fft_zop_1thr", n, lvl, bs, Double(fpBytes) / 1048576,
                             gf(m6Med(t)), Double(fpBytes) / (m6Med(t) * 1e-3) / 1e9, m6Med(t) * 1e3 / Double(bs), "-"))
                ri.deallocate(); ii.deallocate(); ro.deallocate(); io.deallocate()
                vDSP_destroy_fftsetup(setup)
            }
        }
    }
    try! csv.write(toFile: csvPath, atomically: true, encoding: .utf8)
    print("=" * 110)
    print("CSV -> \(csvPath)")
}
