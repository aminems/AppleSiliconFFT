import Accelerate
import Foundation
import Metal
import MetalPerformanceShadersGraph

// =============================================================================
// paper_m6 Stage 3 — one-DRAM-pass N = 8192 / 16384 vs two-pass baselines.
//
//   m6-onepass [csv-path]
//
// Kernels: src/metal/fft_onepass_large.metal (register-resident block, 32 KiB
// threadgroup memory used only as a component-wise exchange buffer).
// Baselines at the same N: MPSGraph fastFourierTransform, and for 16384 the
// repo's two-dispatch four-step (fourstep16384_A_b + _B_b).
// Same footprint levels / HOT / COLD protocol as m6-landscape.
// =============================================================================

@available(macOS 14.0, *)
func runM6OnePass() {
    let dev = MTLCreateSystemDefaultDevice()!
    let queue = dev.makeCommandQueue()!
    let o = MTLCompileOptions(); o.fastMathEnabled = true
    let header = try! String(contentsOfFile: try! FFTDeviceSource.path("src/common/fft_device.h"), encoding: .utf8)
    let onepassSrc = try! String(contentsOfFile: try! FFTDeviceSource.path("src/metal/fft_onepass_large.metal"), encoding: .utf8)
    let lib1: MTLLibrary
    do { lib1 = try dev.makeLibrary(source: header + "\n" + onepassSrc, options: o) }
    catch { fatalError("compile fft_onepass_large.metal failed:\n\(error)") }
    let dx = try! dev.makeLibrary(source: loadCombinedSource(), options: o)
    let fl = try! dev.makeLibrary(source: """
        #include <metal_stdlib>
        using namespace metal;
        kernel void flush_write(device float4* d [[buffer(0)]], constant uint& n [[buffer(1)]],
                                uint g [[thread_position_in_grid]], uint s [[threads_per_grid]]) {
            for (uint i = g; i < n; i += s) d[i] = float4(float(i)); }
        """, options: o)
    func P(_ l: MTLLibrary, _ n: String) -> MTLComputePipelineState {
        try! dev.makeComputePipelineState(function: l.makeFunction(name: n)!)
    }
    let flushP = P(fl, "flush_write")
    let flushBytes = 512 << 20
    let flushBuf = dev.makeBuffer(length: flushBytes, options: .storageModePrivate)!
    func flush() {
        let cb = queue.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
        e.setComputePipelineState(flushP); e.setBuffer(flushBuf, offset: 0, index: 0)
        var n = UInt32(flushBytes / 16); e.setBytes(&n, length: 4, index: 1)
        e.dispatchThreads(MTLSize(width: 12 * 1024 * 32, height: 1, depth: 1),
                          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    }

    typealias Enc = (MTLComputeCommandEncoder, MTLBuffer, MTLBuffer, MTLBuffer, Int) -> Void
    func single(_ p: MTLComputePipelineState, _ threads: Int) -> Enc {
        return { e, a, b, _, bs in
            e.setComputePipelineState(p); e.setBuffer(a, offset: 0, index: 0); e.setBuffer(b, offset: 0, index: 1)
            e.dispatchThreadgroups(MTLSize(width: bs, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        }
    }
    let p8k = P(lib1, "fft8192_onepass"), p16k = P(lib1, "fft16384_onepass")
    let pA = P(dx, "fourstep16384_A_b"), pB = P(dx, "fourstep16384_B_b")
    let impls: [(String, Int, Enc)] = [
        ("onepass8192", 8192, single(p8k, 512)),
        ("onepass16384", 16384, single(p16k, 1024)),
        ("fourstep16384_b(2 pass)", 16384, { e, a, b, tmp, bs in
            single(pA, 512)(e, a, tmp, tmp, 4 * bs)
            e.memoryBarrier(scope: .buffers)
            e.setComputePipelineState(pB); e.setBuffer(tmp, offset: 0, index: 0); e.setBuffer(b, offset: 0, index: 1)
            e.dispatchThreadgroups(MTLSize(width: 8 * bs, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1))
        }),
    ]


    print("=" * 110)
    print("m6-onepass  Device: \(dev.name)")
    for (name, p) in [("fft8192_onepass", p8k), ("fft16384_onepass", p16k)] {
        print("  \(name): maxTotalThreadsPerThreadgroup=\(p.maxTotalThreadsPerThreadgroup)  "
              + "execWidth=\(p.threadExecutionWidth)  staticTGmem=\(p.staticThreadgroupMemoryLength) B")
    }
    print("=" * 110)

    let csvPath = CommandLine.arguments.count >= 3 ? CommandLine.arguments[2] : "m6_onepass.csv"
    var csv = "backend,variant,N,level,batch,footprint_MiB,mode,gflops_med,gflops_q1,gflops_q3,us_per_fft,gbps,relL2\n"
    let maxHalf = (128 << 20) + (1 << 24)
    let bufIn = dev.makeBuffer(length: maxHalf, options: .storageModeShared)!
    let bufOut = dev.makeBuffer(length: maxHalf, options: .storageModeShared)!
    let bufTmp = dev.makeBuffer(length: maxHalf, options: .storageModeShared)!
    let inP = bufIn.contents().bindMemory(to: Float.self, capacity: maxHalf / 4)
    for i in 0..<(maxHalf / 4) { inP[i] = Float.random(in: -1...1) }
    let levels: [(String, Int)] = [("L2", 1 << 20), ("SLC", 8 << 20), ("DRAM", 256 << 20)]

    print(String(format: "%-26@ %6@ %-4@ %6@ %8@ | %9@ %9@ %8@ | %9@ %8@ | %8@",
                 "variant", "N", "lvl", "batch", "MiB", "HOT GF", "GB/s", "us/FFT", "COLD GF", "GB/s", "relL2"))
    print("-" * 110)
    for n in [8192, 16384] {
        let flops = 5.0 * Double(n) * log2(Double(n))
        for (lvl, target) in levels {
            let bs = max(1, target / (2 * n * 8))
            let fp = 2 * bs * n * 8
            func gf(_ ms: Double) -> Double { flops * Double(bs) / (ms * 1e-3) / 1e9 }
            let vb = min(bs, 4)
            let ref = vdspRef(Array(UnsafeBufferPointer(start: inP, count: 2 * n * vb)), n: n, batch: vb)

            func measure(_ name: String, _ backend: String, run: (Int) -> Double) {
                memset(bufOut.contents(), 0, 2 * n * vb * 4)
                _ = run(1)
                let err = relL2(bufOut.contents().bindMemory(to: Float.self, capacity: ref.count), ref)
                let K = max(1, (64 << 20) / fp)
                for _ in 0..<2 { _ = run(K); flush(); _ = run(1) }
                var hot: [Double] = [], cold: [Double] = []
                for _ in 0..<15 { hot.append(run(K) / Double(K)); flush(); cold.append(run(1)) }
                for (mode, t) in [("hot", hot), ("cold", cold)] {
                    let g = t.map(gf), (a, b) = m6Q13(g)
                    csv += String(format: "%@,%@,%d,%@,%d,%.3f,%@,%.2f,%.2f,%.2f,%.4f,%.2f,%.2e\n",
                                  backend, name, n, lvl, bs, Double(fp) / 1048576, mode, m6Med(g), a, b,
                                  m6Med(t) * 1e3 / Double(bs), Double(fp) / (m6Med(t) * 1e-3) / 1e9, err)
                }
                print(String(format: "%-26@ %6d %-4@ %6d %8.2f | %9.1f %9.1f %8.3f | %9.1f %8.1f | %8.1e%@",
                             name, n, lvl, bs, Double(fp) / 1048576,
                             gf(m6Med(hot)), Double(fp) / (m6Med(hot) * 1e-3) / 1e9, m6Med(hot) * 1e3 / Double(bs),
                             gf(m6Med(cold)), Double(fp) / (m6Med(cold) * 1e-3) / 1e9, err, err > 1e-4 ? "  FAIL" : ""))
            }

            for (name, nn, enc) in impls where nn == n {
                measure(name, "ours") { reps in
                    let cb = queue.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
                    for _ in 0..<reps { enc(e, bufIn, bufOut, bufTmp, bs) }
                    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
                    return (cb.gpuEndTime - cb.gpuStartTime) * 1e3
                }
            }
            autoreleasepool {
                let m = MPSFFT(dev: dev, n: n, batch: bs)
                let shape: [NSNumber] = [NSNumber(value: bs), NSNumber(value: n)]
                let tdIn = MPSGraphTensorData(bufIn, shape: shape, dataType: .complexFloat32)
                let tdOut = MPSGraphTensorData(bufOut, shape: shape, dataType: .complexFloat32)
                let ed = MPSGraphExecutableExecutionDescriptor()
                measure("MPSGraph", "mpsgraph") { reps in
                    let cb = MPSCommandBuffer(from: queue)
                    for _ in 0..<reps { _ = m.exe.encode(to: cb, inputs: [tdIn], results: [tdOut], executionDescriptor: ed) }
                    cb.commit(); cb.waitUntilCompleted()
                    return (cb.commandBuffer.gpuEndTime - cb.commandBuffer.gpuStartTime) * 1e3
                }
            }
        }
    }
    try! csv.write(toFile: csvPath, atomically: true, encoding: .utf8)
    print("=" * 110)
    print("single-pass DRAM ceiling at 150 GB/s: N=8192 -> \(Int(150 * 5 * 13 / 16)) GF, N=16384 -> \(Int(150 * 5 * 14 / 16)) GF")
    print("CSV -> \(csvPath)")
}

/// Correctness-only check of every kernel in fft_onepass_large.metal (batch 4).
func runM6OnePassCheck() {
    let dev = MTLCreateSystemDefaultDevice()!, queue = dev.makeCommandQueue()!
    let o = MTLCompileOptions(); o.fastMathEnabled = true
    let src = try! String(contentsOfFile: try! FFTDeviceSource.path("src/common/fft_device.h"), encoding: .utf8) + "\n"
        + (try! String(contentsOfFile: try! FFTDeviceSource.path("src/metal/fft_onepass_large.metal"), encoding: .utf8))
    let lib = try! dev.makeLibrary(source: src, options: o)
    let kernels: [(String, Int, Int)] = [("fft8192_onepass", 8192, 512), ("fft8192_onepass_w2dbg", 8192, 512),
                                         ("fft16384_onepass", 16384, 1024),
                                         ("fft16384_onepass_t512", 16384, 512)]
    for (name, n, t) in kernels {
        guard let f = lib.makeFunction(name: name) else { print("  (missing \(name))"); continue }
        let p = try! dev.makeComputePipelineState(function: f)
        let bs = 4
        var x = [Float](repeating: 0, count: 2 * n * bs)
        for i in 0..<x.count { x[i] = Float.random(in: -1...1) }
        let a = dev.makeBuffer(bytes: x, length: x.count * 4, options: .storageModeShared)!
        let b = dev.makeBuffer(length: x.count * 4, options: .storageModeShared)!
        let cb = queue.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
        e.setComputePipelineState(p); e.setBuffer(a, offset: 0, index: 0); e.setBuffer(b, offset: 0, index: 1)
        e.dispatchThreadgroups(MTLSize(width: bs, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: t, height: 1, depth: 1))
        e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let ref = vdspRef(x, n: n, batch: bs)
        let out = b.contents().bindMemory(to: Float.self, capacity: x.count)
        // locate the first wrong bin to help debugging
        var firstBad = -1
        for i in 0..<(n) where firstBad < 0 {
            let dr = out[2*i] - ref[2*i], di = out[2*i+1] - ref[2*i+1]
            if (dr*dr + di*di).squareRoot() > 1e-3 * Float(n).squareRoot() { firstBad = i }
        }
        print(String(format: "  %-24@ N=%6d relL2 %.2e  firstBadBin %d  maxThreads %d", name as NSString, n,
                     relL2(out, ref), firstBad, p.maxTotalThreadsPerThreadgroup))
    }
}

// -----------------------------------------------------------------------------
// m6-fp16 [csv]: fp16 STORAGE (half2 in device memory, fp32 arithmetic) vs fp32,
// same batch (same FFT count) so the fp16 footprint is half the fp32 one.
// Accuracy: rel L2 vs vDSP run on the fp16-rounded input (= error added by the
// transform + fp16 output rounding), reported also as SQNR dB.
@available(macOS 14.0, *)
func runM6FP16() {
    let dev = MTLCreateSystemDefaultDevice()!, queue = dev.makeCommandQueue()!
    let o = MTLCompileOptions(); o.fastMathEnabled = true
    let one = try! dev.makeLibrary(source:
        (try! String(contentsOfFile: try! FFTDeviceSource.path("src/common/fft_device.h"), encoding: .utf8)) + "\n"
        + (try! String(contentsOfFile: try! FFTDeviceSource.path("src/metal/fft_onepass_large.metal"), encoding: .utf8)), options: o)
    let dx = try! dev.makeLibrary(source: loadCombinedSource(), options: o)
    let fl = try! dev.makeLibrary(source: """
        #include <metal_stdlib>
        using namespace metal;
        kernel void flush_write(device float4* d [[buffer(0)]], constant uint& n [[buffer(1)]],
                                uint g [[thread_position_in_grid]], uint s [[threads_per_grid]]) {
            for (uint i = g; i < n; i += s) d[i] = float4(float(i)); }
        """, options: o)
    func P(_ l: MTLLibrary, _ n: String) -> MTLComputePipelineState {
        try! dev.makeComputePipelineState(function: l.makeFunction(name: n)!)
    }
    let flushP = P(fl, "flush_write"), flushBytes = 512 << 20
    let flushBuf = dev.makeBuffer(length: flushBytes, options: .storageModePrivate)!
    func flush() {
        let cb = queue.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
        e.setComputePipelineState(flushP); e.setBuffer(flushBuf, offset: 0, index: 0)
        var n = UInt32(flushBytes / 16); e.setBytes(&n, length: 4, index: 1)
        e.dispatchThreads(MTLSize(width: 12 * 1024 * 32, height: 1, depth: 1),
                          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    }
    // (name, N, pipeline, threads, isHalf)
    let kernels: [(String, Int, MTLComputePipelineState, Int, Bool)] = [
        ("fft4096_dx", 4096, P(dx, "fft4096_dx"), 512, false),
        ("fft4096_dx_f16", 4096, P(dx, "fft4096_dx_f16"), 512, true),
        ("onepass8192", 8192, P(one, "fft8192_onepass"), 512, false),
        ("onepass8192_f16", 8192, P(one, "fft8192_onepass_f16"), 512, true),
        ("onepass16384", 16384, P(one, "fft16384_onepass"), 1024, false),
        ("onepass16384_f16", 16384, P(one, "fft16384_onepass_f16"), 1024, true),
        ("onepass16384_t512", 16384, P(one, "fft16384_onepass_t512"), 512, false),
        ("onepass16384_t512_f16", 16384, P(one, "fft16384_onepass_t512_f16"), 512, true),
    ]
    let csvPath = CommandLine.arguments.count >= 3 ? CommandLine.arguments[2] : "m6_fp16.csv"
    var csv = "backend,variant,N,level,batch,footprint_MiB,mode,gflops_med,gflops_q1,gflops_q3,us_per_fft,gbps,relL2\n"
    let maxHalf = (128 << 20) + (1 << 24)
    let bufIn = dev.makeBuffer(length: maxHalf, options: .storageModeShared)!
    let bufOut = dev.makeBuffer(length: maxHalf, options: .storageModeShared)!
    let bufInH = dev.makeBuffer(length: maxHalf / 2, options: .storageModeShared)!
    let bufOutH = dev.makeBuffer(length: maxHalf / 2, options: .storageModeShared)!
    let fIn = bufIn.contents().bindMemory(to: Float.self, capacity: maxHalf / 4)
    let hIn = bufInH.contents().bindMemory(to: Float16.self, capacity: maxHalf / 4)
    for i in 0..<(maxHalf / 4) { let v = Float.random(in: -1...1); fIn[i] = v; hIn[i] = Float16(v) }
    let levels: [(String, Int)] = [("L2", 1 << 20), ("SLC", 8 << 20), ("DRAM", 256 << 20)]

    print("=" * 112)
    print("m6-fp16  Device: \(dev.name)  fp16 = half2 device storage, fp32 math; batch fixed by the fp32 footprint target")
    print("=" * 112)
    print(String(format: "%-22@ %6@ %-4@ %6@ %8@ | %9@ %8@ | %9@ %8@ | %9@ %6@",
                 "variant", "N", "lvl", "batch", "MiB", "HOT GF", "GB/s", "COLD GF", "GB/s", "relL2", "SQNR"))
    print("-" * 112)
    for n in [4096, 8192, 16384] {
        let flops = 5.0 * Double(n) * log2(Double(n))
        for (lvl, target) in levels {
            let bs = max(1, target / (2 * n * 8))
            func gf(_ ms: Double) -> Double { flops * Double(bs) / (ms * 1e-3) / 1e9 }
            let vb = min(bs, 4)
            let ref32 = vdspRef(Array(UnsafeBufferPointer(start: fIn, count: 2 * n * vb)), n: n, batch: vb)
            let ref16 = vdspRef((0..<(2 * n * vb)).map { Float(hIn[$0]) }, n: n, batch: vb)

            func measure(_ name: String, _ backend: String, half: Bool, run: (Int) -> Double) {
                let fp = 2 * bs * n * (half ? 4 : 8)
                let outB = half ? bufOutH : bufOut
                memset(outB.contents(), 0, 2 * n * vb * (half ? 2 : 4))
                _ = run(1)
                let err: Double
                if half {
                    let h = outB.contents().bindMemory(to: Float16.self, capacity: 2 * n * vb)
                    let f = (0..<(2 * n * vb)).map { Float(h[$0]) }
                    err = f.withUnsafeBufferPointer { relL2($0.baseAddress!, ref16) }
                } else {
                    err = relL2(outB.contents().bindMemory(to: Float.self, capacity: 2 * n * vb), ref32)
                }
                let K = max(1, (64 << 20) / fp)
                for _ in 0..<2 { _ = run(K); flush(); _ = run(1) }
                var hot: [Double] = [], cold: [Double] = []
                for _ in 0..<15 { hot.append(run(K) / Double(K)); flush(); cold.append(run(1)) }
                for (mode, t) in [("hot", hot), ("cold", cold)] {
                    let g = t.map(gf), (a, b) = m6Q13(g)
                    csv += String(format: "%@,%@,%d,%@,%d,%.3f,%@,%.2f,%.2f,%.2f,%.4f,%.2f,%.2e\n",
                                  backend, name, n, lvl, bs, Double(fp) / 1048576, mode, m6Med(g), a, b,
                                  m6Med(t) * 1e3 / Double(bs), Double(fp) / (m6Med(t) * 1e-3) / 1e9, err)
                }
                print(String(format: "%-22@ %6d %-4@ %6d %8.2f | %9.1f %8.1f | %9.1f %8.1f | %9.1e %6.1f",
                             name, n, lvl, bs, Double(fp) / 1048576,
                             gf(m6Med(hot)), Double(fp) / (m6Med(hot) * 1e-3) / 1e9,
                             gf(m6Med(cold)), Double(fp) / (m6Med(cold) * 1e-3) / 1e9, err, -20 * log10(err)))
            }
            for (name, nn, p, t, half) in kernels where nn == n {
                measure(name, "ours", half: half) { reps in
                    let cb = queue.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
                    e.setComputePipelineState(p)
                    e.setBuffer(half ? bufInH : bufIn, offset: 0, index: 0)
                    e.setBuffer(half ? bufOutH : bufOut, offset: 0, index: 1)
                    for _ in 0..<reps {
                        e.dispatchThreadgroups(MTLSize(width: bs, height: 1, depth: 1),
                                               threadsPerThreadgroup: MTLSize(width: t, height: 1, depth: 1))
                    }
                    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
                    return (cb.gpuEndTime - cb.gpuStartTime) * 1e3
                }
            }
            for half in [false, true] {
                autoreleasepool {
                    let dt: MPSDataType = half ? .complexFloat16 : .complexFloat32
                    let m = MPSFFT(dev: dev, n: n, batch: bs, dtype: dt)
                    let shape: [NSNumber] = [NSNumber(value: bs), NSNumber(value: n)]
                    let tdIn = MPSGraphTensorData(half ? bufInH : bufIn, shape: shape, dataType: dt)
                    let tdOut = MPSGraphTensorData(half ? bufOutH : bufOut, shape: shape, dataType: dt)
                    let ed = MPSGraphExecutableExecutionDescriptor()
                    measure(half ? "MPSGraph_f16" : "MPSGraph", "mpsgraph", half: half) { reps in
                        let cb = MPSCommandBuffer(from: queue)
                        for _ in 0..<reps { _ = m.exe.encode(to: cb, inputs: [tdIn], results: [tdOut], executionDescriptor: ed) }
                        cb.commit(); cb.waitUntilCompleted()
                        return (cb.commandBuffer.gpuEndTime - cb.commandBuffer.gpuStartTime) * 1e3
                    }
                }
            }
        }
    }
    try! csv.write(toFile: csvPath, atomically: true, encoding: .utf8)
    print("=" * 112)
    print("fp16-storage DRAM ceilings at 150 GB/s (8 B/point): N=4096 \(Int(150 * 5 * 12 / 8)) GF, 8192 \(Int(150 * 5 * 13 / 8)), 16384 \(Int(150 * 5 * 14 / 8))")
    print("CSV -> \(csvPath)")
}
