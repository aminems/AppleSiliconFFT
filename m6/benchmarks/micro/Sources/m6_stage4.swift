import Foundation
import Metal

// =============================================================================
// paper_m6 Stage 4 — GPU matrix units via Metal 4 tensor ops (MPP matmul2d).
//
//   m6-mma-peak   dense GEMM TFLOPS for several tile / type configs, square
//                 M = N = K in {1024, 2048, 4096}; validated on sampled entries.
// =============================================================================

@available(macOS 26.0, *)
func m6TensorLibrary(_ dev: MTLDevice) -> MTLLibrary {
    let o = MTLCompileOptions()
    o.languageVersion = .version4_0
    o.fastMathEnabled = true
    let src = try! String(contentsOfFile: try! FFTDeviceSource.path("src/metal/m6_tensor_ops.metal"), encoding: .utf8)
    do { return try dev.makeLibrary(source: src, options: o) }
    catch { fatalError("m6_tensor_ops.metal compile failed:\n\(error)") }
}

@available(macOS 26.0, *)
func runM6MMAPeak() {
    let dev = MTLCreateSystemDefaultDevice()!, queue = dev.makeCommandQueue()!
    let lib = m6TensorLibrary(dev)
    // (kernel, TM, TN, SG, inHalf, outHalf)
    let cfgs: [(String, Int, Int, Int, Bool, Bool)] = [
        ("gemm_h_f_64x32_sg4", 64, 32, 4, true, false),
        ("gemm_h_f_64x64_sg4", 64, 64, 4, true, false),
        ("gemm_h_f_128x64_sg4", 128, 64, 4, true, false),
        ("gemm_h_h_64x32_sg4", 64, 32, 4, true, true),
        ("gemm_f_f_64x32_sg4", 64, 32, 4, false, false),
        ("gemm_f_f_64x64_sg4", 64, 64, 4, false, false),
    ]
    print("=" * 96)
    print("m6-mma-peak: MPP matmul2d GEMM (Metal 4 tensor ops)  Device: \(dev.name)")
    print("reference: FMA peak FP32 4.38 TF, FP16 8.46 TF (m6-bw)")
    print("=" * 96)
    for (name, tm, tn, sg, inH, outH) in cfgs {
        let p = try! dev.makeComputePipelineState(function: lib.makeFunction(name: name)!)
        var line = String(format: "  %-22@", name as NSString)
        for s in [1024, 2048, 4096] {
            let eIn = inH ? 2 : 4, eOut = outH ? 2 : 4
            var a = [Float](repeating: 0, count: s * s), b = a
            for i in 0..<(s * s) { a[i] = Float.random(in: -1...1); b[i] = Float.random(in: -1...1) }
            func buf(_ x: [Float], half: Bool) -> MTLBuffer {
                if half {
                    let h = x.map { Float16($0) }
                    return dev.makeBuffer(bytes: h, length: h.count * 2, options: .storageModeShared)!
                }
                return dev.makeBuffer(bytes: x, length: x.count * 4, options: .storageModeShared)!
            }
            let bA = buf(a, half: inH), bB = buf(b, half: inH)
            let bC = dev.makeBuffer(length: s * s * eOut, options: .storageModeShared)!
            _ = eIn
            var mnk = SIMD3<UInt32>(UInt32(s), UInt32(s), UInt32(s))
            func run(_ reps: Int) -> Double {
                let cb = queue.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
                e.setComputePipelineState(p)
                e.setBuffer(bA, offset: 0, index: 0); e.setBuffer(bB, offset: 0, index: 1)
                e.setBuffer(bC, offset: 0, index: 2); e.setBytes(&mnk, length: 16, index: 3)
                for _ in 0..<reps {
                    e.dispatchThreadgroups(MTLSize(width: s / tn, height: s / tm, depth: 1),
                                           threadsPerThreadgroup: MTLSize(width: 32 * sg, height: 1, depth: 1))
                }
                e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
                return (cb.gpuEndTime - cb.gpuStartTime) * 1e3 / Double(reps)
            }
            _ = run(1)
            // validate 64 sampled entries against a CPU dot product (on the rounded inputs)
            let ra = inH ? a.map { Float(Float16($0)) } : a, rb = inH ? b.map { Float(Float16($0)) } : b
            var maxRel = 0.0
            for _ in 0..<64 {
                let i = Int.random(in: 0..<s), j = Int.random(in: 0..<s)
                var ref = 0.0
                for k in 0..<s { ref += Double(ra[i * s + k]) * Double(rb[k * s + j]) }
                let got: Double = outH
                    ? Double(bC.contents().bindMemory(to: Float16.self, capacity: s * s)[i * s + j])
                    : Double(bC.contents().bindMemory(to: Float.self, capacity: s * s)[i * s + j])
                maxRel = max(maxRel, abs(got - ref) / (Double(s).squareRoot() * 0.33))
            }
            let reps = max(1, Int(200.0 / max(run(1), 0.01)))
            var t: [Double] = []
            for _ in 0..<7 { t.append(run(reps)) }
            let tf = 2.0 * Double(s * s * s) / (m6Med(t) * 1e-3) / 1e12
            line += String(format: " | %4d: %6.2f TF (err %.1e)", s, tf, maxRel)
        }
        print(line)
    }
    print("  err = max |C - C_ref| / (sqrt(K) * rms(a*b)) over 64 sampled entries")
    print("=" * 96)
}

// -----------------------------------------------------------------------------
// m6-tensor-fft [csv]: FFT-4096 on the matrix units (fft4096_tensor_f16, three
// radix-16 stages as 32x32 . 32x256 fp16 matmuls in threadgroup memory) vs the
// ALU kernels fft4096_dx_f16 (fp16 storage) and fft4096_dx (fp32).
@available(macOS 26.0, *)
func runM6TensorFFT() {
    let dev = MTLCreateSystemDefaultDevice()!, queue = dev.makeCommandQueue()!
    let tlib = m6TensorLibrary(dev)
    let o = MTLCompileOptions(); o.fastMathEnabled = true
    let dx = try! dev.makeLibrary(source: loadCombinedSource(), options: o)
    func P(_ l: MTLLibrary, _ n: String) -> MTLComputePipelineState {
        try! dev.makeComputePipelineState(function: l.makeFunction(name: n)!)
    }
    let fl = try! dev.makeLibrary(source: """
        #include <metal_stdlib>
        using namespace metal;
        kernel void flush_write(device float4* d [[buffer(0)]], constant uint& n [[buffer(1)]],
                                uint g [[thread_position_in_grid]], uint s [[threads_per_grid]]) {
            for (uint i = g; i < n; i += s) d[i] = float4(float(i)); }
        """, options: o)
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
    // W16 = [[Fr, -Fi], [Fi, Fr]], F[t'][t] = exp(-2 pi i t t' / 16)
    var w = [Float16](repeating: 0, count: 32 * 32)
    for tp in 0..<16 { for t in 0..<16 {
        let a = -2.0 * Double.pi * Double(t * tp) / 16.0
        let fr = Float16(cos(a)), fi = Float16(sin(a))
        w[tp * 32 + t] = fr; w[tp * 32 + 16 + t] = -fi
        w[(16 + tp) * 32 + t] = fi; w[(16 + tp) * 32 + 16 + t] = fr
    } }
    let wBuf = dev.makeBuffer(bytes: w, length: w.count * 2, options: .storageModeShared)!
    let kernels: [(String, MTLComputePipelineState, Int, Bool)] = [
        ("fft4096_tensor_f16", P(tlib, "fft4096_tensor_f16"), 256, true),
        ("probe_mm_only", P(tlib, "fft4096_tensor_f16_mmonly"), 256, true),
        ("fft4096_dx_f16", P(dx, "fft4096_dx_f16"), 512, true),
        ("fft4096_dx", P(dx, "fft4096_dx"), 512, false),
    ]
    let n = 4096, maxB = 4096
    let fIn = dev.makeBuffer(length: maxB * n * 8, options: .storageModeShared)!
    let fOut = dev.makeBuffer(length: maxB * n * 8, options: .storageModeShared)!
    let hIn = dev.makeBuffer(length: maxB * n * 4, options: .storageModeShared)!
    let hOut = dev.makeBuffer(length: maxB * n * 4, options: .storageModeShared)!
    let fp = fIn.contents().bindMemory(to: Float.self, capacity: maxB * n * 2)
    let hp = hIn.contents().bindMemory(to: Float16.self, capacity: maxB * n * 2)
    for i in 0..<(maxB * n * 2) { let v = Float.random(in: -1...1); fp[i] = v; hp[i] = Float16(v) }
    let flops = 5.0 * Double(n) * 12.0
    let csvPath = CommandLine.arguments.count >= 3 ? CommandLine.arguments[2] : "m6_tensor_fft.csv"
    var csv = "backend,variant,N,level,batch,footprint_MiB,mode,gflops_med,gflops_q1,gflops_q3,us_per_fft,gbps,relL2\n"
    print("=" * 100)
    print("m6-tensor-fft  N=4096  Device: \(dev.name)   (batch fixed by fp32 footprint target)")
    print("=" * 100)
    print(String(format: "%-20@ %-4@ %5@ | %9@ %8@ | %9@ | %9@ %6@", "variant", "lvl", "batch",
                 "HOT GF", "GB/s", "COLD GF", "relL2", "SQNR"))
    print("-" * 100)
    for (lvl, bs) in [("L2", 16), ("SLC", 128), ("DRAM", 4096)] {
        let vb = 4
        let ref32 = vdspRef(Array(UnsafeBufferPointer(start: fp, count: 2 * n * vb)), n: n, batch: vb)
        let ref16 = vdspRef((0..<(2 * n * vb)).map { Float(hp[$0]) }, n: n, batch: vb)
        for (name, p, thr, half) in kernels {
            let inB = half ? hIn : fIn, outB = half ? hOut : fOut
            let bytes = 2 * bs * n * (half ? 4 : 8)
            func run(_ reps: Int) -> Double {
                let cb = queue.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
                e.setComputePipelineState(p)
                e.setBuffer(inB, offset: 0, index: 0); e.setBuffer(outB, offset: 0, index: 1)
                e.setBuffer(wBuf, offset: 0, index: 2)
                for _ in 0..<reps {
                    e.dispatchThreadgroups(MTLSize(width: bs, height: 1, depth: 1),
                                           threadsPerThreadgroup: MTLSize(width: thr, height: 1, depth: 1))
                }
                e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
                return (cb.gpuEndTime - cb.gpuStartTime) * 1e3
            }
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
            let K = max(1, (64 << 20) / bytes)
            for _ in 0..<2 { _ = run(K); flush(); _ = run(1) }
            var hot: [Double] = [], cold: [Double] = []
            for _ in 0..<15 { hot.append(run(K) / Double(K)); flush(); cold.append(run(1)) }
            func gf(_ ms: Double) -> Double { flops * Double(bs) / (ms * 1e-3) / 1e9 }
            for (mode, t) in [("hot", hot), ("cold", cold)] {
                let g = t.map(gf), (a, b) = m6Q13(g)
                csv += String(format: "ours,%@,%d,%@,%d,%.3f,%@,%.2f,%.2f,%.2f,%.4f,%.2f,%.2e\n",
                              name, n, lvl, bs, Double(bytes) / 1048576, mode, m6Med(g), a, b,
                              m6Med(t) * 1e3 / Double(bs), Double(bytes) / (m6Med(t) * 1e-3) / 1e9, err)
            }
            print(String(format: "%-20@ %-4@ %5d | %9.1f %8.1f | %9.1f | %9.1e %6.1f", name, lvl, bs,
                         gf(m6Med(hot)), Double(bytes) / (m6Med(hot) * 1e-3) / 1e9, gf(m6Med(cold)),
                         err, -20 * log10(err)))
        }
    }
    try! csv.write(toFile: csvPath, atomically: true, encoding: .utf8)
    print("=" * 100)
}
