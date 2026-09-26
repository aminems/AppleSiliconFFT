import Accelerate
import Darwin
import Foundation
import Metal

// =============================================================================
// paper_m6 Stage 1 — device characterisation on Apple M6.
//
//   m6-bw     GPU streaming bandwidth vs working set (hot: same buffers reused
//             back-to-back, so anything that fits the SLC is served from it)
//             + GPU peak FP32/FP16 FMA throughput.
//   m6-batch  fft4096_dx batch sweep, HOT (buffers reused, back-to-back
//             dispatches in one command buffer) vs COLD (a 512 MiB flush
//             kernel runs before every timed FFT command buffer).
//   m6-cpu    vDSP_fft_zop N=4096 throughput per thread for T concurrent
//             threads at a given QoS; the CPU number each thread ran on is
//             sampled with pthread_cpu_number_np so tiers can be identified.
// =============================================================================

private let stage1Src = """
#include <metal_stdlib>
using namespace metal;

kernel void bw_copy(device const float4* src [[buffer(0)]],
                    device float4*       dst [[buffer(1)]],
                    constant uint&       n   [[buffer(2)]],
                    uint gid [[thread_position_in_grid]],
                    uint gsz [[threads_per_grid]]) {
    for (uint i = gid; i < n; i += gsz) dst[i] = src[i];
}

kernel void bw_read(device const float4* src [[buffer(0)]],
                    device float*        sink [[buffer(1)]],
                    constant uint&       n   [[buffer(2)]],
                    uint gid [[thread_position_in_grid]],
                    uint gsz [[threads_per_grid]]) {
    float4 acc = 0;
    for (uint i = gid; i < n; i += gsz) acc += src[i];
    if (acc.x == 1234.5f) sink[0] = acc.y;   // never true; defeats DCE
}

kernel void flush_write(device float4* dst [[buffer(0)]],
                        constant uint& n   [[buffer(1)]],
                        uint gid [[thread_position_in_grid]],
                        uint gsz [[threads_per_grid]]) {
    for (uint i = gid; i < n; i += gsz) dst[i] = float4(float(i));
}

// 8 independent float4 FMA chains x ITERS -> 8*4*2*ITERS flops/thread
kernel void fma_f32(device float* out [[buffer(0)]],
                    constant uint& iters [[buffer(1)]],
                    uint gid [[thread_position_in_grid]]) {
    float4 a0=float(gid)*1e-9f, a1=a0+1, a2=a0+2, a3=a0+3, a4=a0+4, a5=a0+5, a6=a0+6, a7=a0+7;
    const float4 m = 0.999f, c = 1e-7f;
    for (uint i = 0; i < iters; ++i) {
        a0=fma(a0,m,c); a1=fma(a1,m,c); a2=fma(a2,m,c); a3=fma(a3,m,c);
        a4=fma(a4,m,c); a5=fma(a5,m,c); a6=fma(a6,m,c); a7=fma(a7,m,c);
    }
    float4 s = a0+a1+a2+a3+a4+a5+a6+a7;
    out[gid] = s.x+s.y+s.z+s.w;
}

kernel void fma_f16(device half* out [[buffer(0)]],
                    constant uint& iters [[buffer(1)]],
                    uint gid [[thread_position_in_grid]]) {
    half4 a0=half(gid&7)*0.01h, a1=a0+1, a2=a0+2, a3=a0+3, a4=a0+4, a5=a0+5, a6=a0+6, a7=a0+7;
    const half4 m = 0.999h, c = 0.0001h;
    for (uint i = 0; i < iters; ++i) {
        a0=fma(a0,m,c); a1=fma(a1,m,c); a2=fma(a2,m,c); a3=fma(a3,m,c);
        a4=fma(a4,m,c); a5=fma(a5,m,c); a6=fma(a6,m,c); a7=fma(a7,m,c);
    }
    half4 s = a0+a1+a2+a3+a4+a5+a6+a7;
    out[gid] = s.x+s.y+s.z+s.w;
}
"""

private struct S1GPU {
    let dev: MTLDevice
    let queue: MTLCommandQueue
    let lib: MTLLibrary
    init(extraSource: String = "") {
        dev = MTLCreateSystemDefaultDevice()!
        queue = dev.makeCommandQueue()!
        let o = MTLCompileOptions(); o.fastMathEnabled = true
        lib = try! dev.makeLibrary(source: extraSource.isEmpty ? stage1Src : extraSource, options: o)
    }
    func pipe(_ name: String, _ l: MTLLibrary? = nil) -> MTLComputePipelineState {
        try! dev.makeComputePipelineState(function: (l ?? lib).makeFunction(name: name)!)
    }
}

private func median(_ x: [Double]) -> Double { let s = x.sorted(); return s[s.count / 2] }
private func quart(_ x: [Double]) -> (Double, Double) {
    let s = x.sorted(); return (s[s.count / 4], s[(3 * s.count) / 4])
}

/// Encode `reps` back-to-back dispatches in ONE command buffer; returns GPU ms total.
private func gpuRun(_ g: S1GPU, _ p: MTLComputePipelineState, reps: Int,
                    grid: MTLSize, tg: MTLSize, byThreadgroups: Bool,
                    bind: (MTLComputeCommandEncoder) -> Void) -> Double {
    let cb = g.queue.makeCommandBuffer()!
    let e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(p)
    bind(e)
    for _ in 0..<reps {
        if byThreadgroups { e.dispatchThreadgroups(grid, threadsPerThreadgroup: tg) }
        else { e.dispatchThreads(grid, threadsPerThreadgroup: tg) }
    }
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    return (cb.gpuEndTime - cb.gpuStartTime) * 1e3
}

// -----------------------------------------------------------------------------
func runM6Bandwidth() {
    let g = S1GPU()
    print("=" * 84)
    print("m6-bw: GPU streaming bandwidth vs working set (hot, buffers reused)  Device: \(g.dev.name)")
    print("=" * 84)
    let copyP = g.pipe("bw_copy"), readP = g.pipe("bw_read")
    let maxBytes = 1 << 30
    let src = g.dev.makeBuffer(length: maxBytes, options: .storageModePrivate)!
    let dst = g.dev.makeBuffer(length: maxBytes, options: .storageModePrivate)!
    let sink = g.dev.makeBuffer(length: 64, options: .storageModeShared)!
    // init src
    _ = gpuRun(g, g.pipe("flush_write"), reps: 1, grid: MTLSize(width: 12 * 1024 * 64, height: 1, depth: 1),
               tg: MTLSize(width: 256, height: 1, depth: 1), byThreadgroups: false) { e in
        e.setBuffer(src, offset: 0, index: 0); var n = UInt32(maxBytes / 16); e.setBytes(&n, length: 4, index: 1)
    }
    let threads = 12 * 1024 * 16   // enough in-flight loads to saturate
    print("  WS/buffer  | copy GB/s (rd+wr)  med [q1,q3]   | read GB/s  med [q1,q3]")
    print("  " + "-" * 76)
    var ws = 256 << 10
    var rows: [String] = []
    while ws <= maxBytes {
        let n = UInt32(ws / 16)
        let reps = max(1, (4 << 30) / ws)          // >= 4 GiB of traffic per sample
        func sample(_ p: MTLComputePipelineState, _ copy: Bool) -> [Double] {
            var r: [Double] = []
            _ = gpuRun(g, p, reps: max(1, reps / 4), grid: MTLSize(width: min(threads, Int(n)), height: 1, depth: 1),
                       tg: MTLSize(width: 256, height: 1, depth: 1), byThreadgroups: false) { e in
                e.setBuffer(src, offset: 0, index: 0); e.setBuffer(copy ? dst : sink, offset: 0, index: 1)
                var nn = n; e.setBytes(&nn, length: 4, index: 2)
            }
            for _ in 0..<9 {
                let ms = gpuRun(g, p, reps: reps, grid: MTLSize(width: min(threads, Int(n)), height: 1, depth: 1),
                                tg: MTLSize(width: 256, height: 1, depth: 1), byThreadgroups: false) { e in
                    e.setBuffer(src, offset: 0, index: 0); e.setBuffer(copy ? dst : sink, offset: 0, index: 1)
                    var nn = n; e.setBytes(&nn, length: 4, index: 2)
                }
                r.append(Double(ws) * Double(reps) * (copy ? 2 : 1) / (ms * 1e-3) / 1e9)
            }
            return r
        }
        let c = sample(copyP, true), rd = sample(readP, false)
        let (cq1, cq3) = quart(c), (rq1, rq3) = quart(rd)
        let label = ws >= (1 << 20) ? "\(ws >> 20) MiB" : "\(ws >> 10) KiB"
        let line = String(format: "  %9@  | %7.1f  [%6.1f,%6.1f]      | %7.1f  [%6.1f,%6.1f]",
                          label as NSString, median(c), cq1, cq3, median(rd), rq1, rq3)
        print(line); rows.append(line)
        ws *= 2
    }

    print("\n-- GPU peak FMA throughput --")
    for (name, isHalf) in [("fma_f32", false), ("fma_f16", true)] {
        let p = g.pipe(name)
        let nThreads = 12 * 1024 * 8
        let out = g.dev.makeBuffer(length: nThreads * 4, options: .storageModePrivate)!
        var iters: UInt32 = 4096
        func once() -> Double {
            gpuRun(g, p, reps: 1, grid: MTLSize(width: nThreads, height: 1, depth: 1),
                   tg: MTLSize(width: 1024, height: 1, depth: 1), byThreadgroups: false) { e in
                e.setBuffer(out, offset: 0, index: 0); e.setBytes(&iters, length: 4, index: 1)
            }
        }
        for _ in 0..<5 { _ = once() }
        let t = (0..<15).map { _ in once() }
        let flops = Double(nThreads) * 8 * 4 * 2 * Double(iters)
        print(String(format: "  %@: %.0f GFLOPS (median of 15; best %.0f)", name as NSString,
                     flops / (median(t) * 1e-3) / 1e9, flops / (t.min()! * 1e-3) / 1e9)
              + (isHalf ? "" : ""))
    }
    print("=" * 84)
}

// -----------------------------------------------------------------------------
func runM6BatchSweep() {
    let g = S1GPU(extraSource: loadCombinedSource())
    let fl = S1GPU()
    let fftP = g.pipe("fft4096_dx")
    let flushP = fl.pipe("flush_write")
    let N = 4096, T = 512
    let batches = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768,
                   1024, 1536, 2048, 3072, 4096, 8192, 16384]
    let maxB = batches.max()!
    let bytesPer = N * 8
    var inF = [Float](repeating: 0, count: maxB * N * 2)
    for i in 0..<inF.count { inF[i] = Float.random(in: -1...1) }
    let inBuf = inF.withUnsafeBytes { g.dev.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
    let outBuf = g.dev.makeBuffer(length: maxB * bytesPer, options: .storageModeShared)!
    let flushBytes = 512 << 20
    let flushBuf = fl.dev.makeBuffer(length: flushBytes, options: .storageModePrivate)!
    let flops = 5.0 * Double(N) * log2(Double(N))

    print("=" * 96)
    print("m6-batch: fft4096_dx (radix-8, 512 thr/TG) HOT vs COLD   Device: \(g.dev.name)")
    print("HOT  = K back-to-back dispatches in one CB (same buffers; >= 64 MiB traffic/sample), per-dispatch time")
    print("COLD = 512 MiB flush-write CB, then one FFT CB timed alone")
    print("GB/s = (in+out bytes) / time.  15 samples each, median [q1,q3]")
    print("=" * 96)
    print("  batch  in+out MiB |  HOT GF  [q1,q3]        GB/s   us/FFT |  COLD GF [q1,q3]        GB/s")
    print("  " + "-" * 92)
    for bs in batches {
        let io = Double(2 * bs * bytesPer)
        let K = max(1, Int((64.0 * 1048576.0) / io))
        func hot() -> Double {
            gpuRun(g, fftP, reps: K, grid: MTLSize(width: bs, height: 1, depth: 1),
                   tg: MTLSize(width: T, height: 1, depth: 1), byThreadgroups: true) { e in
                e.setBuffer(inBuf, offset: 0, index: 0); e.setBuffer(outBuf, offset: 0, index: 1)
            } / Double(K)
        }
        func cold() -> Double {
            _ = gpuRun(fl, flushP, reps: 1, grid: MTLSize(width: 12 * 1024 * 32, height: 1, depth: 1),
                       tg: MTLSize(width: 256, height: 1, depth: 1), byThreadgroups: false) { e in
                e.setBuffer(flushBuf, offset: 0, index: 0); var n = UInt32(flushBytes / 16); e.setBytes(&n, length: 4, index: 1)
            }
            return gpuRun(g, fftP, reps: 1, grid: MTLSize(width: bs, height: 1, depth: 1),
                          tg: MTLSize(width: T, height: 1, depth: 1), byThreadgroups: true) { e in
                e.setBuffer(inBuf, offset: 0, index: 0); e.setBuffer(outBuf, offset: 0, index: 1)
            }
        }
        for _ in 0..<3 { _ = hot(); _ = cold() }
        var h: [Double] = [], c: [Double] = []
        for _ in 0..<15 { h.append(hot()); c.append(cold()) }   // interleaved
        func gf(_ ms: Double) -> Double { flops * Double(bs) / (ms * 1e-3) / 1e9 }
        let hg = h.map(gf), cg = c.map(gf)
        let (hq1, hq3) = quart(hg), (cq1, cq3) = quart(cg)
        print(String(format: "  %5d  %9.2f  | %7.1f [%6.1f,%6.1f] %7.1f  %6.3f | %7.1f [%6.1f,%6.1f] %7.1f",
                     bs, io / 1048576, median(hg), hq1, hq3, io / (median(h) * 1e-3) / 1e9,
                     median(h) * 1e3 / Double(bs),
                     median(cg), cq1, cq3, io / (median(c) * 1e-3) / 1e9))
    }
    print("=" * 96)
}

// -----------------------------------------------------------------------------
private final class CPUWorker: Thread {
    let log2n: vDSP_Length, n: Int, batch: Int, setup: FFTSetup
    let start: DispatchSemaphore, done: DispatchSemaphore
    var ffts = 0
    var cpuHist: [Int: Int] = [:]
    let deadline: UnsafeMutablePointer<Double>
    init(setup: FFTSetup, n: Int, batch: Int, start: DispatchSemaphore, done: DispatchSemaphore,
         deadline: UnsafeMutablePointer<Double>, qos: QualityOfService) {
        self.setup = setup; self.n = n; self.batch = batch
        self.log2n = vDSP_Length(Int(log2(Double(n))))
        self.start = start; self.done = done; self.deadline = deadline
        super.init(); self.qualityOfService = qos
    }
    override func main() {
        let cnt = n * batch
        let ri = UnsafeMutablePointer<Float>.allocate(capacity: cnt), ii = UnsafeMutablePointer<Float>.allocate(capacity: cnt)
        let ro = UnsafeMutablePointer<Float>.allocate(capacity: cnt), io = UnsafeMutablePointer<Float>.allocate(capacity: cnt)
        for i in 0..<cnt { ri[i] = Float.random(in: -1...1); ii[i] = Float.random(in: -1...1) }
        start.wait()
        var k = 0
        while CFAbsoluteTimeGetCurrent() < deadline.pointee {
            for b in 0..<batch {
                var si = DSPSplitComplex(realp: ri + b * n, imagp: ii + b * n)
                var so = DSPSplitComplex(realp: ro + b * n, imagp: io + b * n)
                vDSP_fft_zop(setup, &si, 1, &so, 1, log2n, FFTDirection(kFFTDirection_Forward))
            }
            k += batch
            var cpu: Int = 0
            if pthread_cpu_number_np(&cpu) == 0 { cpuHist[cpu, default: 0] += 1 }
        }
        ffts = k
        ri.deallocate(); ii.deallocate(); ro.deallocate(); io.deallocate()
        done.signal()
    }
}

func runM6CPU() {
    let N = 4096, batch = 16          // 16 x 4096 c64 split = 512 KiB in + 512 KiB out per thread
    let seconds = 3.0
    let setup = vDSP_create_fftsetup(12, FFTRadix(kFFTRadix2))!
    let flops = 5.0 * Double(N) * log2(Double(N))
    print("=" * 96)
    print("m6-cpu: vDSP_fft_zop N=4096, batch \(batch)/thread, \(seconds)s per config; per-thread GFLOPS + CPU ids seen")
    print("=" * 96)
    let configs: [(Int, QualityOfService, String)] =
        [(1, .userInteractive, "UI"), (2, .userInteractive, "UI"), (3, .userInteractive, "UI"),
         (4, .userInteractive, "UI"), (6, .userInteractive, "UI"), (8, .userInteractive, "UI"),
         (12, .userInteractive, "UI"), (1, .background, "BG"), (6, .background, "BG")]
    for (nt, qos, ql) in configs {
        let start = DispatchSemaphore(value: 0), done = DispatchSemaphore(value: 0)
        let dl = UnsafeMutablePointer<Double>.allocate(capacity: 1); dl.pointee = .infinity
        let ws = (0..<nt).map { _ in CPUWorker(setup: setup, n: N, batch: batch, start: start, done: done, deadline: dl, qos: qos) }
        ws.forEach { $0.start() }
        Thread.sleep(forTimeInterval: 0.3)
        dl.pointee = CFAbsoluteTimeGetCurrent() + seconds
        for _ in 0..<nt { start.signal() }
        for _ in 0..<nt { done.wait() }
        let per = ws.map { Double($0.ffts) * flops / seconds / 1e9 }
        let tot = per.reduce(0, +)
        print(String(format: "  T=%2d %@  total %7.1f GF | per-thread:", nt, ql as NSString, tot)
              + per.map { String(format: " %5.1f", $0) }.joined())
        for (i, w) in ws.enumerated() {
            let h = w.cpuHist.sorted { $0.value > $1.value }.prefix(4).map { "cpu\($0.key):\($0.value)" }.joined(separator: " ")
            print("        thr\(i): \(h)")
        }
        dl.deallocate()
        Thread.sleep(forTimeInterval: 1.0)
    }
    vDSP_destroy_fftsetup(setup)
    print("=" * 96)
}

// -----------------------------------------------------------------------------
// m6-cpu-scale: does a workload scale with threads? NEON FMA (private per core)
// vs cblas_sgemm (Accelerate -> SME on M4+) vs vDSP FFT. Flat scaling = shared unit.
private final class LoopWorker: Thread {
    let body: () -> Double          // returns flops done in one call
    let start: DispatchSemaphore, done: DispatchSemaphore
    let deadline: UnsafeMutablePointer<Double>
    var flops = 0.0
    init(qos: QualityOfService, start: DispatchSemaphore, done: DispatchSemaphore,
         deadline: UnsafeMutablePointer<Double>, body: @escaping () -> Double) {
        self.body = body; self.start = start; self.done = done; self.deadline = deadline
        super.init(); qualityOfService = qos
    }
    override func main() {
        start.wait()
        while CFAbsoluteTimeGetCurrent() < deadline.pointee { flops += body() }
        done.signal()
    }
}

@inline(never) private func neonFMA(_ iters: Int) -> SIMD4<Float> {
    var a0 = SIMD4<Float>(repeating: 1), a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3
    var a4 = a0 + 4, a5 = a0 + 5, a6 = a0 + 6, a7 = a0 + 7
    let m = SIMD4<Float>(repeating: 0.9999), c = SIMD4<Float>(repeating: 1e-6)
    for _ in 0..<iters {
        a0 = a0.addingProduct(a0, m) ; a1 = a1.addingProduct(a1, m); a2 = a2.addingProduct(a2, m); a3 = a3.addingProduct(a3, m)
        a4 = a4.addingProduct(a4, m) ; a5 = a5.addingProduct(a5, m); a6 = a6.addingProduct(a6, m); a7 = a7.addingProduct(a7, m)
        a0 += c
    }
    return a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7
}

func runM6CPUScale() {
    let seconds = 2.0
    print("=" * 96)
    print("m6-cpu-scale: total GFLOPS vs threads (UI QoS unless BG). Flat = shared resource.")
    print("=" * 96)
    let makers: [(String, () -> (() -> Double))] = [
        ("neon-fma", {
            { let r = neonFMA(1 << 16); if r.x == 12345 { print(r) }; return Double(1 << 16) * 8 * 4 * 2 } }),
        ("sgemm-512", {
            let n = 512
            let a = [Float](repeating: 0.5, count: n * n), b = a
            var cbuf = [Float](repeating: 0, count: n * n)
            return {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(n), Int32(n), Int32(n),
                            1, a, Int32(n), b, Int32(n), 0, &cbuf, Int32(n))
                return 2.0 * Double(n * n * n)
            } }),
        ("vdsp-fft4096x16", {
            let N = 4096, B = 16
            let setup = vDSP_create_fftsetup(12, FFTRadix(kFFTRadix2))!
            let ri = UnsafeMutablePointer<Float>.allocate(capacity: N * B), ii = UnsafeMutablePointer<Float>.allocate(capacity: N * B)
            let ro = UnsafeMutablePointer<Float>.allocate(capacity: N * B), io = UnsafeMutablePointer<Float>.allocate(capacity: N * B)
            for i in 0..<(N * B) { ri[i] = Float.random(in: -1...1); ii[i] = Float.random(in: -1...1) }
            return {
                for b in 0..<B {
                    var si = DSPSplitComplex(realp: ri + b * N, imagp: ii + b * N)
                    var so = DSPSplitComplex(realp: ro + b * N, imagp: io + b * N)
                    vDSP_fft_zop(setup, &si, 1, &so, 1, 12, FFTDirection(kFFTDirection_Forward))
                }
                return Double(B) * 5 * 4096 * 12
            } }),
    ]
    for (name, mk) in makers {
        var line = String(format: "  %-16@", name as NSString)
        for (nt, qos) in [(1, QualityOfService.userInteractive), (2, .userInteractive), (4, .userInteractive),
                          (6, .userInteractive), (12, .userInteractive), (1, .background), (6, .background)] {
            let start = DispatchSemaphore(value: 0), done = DispatchSemaphore(value: 0)
            let dl = UnsafeMutablePointer<Double>.allocate(capacity: 1); dl.pointee = .infinity
            let ws = (0..<nt).map { _ in LoopWorker(qos: qos, start: start, done: done, deadline: dl, body: mk()) }
            ws.forEach { $0.start() }
            Thread.sleep(forTimeInterval: 0.3)
            dl.pointee = CFAbsoluteTimeGetCurrent() + seconds
            for _ in 0..<nt { start.signal() }
            for _ in 0..<nt { done.wait() }
            let tot = ws.map(\.flops).reduce(0, +) / seconds / 1e9
            line += String(format: " | %@%2d %7.1f", (qos == .background ? "BG" : "UI") as NSString, nt, tot)
            dl.deallocate()
            Thread.sleep(forTimeInterval: 0.7)
        }
        print(line)
    }
    print("=" * 96)
}
