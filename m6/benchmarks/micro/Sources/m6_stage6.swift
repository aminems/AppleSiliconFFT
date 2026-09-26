import Accelerate
import CSMEFFT
import Foundation
import Metal

// =============================================================================
// paper_m6 Stage 6 — Range-Doppler SAR on M6, end to end.
//
//   m6-sar
//
// Scene: 4096 pulses x 4096 range samples, L-band stripmap, point targets with a real
// antenna beam (600 m synthetic aperture, fully inside the frame) and ~1.8 range bins of
// range cell migration, so RCMC matters. Split re/im fp32 planes, row-major [az][rg].
//
// Pipelines (same RDA math; kernels in src/metal/m6_sar.metal, SME in src/cpu/sme_fft.c):
//   CPU  vDSP reference: range compress rows, transpose, azimuth FFT rows,
//        RCMC + azimuth filter (12 threads), azimuth IFFT rows. Output [rg][az].
//   GPU  4 dispatches: fused range compress, transpose, azimuth FFT, fused
//        RCMC + azimuth filter + IFFT. Output [rg][az].
//   HYB  GPU range compress -> SME column FFT -> GPU RCMC + filter -> SME column IFFT.
//        No transpose. Output [az][rg].
// The reference is validated by focus quality (peak position, IRW, PSLR vs theory); the
// GPU and hybrid images are compared with it by relative L2 error.
// =============================================================================

private let c0 = 299_792_458.0

private struct SarGeom {
    let nr = 4096, na = 4096
    let fc = 1.25e9, bw = 100e6, tp = 10e-6, fs = 120e6
    let v = 200.0, prf = 500.0, la = 8.0, rc = 20_000.0
    var lambda: Double { c0 / fc }
    var kr: Double { bw / tp }
    var tStart: Double { 2 * rc / c0 - Double(nr / 2) / fs }
    var r0Start: Double { c0 / 2 * tStart }
    var dr: Double { c0 / (2 * fs) }
    var theta: Double { lambda / la }
    func r0(_ n: Int) -> Double { r0Start + Double(n) * dr }
    func fa(_ k: Int) -> Double { Double(k < na / 2 ? k : k - na) * prf / Double(na) }
    func dFactor(_ fa: Double) -> Double { let s = lambda * fa / (2 * v); return (1 - s * s).squareRoot() }
}

private struct SarTarget { let rg: Int, az: Int; let amp: Double }

// Metal-side struct (m6_sar.metal SarParams)
private struct SarParamsGPU {
    var nr: UInt32, na: UInt32
    var prf: Float, v: Float, lambda: Float, r0Start: Float, dr: Float, scale: Float
}

// ---- scene -------------------------------------------------------------------------------------

private func simulate(_ g: SarGeom, _ targets: [SarTarget], re: UnsafeMutablePointer<Float>, im: UnsafeMutablePointer<Float>) {
    let nr = g.nr, na = g.na
    for i in 0..<(nr * na) { re[i] = 0; im[i] = 0 }
    let half = Int(g.tp * g.fs) / 2
    for t in targets {
        let r0 = g.r0(t.rg)
        let xt = Double(t.az - na / 2) * g.v / g.prf
        let lh = r0 * g.theta / 2
        for a in 0..<na {
            let eta = Double(a - na / 2) / g.prf
            let dx = g.v * eta - xt
            if abs(dx) > lh { continue }
            let r = (r0 * r0 + dx * dx).squareRoot()
            let tau0 = 2 * r / c0
            let i0 = Int(((tau0 - g.tStart) * g.fs).rounded())
            let aphase = -4 * Double.pi * r / g.lambda
            for i in max(0, i0 - half - 1)...min(nr - 1, i0 + half + 1) {
                let dt = g.tStart + Double(i) / g.fs - tau0
                if abs(dt) > g.tp / 2 { continue }
                let ph = Double.pi * g.kr * dt * dt + aphase
                re[a * nr + i] += Float(t.amp * cos(ph))
                im[a * nr + i] += Float(t.amp * sin(ph))
            }
        }
    }
    // complex Gaussian noise, sigma 0.1 per component (deterministic LCG + Box-Muller)
    var s: UInt64 = 0x9E3779B97F4A7C15
    func u() -> Double { s = s &* 6364136223846793005 &+ 1442695040888963407; return (Double(s >> 11) + 0.5) / 9007199254740992.0 }
    for i in 0..<(nr * na) {
        let r = 0.1 * (-2 * log(u())).squareRoot(), th = 2 * Double.pi * u()
        re[i] += Float(r * cos(th)); im[i] += Float(r * sin(th))
    }
}

/// conj(FFT(chirp reference)), reference centred at index 0 (circular).
private func chirpFilter(_ g: SarGeom, setup: FFTSetup) -> (re: [Float], im: [Float]) {
    let n = g.nr, ns = Int(g.tp * g.fs), half = ns / 2
    var re = [Float](repeating: 0, count: n), im = [Float](repeating: 0, count: n)
    for i in 0..<ns {
        let t = Double(i - half) / g.fs
        let ph = Double.pi * g.kr * t * t
        let idx = i < half ? n - half + i : i - half
        re[idx] = Float(cos(ph)); im[idx] = Float(sin(ph))
    }
    re.withUnsafeMutableBufferPointer { r in im.withUnsafeMutableBufferPointer { m in
        var sc = DSPSplitComplex(realp: r.baseAddress!, imagp: m.baseAddress!)
        vDSP_fft_zip(setup, &sc, 1, vDSP_Length(12), FFTDirection(kFFTDirection_Forward))
    } }
    for i in 0..<n { im[i] = -im[i] }
    return (re, im)
}

/// 8-tap windowed-sinc interpolator, 65 fractional phases f/64 (f = 0...64); row f sums to 1.
private func interpTable() -> [Float] {
    var t = [Float](repeating: 0, count: 65 * 8)
    for f in 0...64 {
        let fr = Double(f) / 64
        var w = [Double](repeating: 0, count: 8), sum = 0.0
        for j in 0..<8 {
            let d = Double(j - 3) - fr
            let sinc = abs(d) < 1e-12 ? 1.0 : sin(Double.pi * d) / (Double.pi * d)
            let win = 0.5 * (1 + cos(Double.pi * d / 4.5))
            w[j] = sinc * win; sum += w[j]
        }
        for j in 0..<8 { t[f * 8 + j] = Float(w[j] / sum) }
    }
    return t
}

// ---- CPU vDSP reference RDA (output [rg][az]) --------------------------------------------------

private struct CPUTimes { var rc = 0.0, tr = 0.0, az = 0.0, rcmc = 0.0, ifft = 0.0 }

private func referenceRDA(_ g: SarGeom, rawRe: UnsafePointer<Float>, rawIm: UnsafePointer<Float>,
                          outRe: UnsafeMutablePointer<Float>, outIm: UnsafeMutablePointer<Float>,
                          wRe: UnsafeMutablePointer<Float>, wIm: UnsafeMutablePointer<Float>,
                          hr: (re: [Float], im: [Float]), tab: [Float], setup: FFTSetup) -> CPUTimes {
    let nr = g.nr, na = g.na, log2n = vDSP_Length(12)
    var tm = CPUTimes()
    let inv = 1.0 / Float(nr)
    // 1. range compression, row by row (in cache): FFT, x Hr, IFFT, 1/N
    var t0 = CFAbsoluteTimeGetCurrent()
    wRe.update(from: rawRe, count: nr * na); wIm.update(from: rawIm, count: nr * na)
    hr.re.withUnsafeBufferPointer { hre in hr.im.withUnsafeBufferPointer { him in
        var h = DSPSplitComplex(realp: UnsafeMutablePointer(mutating: hre.baseAddress!),
                                imagp: UnsafeMutablePointer(mutating: him.baseAddress!))
        for a in 0..<na {
            var row = DSPSplitComplex(realp: wRe + a * nr, imagp: wIm + a * nr)
            vDSP_fft_zip(setup, &row, 1, log2n, FFTDirection(kFFTDirection_Forward))
            vDSP_zvmul(&row, 1, &h, 1, &row, 1, vDSP_Length(nr), 1)
            vDSP_fft_zip(setup, &row, 1, log2n, FFTDirection(kFFTDirection_Inverse))
            var s = inv
            vDSP_vsmul(row.realp, 1, &s, row.realp, 1, vDSP_Length(nr))
            vDSP_vsmul(row.imagp, 1, &s, row.imagp, 1, vDSP_Length(nr))
        }
    } }
    tm.rc = CFAbsoluteTimeGetCurrent() - t0
    // 2. transpose [az][rg] -> [rg][az] (into out*)
    t0 = CFAbsoluteTimeGetCurrent()
    vDSP_mtrans(wRe, 1, outRe, 1, vDSP_Length(nr), vDSP_Length(na))
    vDSP_mtrans(wIm, 1, outIm, 1, vDSP_Length(nr), vDSP_Length(na))
    tm.tr = CFAbsoluteTimeGetCurrent() - t0
    // 3. azimuth FFT along rows of [rg][az]
    t0 = CFAbsoluteTimeGetCurrent()
    for n in 0..<nr {
        var row = DSPSplitComplex(realp: outRe + n * na, imagp: outIm + n * na)
        vDSP_fft_zip(setup, &row, 1, log2n, FFTDirection(kFFTDirection_Forward))
    }
    tm.az = CFAbsoluteTimeGetCurrent() - t0
    // 4. RCMC + azimuth filter, exact double phase, 12 threads, into w*
    t0 = CFAbsoluteTimeGetCurrent()
    let sre = UnsafePointer(outRe), sim = UnsafePointer(outIm)
    let dre = wRe, dim = wIm
    tab.withUnsafeBufferPointer { tb in
        DispatchQueue.concurrentPerform(iterations: nr) { n in
            let r0 = g.r0(n)
            for k in 0..<na {
                let d = g.dFactor(g.fa(k))
                let x = Double(n) + (r0 / d - r0) / g.dr
                let m = Int(x.rounded(.down))
                let p64 = (x - Double(m)) * 64
                let f = min(Int(p64), 63), al = p64 - Double(f)
                var ar = 0.0, ai = 0.0
                for j in 0..<8 {
                    let mm = m - 3 + j
                    if mm >= 0 && mm < nr {
                        let w = Double(tb[f * 8 + j]) * (1 - al) + Double(tb[f * 8 + 8 + j]) * al
                        ar += w * Double(sre[mm * na + k]); ai += w * Double(sim[mm * na + k])
                    }
                }
                let ph = 4 * Double.pi * r0 * (d - 1) / g.lambda   // baseband azimuth filter (see m6_sar.metal)
                let c = cos(ph), s = sin(ph)
                dre[n * na + k] = Float(ar * c - ai * s)
                dim[n * na + k] = Float(ar * s + ai * c)
            }
        }
    }
    tm.rcmc = CFAbsoluteTimeGetCurrent() - t0
    // 5. azimuth IFFT along rows, 1/N, into out*
    t0 = CFAbsoluteTimeGetCurrent()
    let invA = 1.0 / Float(na)
    for n in 0..<nr {
        var src = DSPSplitComplex(realp: wRe + n * na, imagp: wIm + n * na)
        var dst = DSPSplitComplex(realp: outRe + n * na, imagp: outIm + n * na)
        vDSP_fft_zop(setup, &src, 1, &dst, 1, log2n, FFTDirection(kFFTDirection_Inverse))
        var s = invA
        vDSP_vsmul(dst.realp, 1, &s, dst.realp, 1, vDSP_Length(na))
        vDSP_vsmul(dst.imagp, 1, &s, dst.imagp, 1, vDSP_Length(na))
    }
    tm.ifft = CFAbsoluteTimeGetCurrent() - t0
    return tm
}

// ---- focus-quality metrics -----------------------------------------------------------------------

/// Band-limited x16 upsampling of a 64-sample cut (zero-padded spectrum); returns magnitudes.
private func upsampleMag(_ xr: [Double], _ xi: [Double], _ u: Int = 16) -> [Double] {
    let n = xr.count, m = n * u
    var Xr = [Double](repeating: 0, count: n), Xi = Xr
    for k in 0..<n { for t in 0..<n {
        let a = -2 * Double.pi * Double(k * t) / Double(n)
        Xr[k] += xr[t] * cos(a) - xi[t] * sin(a); Xi[k] += xr[t] * sin(a) + xi[t] * cos(a)
    } }
    var Yr = [Double](repeating: 0, count: m), Yi = Yr
    for k in 0..<n { let kk = k < n / 2 ? k : m - n + k; Yr[kk] = Xr[k]; Yi[kk] = Xi[k] }
    var out = [Double](repeating: 0, count: m)
    for t in 0..<m {
        var sr = 0.0, si = 0.0
        for k in 0..<m where Yr[k] != 0 || Yi[k] != 0 {
            let a = 2 * Double.pi * Double(k * t) / Double(m)
            sr += Yr[k] * cos(a) - Yi[k] * sin(a); si += Yr[k] * sin(a) + Yi[k] * cos(a)
        }
        out[t] = (sr * sr + si * si).squareRoot() / Double(n)
    }
    return out
}

/// (peak offset from cut centre in samples, 3 dB width in samples, PSLR dB) of an upsampled cut.
private func cutMetrics(_ mag: [Double], _ u: Int = 16) -> (off: Double, irw: Double, pslr: Double) {
    let lo = 8 * u, hi = mag.count - 8 * u     // ignore cut edges (non-periodic window)
    var p = lo
    for i in lo..<hi where mag[i] > mag[p] { p = i }
    let half = mag[p] / 2.0.squareRoot()
    var l = p; while l > 0 && mag[l - 1] >= half { l -= 1 }
    var r = p; while r < mag.count - 1 && mag[r + 1] >= half { r += 1 }
    let irw = Double(r - l + 1) / Double(u)
    var nl = p; while nl > lo && mag[nl - 1] < mag[nl] { nl -= 1 }
    var nrr = p; while nrr < hi - 1 && mag[nrr + 1] < mag[nrr] { nrr += 1 }
    var side = 0.0
    for i in lo..<hi where i < nl || i > nrr { side = max(side, mag[i]) }
    return (Double(p) / Double(u) - Double(mag.count / u / 2), irw, 20 * log10(side / mag[p]))
}

private struct FocusResult { let rg: Double, az: Double, irwR: Double, irwA: Double, pslrR: Double, pslrA: Double }

/// img(rg, az) accessor; returns per-target measured position and cut metrics.
private func focus(_ targets: [SarTarget], _ img: (Int, Int) -> (Float, Float)) -> [FocusResult] {
    targets.map { t in
        var br = t.rg, ba = t.az, best: Float = -1
        for r in (t.rg - 8)...(t.rg + 8) { for a in (t.az - 8)...(t.az + 8) {
            let v = img(r, a); let p = v.0 * v.0 + v.1 * v.1
            if p > best { best = p; br = r; ba = a }
        } }
        var rr = [Double](), ri = [Double](), ar = [Double](), ai = [Double]()
        for d in -32..<32 {
            let x = img(br + d, ba); rr.append(Double(x.0)); ri.append(Double(x.1))
            let y = img(br, ba + d); ar.append(Double(y.0)); ai.append(Double(y.1))
        }
        let mr = cutMetrics(upsampleMag(rr, ri)), ma = cutMetrics(upsampleMag(ar, ai))
        return FocusResult(rg: Double(br) + mr.off, az: Double(ba) + ma.off, irwR: mr.irw, irwA: ma.irw,
                           pslrR: mr.pslr, pslrA: ma.pslr)
    }
}

private func printFocus(_ name: String, _ targets: [SarTarget], _ f: [FocusResult]) {
    print("  \(name): target  expected(rg,az)   measured(rg,az)      IRW rg/az (bins)   PSLR rg/az (dB)")
    for (t, r) in zip(targets, f) {
        print(String(format: "     #%d  (%4d,%4d)   (%8.2f,%8.2f)    %5.2f / %5.2f     %6.2f / %6.2f",
                     targets.firstIndex { $0.rg == t.rg && $0.az == t.az }!, t.rg, t.az, r.rg, r.az, r.irwR, r.irwA, r.pslrR, r.pslrA))
    }
}

private func relL2(_ ar: UnsafePointer<Float>, _ ai: UnsafePointer<Float>, _ n: Int,
                   _ ref: (Int) -> (Float, Float)) -> Double {
    var num = 0.0, den = 0.0
    for i in 0..<n {
        let r = ref(i)
        let dr = Double(ar[i] - r.0), di = Double(ai[i] - r.1)
        num += dr * dr + di * di
        den += Double(r.0) * Double(r.0) + Double(r.1) * Double(r.1)
    }
    return (num / den).squareRoot()
}

/// Relative L2 error in 64x64 windows around the targets (where the image is signal, not noise).
private func relL2Targets(_ targets: [SarTarget], _ img: (Int, Int) -> (Float, Float), _ ref: (Int, Int) -> (Float, Float)) -> Double {
    var num = 0.0, den = 0.0
    for t in targets { for r in (t.rg - 32)..<(t.rg + 32) { for a in (t.az - 32)..<(t.az + 32) {
        let x = img(r, a), y = ref(r, a)
        num += Double((x.0 - y.0) * (x.0 - y.0) + (x.1 - y.1) * (x.1 - y.1))
        den += Double(y.0 * y.0 + y.1 * y.1)
    } } }
    return (num / den).squareRoot()
}

private func med(_ v: [Double]) -> Double { let s = v.sorted(); return s[s.count / 2] }

// ---- driver ----------------------------------------------------------------------------------

func runM6SAR() {
    let g = SarGeom()
    let nr = g.nr, na = g.na, np = nr * na
    let targets = [SarTarget(rg: 2048, az: 2048, amp: 1), SarTarget(rg: 2448, az: 2048, amp: 1),
                   SarTarget(rg: 2048, az: 2548, amp: 1), SarTarget(rg: 1648, az: 1548, amp: 1),
                   SarTarget(rg: 3048, az: 1048, amp: 1), SarTarget(rg: 1048, az: 3048, amp: 0.5)]
    print("m6-sar: Range-Doppler SAR, \(na) pulses x \(nr) range samples, split fp32 planes (128 MiB per complex image)")
    print(String(format: "  L-band %.3f m, B %.0f MHz, fs %.0f MHz, v %.0f m/s, PRF %.0f Hz, antenna %.0f m",
                 g.lambda, g.bw / 1e6, g.fs / 1e6, g.v, g.prf, g.la))
    let lsa = g.rc * g.theta
    let rcm = ((g.rc * g.rc + lsa * lsa / 4).squareRoot() - g.rc) / g.dr
    print(String(format: "  synthetic aperture %.0f m = %.0f pulses; max range migration %.2f bins", lsa, lsa / (g.v / g.prf), rcm))
    print(String(format: "  theory (unweighted): IRW rg %.2f bins, az %.2f bins; PSLR -13.26 dB",
                 0.886 * g.fs / g.bw, 0.886 * g.prf / (2 * g.v / g.la)))

    guard let dev = MTLCreateSystemDefaultDevice(), let q = dev.makeCommandQueue() else { fatalError("no Metal") }
    func buf() -> MTLBuffer { dev.makeBuffer(length: np * 4, options: .storageModeShared)! }
    let rawRe = buf(), rawIm = buf(), aRe = buf(), aIm = buf(), bRe = buf(), bIm = buf()
    func fp(_ b: MTLBuffer) -> UnsafeMutablePointer<Float> { b.contents().bindMemory(to: Float.self, capacity: np) }

    let t0 = CFAbsoluteTimeGetCurrent()
    simulate(g, targets, re: fp(rawRe), im: fp(rawIm))
    print(String(format: "  simulated in %.2f s", CFAbsoluteTimeGetCurrent() - t0))

    let setup = vDSP_create_fftsetup(12, FFTRadix(kFFTRadix2))!
    let hr = chirpFilter(g, setup: setup)
    let tab = interpTable()

    // ---- CPU reference ----
    let refRe = UnsafeMutablePointer<Float>.allocate(capacity: np), refIm = UnsafeMutablePointer<Float>.allocate(capacity: np)
    let wRe = UnsafeMutablePointer<Float>.allocate(capacity: np), wIm = UnsafeMutablePointer<Float>.allocate(capacity: np)
    var cpuT = [CPUTimes]()
    for _ in 0..<3 {
        cpuT.append(referenceRDA(g, rawRe: fp(rawRe), rawIm: fp(rawIm), outRe: refRe, outIm: refIm, wRe: wRe, wIm: wIm,
                                 hr: hr, tab: tab, setup: setup))
    }
    let ct = cpuT.last!
    let refImg: (Int, Int) -> (Float, Float) = { r, a in (refRe[r * na + a], refIm[r * na + a]) }
    print("\nCPU reference (vDSP, 1 thread; RCMC loop 12 threads, double phase):")
    let cpuTot = cpuT.map { $0.rc + $0.tr + $0.az + $0.rcmc + $0.ifft }
    print(String(format: "  range compress %.1f | transpose %.1f | az FFT %.1f | RCMC+filter %.1f | az IFFT %.1f | total %.1f ms (median of 3)",
                 ct.rc * 1e3, ct.tr * 1e3, ct.az * 1e3, ct.rcmc * 1e3, ct.ifft * 1e3, med(cpuTot) * 1e3))
    printFocus("reference", targets, focus(targets, refImg))
    do {   // metric self-test on an ideal sinc at the same oversampling (1.2 samples per 1/B)
        let ov = g.fs / g.bw
        let cr = (-32..<32).map { d -> Double in let x = Double.pi * Double(d) / ov; return d == 0 ? 1 : sin(x) / x }
        let m = cutMetrics(upsampleMag(cr, [Double](repeating: 0, count: 64)))
        print(String(format: "  metric self-test (ideal sinc): IRW %.2f bins, PSLR %.2f dB", m.irw, m.pslr))
        let t = targets[0]
        print("  |ref| along range at target 0: " + (-4...4).map { String(format: "%.0f", hypot(refRe[(t.rg + $0) * na + t.az], refIm[(t.rg + $0) * na + t.az])) }.joined(separator: " "))
    }

    // ---- GPU ----
    let o = MTLCompileOptions(); o.fastMathEnabled = ProcessInfo.processInfo.environment["SAR_NOFASTMATH"] == nil
    let src = (try! String(contentsOfFile: try! FFTDeviceSource.path("src/common/fft_device.h"), encoding: .utf8)) + "\n"
        + (try! String(contentsOfFile: try! FFTDeviceSource.path("src/metal/m6_sar.metal"), encoding: .utf8))
    let lib: MTLLibrary
    do { lib = try dev.makeLibrary(source: src, options: o) } catch { fatalError("m6_sar.metal: \(error)") }
    func P(_ n: String) -> MTLComputePipelineState { try! dev.makeComputePipelineState(function: lib.makeFunction(name: n)!) }
    let pRC = P("sar_range_compress"), pTR = P("sar_transpose"), pFR = P("sar_fft_rows")
    let pRI = P("sar_rcmc_azc_ifft_t"), pRF = P("sar_rcmc_azfilter"), pFC = P("sar_fft_cols_t")

    var hrI = [Float](repeating: 0, count: 2 * nr)
    for i in 0..<nr { hrI[2 * i] = hr.re[i]; hrI[2 * i + 1] = hr.im[i] }
    let hrB = dev.makeBuffer(bytes: hrI, length: 8 * nr, options: .storageModeShared)!
    let tabB = dev.makeBuffer(bytes: tab, length: tab.count * 4, options: .storageModeShared)!
    // azimuth-filter phase in cycles: azA[k] + n azB[k] (see m6_sar.metal)
    var azA = [Float](), azBhi = [Float](), azBlo = [Float]()
    for k in 0..<na {
        let dm1 = g.dFactor(g.fa(k)) - 1
        let a = 2 * g.r0Start * dm1 / g.lambda, b = 2 * g.dr * dm1 / g.lambda
        azA.append(Float(a - a.rounded(.down)))
        let e = b == 0 ? 0 : Int(floor(log2(abs(b)))) - 10          // keep 11 significant bits
        let hi = (b / pow(2, Double(e))).rounded() * pow(2, Double(e))
        azBhi.append(Float(hi)); azBlo.append(Float(b - hi))
    }
    let azAB = dev.makeBuffer(bytes: azA, length: na * 4, options: .storageModeShared)!
    let azBhiB = dev.makeBuffer(bytes: azBhi, length: na * 4, options: .storageModeShared)!
    let azBloB = dev.makeBuffer(bytes: azBlo, length: na * 4, options: .storageModeShared)!
    func params(_ scale: Float) -> SarParamsGPU {
        SarParamsGPU(nr: UInt32(nr), na: UInt32(na), prf: Float(g.prf), v: Float(g.v), lambda: Float(g.lambda),
                     r0Start: Float(g.r0Start), dr: Float(g.dr), scale: scale)
    }

    typealias Step = (MTLComputeCommandEncoder) -> Void
    func io(_ e: MTLComputeCommandEncoder, _ i0: MTLBuffer, _ i1: MTLBuffer, _ o0: MTLBuffer, _ o1: MTLBuffer) {
        e.setBuffer(i0, offset: 0, index: 0); e.setBuffer(i1, offset: 0, index: 1)
        e.setBuffer(o0, offset: 0, index: 2); e.setBuffer(o1, offset: 0, index: 3)
    }
    let t512 = MTLSize(width: 512, height: 1, depth: 1)
    let stRC: Step = { e in e.setComputePipelineState(pRC); io(e, rawRe, rawIm, aRe, aIm)
        e.setBuffer(hrB, offset: 0, index: 4)
        e.dispatchThreadgroups(MTLSize(width: na, height: 1, depth: 1), threadsPerThreadgroup: t512) }
    let stTR: Step = { e in e.setComputePipelineState(pTR); io(e, aRe, aIm, bRe, bIm)
        var d = SIMD2<UInt32>(UInt32(nr), UInt32(na)); e.setBytes(&d, length: 8, index: 4)
        e.dispatchThreadgroups(MTLSize(width: nr / 32, height: na / 32, depth: 2),
                               threadsPerThreadgroup: MTLSize(width: 32, height: 8, depth: 1)) }
    let stFR: Step = { e in e.setComputePipelineState(pFR); io(e, bRe, bIm, aRe, aIm)
        e.dispatchThreadgroups(MTLSize(width: nr, height: 1, depth: 1), threadsPerThreadgroup: t512) }
    let stFC: Step = { e in e.setComputePipelineState(pFC); io(e, aRe, aIm, bRe, bIm)   // replaces stTR + stFR
        var n = UInt32(nr); e.setBytes(&n, length: 4, index: 4)
        e.dispatchThreadgroups(MTLSize(width: nr, height: 1, depth: 1), threadsPerThreadgroup: t512) }
    let stRIb: Step = { e in e.setComputePipelineState(pRI); io(e, bRe, bIm, aRe, aIm)   // stRI reading b, writing a
        var p = params(1); e.setBytes(&p, length: MemoryLayout<SarParamsGPU>.stride, index: 4)
        e.setBuffer(tabB, offset: 0, index: 5); e.setBuffer(azAB, offset: 0, index: 6)
        e.setBuffer(azBhiB, offset: 0, index: 7); e.setBuffer(azBloB, offset: 0, index: 8)
        e.dispatchThreadgroups(MTLSize(width: nr, height: 1, depth: 1), threadsPerThreadgroup: t512) }
    let stRI: Step = { e in e.setComputePipelineState(pRI); io(e, aRe, aIm, bRe, bIm)
        var p = params(1); e.setBytes(&p, length: MemoryLayout<SarParamsGPU>.stride, index: 4)
        e.setBuffer(tabB, offset: 0, index: 5); e.setBuffer(azAB, offset: 0, index: 6)
        e.setBuffer(azBhiB, offset: 0, index: 7); e.setBuffer(azBloB, offset: 0, index: 8)
        e.dispatchThreadgroups(MTLSize(width: nr, height: 1, depth: 1), threadsPerThreadgroup: t512) }

    func runGPU(_ steps: [Step]) -> Double {   // one command buffer, GPU time
        let cb = q.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
        for s in steps { s(e) }
        e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        return cb.gpuEndTime - cb.gpuStartTime
    }
    _ = runGPU([stRC, stTR, stFR, stRI])
    let stRF: Step = { e in e.setComputePipelineState(pRF); io(e, bRe, bIm, aRe, aIm)
        var p = params(1 / Float(na)); e.setBytes(&p, length: MemoryLayout<SarParamsGPU>.stride, index: 4)
        e.setBuffer(tabB, offset: 0, index: 5); e.setBuffer(azAB, offset: 0, index: 6)
        e.setBuffer(azBhiB, offset: 0, index: 7); e.setBuffer(azBloB, offset: 0, index: 8)
        e.dispatchThreads(MTLSize(width: nr, height: na, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)) }
    var pendingRF = false
    var diagSpec: (UnsafeMutablePointer<Float>, UnsafeMutablePointer<Float>)? = nil
    if ProcessInfo.processInfo.environment["SAR_DIAG"] != nil {   // per-stage error vs vDSP
        _ = runGPU([stRC])   // aRe/aIm = GPU range-compressed
        let cr = UnsafeMutablePointer<Float>.allocate(capacity: np), ci = UnsafeMutablePointer<Float>.allocate(capacity: np)
        cr.update(from: fp(rawRe), count: np); ci.update(from: fp(rawIm), count: np)
        hr.re.withUnsafeBufferPointer { hre in hr.im.withUnsafeBufferPointer { him in
            var h = DSPSplitComplex(realp: UnsafeMutablePointer(mutating: hre.baseAddress!), imagp: UnsafeMutablePointer(mutating: him.baseAddress!))
            for a in 0..<na {
                var row = DSPSplitComplex(realp: cr + a * nr, imagp: ci + a * nr)
                vDSP_fft_zip(setup, &row, 1, 12, FFTDirection(kFFTDirection_Forward))
                vDSP_zvmul(&row, 1, &h, 1, &row, 1, vDSP_Length(nr), 1)
                vDSP_fft_zip(setup, &row, 1, 12, FFTDirection(kFFTDirection_Inverse))
                var s = 1 / Float(nr)
                vDSP_vsmul(row.realp, 1, &s, row.realp, 1, vDSP_Length(nr)); vDSP_vsmul(row.imagp, 1, &s, row.imagp, 1, vDSP_Length(nr))
            }
        } }
        print(String(format: "  diag: range compress GPU vs vDSP rel L2 %.2e", relL2(fp(aRe), fp(aIm), np) { (cr[$0], ci[$0]) }))
        // feed the vDSP range-compressed data through GPU transpose + az FFT and compare the az spectrum
        fp(aRe).update(from: cr, count: np); fp(aIm).update(from: ci, count: np)
        _ = runGPU([stTR, stFR])   // aRe/aIm = GPU az spectrum [rg][fa]
        vDSP_mtrans(cr, 1, fp(bRe), 1, vDSP_Length(nr), vDSP_Length(na)); vDSP_mtrans(ci, 1, fp(bIm), 1, vDSP_Length(nr), vDSP_Length(na))
        for n in 0..<nr { var row = DSPSplitComplex(realp: fp(bRe) + n * na, imagp: fp(bIm) + n * na)
            vDSP_fft_zip(setup, &row, 1, 12, FFTDirection(kFFTDirection_Forward)) }
        print(String(format: "  diag: az FFT GPU vs vDSP rel L2 %.2e", relL2(fp(aRe), fp(aIm), np) { (fp(bRe)[$0], fp(bIm)[$0]) }))
        // RCMC + filter alone: GPU row kernel (hybrid layout) vs double reference, per Doppler bin
        let sr = UnsafeMutablePointer<Float>.allocate(capacity: np), si = UnsafeMutablePointer<Float>.allocate(capacity: np)
        sr.update(from: fp(bRe), count: np); si.update(from: fp(bIm), count: np)          // [rg][fa]
        vDSP_mtrans(sr, 1, fp(bRe), 1, vDSP_Length(na), vDSP_Length(nr)); vDSP_mtrans(si, 1, fp(bIm), 1, vDSP_Length(na), vDSP_Length(nr))
        pendingRF = true
        diagSpec = (sr, si)
        cr.deallocate(); ci.deallocate()
    }
    if pendingRF, let (sr, si) = diagSpec {
        _ = runGPU([stRF])   // aRe/aIm = GPU RCMC+filter, [fa][rg], scaled 1/na
        for k in [0, 5, 25, 60, 200, 1000, 2048, 3000, 4090] {
            let d = g.dFactor(g.fa(k)); var num = 0.0, den = 0.0
            for n in 0..<nr {
                let r0 = g.r0(n), x = Double(n) + (r0 / d - r0) / g.dr
                let m = Int(x.rounded(.down)); let p64 = (x - Double(m)) * 64
                let f = min(Int(p64), 63), al = p64 - Double(f)
                var ar = 0.0, ai = 0.0
                for j in 0..<8 { let mm = m - 3 + j; if mm >= 0 && mm < nr {
                    let w = Double(tab[f * 8 + j]) * (1 - al) + Double(tab[f * 8 + 8 + j]) * al
                    ar += w * Double(sr[mm * na + k]); ai += w * Double(si[mm * na + k]) } }
                let ph = 4 * Double.pi * r0 * (d - 1) / g.lambda
                let yr = (ar * cos(ph) - ai * sin(ph)) / Double(na), yi = (ar * sin(ph) + ai * cos(ph)) / Double(na)
                let gr = Double(fp(aRe)[k * nr + n]), gi = Double(fp(aIm)[k * nr + n])
                num += (gr - yr) * (gr - yr) + (gi - yi) * (gi - yi); den += yr * yr + yi * yi
            }
            print(String(format: "  diag: RCMC+filter Doppler bin %4d (fa %7.2f Hz, shift %6.2f bins): rel L2 %.2e, energy %.2e",
                         k, g.fa(k), (1 / d - 1) * g.rc / g.dr, (num / den).squareRoot(), den))
        }
        sr.deallocate(); si.deallocate()
        _ = runGPU([stRC, stTR, stFR, stRI])
    }
    let gpuImg: (Int, Int) -> (Float, Float) = { r, a in (fp(bRe)[r * na + a], fp(bIm)[r * na + a]) }
    let errG = relL2(fp(bRe), fp(bIm), np) { (refRe[$0], refIm[$0]) }
    var gTot = [Double](), gStep = [[Double]](repeating: [], count: 4)
    for _ in 0..<7 {
        gTot.append(runGPU([stRC, stTR, stFR, stRI]))
        for (i, s) in [stRC, stTR, stFR, stRI].enumerated() { gStep[i].append(runGPU([s])) }
    }
    let passBytes = Double(np * 16)   // read + write of one complex fp32 image
    print("\nGPU pipeline (4 dispatches, 1 command buffer; GPU time, median of 7):")
    print(String(format: "  range compress %.2f | transpose %.2f | az FFT %.2f | RCMC+filter+IFFT %.2f | total %.2f ms  (%.1fx CPU ref)",
                 med(gStep[0]) * 1e3, med(gStep[1]) * 1e3, med(gStep[2]) * 1e3, med(gStep[3]) * 1e3, med(gTot) * 1e3,
                 med(cpuTot) / med(gTot)))
    print(String(format: "  effective %.0f GB/s per pass (4 passes x 256 MiB); DRAM-bound estimate at 150 GB/s: %.2f ms",
                 4 * passBytes / med(gTot) * 1e-9, 4 * passBytes / 150e9 * 1e3))
    print(String(format: "  rel L2 vs reference: whole image %.2e, 64x64 around targets %.2e", errG,
                 relL2Targets(targets, gpuImg, refImg)))
    printFocus("GPU", targets, focus(targets, gpuImg))

    // fp16-storage variant: intermediates in half planes (40 B/point instead of 64)
    let pRCh = P("sar_range_compress_h"), pTRh = P("sar_transpose_h"), pFRh = P("sar_fft_rows_h"), pRIh = P("sar_rcmc_azc_ifft_th")
    func hbuf() -> MTLBuffer { dev.makeBuffer(length: np * 2, options: .storageModePrivate)! }
    let hAr = hbuf(), hAi = hbuf(), hBr = hbuf(), hBi = hbuf()
    let hSteps: [Step] = [
        { e in e.setComputePipelineState(pRCh); io(e, rawRe, rawIm, hAr, hAi); e.setBuffer(hrB, offset: 0, index: 4)
            e.dispatchThreadgroups(MTLSize(width: na, height: 1, depth: 1), threadsPerThreadgroup: t512) },
        { e in e.setComputePipelineState(pTRh); io(e, hAr, hAi, hBr, hBi)
            var d = SIMD2<UInt32>(UInt32(nr), UInt32(na)); e.setBytes(&d, length: 8, index: 4)
            e.dispatchThreadgroups(MTLSize(width: nr / 32, height: na / 32, depth: 2),
                                   threadsPerThreadgroup: MTLSize(width: 32, height: 8, depth: 1)) },
        { e in e.setComputePipelineState(pFRh); io(e, hBr, hBi, hAr, hAi)
            e.dispatchThreadgroups(MTLSize(width: nr, height: 1, depth: 1), threadsPerThreadgroup: t512) },
        { e in e.setComputePipelineState(pRIh); io(e, hAr, hAi, bRe, bIm)
            var p = params(Float(na)); e.setBytes(&p, length: MemoryLayout<SarParamsGPU>.stride, index: 4)
            e.setBuffer(tabB, offset: 0, index: 5); e.setBuffer(azAB, offset: 0, index: 6)
            e.setBuffer(azBhiB, offset: 0, index: 7); e.setBuffer(azBloB, offset: 0, index: 8)
            e.dispatchThreadgroups(MTLSize(width: nr, height: 1, depth: 1), threadsPerThreadgroup: t512) },
    ]
    _ = runGPU(hSteps)
    let errH16 = relL2(fp(bRe), fp(bIm), np) { (refRe[$0], refIm[$0]) }
    var h16 = [Double](), h16s = [[Double]](repeating: [], count: 4)
    for _ in 0..<7 { h16.append(runGPU(hSteps)); for (i, s) in hSteps.enumerated() { h16s[i].append(runGPU([s])) } }
    print(String(format: "\nGPU pipeline, fp16 intermediates: range compress %.2f | transpose %.2f | az FFT %.2f | RCMC+filter+IFFT %.2f | total %.2f ms (%.1fx CPU ref, %.2fx fp32 GPU)",
                 med(h16s[0]) * 1e3, med(h16s[1]) * 1e3, med(h16s[2]) * 1e3, med(h16s[3]) * 1e3, med(h16) * 1e3,
                 med(cpuTot) / med(h16), med(gTot) / med(h16)))
    print(String(format: "  bytes 40/point (vs 64): floor %.2f ms at 150 GB/s; rel L2 vs reference: whole image %.2e, around targets %.2e",
                 Double(np) * 40 / 150e9 * 1e3, errH16, relL2Targets(targets, gpuImg, refImg)))
    printFocus("GPU fp16", targets, focus(targets, gpuImg))

    // 3-pass variant: transpose fused into the azimuth FFT (strided column reads)
    _ = runGPU([stRC, stFC, stRIb])
    let err3 = relL2(fp(aRe), fp(aIm), np) { (refRe[$0], refIm[$0]) }
    var g3 = [Double](), gFC = [Double]()
    for _ in 0..<7 { g3.append(runGPU([stRC, stFC, stRIb])); gFC.append(runGPU([stFC])) }
    print(String(format: "GPU 3-pass variant (azimuth FFT reads columns directly): az FFT %.2f | total %.2f ms (%.1fx CPU ref); floor 3 x 256 MiB at 150 GB/s = %.2f ms; rel L2 %.2e",
                 med(gFC) * 1e3, med(g3) * 1e3, med(cpuTot) / med(g3), 3 * passBytes / 150e9 * 1e3, err3))

    // ---- hybrid GPU + SME ----
    let planF = sme_fft_plan_create_dir(Int32(na), -1)!, planI = sme_fft_plan_create_dir(Int32(na), 1)!
    let nthr = 2
    let scratch = (0..<nthr).map { _ in
        UnsafeMutableRawPointer.allocate(byteCount: sme_fft_columns_scratch_bytes(planF), alignment: 16384)
            .bindMemory(to: Float.self, capacity: 1) }
    func smeCols(_ plan: OpaquePointer, _ iR: MTLBuffer, _ iI: MTLBuffer, _ oR: MTLBuffer, _ oI: MTLBuffer, _ t: Int) {
        let per = nr / t
        DispatchQueue.concurrentPerform(iterations: t) { k in
            if t > 1 { pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) }
            let off = k * per
            sme_fft_columns(plan, fp(iR) + off, fp(iI) + off, fp(oR) + off, fp(oI) + off, nr, Int32(per), scratch[k])
        }
    }
    func runHybrid(_ t: Int) -> [Double] {
        var ts = [Double]()
        var c = CFAbsoluteTimeGetCurrent()
        _ = runGPU([stRC]);                             ts.append(CFAbsoluteTimeGetCurrent() - c); c = CFAbsoluteTimeGetCurrent()
        smeCols(planF, aRe, aIm, bRe, bIm, t);          ts.append(CFAbsoluteTimeGetCurrent() - c); c = CFAbsoluteTimeGetCurrent()
        _ = runGPU([stRF]);                             ts.append(CFAbsoluteTimeGetCurrent() - c); c = CFAbsoluteTimeGetCurrent()
        smeCols(planI, aRe, aIm, bRe, bIm, t);          ts.append(CFAbsoluteTimeGetCurrent() - c)
        return ts
    }
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)
    _ = runHybrid(1)
    let hybImg: (Int, Int) -> (Float, Float) = { r, a in (fp(bRe)[a * nr + r], fp(bIm)[a * nr + r]) }
    let errH = relL2(fp(bRe), fp(bIm), np) { i in (refRe[(i % nr) * na + i / nr], refIm[(i % nr) * na + i / nr]) }
    print("\nHybrid pipeline (GPU rows + SME2 columns, no transpose; wall clock incl. sync, median of 7):")
    for t in [1, 2] {
        var rs = [[Double]]()
        for _ in 0..<7 { rs.append(runHybrid(t)) }
        let m = (0..<4).map { i in med(rs.map { $0[i] }) }, tot = med(rs.map { $0.reduce(0, +) })
        print(String(format: "  SME threads %d: GPU range compress %.2f | SME az FFT %.2f | GPU RCMC+filter %.2f | SME az IFFT %.2f | total %.2f ms",
                     t, m[0] * 1e3, m[1] * 1e3, m[2] * 1e3, m[3] * 1e3, tot * 1e3))
    }
    print(String(format: "  rel L2 vs reference: whole image %.2e, 64x64 around targets %.2e", errH,
                 relL2Targets(targets, hybImg, refImg)))
    printFocus("hybrid", targets, focus(targets, hybImg))

    // SME column FFT vs the GPU's transpose + row FFT for the azimuth FFT alone
    print(String(format: "\nAzimuth FFT alone: GPU transpose + row FFT %.2f ms; SME column FFT (1 thr) see above; vDSP transpose + rows %.1f ms",
                 (med(gStep[1]) + med(gStep[2])) * 1e3, (ct.tr + ct.az) * 1e3))

    // ---- co-scheduling: GPU pipeline on scene A while SME column FFTs run on scene B ----
    let cRe = buf(), cIm = buf()
    cRe.contents().copyMemory(from: rawRe.contents(), byteCount: np * 4)
    cIm.contents().copyMemory(from: rawIm.contents(), byteCount: np * 4)
    let dur = 2.0
    func gpuLoop(_ until: Double) -> Int {   // back-to-back pipelines, 2 command buffers in flight
        let sem = DispatchSemaphore(value: 2); var n = 0
        while CFAbsoluteTimeGetCurrent() < until {
            sem.wait()
            let cb = q.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
            for s in [stRC, stTR, stFR, stRI] { s(e) }
            e.endEncoding(); cb.addCompletedHandler { _ in sem.signal() }; cb.commit(); n += 1
        }
        for _ in 0..<2 { sem.wait() }
        for _ in 0..<2 { sem.signal() }   // libdispatch traps if freed below its initial value
        return n
    }
    func smeLoop(_ until: Double) -> Int {   // in-place forward column FFTs of a 4096 x 4096 image
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)
        var n = 0
        while CFAbsoluteTimeGetCurrent() < until {
            sme_fft_columns(planF, fp(cRe), fp(cIm), fp(cRe), fp(cIm), nr, Int32(nr), scratch[0]); n += 1
        }
        return n
    }
    var s0 = CFAbsoluteTimeGetCurrent()
    let gA = Double(gpuLoop(s0 + dur)) / (CFAbsoluteTimeGetCurrent() - s0)
    s0 = CFAbsoluteTimeGetCurrent()
    let sA = Double(smeLoop(s0 + dur)) / (CFAbsoluteTimeGetCurrent() - s0)
    s0 = CFAbsoluteTimeGetCurrent()
    var sN = 0
    let th = Thread { sN = smeLoop(s0 + dur) }; th.stackSize = 1 << 22; th.start()
    let gN = gpuLoop(s0 + dur)
    while th.isExecuting || !th.isFinished { usleep(1000) }
    let el = CFAbsoluteTimeGetCurrent() - s0
    let gC = Double(gN) / el, sC = Double(sN) / el
    print("\nCo-scheduling (\(dur) s windows; GPU = full 4-pass SAR scenes, SME = 4096-column FFT images, 256 MiB traffic each):")
    print(String(format: "  alone:      GPU %.1f scenes/s (%.0f GB/s) | SME %.1f images/s (%.0f GB/s)", gA, gA * 4 * passBytes * 1e-9, sA, sA * passBytes * 1e-9))
    print(String(format: "  concurrent: GPU %.1f scenes/s (%.0f GB/s) | SME %.1f images/s (%.0f GB/s) | total %.0f GB/s",
                 gC, gC * 4 * passBytes * 1e-9, sC, sC * passBytes * 1e-9, (gC * 4 + sC) * passBytes * 1e-9))

    sme_fft_plan_destroy(planF); sme_fft_plan_destroy(planI)
    for s in scratch { UnsafeMutableRawPointer(s).deallocate() }
    refRe.deallocate(); refIm.deallocate(); wRe.deallocate(); wIm.deallocate()
    vDSP_destroy_fftsetup(setup)
}
