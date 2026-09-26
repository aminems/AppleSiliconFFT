// =============================================================================
// m6_sar.metal — paper_m6 Stage 6: Range-Doppler SAR processing on M6, split re/im planes.
//
// Compiled at runtime after src/common/fft_device.h (fftdx::FFT<4096, 8>).
// Scene: NA pulses x NR range samples, row-major [az][rg], fp32, separate re and im planes.
//
// GPU-only path (4 DRAM passes, output transposed [rg][az]):
//   sar_range_compress   rows:  FFT -> x conj(chirp spectrum) -> IFFT        (fused)
//   sar_transpose        [az][rg] -> [rg][az]
//   sar_fft_rows         azimuth FFT along rows of [rg][az]
//   sar_rcmc_azc_ifft_t  per range row: RCMC (8-tap sinc over neighbouring rows)
//                        x azimuth filter -> azimuth IFFT                      (fused)
// Hybrid path (GPU rows + SME2 CPU columns, 4 passes, no transpose, output [az][rg]):
//   sar_range_compress -> [CPU SME column FFT] -> sar_rcmc_azfilter -> [CPU SME column IFFT]
//
// RCMC: output range bin n reads input at x = n + dn, dn = (1/D(fa) - 1) R0(n) 2 fs / c,
// D(fa) = sqrt(1 - (lambda fa / 2v)^2). Taps in[m-3 .. m+4], m = floor(x), weights blended
// linearly between the two nearest rows of a 65-row (phase 0..64/64) windowed-sinc table.
// (Nearest-row selection is discontinuous: fp32 vs double positions pick different rows for ~2%
// of samples, a 1.5e-3 image difference.) Azimuth filter, baseband form: exp(+j 4 pi R0(n) (D - 1) / lambda).
// (The full exp(+j 4 pi R0 D / lambda) differs by a constant phase per range bin, i.e. a range-
// spectrum shift of 2 dr / lambda cycles per sample, which wraps the range band.)
// The phase reaches ~1e4 rad at |fa| = PRF/2, so it is evaluated in cycles as
//   cyc = azA[k] + n * (azBhi[k] + azBlo[k])   (mod 1),  R0(n) = r0_start + n dr,
// with azA = fract(2 r0_start (D-1) / lambda) and azB = 2 dr (D-1) / lambda from the host in double,
// azBhi rounded to 11 significant bits so that n * azBhi (n < 4096) is exact in fp32. A plain fp32
// product gives ~1e-3 rad phase errors (1.5e-3 relative image error vs a double-phase reference).
// =============================================================================

struct SarParams {
    uint  nr, na;        // range samples, pulses
    float prf, v, lambda_;
    float r0_start, dr;  // slant range of bin 0 and per bin (c / 2fs)
    float scale;         // output scale (1/NA for the SME inverse, which is unnormalized)
};

constant uint SAR_N = 4096;
typedef fftdx::FFT<SAR_N, 8u> SarFFT;

inline float sar_fa(uint k, constant SarParams& p) {
    int kk = int(k) < int(p.na / 2) ? int(k) : int(k) - int(p.na);
    return float(kk) * p.prf / float(p.na);
}
inline float sar_dminus1(float fa, constant SarParams& p) {   // D - 1, accurate for small fa
    float s = p.lambda_ * fa / (2.0f * p.v);
    float s2 = s * s;
    return -s2 / (1.0f + sqrt(1.0f - s2));
}

// ---- 1. range compression, one threadgroup (512 threads) per pulse ---------------------------
struct RCLoad {
    device const float* re; device const float* im;
    float2 operator()(uint i) const { return float2(re[i], im[i]); }
};
struct RCMid {
    device const float2* h;
    float2 operator()(uint k, float2 X) const { return fftdx::cmul(X, h[k]); }
};
struct RCStore {
    device float* re; device float* im;
    void operator()(uint i, float2 y) const { re[i] = y.x; im[i] = y.y; }
};

kernel void sar_range_compress(device const float* ire [[buffer(0)]], device const float* iim [[buffer(1)]],
                               device float* ore [[buffer(2)]], device float* oim [[buffer(3)]],
                               device const float2* hr [[buffer(4)]],
                               uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float2 smem[SAR_N];
    const size_t b = size_t(row) * SAR_N;
    SarFFT::convolve(smem, tid, RCLoad{ire + b, iim + b}, RCMid{hr}, RCStore{ore + b, oim + b});
}

// ---- 2. transpose, 32x32 tiles, z = plane (0 re, 1 im) ------------------------------------
kernel void sar_transpose(device const float* ire [[buffer(0)]], device const float* iim [[buffer(1)]],
                          device float* ore [[buffer(2)]], device float* oim [[buffer(3)]],
                          constant uint2& dims [[buffer(4)]],   // (cols, rows) of the input
                          uint3 tg [[threadgroup_position_in_grid]], uint3 t [[thread_position_in_threadgroup]]) {
    threadgroup float tile[32][33];
    device const float* in = tg.z == 0 ? ire : iim;
    device float* out = tg.z == 0 ? ore : oim;
    const uint cols = dims.x, rows = dims.y;
    const uint x0 = tg.x * 32, y0 = tg.y * 32;
    for (uint k = t.y; k < 32; k += 8) tile[k][t.x] = in[size_t(y0 + k) * cols + x0 + t.x];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = t.y; k < 32; k += 8) out[size_t(x0 + k) * rows + y0 + t.x] = tile[t.x][k];
}

// ---- 3. plain forward FFT along rows ------------------------------------------------------
kernel void sar_fft_rows(device const float* ire [[buffer(0)]], device const float* iim [[buffer(1)]],
                         device float* ore [[buffer(2)]], device float* oim [[buffer(3)]],
                         uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float2 smem[SAR_N];
    const size_t b = size_t(row) * SAR_N;
    SarFFT::execute(smem, tid, RCLoad{ire + b, iim + b}, RCStore{ore + b, oim + b});
}

// RCMC + azimuth filter for output (range bin n, Doppler bin k). `at(m)` returns input at range m.
template <class At>
inline float2 rcmc_azf(uint n, uint k, constant SarParams& p, device const float* tab,
                       device const float* azA, device const float* azBhi, device const float* azBlo, At at) {
    const float fa = sar_fa(k, p);
    const float dm1 = sar_dminus1(fa, p);                      // D - 1
    const float r0 = p.r0_start + float(n) * p.dr;
    const float dn = (-dm1 / (1.0f + dm1)) * r0 / p.dr;        // (1/D - 1) R0 / dr
    const float fl = floor(dn);                                // n is an integer: take the fraction
    const int m = int(n) + int(fl);                            // from dn alone (exact), not from n + dn
    const float ph64 = (dn - fl) * 64.0f;                      // table phase, blended linearly
    const int f = min(int(ph64), 63);
    const float al = ph64 - float(f);
    device const float* w0 = tab + f * 8;
    device const float* w1 = w0 + 8;
    float2 acc = 0.0f;
    for (int j = 0; j < 8; ++j) {
        int mm = m - 3 + j;
        if (mm >= 0 && mm < int(p.nr)) acc += mix(w0[j], w1[j], al) * at(uint(mm));
    }
    float ch = float(n) * azBhi[k];                            // exact
    ch -= floor(ch);                                           // exact
    float cyc = ch + float(n) * azBlo[k] + azA[k];
    cyc -= floor(cyc);
    float s, c; s = sincos(2.0f * M_PI_F * cyc, c);
    return fftdx::cmul(acc, float2(c, s)) * p.scale;
}

// ---- 4. GPU path: [rg][fa] layout, one threadgroup per range row: RCMC + az filter + IFFT ------
struct ColAt {   // input sample at (range row m, Doppler column k) of the transposed layout
    device const float* re; device const float* im; uint na; uint k;
    float2 operator()(uint m) const { size_t i = size_t(m) * na + k; return float2(re[i], im[i]); }
};

kernel void sar_rcmc_azc_ifft_t(device const float* ire [[buffer(0)]], device const float* iim [[buffer(1)]],
                                device float* ore [[buffer(2)]], device float* oim [[buffer(3)]],
                                constant SarParams& p [[buffer(4)]], device const float* tab [[buffer(5)]],
                                device const float* azA [[buffer(6)]], device const float* azBhi [[buffer(7)]],
                                device const float* azBlo [[buffer(8)]],
                                uint n [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float2 smem[SAR_N];
    constexpr uint T = SarFFT::threads();
    float2 td[8];
    for (uint j = 0; j < 8; ++j) {
        const uint k = tid + j * T;
        td[j] = fftdx::cconj(rcmc_azf(n, k, p, tab, azA, azBhi, azBlo, ColAt{ire, iim, p.na, k}));
    }
    SarFFT::forward<false>(td, smem, tid);                     // conj-FFT = N x IFFT, conjugated
    const float inv = 1.0f / float(SAR_N);
    const size_t b = size_t(n) * p.na;
    for (uint j = 0; j < 8; ++j) {
        const uint k = tid + j * T;
        ore[b + k] = td[j].x * inv;
        oim[b + k] = -td[j].y * inv;
    }
}

// ---- 5. hybrid path: [fa][rg] layout, one thread per output: RCMC + az filter (row op) ----------
struct RowAt {
    device const float* re; device const float* im;
    float2 operator()(uint m) const { return float2(re[m], im[m]); }
};

kernel void sar_rcmc_azfilter(device const float* ire [[buffer(0)]], device const float* iim [[buffer(1)]],
                              device float* ore [[buffer(2)]], device float* oim [[buffer(3)]],
                              constant SarParams& p [[buffer(4)]], device const float* tab [[buffer(5)]],
                              device const float* azA [[buffer(6)]], device const float* azBhi [[buffer(7)]],
                              device const float* azBlo [[buffer(8)]],
                              uint2 g [[thread_position_in_grid]]) {   // g.x = range bin n, g.y = Doppler bin k
    const size_t b = size_t(g.y) * p.nr;
    float2 v = rcmc_azf(g.x, g.y, p, tab, azA, azBhi, azBlo, RowAt{ire + b, iim + b});
    ore[b + g.x] = v.x;
    oim[b + g.x] = v.y;
}

// ---- 3'. transpose fused into the azimuth FFT: one threadgroup per range bin reads its column of
//          [az][rg] with stride nr (4 B per row; neighbouring threadgroups share the lines) and
//          writes the spectrum as a contiguous row of [rg][fa]. Replaces sar_transpose + sar_fft_rows.
struct StridedLoad {
    device const float* re; device const float* im; uint stride;
    float2 operator()(uint i) const { size_t j = size_t(i) * stride; return float2(re[j], im[j]); }
};

kernel void sar_fft_cols_t(device const float* ire [[buffer(0)]], device const float* iim [[buffer(1)]],
                           device float* ore [[buffer(2)]], device float* oim [[buffer(3)]],
                           constant uint& nr [[buffer(4)]],
                           uint col [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float2 smem[SAR_N];
    const size_t b = size_t(col) * SAR_N;
    SarFFT::execute(smem, tid, StridedLoad{ire + col, iim + col, nr}, RCStore{ore + b, oim + b});
}

// ---- fp16-storage variants of the GPU path: the three intermediate images are half planes
//      (FFT arithmetic and RCMC accumulation stay fp32). The azimuth FFT stores X / 4096 so that
//      the compressed-azimuth peak (~1e6 for these targets) fits fp16; the last kernel undoes it
//      through SarParams.scale = 4096. Bytes per point per pipeline: 40 instead of 64.
struct HLoad {
    device const half* re; device const half* im;
    float2 operator()(uint i) const { return float2(float(re[i]), float(im[i])); }
};
struct HStore {
    device half* re; device half* im; float s;
    void operator()(uint i, float2 y) const { re[i] = half(y.x * s); im[i] = half(y.y * s); }
};

kernel void sar_range_compress_h(device const float* ire [[buffer(0)]], device const float* iim [[buffer(1)]],
                                 device half* ore [[buffer(2)]], device half* oim [[buffer(3)]],
                                 device const float2* hr [[buffer(4)]],
                                 uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float2 smem[SAR_N];
    const size_t b = size_t(row) * SAR_N;
    SarFFT::convolve(smem, tid, RCLoad{ire + b, iim + b}, RCMid{hr}, HStore{ore + b, oim + b, 1.0f});
}

kernel void sar_transpose_h(device const half* ire [[buffer(0)]], device const half* iim [[buffer(1)]],
                            device half* ore [[buffer(2)]], device half* oim [[buffer(3)]],
                            constant uint2& dims [[buffer(4)]],
                            uint3 tg [[threadgroup_position_in_grid]], uint3 t [[thread_position_in_threadgroup]]) {
    threadgroup half tile[32][34];
    device const half* in = tg.z == 0 ? ire : iim;
    device half* out = tg.z == 0 ? ore : oim;
    const uint cols = dims.x, rows = dims.y;
    const uint x0 = tg.x * 32, y0 = tg.y * 32;
    for (uint k = t.y; k < 32; k += 8) tile[k][t.x] = in[size_t(y0 + k) * cols + x0 + t.x];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = t.y; k < 32; k += 8) out[size_t(x0 + k) * rows + y0 + t.x] = tile[t.x][k];
}

kernel void sar_fft_rows_h(device const half* ire [[buffer(0)]], device const half* iim [[buffer(1)]],
                           device half* ore [[buffer(2)]], device half* oim [[buffer(3)]],
                           uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float2 smem[SAR_N];
    const size_t b = size_t(row) * SAR_N;
    SarFFT::execute(smem, tid, HLoad{ire + b, iim + b}, HStore{ore + b, oim + b, 1.0f / float(SAR_N)});
}

struct ColAtH {
    device const half* re; device const half* im; uint na; uint k;
    float2 operator()(uint m) const { size_t i = size_t(m) * na + k; return float2(float(re[i]), float(im[i])); }
};

kernel void sar_rcmc_azc_ifft_th(device const half* ire [[buffer(0)]], device const half* iim [[buffer(1)]],
                                 device float* ore [[buffer(2)]], device float* oim [[buffer(3)]],
                                 constant SarParams& p [[buffer(4)]], device const float* tab [[buffer(5)]],
                                 device const float* azA [[buffer(6)]], device const float* azBhi [[buffer(7)]],
                                 device const float* azBlo [[buffer(8)]],
                                 uint n [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float2 smem[SAR_N];
    constexpr uint T = SarFFT::threads();
    float2 td[8];
    for (uint j = 0; j < 8; ++j) {
        const uint k = tid + j * T;
        td[j] = fftdx::cconj(rcmc_azf(n, k, p, tab, azA, azBhi, azBlo, ColAtH{ire, iim, p.na, k}));
    }
    SarFFT::forward<false>(td, smem, tid);
    const float inv = 1.0f / float(SAR_N);
    const size_t b = size_t(n) * p.na;
    for (uint j = 0; j < 8; ++j) {
        const uint k = tid + j * T;
        ore[b + k] = td[j].x * inv;
        oim[b + k] = -td[j].y * inv;
    }
}
