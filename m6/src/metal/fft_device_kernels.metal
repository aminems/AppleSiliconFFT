#include "../common/fft_device.h"
using namespace metal;

// =============================================================================
// KERNEL INDEX — these are demonstrator / benchmark / reusable-pattern kernels
// built on the fft_device.h primitive. They are NOT the primitive itself (that
// is the header); think of them as worked examples + the bodies the dx
// benchmarks and the SAR --dx pipeline dispatch.
//
//   Reusable in production (dispatched by SARRadar --dx):
//     range_compression_dx, multiply_ifft_dx
//   Building blocks / patterns (1D, fp16, 2D, four-step, conv):
//     fft4096_dx, fft4096_dx_f16(smem), fft2d_rows/cols_*, fft2dconv_*,
//     fourstep16384_*_b(h), transpose2d_tiled, ifft4096_dx
//   Callback-API demos:
//     fft4096_dx_cb, range_compression_dx_cb, fft4096_hann_dx_cb
//   Test/validation only (exercised by dx-test / dx-radix):
//     fft4_dx_r4, fft8_dx_r8, fft64_dx_r8, fft256_dx_r4, fft4096_dx_r4/_r16,
//     range_compress_detect_dx, emul_filter_4096, mag2_4096
//
// To use the primitive in your OWN kernel, see docs/FFT_DEVICE_API.md — you do
// not need this file, only fft_device.h.
// =============================================================================
//
// =============================================================================
// Demo / validation kernels for the device-side FFT primitive (fft_device.h).
//
// fft4096_dx           — bare N=4096 radix-8 transform through the primitive.
//                        Must reproduce fft_4096_batched.metal at the same
//                        GFLOPS (zero-overhead abstraction check).
// range_compression_dx — IFFT( FFT(x) .* filter ) in ONE dispatch, with the
//                        FFT->multiply->IFFT midpoint handed off in registers
//                        (no extra threadgroup round-trip). The cuFFTDx-style
//                        fused convolution that the SAR range/azimuth stages use.
//
// We fully-qualify fftdx:: (no `using namespace fftdx`) so these can be compiled
// in the same translation unit as fft_4096_batched.metal — whose file-scope
// cmul()/SQRT2_2 would otherwise be ambiguous with the namespaced versions.
// =============================================================================

kernel void fft4096_dx(
    device const float2* input  [[buffer(0)]],
    device float2*       output [[buffer(1)]],
    uint tid   [[thread_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]]
) {
    constexpr uint N = 4096u, R = 8u, T = N / R;
    threadgroup float2 smem[N];
    const uint base = tg_id * N;

    float2 td[R];
    #pragma unroll
    for (uint j = 0; j < R; ++j) td[j] = input[base + tid + j * T];

    fftdx::FFT<N, R>::forward(td, smem, tid);

    #pragma unroll
    for (uint j = 0; j < R; ++j) output[base + tid + j * T] = td[j];
}

kernel void range_compression_dx(
    device const float2* input  [[buffer(0)]],
    device float2*       output [[buffer(1)]],
    device const float2* filter [[buffer(2)]],   // matched filter, freq domain, natural order
    uint tid   [[thread_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]]
) {
    constexpr uint N = 4096u, R = 8u, T = N / R;
    threadgroup float2 smem[N];
    const uint base = tg_id * N;

    float2 td[R];
    #pragma unroll
    for (uint j = 0; j < R; ++j) td[j] = input[base + tid + j * T];

    // Forward FFT: td[j] now holds X[tid + j*T] in natural order.
    fftdx::FFT<N, R>::forward(td, smem, tid);

    // Matched-filter multiply in the frequency domain, then conjugate for the
    // IFFT (IFFT(z) = (1/N) * conj(FFT(conj(z)))). Stays entirely in registers.
    #pragma unroll
    for (uint j = 0; j < R; ++j) {
        float2 xf = fftdx::cmul(td[j], filter[tid + j * T]);
        td[j] = fftdx::cconj(xf);
    }

    // conj-FFT (GUARD=true: chained transform reusing smem).
    fftdx::FFT<N, R>::forward<true>(td, smem, tid);

    // conj + 1/N scale completes the IFFT; natural-order time output.
    constexpr float inv = 1.0f / float(N);
    #pragma unroll
    for (uint j = 0; j < R; ++j)
        output[base + tid + j * T] = float2(td[j].x * inv, -td[j].y * inv);
}

// =============================================================================
// Callback-API demo (fft_device.h execute()/convolve()). With the load/mid/store
// functors, a fused kernel collapses to one call — and is bit-identical to the
// hand-written versions above (same forward(), same math).
// =============================================================================

// Plain functor structs (the cuFFTDx-callback analog). Each just wraps a
// device pointer + base offset; passed by value, inlined by the compiler.
struct DXLoad   { device const float2* p; uint base; float2 operator()(uint i) const { return p[base + i]; } };
struct DXStore  { device float2*       p; uint base; void   operator()(uint i, float2 v) const { p[base + i] = v; } };
struct DXFilter { device const float2* f; float2 operator()(uint k, float2 X) const { return fftdx::cmul(X, f[k]); } };

// Hann-window-on-load: a NEW fusion that the callback API makes a one-liner —
// the window multiply is folded into the FFT's input read (zero extra traffic).
struct DXHannLoad {
    device const float2* p; uint base;
    float2 operator()(uint i) const {
        float w = 0.5f * (1.0f - precise::cos(2.0f * M_PI_F * float(i) / float(4096u - 1u)));
        return p[base + i] * w;
    }
};

kernel void fft4096_dx_cb(
    device const float2* input  [[buffer(0)]],
    device float2*       output [[buffer(1)]],
    uint tid   [[thread_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]]
) {
    threadgroup float2 smem[4096];
    const uint base = tg_id * 4096u;
    fftdx::FFT<4096u, 8u>::execute(smem, tid, DXLoad{input, base}, DXStore{output, base});
}

kernel void range_compression_dx_cb(
    device const float2* input  [[buffer(0)]],
    device float2*       output [[buffer(1)]],
    device const float2* filter [[buffer(2)]],
    uint tid   [[thread_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]]
) {
    threadgroup float2 smem[4096];
    const uint base = tg_id * 4096u;
    fftdx::FFT<4096u, 8u>::convolve(smem, tid,
        DXLoad{input, base}, DXFilter{filter}, DXStore{output, base});
}

// Windowed FFT: Hann window fused into the load. One call.
kernel void fft4096_hann_dx_cb(
    device const float2* input  [[buffer(0)]],
    device float2*       output [[buffer(1)]],
    uint tid   [[thread_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]]
) {
    threadgroup float2 smem[4096];
    const uint base = tg_id * 4096u;
    fftdx::FFT<4096u, 8u>::execute(smem, tid, DXHannLoad{input, base}, DXStore{output, base});
}

// =============================================================================
// fp16-storage / fp32-accumulate variants (radar-usable ~60 dB). td stays
// float2 (fp32 accumulate); only the inter-stage threadgroup buffer is half2
// (16 KiB vs 32). Apple's free fp16<->fp32 conversion makes the boundary cast
// cost nothing, and the smaller smem can raise occupancy past 1 TG/core.
// =============================================================================

// fp32 device I/O, fp16 smem — isolates the cost of fp16 inter-stage storage.
kernel void fft4096_dx_f16smem(
    device const float2* input  [[buffer(0)]],
    device float2*       output [[buffer(1)]],
    uint tid   [[thread_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]]
) {
    constexpr uint N = 4096u, R = 8u, T = N / R;
    threadgroup half2 smem[N];                     // 16 KiB
    const uint base = tg_id * N;
    float2 td[R];
    #pragma unroll
    for (uint j = 0; j < R; ++j) td[j] = input[base + tid + j * T];
    fftdx::FFT<N, R, half2>::forward(td, smem, tid);
    #pragma unroll
    for (uint j = 0; j < R; ++j) output[base + tid + j * T] = td[j];
}

// Fully fp16: half2 device I/O + half2 smem, fp32 accumulate. Halves device
// bandwidth — the realistic fp16 SAR datapath.
kernel void fft4096_dx_f16(
    device const half2* input  [[buffer(0)]],
    device half2*       output [[buffer(1)]],
    uint tid   [[thread_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]]
) {
    constexpr uint N = 4096u, R = 8u, T = N / R;
    threadgroup half2 smem[N];
    const uint base = tg_id * N;
    float2 td[R];
    #pragma unroll
    for (uint j = 0; j < R; ++j) td[j] = float2(input[base + tid + j * T]);  // fp16 -> fp32 reg
    fftdx::FFT<N, R, half2>::forward(td, smem, tid);
    #pragma unroll
    for (uint j = 0; j < R; ++j) output[base + tid + j * T] = half2(td[j]);  // fp32 reg -> fp16
}

// =============================================================================
// SAR range compression, end-to-end fusion test (N=4096).
//
// FULLY FUSED (one dispatch): Hann window -> FFT -> matched-filter multiply ->
// IFFT -> magnitude-squared detection. The store callback writes a real float
// |y|^2, so the whole windowed-correlate-and-detect collapses to ONE kernel
// launch with NO intermediate global-memory buffers — strictly more fused than
// the repo's hand-written fused_range_compression (which still emits complex
// output and leaves windowing + detection as separate dispatches).
// =============================================================================

// Magnitude-squared store: convolve() hands us the 1/N-scaled time sample; we
// write |y|^2 as a real float (halves output bandwidth, removes a dispatch).
struct DXMagStore { device float* p; uint base; void operator()(uint i, float2 y) const { p[base + i] = y.x * y.x + y.y * y.y; } };

kernel void range_compress_detect_dx(
    device const float2* input  [[buffer(0)]],   // raw echo (nLines x 4096), complex
    device float*        output [[buffer(1)]],   // detected magnitude^2 (nLines x 4096), real
    device const float2* filter [[buffer(2)]],   // matched filter conj(chirp spectrum), 4096
    uint tid   [[thread_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]]
) {
    threadgroup float2 smem[4096];
    const uint base = tg_id * 4096u;
    fftdx::FFT<4096u, 8u>::convolve(smem, tid,
        DXHannLoad{input, base}, DXFilter{filter}, DXMagStore{output, base});
}

// Multiply + IFFT in one dispatch (frequency-domain data already FFT'd) — the
// primitive equivalent of fused_multiply_ifft, used by the RDA azimuth stage.
// Same 4-buffer signature: input(freq), output(time), filter, params[singleRow].
struct DXFreqMulConjLoad {
    device const float2* x; device const float2* f; uint base; uint fbase;
    float2 operator()(uint i) const { return fftdx::cconj(fftdx::cmul(x[base + i], f[fbase + i])); }
};
kernel void multiply_ifft_dx(
    device const float2* input  [[buffer(0)]],
    device float2*       output [[buffer(1)]],
    device const float2* filter [[buffer(2)]],
    device const uint*   params [[buffer(3)]],
    uint tid   [[thread_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]]
) {
    constexpr uint N = 4096u, R = 8u, T = N / R;
    threadgroup float2 smem[N];
    const uint base  = tg_id * N;
    const uint fbase = params[0] ? 0u : base;   // single-row vs per-row filter
    float2 td[R];
    #pragma unroll
    for (uint j = 0; j < R; ++j) td[j] = DXFreqMulConjLoad{input, filter, base, fbase}(tid + j * T);
    fftdx::FFT<N, R>::forward(td, smem, tid);    // conj-FFT (the IFFT body)
    constexpr float inv = 1.0f / float(N);
    #pragma unroll
    for (uint j = 0; j < R; ++j) output[base + tid + j * T] = float2(td[j].x * inv, -td[j].y * inv);
}

// ---- Unfused equivalent: the SAME computation as 4 dispatches + 3 global
//      round-trips, for the wall-clock comparison. Bit-matches the fused kernel.

// (1) windowed forward FFT == fft4096_hann_dx_cb above.

// (2) elementwise frequency-domain matched-filter multiply.
kernel void emul_filter_4096(
    device const float2* in     [[buffer(0)]],
    device const float2* filter [[buffer(1)]],
    device float2*       out    [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    out[gid] = fftdx::cmul(in[gid], filter[gid & 4095u]);
}

// (3) inverse FFT: IFFT(z) = (1/N) conj(FFT(conj z)).
kernel void ifft4096_dx(
    device const float2* input  [[buffer(0)]],
    device float2*       output [[buffer(1)]],
    uint tid   [[thread_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]]
) {
    constexpr uint N = 4096u, R = 8u, T = N / R;
    threadgroup float2 smem[N];
    const uint base = tg_id * N;
    float2 td[R];
    #pragma unroll
    for (uint j = 0; j < R; ++j) td[j] = fftdx::cconj(input[base + tid + j * T]);
    fftdx::FFT<N, R>::forward(td, smem, tid);
    constexpr float inv = 1.0f / float(N);
    #pragma unroll
    for (uint j = 0; j < R; ++j) output[base + tid + j * T] = float2(td[j].x * inv, -td[j].y * inv);
}

// (4) magnitude-squared detection.
kernel void mag2_4096(
    device const float2* in  [[buffer(0)]],
    device float*        out [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    float2 v = in[gid];
    out[gid] = v.x * v.x + v.y * v.y;
}

// =============================================================================
// Radix-genericity checks: the SAME primitive instantiated at different
// (N, R), including the radix-4 leaf and the exact sub-FFT sizes a four-step
// N=16384 = 256 x 64 needs (256 = 4^4 via radix-4, 64 = 8^2 via radix-8).
// Each is validated against vDSP in the `dx-radix` benchmark.
// =============================================================================
kernel void fft4096_dx_r4(                       // N=4096 = 4^6, radix-4 (1024 thr/TG)
    device const float2* input [[buffer(0)]], device float2* output [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]], uint tg_id [[threadgroup_position_in_grid]]
) {
    constexpr uint N = 4096u, R = 4u, T = N / R;
    threadgroup float2 smem[N]; const uint base = tg_id * N;
    float2 td[R];
    #pragma unroll
    for (uint j = 0; j < R; ++j) td[j] = input[base + tid + j * T];
    fftdx::FFT<N, R>::forward(td, smem, tid);
    #pragma unroll
    for (uint j = 0; j < R; ++j) output[base + tid + j * T] = td[j];
}

kernel void fft256_dx_r4(                        // N=256 = 4^4, radix-4 (64 thr/TG)
    device const float2* input [[buffer(0)]], device float2* output [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]], uint tg_id [[threadgroup_position_in_grid]]
) {
    constexpr uint N = 256u, R = 4u, T = N / R;
    threadgroup float2 smem[N]; const uint base = tg_id * N;
    float2 td[R];
    #pragma unroll
    for (uint j = 0; j < R; ++j) td[j] = input[base + tid + j * T];
    fftdx::FFT<N, R>::forward(td, smem, tid);
    #pragma unroll
    for (uint j = 0; j < R; ++j) output[base + tid + j * T] = td[j];
}

kernel void fft4096_dx_r16(                       // N=4096 = 16^3, radix-16 (256 thr/TG)
    device const float2* input [[buffer(0)]], device float2* output [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]], uint tg_id [[threadgroup_position_in_grid]]
) {
    constexpr uint N = 4096u, R = 16u, T = N / R;
    threadgroup float2 smem[N]; const uint base = tg_id * N;
    float2 td[R];
    #pragma unroll
    for (uint j = 0; j < R; ++j) td[j] = input[base + tid + j * T];
    fftdx::FFT<N, R>::forward(td, smem, tid);
    #pragma unroll
    for (uint j = 0; j < R; ++j) output[base + tid + j * T] = td[j];
}

// Single-stage edge cases (N == R): exercise the terminal StageImpl path with
// no inter-stage smem exchange (1 thread/TG, EPT = R).
kernel void fft8_dx_r8(                            // N=8 = 8^1, radix-8 (1 thr/TG)
    device const float2* input [[buffer(0)]], device float2* output [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]], uint tg_id [[threadgroup_position_in_grid]]
) {
    constexpr uint N = 8u, R = 8u, T = N / R;       // T = 1
    threadgroup float2 smem[N]; const uint base = tg_id * N;
    float2 td[R];
    #pragma unroll
    for (uint j = 0; j < R; ++j) td[j] = input[base + tid + j * T];
    fftdx::FFT<N, R>::forward(td, smem, tid);
    #pragma unroll
    for (uint j = 0; j < R; ++j) output[base + tid + j * T] = td[j];
}
kernel void fft4_dx_r4(                            // N=4 = 4^1, radix-4 (1 thr/TG)
    device const float2* input [[buffer(0)]], device float2* output [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]], uint tg_id [[threadgroup_position_in_grid]]
) {
    constexpr uint N = 4u, R = 4u, T = N / R;       // T = 1
    threadgroup float2 smem[N]; const uint base = tg_id * N;
    float2 td[R];
    #pragma unroll
    for (uint j = 0; j < R; ++j) td[j] = input[base + tid + j * T];
    fftdx::FFT<N, R>::forward(td, smem, tid);
    #pragma unroll
    for (uint j = 0; j < R; ++j) output[base + tid + j * T] = td[j];
}

kernel void fft64_dx_r8(                          // N=64 = 8^2, radix-8 (8 thr/TG)
    device const float2* input [[buffer(0)]], device float2* output [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]], uint tg_id [[threadgroup_position_in_grid]]
) {
    constexpr uint N = 64u, R = 8u, T = N / R;
    threadgroup float2 smem[N]; const uint base = tg_id * N;
    float2 td[R];
    #pragma unroll
    for (uint j = 0; j < R; ++j) td[j] = input[base + tid + j * T];
    fftdx::FFT<N, R>::forward(td, smem, tid);
    #pragma unroll
    for (uint j = 0; j < R; ++j) output[base + tid + j * T] = td[j];
}

// =============================================================================
// Four-step N = 16384 = N1(256) x N2(64), COMPOSED from the device primitive.
// Proves the primitive scales past the 32 KiB single-threadgroup ceiling
// (16384 complex = 128 KiB). Index map: n = n2*N1 + n1, k = k1*N2 + k2.
//   X[k1*N2+k2] = sum_n1 W_N1^{n1 k1} W_N^{n1 k2} ( sum_n2 x[n2*N1+n1] W_N2^{n2 k2} )
// Stage A: N1 FFTs of length N2=64 (over n2, stride N1) + mid twiddle W_N^{n1 k2}.
// Stage B: N2 FFTs of length N1=256 (over n1, stride N2), scattered to output.
// The callback API expresses the strided gather (load) and the twiddle/transpose
// (store) directly — the four-step "glue" is just two functor pairs.
// =============================================================================
constant uint FS_N  = 16384u;
constant uint FS_N1 = 256u;     // stage-B FFT length (radix-4: 4^4)
constant uint FS_N2 = 64u;      // stage-A FFT length (radix-8: 8^2)

// Stage A: thread block tg_id encodes (batch b, column n1); FFT-64 over n2.
struct FSLoadA  { device const float2* x; uint base; uint n1;
                  float2 operator()(uint n2) const { return x[base + n2 * FS_N1 + n1]; } };
struct FSStoreA { device float2* z; uint base; uint n1;
                  void operator()(uint k2, float2 Y) const {
                      float ang = -2.0f * M_PI_F * float(n1 * k2) / float(FS_N);   // W_N^{n1 k2}
                      float s, c; s = sincos(ang, c);
                      z[base + n1 * FS_N2 + k2] = fftdx::cmul(Y, float2(c, s));
                  } };

kernel void fourstep16384_A(
    device const float2* input [[buffer(0)]],
    device float2*       zbuf  [[buffer(1)]],
    uint tid   [[thread_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]]
) {
    threadgroup float2 smem[FS_N2];
    const uint b    = tg_id / FS_N1;        // batch index
    const uint n1   = tg_id % FS_N1;        // column
    const uint base = b * FS_N;
    fftdx::FFT<FS_N2, 8u>::execute(smem, tid, FSLoadA{input, base, n1}, FSStoreA{zbuf, base, n1});
}

// Stage B: tg_id encodes (batch b, row k2); FFT-256 over n1, scatter to output.
struct FSLoadB  { device const float2* z; uint base; uint k2;
                  float2 operator()(uint n1) const { return z[base + n1 * FS_N2 + k2]; } };
struct FSStoreB { device float2* out; uint base; uint k2;
                  void operator()(uint k1, float2 X) const { out[base + k1 * FS_N2 + k2] = X; } };

kernel void fourstep16384_B(
    device const float2* zbuf   [[buffer(0)]],
    device float2*       output [[buffer(1)]],
    uint tid   [[thread_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]]
) {
    threadgroup float2 smem[FS_N1];
    const uint b    = tg_id / FS_N2;        // batch index
    const uint k2   = tg_id % FS_N2;        // row
    const uint base = b * FS_N;
    fftdx::FFT<FS_N1, 4u>::execute(smem, tid, FSLoadB{zbuf, base, k2}, FSStoreB{output, base, k2});
}

// ---- Batched four-step: pack many sub-FFTs per threadgroup so each TG runs
//      512 threads (vs 8 / 64), fixing the small-sub-FFT occupancy waste. Each
//      sub-FFT owns a private smem slice; whole-TG barriers stay correct because
//      all sub-FFTs march the same stages in lockstep.
constant uint FS_GA = 64u;   // stage-A sub-FFTs (64-pt) per TG -> 64*8  = 512 threads
constant uint FS_GB = 8u;    // stage-B sub-FFTs (256-pt) per TG -> 8*64 = 512 threads

kernel void fourstep16384_A_b(
    device const float2* input [[buffer(0)]],
    device float2*       zbuf  [[buffer(1)]],
    uint tid   [[thread_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]]
) {
    threadgroup float2 smem[FS_GA * FS_N2];     // 64*64 = 4096 float2 = 32 KiB
    constexpr uint T = FS_N2 / 8u;              // 8 threads / sub-FFT
    const uint f    = tid / T;                  // sub-FFT within TG  [0, FS_GA)
    const uint lt   = tid % T;                  // local thread id    [0, T)
    const uint gcol = tg_id * FS_GA + f;        // global column = b*N1 + n1
    const uint b    = gcol / FS_N1;
    const uint n1   = gcol % FS_N1;
    const uint base = b * FS_N;
    fftdx::FFT<FS_N2, 8u>::execute(smem + f * FS_N2, lt,
        FSLoadA{input, base, n1}, FSStoreA{zbuf, base, n1});
}

kernel void fourstep16384_B_b(
    device const float2* zbuf   [[buffer(0)]],
    device float2*       output [[buffer(1)]],
    uint tid   [[thread_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]]
) {
    threadgroup float2 smem[FS_GB * FS_N1];     // 8*256 = 2048 float2 = 16 KiB
    constexpr uint T = FS_N1 / 4u;              // 64 threads / sub-FFT
    const uint f    = tid / T;                  // sub-FFT within TG  [0, FS_GB)
    const uint lt   = tid % T;                  // local thread id    [0, T)
    const uint grow = tg_id * FS_GB + f;        // global row = b*N2 + k2
    const uint b    = grow / FS_N2;
    const uint k2   = grow % FS_N2;
    const uint base = b * FS_N;
    fftdx::FFT<FS_N1, 4u>::execute(smem + f * FS_N1, lt,
        FSLoadB{zbuf, base, k2}, FSStoreB{output, base, k2});
}

// ---- fp16-Z four-step: store the intermediate transpose buffer (and the
//      sub-FFT smem) in fp16 to halve the stage-A->B round-trip. fp32 accumulate
//      in registers throughout; only Z and smem are fp16.
struct FSStoreA_h { device half2* z; uint base; uint n1;
    void operator()(uint k2, float2 Y) const {
        float ang = -2.0f * M_PI_F * float(n1 * k2) / float(FS_N);
        float s, c; s = sincos(ang, c);
        z[base + n1 * FS_N2 + k2] = half2(fftdx::cmul(Y, float2(c, s)));
    } };
struct FSLoadB_h { device const half2* z; uint base; uint k2;
    float2 operator()(uint n1) const { return float2(z[base + n1 * FS_N2 + k2]); } };

kernel void fourstep16384_A_bh(
    device const float2* input [[buffer(0)]],
    device half2*        zbuf  [[buffer(1)]],
    uint tid   [[thread_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]]
) {
    threadgroup half2 smem[FS_GA * FS_N2];      // fp16 smem, 16 KiB
    constexpr uint T = FS_N2 / 8u;
    const uint f    = tid / T;
    const uint lt   = tid % T;
    const uint gcol = tg_id * FS_GA + f;
    const uint b    = gcol / FS_N1;
    const uint n1   = gcol % FS_N1;
    const uint base = b * FS_N;
    fftdx::FFT<FS_N2, 8u, half2>::execute(smem + f * FS_N2, lt,
        FSLoadA{input, base, n1}, FSStoreA_h{zbuf, base, n1});
}

kernel void fourstep16384_B_bh(
    device const half2*  zbuf   [[buffer(0)]],
    device float2*       output [[buffer(1)]],
    uint tid   [[thread_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]]
) {
    threadgroup half2 smem[FS_GB * FS_N1];      // fp16 smem, 8 KiB
    constexpr uint T = FS_N1 / 4u;
    const uint f    = tid / T;
    const uint lt   = tid % T;
    const uint grow = tg_id * FS_GB + f;
    const uint b    = grow / FS_N2;
    const uint k2   = grow % FS_N2;
    const uint base = b * FS_N;
    fftdx::FFT<FS_N1, 4u, half2>::execute(smem + f * FS_N1, lt,
        FSLoadB_h{zbuf, base, k2}, FSStoreB{output, base, k2});
}

// =============================================================================
// Real-input FFT (R2C) via the half-length-complex-FFT packing trick.
//
// A real signal of length 2M packs into M complex via even/odd interleave
// z[n] = x[2n] + i x[2n+1]; one M-point complex FFT then yields the full
// Hermitian half-spectrum X[0..M] after a "mirror" recombination that pairs
// bin k with bin M-k.  Concretely, with Z = FFT_M(z), W^k = exp(-i pi k / M):
//   E[k] = (Z[k] + conj(Z[M-k]))/2          (FFT of even samples)
//   O[k] = -i (Z[k] - conj(Z[M-k]))/2       (FFT of odd samples)
//   X[k] = E[k] + W^k O[k],  k = 0..M-1 ;  X[M] = Re(Z[0]) - Im(Z[0]).
//
// The point of interest on Apple GPUs: bin k needs bin M-k, held by a thread
// at the opposite end of the threadgroup.  We fold that mirror exchange into a
// SINGLE threadgroup-memory round-trip at the OUTPUT boundary (write Z to smem,
// one barrier, read the mirror partner, recombine, store) rather than a separate
// global pass.  The transform of 2M=8192 real points then fits in the same
// 32 KiB (M=4096 complex) that a complex kernel needs for only 4096 points —
// the real-data local-FFT ceiling doubles from 2^12 to 2^13.
//
// rfft8192_r2c_dx     : 8192 real -> 4097 complex (M=4096, fp32 smem, 32 KiB).
// rfft1024_r2c_dx     : 1024 real -> 513  complex (M=512).  Pairs with the
//                       complex-1024 kernel below for a SINGLE-TG real-vs-complex
//                       comparison (2*512=1024 = 4^5 fits one TG as complex too).
// rfft8192_r2c_f16smem: fp16 inter-stage smem (16 KiB) -> 8192 real in 16 KiB,
//                       lifting the real ceiling to 2^14; fp32 accumulate.
// =============================================================================

// Templated R2C body. SMEM = float2 (fp32) or half2 (fp16 inter-stage storage);
// recombination math stays fp32 in registers. M must be a power of R; M a power
// of two so (M-k)&(M-1) is the mod-M mirror index.
template<uint M, uint R, typename SMEM>
inline void rfft_r2c_body(device const float* input, device float2* output,
                          threadgroup SMEM* smem, uint tid, uint tg_id) {
    constexpr uint T = M / R, NR = 2u * M;
    const uint rbase = tg_id * NR;
    const uint obase = tg_id * (M + 1u);

    float2 td[R];
    #pragma unroll
    for (uint j = 0; j < R; ++j) {
        uint n = tid + j * T;
        td[j] = float2(input[rbase + 2u * n], input[rbase + 2u * n + 1u]);
    }
    fftdx::FFT<M, R, SMEM>::forward(td, smem, tid);   // Z[tid + j*T], natural order

    #pragma unroll
    for (uint j = 0; j < R; ++j) smem[tid + j * T] = SMEM(td[j]);  // publish Z
    threadgroup_barrier(mem_flags::mem_threadgroup);

    #pragma unroll
    for (uint j = 0; j < R; ++j) {
        uint k = tid + j * T;
        float2 Zk  = td[j];
        float2 cZMk = fftdx::cconj(float2(smem[(M - k) & (M - 1u)]));  // conj Z[M-k]
        float2 Ek = 0.5f * (Zk + cZMk);
        float2 d  = Zk - cZMk;
        float2 Ok = 0.5f * float2(d.y, -d.x);          // -i * 0.5 * d
        float s, c; s = sincos(-M_PI_F * float(k) / float(M), c);
        output[obase + k] = Ek + fftdx::cmul(float2(c, s), Ok);
    }
    if (tid == 0u) {                                    // Nyquist bin X[M] (real)
        float2 Z0 = float2(smem[0]);
        output[obase + M] = float2(Z0.x - Z0.y, 0.0f);
    }
}

kernel void rfft8192_r2c_dx(
    device const float* input [[buffer(0)]], device float2* output [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]], uint tg_id [[threadgroup_position_in_grid]]
) {
    threadgroup float2 smem[4096];                      // 32 KiB
    rfft_r2c_body<4096u, 8u, float2>(input, output, smem, tid, tg_id);
}

kernel void rfft8192_r2c_f16smem(
    device const float* input [[buffer(0)]], device float2* output [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]], uint tg_id [[threadgroup_position_in_grid]]
) {
    threadgroup half2 smem[4096];                       // 16 KiB (8192 real in 16 KiB)
    rfft_r2c_body<4096u, 8u, half2>(input, output, smem, tid, tg_id);
}

kernel void rfft1024_r2c_dx(
    device const float* input [[buffer(0)]], device float2* output [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]], uint tg_id [[threadgroup_position_in_grid]]
) {
    threadgroup float2 smem[512];                       // 4 KiB
    rfft_r2c_body<512u, 8u, float2>(input, output, smem, tid, tg_id);
}

// Complex-1024 single-TG FFT (1024 = 4^5, radix-4): the honest "what we do today"
// baseline for 1024 real points (load reals into the complex input, imag=0).
kernel void fft1024_dx_r4(
    device const float2* input [[buffer(0)]], device float2* output [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]], uint tg_id [[threadgroup_position_in_grid]]
) {
    constexpr uint N = 1024u, R = 4u, T = N / R;
    threadgroup float2 smem[N];                         // 8 KiB
    const uint base = tg_id * N;
    float2 td[R];
    #pragma unroll
    for (uint j = 0; j < R; ++j) td[j] = input[base + tid + j * T];
    fftdx::FFT<N, R>::forward(td, smem, tid);
    #pragma unroll
    for (uint j = 0; j < R; ++j) output[base + tid + j * T] = td[j];
}

// C2R inverse: half-spectrum X[0..M] (M+1 complex) -> 2M real samples, single TG.
// Inverts the packing: Z[k] = E[k] + i O[k] with
//   E[k] = (X[k] + conj(X[M-k]))/2,  O[k] = W^{-k} (X[k] - conj(X[M-k]))/2,
//   W^{-k} = exp(+i pi k / M);  then z = IFFT_M(Z), x[2n]=Re z[n], x[2n+1]=Im z[n].
// The mirror partner X[M-k] is read at the LOAD boundary (one smem round-trip),
// the symmetric counterpart of the forward fold.
kernel void rifft8192_c2r_dx(
    device const float2* input  [[buffer(0)]],   // 4097 complex bins / transform
    device float*        output [[buffer(1)]],   // 8192 reals / transform
    uint tid   [[thread_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]]
) {
    constexpr uint M = 4096u, R = 8u, T = M / R;
    constexpr uint NR = 2u * M;
    threadgroup float2 smem[M];
    const uint ibase = tg_id * (M + 1u);
    const uint obase = tg_id * NR;

    // Publish X[0..M-1] to smem for mirror access (X[M] read directly for k=0).
    #pragma unroll
    for (uint j = 0; j < R; ++j) smem[tid + j * T] = input[ibase + tid + j * T];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float2 td[R];
    #pragma unroll
    for (uint j = 0; j < R; ++j) {
        uint k = tid + j * T;
        float2 Xk  = smem[k];
        float2 XMk = (k == 0u) ? input[ibase + M] : smem[M - k];   // X[M-k]
        float2 cXMk = fftdx::cconj(XMk);
        float2 Ek = 0.5f * (Xk + cXMk);
        float2 dd = 0.5f * (Xk - cXMk);
        float s, c; s = sincos(M_PI_F * float(k) / float(M), c);    // W^{-k}
        float2 Ok = fftdx::cmul(float2(c, s), dd);
        float2 Zk = Ek + float2(-Ok.y, Ok.x);                       // E + i O
        td[j] = fftdx::cconj(Zk);                                   // IFFT via conj-FFT-conj
    }
    fftdx::FFT<M, R>::forward<true>(td, smem, tid);                 // GUARD: reuses smem
    constexpr float inv = 1.0f / float(M);
    #pragma unroll
    for (uint j = 0; j < R; ++j) {
        uint n = tid + j * T;
        output[obase + 2u * n]      = td[j].x * inv;                // Re z[n]
        output[obase + 2u * n + 1u] = -td[j].y * inv;               // Im z[n]
    }
}

// =============================================================================
// FUSED REAL range compression (the real analog of range_compression_dx).
// One dispatch, real in -> real out: R2C -> matched-filter multiply on the
// Hermitian half-spectrum -> C2R, with BOTH the forward and inverse mirror folds
// and the spectral multiply kept register/smem-resident (no global round-trip).
// `filter` is the matched-filter half-spectrum H[0..M] = conj(FFT(href))[0..M]
// of a REAL reference, so the product stays Hermitian and the output is real.
// This is real pulse compression / the real-IF radar front-end use of R2C+C2R
// (NOT the complex-I/Q SAR RDA, which stays complex throughout).
// =============================================================================
kernel void range_compress_real_dx(
    device const float*  input  [[buffer(0)]],   // 8192 real / line
    device float*        output [[buffer(1)]],   // 8192 real / line (compressed)
    device const float2* filter [[buffer(2)]],   // 4097 complex matched-filter half-spectrum
    uint tid   [[thread_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]]
) {
    // radix-16 (T=256 threads/TG): the fused two-FFT + two-fold body is register
    // heavy; radix-16 needs only M/16=256 threads, fitting the reduced occupancy.
    constexpr uint M = 4096u, R = 16u, T = M / R, NR = 2u * M;
    threadgroup float2 smem[M];                    // 32 KiB (full budget; no extra tg scalar)
    const uint rbase = tg_id * NR, obase = tg_id * NR;
    // XH[M] (Nyquist) is computed by tid 0 and consumed only by the k=0 inverse,
    // which is also tid 0 (k=tid+j*T=0 => tid=0,j=0) — so keep it in a register.
    float2 xhNyq = float2(0.0f, 0.0f);

    // ---- R2C forward ----
    float2 td[R];
    #pragma unroll
    for (uint j = 0; j < R; ++j) {
        uint n = tid + j * T;
        td[j] = float2(input[rbase + 2u * n], input[rbase + 2u * n + 1u]);
    }
    fftdx::FFT<M, R>::forward(td, smem, tid);
    #pragma unroll
    for (uint j = 0; j < R; ++j) smem[tid + j * T] = td[j];        // publish Z
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // form X[k], multiply by H[k] -> XH[k], in place in td (one register array)
    if (tid == 0u) {
        float2 Z0 = smem[0];
        xhNyq = fftdx::cmul(float2(Z0.x - Z0.y, 0.0f), filter[M]);
    }
    #pragma unroll
    for (uint j = 0; j < R; ++j) {
        uint k = tid + j * T;
        float2 cZMk = fftdx::cconj(smem[(M - k) & (M - 1u)]);
        float2 Ek = 0.5f * (td[j] + cZMk);
        float2 d  = td[j] - cZMk;
        float2 Ok = 0.5f * float2(d.y, -d.x);
        float s, c; s = sincos(-M_PI_F * float(k) / float(M), c);
        float2 Xk = Ek + fftdx::cmul(float2(c, s), Ok);
        td[j] = fftdx::cmul(Xk, filter[k]);                        // XH[k]
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);              // Z reads done
    #pragma unroll
    for (uint j = 0; j < R; ++j) smem[tid + j * T] = td[j];        // publish XH
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- C2R inverse on XH, in place in td ----
    #pragma unroll
    for (uint j = 0; j < R; ++j) {
        uint k = tid + j * T;
        float2 XHMk = (k == 0u) ? xhNyq : smem[M - k];
        float2 cXHMk = fftdx::cconj(XHMk);
        float2 Ek = 0.5f * (td[j] + cXHMk);
        float2 dd = 0.5f * (td[j] - cXHMk);
        float s, c; s = sincos(M_PI_F * float(k) / float(M), c);   // W^{-k}
        float2 Ok = fftdx::cmul(float2(c, s), dd);
        float2 Zp = Ek + float2(-Ok.y, Ok.x);                      // E + iO
        td[j] = fftdx::cconj(Zp);
    }
    fftdx::FFT<M, R>::forward<true>(td, smem, tid);               // IFFT (GUARD reuses smem)
    constexpr float inv = 1.0f / float(M);
    #pragma unroll
    for (uint j = 0; j < R; ++j) {
        uint n = tid + j * T;
        output[obase + 2u * n]      = td[j].x * inv;
        output[obase + 2u * n + 1u] = -td[j].y * inv;
    }
}

// =============================================================================
// 2D FFT (separable: batched row FFTs, then batched column FFTs). The column
// pass is just the primitive with a stride-N load/store functor — no transpose
// kernel needed, the strided access IS the transpose. Row-major NxN images,
// batched over images via tg_id. Output natural order in both axes.
//
//   pass 1 (rows): in[img*N*N + row*N + c]  -> tmp,  N row-FFTs / image
//   pass 2 (cols): tmp[img*N*N + r*N + col] -> out,  N col-FFTs / image (stride N)
// =============================================================================
struct Col2DLoad  { device const float2* x; uint base; uint col; uint stride;
                    float2 operator()(uint row) const { return x[base + row * stride + col]; } };
struct Col2DStore { device float2* y; uint base; uint col; uint stride;
                    void operator()(uint row, float2 v) const { y[base + row * stride + col] = v; } };

// ---- 256 x 256 (radix-4) ----
kernel void fft2d_rows_256(
    device const float2* input [[buffer(0)]], device float2* tmp [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]], uint tg_id [[threadgroup_position_in_grid]]
) {
    constexpr uint N = 256u, R = 4u;
    threadgroup float2 smem[N];
    const uint row = tg_id % N, img = tg_id / N;
    const uint rb  = img * N * N + row * N;
    fftdx::FFT<N, R>::execute(smem, tid, DXLoad{input, rb}, DXStore{tmp, rb});
}
kernel void fft2d_cols_256(
    device const float2* tmp [[buffer(0)]], device float2* output [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]], uint tg_id [[threadgroup_position_in_grid]]
) {
    constexpr uint N = 256u, R = 4u;
    threadgroup float2 smem[N];
    const uint col = tg_id % N, img = tg_id / N;
    const uint ib  = img * N * N;
    fftdx::FFT<N, R>::execute(smem, tid, Col2DLoad{tmp, ib, col, N}, Col2DStore{output, ib, col, N});
}

// ---- 1024 x 1024 (radix-4: 1024 = 4^5) ----
kernel void fft2d_rows_1024(
    device const float2* input [[buffer(0)]], device float2* tmp [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]], uint tg_id [[threadgroup_position_in_grid]]
) {
    constexpr uint N = 1024u, R = 4u;
    threadgroup float2 smem[N];
    const uint row = tg_id % N, img = tg_id / N;
    const uint rb  = img * N * N + row * N;
    fftdx::FFT<N, R>::execute(smem, tid, DXLoad{input, rb}, DXStore{tmp, rb});
}
kernel void fft2d_cols_1024(
    device const float2* tmp [[buffer(0)]], device float2* output [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]], uint tg_id [[threadgroup_position_in_grid]]
) {
    constexpr uint N = 1024u, R = 4u;
    threadgroup float2 smem[N];
    const uint col = tg_id % N, img = tg_id / N;
    const uint ib  = img * N * N;
    fftdx::FFT<N, R>::execute(smem, tid, Col2DLoad{tmp, ib, col, N}, Col2DStore{output, ib, col, N});
}

// ---- Non-square 2D FFT (N1 rows x N2 cols), e.g. 512 x 256: rows use a
//      length-N2 FFT (contiguous), columns a length-N1 FFT (stride N2). Shows
//      the two axes can differ in size AND radix (rows radix-4, cols radix-8).
kernel void fft2d_rows_512x256(
    device const float2* input [[buffer(0)]], device float2* tmp [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]], uint tg_id [[threadgroup_position_in_grid]]
) {
    constexpr uint N1 = 512u, N2 = 256u, R = 4u;     // row FFT length = N2
    threadgroup float2 smem[N2];
    const uint row = tg_id % N1, img = tg_id / N1;
    const uint rb  = img * N1 * N2 + row * N2;
    fftdx::FFT<N2, R>::execute(smem, tid, DXLoad{input, rb}, DXStore{tmp, rb});
}
kernel void fft2d_cols_512x256(
    device const float2* tmp [[buffer(0)]], device float2* output [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]], uint tg_id [[threadgroup_position_in_grid]]
) {
    constexpr uint N1 = 512u, N2 = 256u, R = 8u;     // col FFT length = N1, stride N2
    threadgroup float2 smem[N1];
    const uint col = tg_id % N2, img = tg_id / N2;
    const uint ib  = img * N1 * N2;
    fftdx::FFT<N1, R>::execute(smem, tid, Col2DLoad{tmp, ib, col, N2}, Col2DStore{output, ib, col, N2});
}

// Tiled, bank-conflict-free (32x33) transpose for the coalesced 2D-FFT variant
// (rows -> transpose -> rows -> transpose: every FFT pass is contiguous). N is a
// runtime constant; batched over images via the grid's z dimension.
kernel void transpose2d_tiled(
    device const float2* in  [[buffer(0)]],
    device float2*       out [[buffer(1)]],
    constant uint&       N   [[buffer(2)]],
    uint3 lid  [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    constexpr uint TT = 32u;
    threadgroup float2 tile[TT][TT + 1u];
    const uint base = tgid.z * N * N;
    const uint x = tgid.x * TT + lid.x;     // column (read)
    const uint y = tgid.y * TT + lid.y;     // row    (read)
    if (x < N && y < N) tile[lid.y][lid.x] = in[base + y * N + x];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint tx = tgid.y * TT + lid.x;
    const uint ty = tgid.x * TT + lid.y;
    if (tx < N && ty < N) out[base + ty * N + tx] = tile[lid.x][lid.y];
}

// =============================================================================
// Fused 2D convolution (1024x1024): out = IFFT2D( FFT2D(image) .* filter ).
// 4 passes: rows-fwd -> cols-fwd*filter (multiply FUSED into the store) ->
// cols-ifft -> rows-ifft. The pointwise multiply costs no extra dispatch /
// round-trip; the two IFFT passes are conj-FFT-conj with the 1/N split per axis.
// `filter` is a shared NxN frequency-domain mask (same for every image).
// =============================================================================
struct Col2DMulStore { device float2* y; device const float2* f; uint base; uint col; uint stride;
    void operator()(uint k, float2 X) const { y[base + k*stride + col] = fftdx::cmul(X, f[k*stride + col]); } };
struct Col2DConjLoad { device const float2* x; uint base; uint col; uint stride;
    float2 operator()(uint row) const { return fftdx::cconj(x[base + row*stride + col]); } };
struct Col2DConjScaleStore { device float2* y; uint base; uint col; uint stride; float inv;
    void operator()(uint row, float2 v) const { y[base + row*stride + col] = float2(v.x*inv, -v.y*inv); } };
struct DXConjLoad { device const float2* p; uint base;
    float2 operator()(uint i) const { return fftdx::cconj(p[base + i]); } };
struct DXConjScaleStore { device float2* p; uint base; float inv;
    void operator()(uint i, float2 v) const { p[base + i] = float2(v.x*inv, -v.y*inv); } };

// pass 2: forward column FFT, multiply by filter at the store (fused).
kernel void fft2dconv_colsmul_1024(
    device const float2* tmp [[buffer(0)]], device float2* output [[buffer(1)]],
    device const float2* filter [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]], uint tg_id [[threadgroup_position_in_grid]]
) {
    constexpr uint N = 1024u, R = 4u;
    threadgroup float2 smem[N];
    const uint col = tg_id % N, img = tg_id / N;
    const uint ib  = img * N * N;
    fftdx::FFT<N, R>::execute(smem, tid, Col2DLoad{tmp, ib, col, N}, Col2DMulStore{output, filter, ib, col, N});
}
// pass 3: inverse column FFT (conj-FFT-conj, x1/N).
kernel void fft2dconv_colsifft_1024(
    device const float2* in [[buffer(0)]], device float2* output [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]], uint tg_id [[threadgroup_position_in_grid]]
) {
    constexpr uint N = 1024u, R = 4u;
    threadgroup float2 smem[N];
    const uint col = tg_id % N, img = tg_id / N;
    const uint ib  = img * N * N;
    fftdx::FFT<N, R>::execute(smem, tid, Col2DConjLoad{in, ib, col, N},
        Col2DConjScaleStore{output, ib, col, N, 1.0f/float(N)});
}
// pass 4: inverse row FFT (conj-FFT-conj, x1/N) -> final image.
kernel void fft2dconv_rowsifft_1024(
    device const float2* in [[buffer(0)]], device float2* output [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]], uint tg_id [[threadgroup_position_in_grid]]
) {
    constexpr uint N = 1024u, R = 4u;
    threadgroup float2 smem[N];
    const uint row = tg_id % N, img = tg_id / N;
    const uint rb  = img * N * N + row * N;
    fftdx::FFT<N, R>::execute(smem, tid, DXConjLoad{in, rb}, DXConjScaleStore{output, rb, 1.0f/float(N)});
}
