#pragma once
#include <metal_stdlib>
using namespace metal;

// =============================================================================
// fft_device.h — Metal device-side, compile-time-specialized FFT primitive.
//
// The Apple-Metal analog of NVIDIA cuFFTDx: an FFT you call from *inside* your
// own compute kernel, operating on register-resident data with threadgroup
// memory used only for the inter-stage exchange. This lets you FUSE the FFT
// with surrounding element-wise work (windowing, matched-filter multiply,
// magnitude detection) at the load/store boundary with zero extra global-memory
// round-trips — the core of the SAR-pipeline win on unified memory.
//
// Data model (cuFFTDx-faithful): each of THREADS = N/R threads owns EPT = R
// complex elements in registers (`thread float2 td[R]`); the transform runs
// entirely on those registers, touching threadgroup `smem` only to reshuffle
// between radix stages. The caller loads td[] (optionally applying a window /
// format-convert), calls forward(), and stores td[] (optionally applying
// detection / scale). The frequency/time index owned by register j of thread
// tid is `tid + j*THREADS`, both on input and output.
//
// Algorithm: ordered (self-sorting) radix-R Stockham, decimation-in-time.
// Output is natural order (no bit-reversal pass). The per-stage strides,
// twiddle exponents and barrier discipline are constructed to be bit-compatible
// with src/metal/fft_4096_batched.metal (the 138 GFLOPS hand-tuned radix-8
// baseline): uniform read at stride THREADS every stage, write stride R^p,
// twiddle pos = tid & (R^p - 1) with exponent pos*(N/R^(p+1)).
//
// Status: radix-8 path implemented (milestone 1). The stage-recursion / stride /
// barrier machinery is radix-generic; only the butterfly+twiddle leaf is R=8
// today (radix-4/16 are a drop-in `radix<R>()` away — see TODO below).
// =============================================================================

namespace fftdx {

constant float SQRT2_2 = 0.70710678118654752f;

inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
inline float2 cconj(float2 a) { return float2(a.x, -a.y); }

// ---- compile-time helpers ---------------------------------------------------
constexpr uint ce_ipow(uint b, uint e) { return e == 0u ? 1u : b * ce_ipow(b, e - 1u); }
constexpr uint ce_ilog(uint n, uint b) { return n <= 1u ? 0u : 1u + ce_ilog(n / b, b); }
constexpr uint ce_ilog2(uint n)        { return n <= 1u ? 0u : 1u + ce_ilog2(n >> 1); }

// ---- radix-8 DIT butterfly (identical math to fft_4096_batched.metal) -------
// Split-radix-style radix-8 (~32 FLOPs vs ~320 naive): octant twiddles folded
// into adds + a single 1/sqrt(2) scale, ±j folded into swap/negate.
inline void radix8(thread float2* x) {
    float2 t0 = x[0] + x[4];  float2 t1 = x[1] + x[5];
    float2 t2 = x[2] + x[6];  float2 t3 = x[3] + x[7];
    float2 t4 = x[0] - x[4];  float2 t5 = x[1] - x[5];
    float2 t6 = x[2] - x[6];  float2 t7 = x[3] - x[7];

    float2 t5w = float2(SQRT2_2 * (t5.x + t5.y), SQRT2_2 * (t5.y - t5.x));
    float2 t6w = float2(t6.y, -t6.x);
    float2 t7w = float2(SQRT2_2 * (-t7.x + t7.y), SQRT2_2 * (-t7.y - t7.x));

    float2 u0 = t0 + t2;  float2 u1 = t1 + t3;
    float2 u2 = t0 - t2;  float2 u3 = t1 - t3;
    float2 u3r = float2(u3.y, -u3.x);

    float2 v0 = t4 + t6w;  float2 v1 = t5w + t7w;
    float2 v2 = t4 - t6w;  float2 v3 = t5w - t7w;
    float2 v3r = float2(v3.y, -v3.x);

    x[0] = u0 + u1;  x[4] = u0 - u1;
    x[2] = u2 + u3r; x[6] = u2 - u3r;
    x[1] = v0 + v1;  x[5] = v0 - v1;
    x[3] = v2 + v3r; x[7] = v2 - v3r;
}

// Apply the per-element twiddles W^(j*base) to x[1..7] given w1 = W^base.
inline void apply_twiddle8(thread float2* x, float2 w1) {
    float2 w2 = cmul(w1, w1);
    float2 w3 = cmul(w2, w1);
    float2 w4 = cmul(w2, w2);
    float2 w5 = cmul(w4, w1);
    float2 w6 = cmul(w4, w2);
    float2 w7 = cmul(w4, w3);
    x[1] = cmul(x[1], w1); x[2] = cmul(x[2], w2); x[3] = cmul(x[3], w3);
    x[4] = cmul(x[4], w4); x[5] = cmul(x[5], w5); x[6] = cmul(x[6], w6);
    x[7] = cmul(x[7], w7);
}

// ---- radix-4 DIT butterfly (x[j] = natural DFT-4 output j) ------------------
inline void radix4(thread float2* x) {
    float2 a = x[0] + x[2];
    float2 b = x[0] - x[2];
    float2 c = x[1] + x[3];
    float2 d = x[1] - x[3];
    float2 dj = float2(d.y, -d.x);   // -j * d  (forward transform)
    x[0] = a + c;
    x[1] = b + dj;
    x[2] = a - c;
    x[3] = b - dj;
}
inline void apply_twiddle4(thread float2* x, float2 w1) {
    float2 w2 = cmul(w1, w1);
    float2 w3 = cmul(w2, w1);
    x[1] = cmul(x[1], w1); x[2] = cmul(x[2], w2); x[3] = cmul(x[3], w3);
}

// ---- radix-16 DIT butterfly (16 = 4x4 Cooley-Tukey, natural output order) ---
// Step 1: four DFT-4s over the strided columns {x[r], x[r+4], x[r+8], x[r+12]}.
// Step 2: per output residue k1, apply W16^{r k1} then a DFT-4 over r; output
// index is k1 + 4*k2 (natural). Reuses the validated radix4().
inline void radix16(thread float2* x) {
    const float2 w16[4] = { float2(1.0f, 0.0f),
                            float2(0.92387953f, -0.38268343f),    // W16^1
                            float2(0.70710678f, -0.70710678f),    // W16^2
                            float2(0.38268343f, -0.92387953f) };  // W16^3
    float2 Y[16];
    #pragma unroll
    for (uint r = 0; r < 4u; ++r) {
        float2 t[4] = { x[r], x[r + 4u], x[r + 8u], x[r + 12u] };
        radix4(t);
        Y[r*4u+0u]=t[0]; Y[r*4u+1u]=t[1]; Y[r*4u+2u]=t[2]; Y[r*4u+3u]=t[3];
    }
    #pragma unroll
    for (uint k1 = 0; k1 < 4u; ++k1) {
        float2 b  = w16[k1];
        float2 b2 = cmul(b, b);
        float2 b3 = cmul(b2, b);
        float2 z[4] = { Y[0u*4u+k1], cmul(Y[1u*4u+k1], b), cmul(Y[2u*4u+k1], b2), cmul(Y[3u*4u+k1], b3) };
        radix4(z);
        x[k1]        = z[0];   // k2=0 -> k1 + 4*0
        x[k1 + 4u]   = z[1];   // k2=1
        x[k1 + 8u]   = z[2];
        x[k1 + 12u]  = z[3];
    }
}
inline void apply_twiddle16(thread float2* x, float2 w1) {
    float2 w = w1;
    #pragma unroll
    for (uint j = 1; j < 16u; ++j) { x[j] = cmul(x[j], w); w = cmul(w, w1); }
}

// ---- radix-R leaf dispatch (the only R-specific piece; strides/twiddle-
//      exponents/barriers in StageImpl are radix-generic) --------------------
template<uint R> struct RadixLeaf;
template<> struct RadixLeaf<4u> {
    static void dft(thread float2* x)                 { radix4(x); }
    static void twiddle(thread float2* x, float2 w1)  { apply_twiddle4(x, w1); }
};
template<> struct RadixLeaf<8u> {
    static void dft(thread float2* x)                 { radix8(x); }
    static void twiddle(thread float2* x, float2 w1)  { apply_twiddle8(x, w1); }
};
template<> struct RadixLeaf<16u> {
    static void dft(thread float2* x)                 { radix16(x); }
    static void twiddle(thread float2* x, float2 w1)  { apply_twiddle16(x, w1); }
};

// ---- one radix stage, recursively chained at compile time -------------------
// StageImpl<N,R,P,LAST>: thread holds this stage's R inputs in td[] (loaded at
// stride THREADS). Applies twiddle (P>0), the radix butterfly, then — unless
// this is the last stage — scatters to smem (write stride R^P) and re-gathers
// the next stage's inputs (read stride THREADS), recursing to P+1.
// SMEM is the threadgroup-storage element type (float2 or half2). The transform
// state td[] is ALWAYS float2 — i.e. fp32 ACCUMULATE in registers — and only the
// inter-stage exchange is stored in SMEM. With SMEM=half2 this is the classic
// "fp16 storage / fp32 accumulate" scheme: 16 KiB smem (vs 32) at N=4096, and
// Apple's free fp16<->fp32 conversion makes the boundary cast cost nothing.
// SMEM=float2 (default) is bit-identical to the original fp32 primitive.
template<uint N, uint R, typename SMEM, uint P, bool LAST>
struct StageImpl {
    static void run(thread float2* td, threadgroup SMEM* smem, uint tid, bool guard) {
        constexpr uint  T    = N / R;                 // threads == read stride
        constexpr uint  RP   = ce_ipow(R, P);         // write stride this stage
        constexpr uint  RP1  = ce_ipow(R, P + 1u);
        constexpr uint  SH   = P * ce_ilog2(R);        // group shift
        constexpr uint  MASK = RP - 1u;                // pos mask (0 when P==0)
        constexpr uint  TWS  = N / RP1;                // twiddle base exponent scale
        constexpr float W0   = -2.0f * M_PI_F / float(N);

        if (P > 0u) {
            uint pos = tid & MASK;
            float ang = W0 * float(pos * TWS);
            float s, c; s = sincos(ang, c);
            RadixLeaf<R>::twiddle(td, float2(c, s));
        }
        RadixLeaf<R>::dft(td);

        // Exchange to next stage. Pre-write barrier guards the threadgroup
        // buffer against WAR: required for every stage P>0; for P==0 it is only
        // needed when `guard` is set (a second transform chained on the same
        // smem, e.g. the conj-FFT of an in-place IFFT).
        if (P > 0u || guard) threadgroup_barrier(mem_flags::mem_threadgroup);
        uint pos = tid & MASK;
        uint grp = tid >> SH;
        uint wr  = grp * RP1 + pos;
        #pragma unroll
        for (uint j = 0; j < R; ++j) smem[wr + j * RP] = SMEM(td[j]);   // fp32 -> SMEM (free if half2)
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint j = 0; j < R; ++j) td[j] = float2(smem[tid + j * T]); // SMEM -> fp32 accumulate

        StageImpl<N, R, SMEM, P + 1u, (P + 1u == ce_ilog(N, R) - 1u)>::run(td, smem, tid, guard);
    }
};

// Terminal (last) stage: twiddle + butterfly, leave the result in registers
// (natural order). No smem write — the caller stores td[].
template<uint N, uint R, typename SMEM, uint P>
struct StageImpl<N, R, SMEM, P, true> {
    static void run(thread float2* td, threadgroup SMEM* smem, uint tid, bool guard) {
        constexpr uint  RP1  = ce_ipow(R, P + 1u);
        constexpr uint  MASK = ce_ipow(R, P) - 1u;
        constexpr uint  TWS  = N / RP1;
        constexpr float W0   = -2.0f * M_PI_F / float(N);
        if (P > 0u) {
            uint pos = tid & MASK;
            float ang = W0 * float(pos * TWS);
            float s, c; s = sincos(ang, c);
            RadixLeaf<R>::twiddle(td, float2(c, s));
        }
        RadixLeaf<R>::dft(td);
        (void)smem; (void)guard;
    }
};

// ---- public API -------------------------------------------------------------
// SMEM = threadgroup storage element: float2 (fp32, default) or half2 (fp16
// storage + fp32 accumulate; halves smem and, on fp16 device I/O, halves
// bandwidth — radar-usable ~60 dB vs pure-fp16's ~42 dB).
//
// CALLER CONTRACT (match all of these — see docs/FFT_DEVICE_API.md):
//   - dispatch threadsPerThreadgroup == threads() (== N/R);
//   - allocate `threadgroup SMEM smem[smemElems()]` (== N elements);
//   - one transform per threadgroup (or batch with private smem slices + a
//     local tid in [0, N/R));
//   - register j of thread `tid` carries index tid + j*threads(), in and out;
//   - output is natural order; forward() is UNNORMALIZED (convolve() applies 1/N).
//   - N must be an exact power of R (static_assert-enforced); R in {4,8,16}.
template<uint N, uint R, typename SMEM = float2>
struct FFT {
    static_assert(R == 4u || R == 8u || R == 16u, "fft_device.h implements radix-4/8/16 leaves");
    static_assert(ce_ipow(R, ce_ilog(N, R)) == N, "N must be an exact power of R");

    // Compile-time descriptors. MSL forbids program-scope constexpr data
    // members outside the constant address space, so expose them as constexpr
    // member functions instead.
    static constexpr uint threads()   { return N / R; }   // threads/FFT (== read stride)
    static constexpr uint ept()       { return R; }        // complex elements per thread
    static constexpr uint stages()    { return ce_ilog(N, R); }
    static constexpr uint smemElems() { return N; }        // SMEM count the caller allocates

    // Forward transform, in place on registers. Caller pre-loads td[0..R-1] with
    // this thread's inputs at indices {tid + j*THREADS}; on return td holds the
    // natural-order outputs at the same indices.
    //
    // GUARD: pass true when chaining a second transform that reuses `smem`
    // (e.g. the conj-FFT inside an IFFT) so the entry adds the needed WAR barrier.
    template<bool GUARD = false>
    static void forward(thread float2* td, threadgroup SMEM* smem, uint tid) {
        StageImpl<N, R, SMEM, 0u, (ce_ilog(N, R) == 1u)>::run(td, smem, tid, GUARD);
    }

    // ---- fused convenience entry points (the cuFFTDx callback analog) -------
    // Callbacks are plain structs exposing an operator(); pass them by value
    // (each typically just wraps a device pointer + base offset). Index args are
    // NATURAL (un-permuted) frequency/time indices in [0, N). Because they are
    // template type parameters, the compiler inlines them — no call overhead and
    // no extra global-memory round-trip relative to a hand-written fused kernel.

    // Bare transform with load/store callbacks.
    //   Load:  float2 operator()(uint idx) const;            // input sample at idx
    //   Store: void   operator()(uint idx, float2 v) const;  // output bin at idx
    template<class Load, class Store>
    static void execute(threadgroup SMEM* smem, uint tid, Load load, Store store) {
        constexpr uint T = N / R;
        float2 td[R];
        #pragma unroll
        for (uint j = 0; j < R; ++j) td[j] = load(tid + j * T);
        forward<false>(td, smem, tid);
        #pragma unroll
        for (uint j = 0; j < R; ++j) store(tid + j * T, td[j]);
    }

    // Circular convolution y = IFFT( Mid(FFT(x)) ) in one call, with the
    // FFT->Mid->IFFT midpoint handed off in registers (no extra smem round-trip).
    // This is the SAR range/azimuth-compression shape; `Mid` is the matched
    // filter (or any frequency-domain pointwise op).
    //   Load:  float2 operator()(uint idx) const;            // input sample
    //   Mid:   float2 operator()(uint k, float2 X) const;    // freq-domain op, e.g. cmul(X, filter[k])
    //   Store: void   operator()(uint idx, float2 y) const;  // y is the 1/N-scaled time sample
    template<class Load, class Mid, class Store>
    static void convolve(threadgroup SMEM* smem, uint tid, Load load, Mid mid, Store store) {
        constexpr uint  T   = N / R;
        constexpr float inv = 1.0f / float(N);
        float2 td[R];
        #pragma unroll
        for (uint j = 0; j < R; ++j) td[j] = load(tid + j * T);
        forward<false>(td, smem, tid);                 // X = FFT(x)
        #pragma unroll
        for (uint j = 0; j < R; ++j) td[j] = cconj(mid(tid + j * T, td[j]));  // conj(Mid(X)) for the IFFT
        forward<true>(td, smem, tid);                  // conj-FFT (chained: GUARD)
        #pragma unroll
        for (uint j = 0; j < R; ++j) store(tid + j * T, float2(td[j].x * inv, -td[j].y * inv));
    }
};

} // namespace fftdx
