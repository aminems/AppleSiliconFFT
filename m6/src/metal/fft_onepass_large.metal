// =============================================================================
// fft_onepass_large.metal — one-DRAM-pass FFT for N = 8192 and 16384 (fp32).
//
// paper_m6 Stage 3. On M6 every GPU FFT library halves in throughput at
// N = 8192: the c64 block (64 KiB) no longer fits the 32 KiB threadgroup
// memory, so they fall back to two DRAM passes (four-step). Here the block
// lives in REGISTERS (T threads x E complex each; register file >> 32 KiB),
// and threadgroup memory is used only as a narrow exchange buffer between
// Stockham stages: the exchange is done one real component at a time
// (8192 floats = 32 KiB) and, for N = 16384, in two position halves.
// Every input element is read from device memory once and every output
// written once: one DRAM pass, exact fp32 (no fp16 storage).
//
// Algorithm: mixed-radix Stockham DIT (natural order in and out).
//   stage with radix R after sub-length LS (product of previous radices):
//   butterfly j in [0, N/R), k = j mod LS
//   inputs  x[j + t*N/R], t = 0..R-1, twiddled by W_{LS*R}^{k t}
//   outputs y[(j/LS)*LS*R + k + t*LS]
// Thread tid owns butterflies j = tid + b*T, b = 0..E/R-1, data in d[b*R + t].
// Requires fft_device.h (radix8/16, apply_twiddle8/16, cmul) prepended.
// =============================================================================

namespace onepass {

constant uint XCHG_FLOATS = 8192u;     // 32 KiB threadgroup exchange buffer

template<uint R> inline void dft(thread float2* a);
template<> inline void dft<8u>(thread float2* a)  { fftdx::radix8(a); }
template<> inline void dft<16u>(thread float2* a) { fftdx::radix16(a); }
template<uint R> inline void twiddle(thread float2* a, float2 w1);
template<> inline void twiddle<8u>(thread float2* a, float2 w1)  { fftdx::apply_twiddle8(a, w1); }
template<> inline void twiddle<16u>(thread float2* a, float2 w1) { fftdx::apply_twiddle16(a, w1); }

// Butterflies of one stage, in registers.
template<uint N, uint T, uint E, uint R, uint LS>
inline void compute(thread float2* d, uint tid) {
    #pragma unroll
    for (uint b = 0; b < E / R; ++b) {
        if (LS > 1u) {
            const uint j = tid + b * T;
            const uint k = j & (LS - 1u);
            float s, c;
            s = sincos(-2.0f * M_PI_F * float(k) / float(LS * R), c);
            twiddle<R>(d + b * R, float2(c, s));
        }
        dft<R>(d + b * R);
    }
}

// Output position of register (b, t) after a stage (R, LS).
template<uint T, uint R, uint LS>
inline uint out_pos(uint tid, uint b, uint t) {
    const uint j = tid + b * T;
    const uint k = j & (LS - 1u);
    return (j / LS) * LS * R + k + t * LS;
}
// Input position of register (b, t) for a stage of radix R2.
template<uint N, uint T, uint R2>
inline uint in_pos(uint tid, uint b, uint t) { return tid + b * T + t * (N / R2); }

// Reshuffle registers from the output layout of stage (R, LS) to the input
// layout of the next stage (radix R2) through a 32 KiB float buffer.
// Rounds: 2 components x (N / 8192) position windows.
template<uint N, uint T, uint E, uint R, uint LS, uint R2, uint XF = XCHG_FLOATS>
inline void exchange(thread float2* d, threadgroup float* x, uint tid) {
    constexpr uint W = N / XF;            // position windows (1 or 2)
    // Read into nd[], not d[]: with W > 1 a register can be read in window 0
    // while its old value is still to be written in window 1.
    float2 nd[E];
    #pragma unroll
    for (uint comp = 0; comp < 2u; ++comp) {
        #pragma unroll
        for (uint w = 0; w < W; ++w) {
            #pragma unroll
            for (uint b = 0; b < E / R; ++b)
                #pragma unroll
                for (uint t = 0; t < R; ++t) {
                    const uint p = out_pos<T, R, LS>(tid, b, t);
                    if (W == 1u || (p / XF) == w)
                        x[p % XF] = comp == 0u ? d[b * R + t].x : d[b * R + t].y;
                }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            #pragma unroll
            for (uint b = 0; b < E / R2; ++b)
                #pragma unroll
                for (uint t = 0; t < R2; ++t) {
                    const uint p = in_pos<N, T, R2>(tid, b, t);
                    if (W == 1u || (p / XF) == w) {
                        if (comp == 0u) nd[b * R2 + t].x = x[p % XF];
                        else            nd[b * R2 + t].y = x[p % XF];
                    }
                }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    #pragma unroll
    for (uint i = 0; i < E; ++i) d[i] = nd[i];
}

} // namespace onepass

// ---- N = 8192 = 16 * 8 * 8 * 8, T = 512 threads x 16 complex ---------------
// IO = float2 (fp32 storage) or half2 (fp16 storage: halves DRAM bytes; all
// arithmetic and the exchange stay fp32).
template<typename IO>
inline void fft8192_body(device const IO* input, device IO* output,
                         threadgroup float* x, uint tid, uint tg_id) {
    using namespace onepass;
    constexpr uint N = 8192u, T = 512u, E = 16u;
    const uint base = tg_id * N;
    float2 d[E];
    #pragma unroll
    for (uint t = 0; t < 16u; ++t) d[t] = float2(input[base + in_pos<N, T, 16u>(tid, 0u, t)]);

    compute<N, T, E, 16u, 1u>(d, tid);
    exchange<N, T, E, 16u, 1u, 8u>(d, x, tid);
    compute<N, T, E, 8u, 16u>(d, tid);
    exchange<N, T, E, 8u, 16u, 8u>(d, x, tid);
    compute<N, T, E, 8u, 128u>(d, tid);
    exchange<N, T, E, 8u, 128u, 8u>(d, x, tid);
    compute<N, T, E, 8u, 1024u>(d, tid);

    #pragma unroll
    for (uint b = 0; b < 2u; ++b)
        #pragma unroll
        for (uint t = 0; t < 8u; ++t)
            output[base + out_pos<T, 8u, 1024u>(tid, b, t)] = IO(d[b * 8u + t]);
}

// ---- N = 16384 = 16 * 16 * 8 * 8, T = 1024 threads x 16 complex ------------
template<typename IO>
inline void fft16384_body(device const IO* input, device IO* output,
                          threadgroup float* x, uint tid, uint tg_id) {
    using namespace onepass;
    constexpr uint N = 16384u, T = 1024u, E = 16u;
    const uint base = tg_id * N;
    float2 d[E];
    #pragma unroll
    for (uint t = 0; t < 16u; ++t) d[t] = float2(input[base + in_pos<N, T, 16u>(tid, 0u, t)]);

    compute<N, T, E, 16u, 1u>(d, tid);
    exchange<N, T, E, 16u, 1u, 16u>(d, x, tid);
    compute<N, T, E, 16u, 16u>(d, tid);
    exchange<N, T, E, 16u, 16u, 8u>(d, x, tid);
    compute<N, T, E, 8u, 256u>(d, tid);
    exchange<N, T, E, 8u, 256u, 8u>(d, x, tid);
    compute<N, T, E, 8u, 2048u>(d, tid);

    #pragma unroll
    for (uint b = 0; b < 2u; ++b)
        #pragma unroll
        for (uint t = 0; t < 8u; ++t)
            output[base + out_pos<T, 8u, 2048u>(tid, b, t)] = IO(d[b * 8u + t]);
}

#define ONEPASS_KERNEL(NAME, BODY, IO)                                            \
kernel void NAME(device const IO* input [[buffer(0)]], device IO* output [[buffer(1)]], \
                 uint tid [[thread_index_in_threadgroup]],                        \
                 uint tg_id [[threadgroup_position_in_grid]]) {                   \
    threadgroup float x[onepass::XCHG_FLOATS];                                    \
    BODY<IO>(input, output, x, tid, tg_id);                                       \
}
ONEPASS_KERNEL(fft8192_onepass,      fft8192_body,  float2)
ONEPASS_KERNEL(fft8192_onepass_f16,  fft8192_body,  half2)
ONEPASS_KERNEL(fft16384_onepass,     fft16384_body, float2)
ONEPASS_KERNEL(fft16384_onepass_f16, fft16384_body, half2)


// ---- N = 16384 variant: T = 512 threads x 32 complex (fewer threads, more
//      registers/thread; radix-16 stages do 2 butterflies/thread, radix-8 do 4).
template<typename IO>
inline void fft16384_t512_body(device const IO* input, device IO* output,
                               threadgroup float* x, uint tid, uint tg_id) {
    using namespace onepass;
    constexpr uint N = 16384u, T = 512u, E = 32u;
    const uint base = tg_id * N;
    float2 d[E];
    #pragma unroll
    for (uint b = 0; b < 2u; ++b)
        #pragma unroll
        for (uint t = 0; t < 16u; ++t) d[b * 16u + t] = float2(input[base + in_pos<N, T, 16u>(tid, b, t)]);

    compute<N, T, E, 16u, 1u>(d, tid);
    exchange<N, T, E, 16u, 1u, 16u>(d, x, tid);
    compute<N, T, E, 16u, 16u>(d, tid);
    exchange<N, T, E, 16u, 16u, 8u>(d, x, tid);
    compute<N, T, E, 8u, 256u>(d, tid);
    exchange<N, T, E, 8u, 256u, 8u>(d, x, tid);
    compute<N, T, E, 8u, 2048u>(d, tid);

    #pragma unroll
    for (uint b = 0; b < 4u; ++b)
        #pragma unroll
        for (uint t = 0; t < 8u; ++t)
            output[base + out_pos<T, 8u, 2048u>(tid, b, t)] = IO(d[b * 8u + t]);
}
ONEPASS_KERNEL(fft16384_onepass_t512,     fft16384_t512_body, float2)
ONEPASS_KERNEL(fft16384_onepass_t512_f16, fft16384_t512_body, half2)

// DEBUG: N = 8192 forced through the 2-window exchange path (XF = 4096).
kernel void fft8192_onepass_w2dbg(
    device const float2* input  [[buffer(0)]],
    device float2*       output [[buffer(1)]],
    uint tid   [[thread_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]]
) {
    using namespace onepass;
    constexpr uint N = 8192u, T = 512u, E = 16u, XF = 4096u;
    threadgroup float x[XCHG_FLOATS];
    const uint base = tg_id * N;
    float2 d[E];
    #pragma unroll
    for (uint t = 0; t < 16u; ++t) d[t] = input[base + in_pos<N, T, 16u>(tid, 0u, t)];
    compute<N, T, E, 16u, 1u>(d, tid);
    exchange<N, T, E, 16u, 1u, 8u, XF>(d, x, tid);
    compute<N, T, E, 8u, 16u>(d, tid);
    exchange<N, T, E, 8u, 16u, 8u, XF>(d, x, tid);
    compute<N, T, E, 8u, 128u>(d, tid);
    exchange<N, T, E, 8u, 128u, 8u, XF>(d, x, tid);
    compute<N, T, E, 8u, 1024u>(d, tid);
    #pragma unroll
    for (uint b = 0; b < 2u; ++b)
        #pragma unroll
        for (uint t = 0; t < 8u; ++t)
            output[base + out_pos<T, 8u, 1024u>(tid, b, t)] = d[b * 8u + t];
}
