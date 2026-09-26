#include <metal_stdlib>
using namespace metal;

// ============================================================================
// N=4096 Batched FFT — Scalar radix-8 Stockham (4 passes)
//
// Split-radix DIT radix-8 approach (same as fft_4096_mma.metal), optimized
// for batched operation via multiple threadgroup dispatch.
//
// 512 threads per threadgroup, 8 elements per thread.
// 4 radix-8 passes: pass 0 from device memory, passes 1-3 via TG memory.
//
// Batching: dispatch batchSize threadgroups, each processing one N=4096 FFT.
// At batch=256, achieves ~138 GFLOPS on M1 (saturates GPU compute).
//
// MMA variant analysis: simdgroup_matrix MMA for radix-8 requires ping-pong
// TG buffers (64 KiB), exceeding the 32 KiB limit on M1/M2/M3. See
// research/theory/simdgroup_matrix_kernel_design.md for the full analysis.
// ============================================================================

constant uint N_FFT = 4096;
constant float SQRT2_2 = 0.70710678118654752f;
constant float TWO_PI_OVER_N = -2.0f * M_PI_F / float(N_FFT);

inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Split-radix DIT radix-8 butterfly (~32 FLOPs vs ~320 for naive)
inline void radix8_butterfly(thread float2& x0, thread float2& x1, thread float2& x2, thread float2& x3,
                             thread float2& x4, thread float2& x5, thread float2& x6, thread float2& x7) {
    float2 t0 = x0 + x4;  float2 t1 = x1 + x5;
    float2 t2 = x2 + x6;  float2 t3 = x3 + x7;
    float2 t4 = x0 - x4;  float2 t5 = x1 - x5;
    float2 t6 = x2 - x6;  float2 t7 = x3 - x7;

    float2 t5w = float2(SQRT2_2 * (t5.x + t5.y), SQRT2_2 * (t5.y - t5.x));
    float2 t6w = float2(t6.y, -t6.x);
    float2 t7w = float2(SQRT2_2 * (-t7.x + t7.y), SQRT2_2 * (-t7.y - t7.x));

    float2 u0 = t0 + t2;  float2 u1 = t1 + t3;
    float2 u2 = t0 - t2;  float2 u3 = t1 - t3;
    float2 u3r = float2(u3.y, -u3.x);

    float2 v0 = t4 + t6w;  float2 v1 = t5w + t7w;
    float2 v2 = t4 - t6w;  float2 v3 = t5w - t7w;
    float2 v3r = float2(v3.y, -v3.x);

    x0 = u0 + u1;  x4 = u0 - u1;
    x2 = u2 + u3r; x6 = u2 - u3r;
    x1 = v0 + v1;  x5 = v0 - v1;
    x3 = v2 + v3r; x7 = v2 - v3r;
}

inline void apply_twiddle8(thread float2& x1, thread float2& x2, thread float2& x3,
                           thread float2& x4, thread float2& x5, thread float2& x6,
                           thread float2& x7, float2 w1) {
    float2 w2 = cmul(w1, w1);
    float2 w3 = cmul(w2, w1);
    float2 w4 = cmul(w2, w2);
    float2 w5 = cmul(w4, w1);
    float2 w6 = cmul(w4, w2);
    float2 w7 = cmul(w4, w3);
    x1 = cmul(x1, w1); x2 = cmul(x2, w2); x3 = cmul(x3, w3);
    x4 = cmul(x4, w4); x5 = cmul(x5, w5); x6 = cmul(x6, w6);
    x7 = cmul(x7, w7);
}

// ============================================================================
// Scalar batched kernel (proven approach)
// ============================================================================

kernel void fft_4096_batched(
    device const float2* input    [[buffer(0)]],
    device float2*       output   [[buffer(1)]],
    device const float*  dft_real [[buffer(2)]],   // unused, kept for API compat
    device const float*  dft_imag [[buffer(3)]],   // unused, kept for API compat
    uint tid        [[thread_index_in_threadgroup]],
    uint tg_id      [[threadgroup_position_in_grid]]
) {
    const uint N = N_FFT;
    const uint base = tg_id * N;

    threadgroup float2 buf[4096];

    // ======== PASS 0: stride=1, read from device ========
    {
        float2 x0 = input[base + tid];
        float2 x1 = input[base + tid + 512u];
        float2 x2 = input[base + tid + 1024u];
        float2 x3 = input[base + tid + 1536u];
        float2 x4 = input[base + tid + 2048u];
        float2 x5 = input[base + tid + 2560u];
        float2 x6 = input[base + tid + 3072u];
        float2 x7 = input[base + tid + 3584u];

        radix8_butterfly(x0, x1, x2, x3, x4, x5, x6, x7);

        uint wr = tid << 3;
        buf[wr + 0] = x0; buf[wr + 1] = x1;
        buf[wr + 2] = x2; buf[wr + 3] = x3;
        buf[wr + 4] = x4; buf[wr + 5] = x5;
        buf[wr + 6] = x6; buf[wr + 7] = x7;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ======== PASS 1: stride=8, tw_scale=64 ========
    {
        uint pos = tid & 7u;
        uint grp = tid >> 3;

        float2 x0 = buf[tid];
        float2 x1 = buf[tid + 512u];
        float2 x2 = buf[tid + 1024u];
        float2 x3 = buf[tid + 1536u];
        float2 x4 = buf[tid + 2048u];
        float2 x5 = buf[tid + 2560u];
        float2 x6 = buf[tid + 3072u];
        float2 x7 = buf[tid + 3584u];

        {
            float a = TWO_PI_OVER_N * float(pos << 6);
            float s, c; s = sincos(a, c);
            apply_twiddle8(x1, x2, x3, x4, x5, x6, x7, float2(c, s));
        }

        radix8_butterfly(x0, x1, x2, x3, x4, x5, x6, x7);

        uint wr = grp * 64u + pos;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf[wr +  0] = x0; buf[wr +  8] = x1;
        buf[wr + 16] = x2; buf[wr + 24] = x3;
        buf[wr + 32] = x4; buf[wr + 40] = x5;
        buf[wr + 48] = x6; buf[wr + 56] = x7;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ======== PASS 2: stride=64, tw_scale=8 ========
    {
        uint pos = tid & 63u;
        uint grp = tid >> 6;

        float2 x0 = buf[tid];
        float2 x1 = buf[tid + 512u];
        float2 x2 = buf[tid + 1024u];
        float2 x3 = buf[tid + 1536u];
        float2 x4 = buf[tid + 2048u];
        float2 x5 = buf[tid + 2560u];
        float2 x6 = buf[tid + 3072u];
        float2 x7 = buf[tid + 3584u];

        {
            float a = TWO_PI_OVER_N * float(pos << 3);
            float s, c; s = sincos(a, c);
            apply_twiddle8(x1, x2, x3, x4, x5, x6, x7, float2(c, s));
        }

        radix8_butterfly(x0, x1, x2, x3, x4, x5, x6, x7);

        uint wr = grp * 512u + pos;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf[wr +   0] = x0; buf[wr +  64] = x1;
        buf[wr + 128] = x2; buf[wr + 192] = x3;
        buf[wr + 256] = x4; buf[wr + 320] = x5;
        buf[wr + 384] = x6; buf[wr + 448] = x7;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ======== PASS 3: stride=512, tw_scale=1, write to device ========
    {
        float2 x0 = buf[tid];
        float2 x1 = buf[tid + 512u];
        float2 x2 = buf[tid + 1024u];
        float2 x3 = buf[tid + 1536u];
        float2 x4 = buf[tid + 2048u];
        float2 x5 = buf[tid + 2560u];
        float2 x6 = buf[tid + 3072u];
        float2 x7 = buf[tid + 3584u];

        {
            float a = TWO_PI_OVER_N * float(tid);
            float s, c; s = sincos(a, c);
            apply_twiddle8(x1, x2, x3, x4, x5, x6, x7, float2(c, s));
        }

        radix8_butterfly(x0, x1, x2, x3, x4, x5, x6, x7);

        output[base + tid]        = x0;
        output[base + tid + 512u]  = x1;
        output[base + tid + 1024u] = x2;
        output[base + tid + 1536u] = x3;
        output[base + tid + 2048u] = x4;
        output[base + tid + 2560u] = x5;
        output[base + tid + 3072u] = x6;
        output[base + tid + 3584u] = x7;
    }
}

// MMA variant: requires 64 KiB TG memory (ping-pong buffers) which exceeds
// the M1/M2/M3 32 KiB limit. See simdgroup_matrix_kernel_design.md for the
// theoretical analysis of the MMA approach and why the scalar radix-8 above
// is the pragmatic choice for current Apple Silicon.
