#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Multi-size Stockham FFT kernels: N = 256, 512, 1024, 2048, 4096, 8192, 16384
//
// Single-threadgroup kernels (N <= 4096): pure radix-4 Stockham in threadgroup mem
// Multi-threadgroup (N > 4096): four-step FFT decomposition through device memory
//
// Carried from N=4096 kernel:
//   1. Single sincos per butterfly (w2=w1^2, w3=w1^3 via cmul)
//   2. Fully unrolled passes with compile-time constant strides
//   3. First pass reads from device, last pass writes to device
//   4. Branchless twiddle application
// ============================================================================

// --- Shared helpers ---

inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

inline void radix4(thread float2& x0, thread float2& x1,
                   thread float2& x2, thread float2& x3) {
    float2 t0 = x0 + x2;
    float2 t1 = x1 + x3;
    float2 t2 = x0 - x2;
    float2 t3 = x1 - x3;
    float2 t3r = float2(t3.y, -t3.x);
    x0 = t0 + t1;
    x1 = t2 + t3r;
    x2 = t0 - t1;
    x3 = t2 - t3r;
}

inline void radix2(thread float2& x0, thread float2& x1) {
    float2 t = x0;
    x0 = t + x1;
    x1 = t - x1;
}

inline void apply_twiddle3(thread float2& x1, thread float2& x2, thread float2& x3,
                           float2 w1) {
    float2 w2 = cmul(w1, w1);
    float2 w3 = cmul(w2, w1);
    x1 = cmul(x1, w1);
    x2 = cmul(x2, w2);
    x3 = cmul(x3, w3);
}


// ============================================================================
// N=256: 4 radix-4 passes, 64 threads, threadgroup buf[256]
// ============================================================================

kernel void fft_256_stockham(
    device const float2* input  [[buffer(0)]],
    device float2*       output [[buffer(1)]],
    uint tid        [[thread_index_in_threadgroup]],
    uint tg_id      [[threadgroup_position_in_grid]]
) {
    const uint N = 256;
    const uint T = 64;  // threads
    const float TWO_PI_OVER_N = -2.0f * M_PI_F / float(N);
    const uint base = tg_id * N;
    threadgroup float2 buf[256];

    // Pass 0: stride=1, no twiddles — read from device
    {
        float2 x0 = input[base + tid];
        float2 x1 = input[base + tid + T];
        float2 x2 = input[base + tid + 2*T];
        float2 x3 = input[base + tid + 3*T];
        radix4(x0, x1, x2, x3);
        uint wr = tid << 2;
        buf[wr]     = x0;
        buf[wr + 1] = x1;
        buf[wr + 2] = x2;
        buf[wr + 3] = x3;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Pass 1: stride=4, tw_shift=4 (N/4/stride = 256/4/4 = 16 -> pos<<4 but tw = pos * (N/4/1) ... )
    // tw_scale for pass p: stride_p = 4^p, tw_k = pos * (N / (4 * stride_p))
    // Pass 1: stride=4, tw_factor = N/(4*4) = 16 -> angle = TWO_PI_OVER_N * pos * 16
    {
        uint pos = tid & 3u;
        uint grp = tid >> 2;
        float2 x0 = buf[tid];
        float2 x1 = buf[tid + T];
        float2 x2 = buf[tid + 2*T];
        float2 x3 = buf[tid + 3*T];
        {
            float a1 = TWO_PI_OVER_N * float(pos * 16u);
            float s1, c1; s1 = sincos(a1, c1);
            apply_twiddle3(x1, x2, x3, float2(c1, s1));
        }
        radix4(x0, x1, x2, x3);
        uint wr = grp * 16u + pos;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf[wr]      = x0;
        buf[wr + 4]  = x1;
        buf[wr + 8]  = x2;
        buf[wr + 12] = x3;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Pass 2: stride=16, tw_factor = N/(4*16) = 4 -> angle = TWO_PI_OVER_N * pos * 4
    {
        uint pos = tid & 15u;
        uint grp = tid >> 4;
        float2 x0 = buf[tid];
        float2 x1 = buf[tid + T];
        float2 x2 = buf[tid + 2*T];
        float2 x3 = buf[tid + 3*T];
        {
            float a1 = TWO_PI_OVER_N * float(pos * 4u);
            float s1, c1; s1 = sincos(a1, c1);
            apply_twiddle3(x1, x2, x3, float2(c1, s1));
        }
        radix4(x0, x1, x2, x3);
        uint wr = grp * 64u + pos;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf[wr]      = x0;
        buf[wr + 16] = x1;
        buf[wr + 32] = x2;
        buf[wr + 48] = x3;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Pass 3: stride=64, tw_factor = N/(4*64) = 1 -> angle = TWO_PI_OVER_N * pos
    // Last pass — write to device
    {
        float2 x0 = buf[tid];
        float2 x1 = buf[tid + T];
        float2 x2 = buf[tid + 2*T];
        float2 x3 = buf[tid + 3*T];
        float a1 = TWO_PI_OVER_N * float(tid);
        float s1, c1; s1 = sincos(a1, c1);
        apply_twiddle3(x1, x2, x3, float2(c1, s1));
        radix4(x0, x1, x2, x3);
        output[base + tid]      = x0;
        output[base + tid + T]  = x1;
        output[base + tid + 2*T] = x2;
        output[base + tid + 3*T] = x3;
    }
}


// ============================================================================
// N=512: 4 radix-4 passes + 1 radix-2 pass, 128 threads, buf[512]
// ============================================================================

kernel void fft_512_stockham(
    device const float2* input  [[buffer(0)]],
    device float2*       output [[buffer(1)]],
    uint tid        [[thread_index_in_threadgroup]],
    uint tg_id      [[threadgroup_position_in_grid]]
) {
    const uint N = 512;
    const uint T = 128;
    const float TWO_PI_OVER_N = -2.0f * M_PI_F / float(N);
    const uint base = tg_id * N;
    threadgroup float2 buf[512];

    // Pass 0: radix-4, stride=1, no twiddles — read from device
    {
        float2 x0 = input[base + tid];
        float2 x1 = input[base + tid + T];
        float2 x2 = input[base + tid + 2*T];
        float2 x3 = input[base + tid + 3*T];
        radix4(x0, x1, x2, x3);
        uint wr = tid << 2;
        buf[wr]     = x0;
        buf[wr + 1] = x1;
        buf[wr + 2] = x2;
        buf[wr + 3] = x3;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Pass 1: radix-4, stride=4, tw_factor = N/(4*4) = 32
    {
        uint pos = tid & 3u;
        uint grp = tid >> 2;
        float2 x0 = buf[tid];
        float2 x1 = buf[tid + T];
        float2 x2 = buf[tid + 2*T];
        float2 x3 = buf[tid + 3*T];
        {
            float a1 = TWO_PI_OVER_N * float(pos * 32u);
            float s1, c1; s1 = sincos(a1, c1);
            apply_twiddle3(x1, x2, x3, float2(c1, s1));
        }
        radix4(x0, x1, x2, x3);
        uint wr = grp * 16u + pos;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf[wr]      = x0;
        buf[wr + 4]  = x1;
        buf[wr + 8]  = x2;
        buf[wr + 12] = x3;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Pass 2: radix-4, stride=16, tw_factor = N/(4*16) = 8
    {
        uint pos = tid & 15u;
        uint grp = tid >> 4;
        float2 x0 = buf[tid];
        float2 x1 = buf[tid + T];
        float2 x2 = buf[tid + 2*T];
        float2 x3 = buf[tid + 3*T];
        {
            float a1 = TWO_PI_OVER_N * float(pos * 8u);
            float s1, c1; s1 = sincos(a1, c1);
            apply_twiddle3(x1, x2, x3, float2(c1, s1));
        }
        radix4(x0, x1, x2, x3);
        uint wr = grp * 64u + pos;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf[wr]      = x0;
        buf[wr + 16] = x1;
        buf[wr + 32] = x2;
        buf[wr + 48] = x3;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Pass 3: radix-4, stride=64, tw_factor = N/(4*64) = 2
    {
        uint pos = tid & 63u;
        uint grp = tid >> 6;
        float2 x0 = buf[tid];
        float2 x1 = buf[tid + T];
        float2 x2 = buf[tid + 2*T];
        float2 x3 = buf[tid + 3*T];
        {
            float a1 = TWO_PI_OVER_N * float(pos * 2u);
            float s1, c1; s1 = sincos(a1, c1);
            apply_twiddle3(x1, x2, x3, float2(c1, s1));
        }
        radix4(x0, x1, x2, x3);
        // After this pass, data is in stride-256 Stockham layout
        // We need one more radix-2 pass to finish 512 = 4^4 * 2
        uint wr = grp * 256u + pos;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf[wr]       = x0;
        buf[wr + 64]  = x1;
        buf[wr + 128] = x2;
        buf[wr + 192] = x3;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Pass 4: radix-2, stride=256, tw_factor = 1 — write to device
    // 256 butterflies, 128 threads -> 2 butterflies per thread
    {
        // Butterfly 1: positions tid and tid+256
        float2 x0 = buf[tid];
        float2 x1 = buf[tid + 256];
        float a1 = TWO_PI_OVER_N * float(tid);
        float s1, c1; s1 = sincos(a1, c1);
        x1 = cmul(x1, float2(c1, s1));
        radix2(x0, x1);
        output[base + tid]       = x0;
        output[base + tid + 256] = x1;

        // Butterfly 2: positions tid+128 and tid+128+256
        float2 y0 = buf[tid + 128];
        float2 y1 = buf[tid + 128 + 256];
        float a2 = TWO_PI_OVER_N * float(tid + 128);
        float s2, c2; s2 = sincos(a2, c2);
        y1 = cmul(y1, float2(c2, s2));
        radix2(y0, y1);
        output[base + tid + 128]       = y0;
        output[base + tid + 128 + 256] = y1;
    }
}


// ============================================================================
// N=1024: 5 radix-4 passes, 256 threads, buf[1024]
// ============================================================================

kernel void fft_1024_stockham(
    device const float2* input  [[buffer(0)]],
    device float2*       output [[buffer(1)]],
    uint tid        [[thread_index_in_threadgroup]],
    uint tg_id      [[threadgroup_position_in_grid]]
) {
    const uint N = 1024;
    const uint T = 256;
    const float TWO_PI_OVER_N = -2.0f * M_PI_F / float(N);
    const uint base = tg_id * N;
    threadgroup float2 buf[1024];

    // Pass 0: stride=1, no twiddles — read from device
    {
        float2 x0 = input[base + tid];
        float2 x1 = input[base + tid + T];
        float2 x2 = input[base + tid + 2*T];
        float2 x3 = input[base + tid + 3*T];
        radix4(x0, x1, x2, x3);
        uint wr = tid << 2;
        buf[wr]     = x0;
        buf[wr + 1] = x1;
        buf[wr + 2] = x2;
        buf[wr + 3] = x3;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Pass 1: stride=4, tw_factor = N/(4*4) = 64
    {
        uint pos = tid & 3u;
        uint grp = tid >> 2;
        float2 x0 = buf[tid];
        float2 x1 = buf[tid + T];
        float2 x2 = buf[tid + 2*T];
        float2 x3 = buf[tid + 3*T];
        {
            float a1 = TWO_PI_OVER_N * float(pos * 64u);
            float s1, c1; s1 = sincos(a1, c1);
            apply_twiddle3(x1, x2, x3, float2(c1, s1));
        }
        radix4(x0, x1, x2, x3);
        uint wr = grp * 16u + pos;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf[wr]      = x0;
        buf[wr + 4]  = x1;
        buf[wr + 8]  = x2;
        buf[wr + 12] = x3;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Pass 2: stride=16, tw_factor = N/(4*16) = 16
    {
        uint pos = tid & 15u;
        uint grp = tid >> 4;
        float2 x0 = buf[tid];
        float2 x1 = buf[tid + T];
        float2 x2 = buf[tid + 2*T];
        float2 x3 = buf[tid + 3*T];
        {
            float a1 = TWO_PI_OVER_N * float(pos * 16u);
            float s1, c1; s1 = sincos(a1, c1);
            apply_twiddle3(x1, x2, x3, float2(c1, s1));
        }
        radix4(x0, x1, x2, x3);
        uint wr = grp * 64u + pos;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf[wr]      = x0;
        buf[wr + 16] = x1;
        buf[wr + 32] = x2;
        buf[wr + 48] = x3;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Pass 3: stride=64, tw_factor = N/(4*64) = 4
    {
        uint pos = tid & 63u;
        uint grp = tid >> 6;
        float2 x0 = buf[tid];
        float2 x1 = buf[tid + T];
        float2 x2 = buf[tid + 2*T];
        float2 x3 = buf[tid + 3*T];
        {
            float a1 = TWO_PI_OVER_N * float(pos * 4u);
            float s1, c1; s1 = sincos(a1, c1);
            apply_twiddle3(x1, x2, x3, float2(c1, s1));
        }
        radix4(x0, x1, x2, x3);
        uint wr = grp * 256u + pos;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf[wr]       = x0;
        buf[wr + 64]  = x1;
        buf[wr + 128] = x2;
        buf[wr + 192] = x3;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Pass 4: stride=256, tw_factor = 1 — write to device
    {
        float2 x0 = buf[tid];
        float2 x1 = buf[tid + T];
        float2 x2 = buf[tid + 2*T];
        float2 x3 = buf[tid + 3*T];
        float a1 = TWO_PI_OVER_N * float(tid);
        float s1, c1; s1 = sincos(a1, c1);
        apply_twiddle3(x1, x2, x3, float2(c1, s1));
        radix4(x0, x1, x2, x3);
        output[base + tid]      = x0;
        output[base + tid + T]  = x1;
        output[base + tid + 2*T] = x2;
        output[base + tid + 3*T] = x3;
    }
}


// ============================================================================
// N=2048: 5 radix-4 passes + 1 radix-2 pass, 512 threads, buf[2048]
// ============================================================================

kernel void fft_2048_stockham(
    device const float2* input  [[buffer(0)]],
    device float2*       output [[buffer(1)]],
    uint tid        [[thread_index_in_threadgroup]],
    uint tg_id      [[threadgroup_position_in_grid]]
) {
    const uint N = 2048;
    const uint T = 512;
    const float TWO_PI_OVER_N = -2.0f * M_PI_F / float(N);
    const uint base = tg_id * N;
    threadgroup float2 buf[2048];

    // Pass 0: radix-4, stride=1, no twiddles — read from device
    {
        float2 x0 = input[base + tid];
        float2 x1 = input[base + tid + T];
        float2 x2 = input[base + tid + 2*T];
        float2 x3 = input[base + tid + 3*T];
        radix4(x0, x1, x2, x3);
        uint wr = tid << 2;
        buf[wr]     = x0;
        buf[wr + 1] = x1;
        buf[wr + 2] = x2;
        buf[wr + 3] = x3;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Pass 1: stride=4, tw_factor = N/(4*4) = 128
    {
        uint pos = tid & 3u;
        uint grp = tid >> 2;
        float2 x0 = buf[tid];
        float2 x1 = buf[tid + T];
        float2 x2 = buf[tid + 2*T];
        float2 x3 = buf[tid + 3*T];
        {
            float a1 = TWO_PI_OVER_N * float(pos * 128u);
            float s1, c1; s1 = sincos(a1, c1);
            apply_twiddle3(x1, x2, x3, float2(c1, s1));
        }
        radix4(x0, x1, x2, x3);
        uint wr = grp * 16u + pos;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf[wr]      = x0;
        buf[wr + 4]  = x1;
        buf[wr + 8]  = x2;
        buf[wr + 12] = x3;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Pass 2: stride=16, tw_factor = N/(4*16) = 32
    {
        uint pos = tid & 15u;
        uint grp = tid >> 4;
        float2 x0 = buf[tid];
        float2 x1 = buf[tid + T];
        float2 x2 = buf[tid + 2*T];
        float2 x3 = buf[tid + 3*T];
        {
            float a1 = TWO_PI_OVER_N * float(pos * 32u);
            float s1, c1; s1 = sincos(a1, c1);
            apply_twiddle3(x1, x2, x3, float2(c1, s1));
        }
        radix4(x0, x1, x2, x3);
        uint wr = grp * 64u + pos;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf[wr]      = x0;
        buf[wr + 16] = x1;
        buf[wr + 32] = x2;
        buf[wr + 48] = x3;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Pass 3: stride=64, tw_factor = N/(4*64) = 8
    {
        uint pos = tid & 63u;
        uint grp = tid >> 6;
        float2 x0 = buf[tid];
        float2 x1 = buf[tid + T];
        float2 x2 = buf[tid + 2*T];
        float2 x3 = buf[tid + 3*T];
        {
            float a1 = TWO_PI_OVER_N * float(pos * 8u);
            float s1, c1; s1 = sincos(a1, c1);
            apply_twiddle3(x1, x2, x3, float2(c1, s1));
        }
        radix4(x0, x1, x2, x3);
        uint wr = grp * 256u + pos;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf[wr]       = x0;
        buf[wr + 64]  = x1;
        buf[wr + 128] = x2;
        buf[wr + 192] = x3;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Pass 4: stride=256, tw_factor = N/(4*256) = 2
    {
        uint pos = tid & 255u;
        uint grp = tid >> 8;
        float2 x0 = buf[tid];
        float2 x1 = buf[tid + T];
        float2 x2 = buf[tid + 2*T];
        float2 x3 = buf[tid + 3*T];
        {
            float a1 = TWO_PI_OVER_N * float(pos * 2u);
            float s1, c1; s1 = sincos(a1, c1);
            apply_twiddle3(x1, x2, x3, float2(c1, s1));
        }
        radix4(x0, x1, x2, x3);
        uint wr = grp * 1024u + pos;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf[wr]        = x0;
        buf[wr + 256]  = x1;
        buf[wr + 512]  = x2;
        buf[wr + 768]  = x3;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Pass 5: radix-2, stride=1024, tw_factor = 1 — write to device
    // 1024 butterflies, 512 threads -> 2 butterflies per thread
    {
        // Butterfly 1: positions tid and tid+1024
        float2 x0 = buf[tid];
        float2 x1 = buf[tid + 1024];
        float a1 = TWO_PI_OVER_N * float(tid);
        float s1, c1; s1 = sincos(a1, c1);
        x1 = cmul(x1, float2(c1, s1));
        radix2(x0, x1);
        output[base + tid]        = x0;
        output[base + tid + 1024] = x1;

        // Butterfly 2: positions tid+512 and tid+512+1024
        float2 y0 = buf[tid + 512];
        float2 y1 = buf[tid + 512 + 1024];
        float a2 = TWO_PI_OVER_N * float(tid + 512);
        float s2, c2; s2 = sincos(a2, c2);
        y1 = cmul(y1, float2(c2, s2));
        radix2(y0, y1);
        output[base + tid + 512]        = y0;
        output[base + tid + 512 + 1024] = y1;
    }
}


// ============================================================================
// N=4096: 6 radix-4 passes, 1024 threads, buf[4096]
// (same as fft_stockham_4096.metal but renamed for consistency)
// ============================================================================

kernel void fft_4096_stockham(
    device const float2* input  [[buffer(0)]],
    device float2*       output [[buffer(1)]],
    uint tid        [[thread_index_in_threadgroup]],
    uint tg_id      [[threadgroup_position_in_grid]]
) {
    const uint N_FFT = 4096;
    const float TWO_PI_OVER_N = -2.0f * M_PI_F / float(N_FFT);
    const uint base = tg_id * N_FFT;
    threadgroup float2 buf[4096];

    // Pass 0: stride=1, no twiddles — read from device
    {
        float2 x0 = input[base + tid];
        float2 x1 = input[base + tid + 1024];
        float2 x2 = input[base + tid + 2048];
        float2 x3 = input[base + tid + 3072];
        radix4(x0, x1, x2, x3);
        uint wr = tid << 2;
        buf[wr]     = x0;
        buf[wr + 1] = x1;
        buf[wr + 2] = x2;
        buf[wr + 3] = x3;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Pass 1: stride=4, tw_scale=256
    {
        uint pos = tid & 3u;
        uint grp = tid >> 2;
        float2 x0 = buf[tid];
        float2 x1 = buf[tid + 1024];
        float2 x2 = buf[tid + 2048];
        float2 x3 = buf[tid + 3072];
        {
            float a1 = TWO_PI_OVER_N * float(pos << 8);
            float s1, c1; s1 = sincos(a1, c1);
            apply_twiddle3(x1, x2, x3, float2(c1, s1));
        }
        radix4(x0, x1, x2, x3);
        uint wr = grp * 16u + pos;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf[wr]      = x0;
        buf[wr + 4]  = x1;
        buf[wr + 8]  = x2;
        buf[wr + 12] = x3;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Pass 2: stride=16, tw_scale=64
    {
        uint pos = tid & 15u;
        uint grp = tid >> 4;
        float2 x0 = buf[tid];
        float2 x1 = buf[tid + 1024];
        float2 x2 = buf[tid + 2048];
        float2 x3 = buf[tid + 3072];
        {
            float a1 = TWO_PI_OVER_N * float(pos << 6);
            float s1, c1; s1 = sincos(a1, c1);
            apply_twiddle3(x1, x2, x3, float2(c1, s1));
        }
        radix4(x0, x1, x2, x3);
        uint wr = grp * 64u + pos;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf[wr]      = x0;
        buf[wr + 16] = x1;
        buf[wr + 32] = x2;
        buf[wr + 48] = x3;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Pass 3: stride=64, tw_scale=16
    {
        uint pos = tid & 63u;
        uint grp = tid >> 6;
        float2 x0 = buf[tid];
        float2 x1 = buf[tid + 1024];
        float2 x2 = buf[tid + 2048];
        float2 x3 = buf[tid + 3072];
        {
            float a1 = TWO_PI_OVER_N * float(pos << 4);
            float s1, c1; s1 = sincos(a1, c1);
            apply_twiddle3(x1, x2, x3, float2(c1, s1));
        }
        radix4(x0, x1, x2, x3);
        uint wr = grp * 256u + pos;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf[wr]       = x0;
        buf[wr + 64]  = x1;
        buf[wr + 128] = x2;
        buf[wr + 192] = x3;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Pass 4: stride=256, tw_scale=4
    {
        uint pos = tid & 255u;
        uint grp = tid >> 8;
        float2 x0 = buf[tid];
        float2 x1 = buf[tid + 1024];
        float2 x2 = buf[tid + 2048];
        float2 x3 = buf[tid + 3072];
        float a1 = TWO_PI_OVER_N * float(pos << 2);
        float s1, c1; s1 = sincos(a1, c1);
        apply_twiddle3(x1, x2, x3, float2(c1, s1));
        radix4(x0, x1, x2, x3);
        uint wr = grp * 1024u + pos;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf[wr]        = x0;
        buf[wr + 256]  = x1;
        buf[wr + 512]  = x2;
        buf[wr + 768]  = x3;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Pass 5: stride=1024, tw_scale=1 — write to device
    {
        float2 x0 = buf[tid];
        float2 x1 = buf[tid + 1024];
        float2 x2 = buf[tid + 2048];
        float2 x3 = buf[tid + 3072];
        float a1 = TWO_PI_OVER_N * float(tid);
        float s1, c1; s1 = sincos(a1, c1);
        apply_twiddle3(x1, x2, x3, float2(c1, s1));
        radix4(x0, x1, x2, x3);
        output[base + tid]        = x0;
        output[base + tid + 1024] = x1;
        output[base + tid + 2048] = x2;
        output[base + tid + 3072] = x3;
    }
}


// ============================================================================
// Corner-turn column FFT: FFT down the columns of a 4096x4096 (row-major)
// matrix in ONE dispatch via column-strided device I/O — no separate transpose
// passes. Replaces transpose -> fft_4096_stockham -> transpose (6 NxN device
// transfers) with a single kernel (2 NxN strided transfers). Threadgroup
// tg_id = column index; only pass-0 load and pass-5 store differ from
// fft_4096_stockham (strided by NR instead of contiguous).
// ============================================================================
kernel void fft_4096_stockham_cols(
    device const float2* input  [[buffer(0)]],
    device float2*       output [[buffer(1)]],
    uint tid        [[thread_index_in_threadgroup]],
    uint tg_id      [[threadgroup_position_in_grid]]
) {
    const uint N_FFT = 4096;
    const uint NR = 4096;                 // matrix stride (cols per row), square SAR
    const float TWO_PI_OVER_N = -2.0f * M_PI_F / float(N_FFT);
    const uint col = tg_id;
    threadgroup float2 buf[4096];

    // Pass 0: column-strided load (the corner turn)
    {
        float2 x0 = input[(tid)        * NR + col];
        float2 x1 = input[(tid + 1024) * NR + col];
        float2 x2 = input[(tid + 2048) * NR + col];
        float2 x3 = input[(tid + 3072) * NR + col];
        radix4(x0, x1, x2, x3);
        uint wr = tid << 2;
        buf[wr] = x0; buf[wr + 1] = x1; buf[wr + 2] = x2; buf[wr + 3] = x3;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Pass 1: stride=4
    {
        uint pos = tid & 3u; uint grp = tid >> 2;
        float2 x0 = buf[tid]; float2 x1 = buf[tid + 1024];
        float2 x2 = buf[tid + 2048]; float2 x3 = buf[tid + 3072];
        { float a1 = TWO_PI_OVER_N * float(pos << 8); float s1, c1; s1 = sincos(a1, c1);
          apply_twiddle3(x1, x2, x3, float2(c1, s1)); }
        radix4(x0, x1, x2, x3);
        uint wr = grp * 16u + pos;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf[wr] = x0; buf[wr + 4] = x1; buf[wr + 8] = x2; buf[wr + 12] = x3;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Pass 2: stride=16
    {
        uint pos = tid & 15u; uint grp = tid >> 4;
        float2 x0 = buf[tid]; float2 x1 = buf[tid + 1024];
        float2 x2 = buf[tid + 2048]; float2 x3 = buf[tid + 3072];
        { float a1 = TWO_PI_OVER_N * float(pos << 6); float s1, c1; s1 = sincos(a1, c1);
          apply_twiddle3(x1, x2, x3, float2(c1, s1)); }
        radix4(x0, x1, x2, x3);
        uint wr = grp * 64u + pos;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf[wr] = x0; buf[wr + 16] = x1; buf[wr + 32] = x2; buf[wr + 48] = x3;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Pass 3: stride=64
    {
        uint pos = tid & 63u; uint grp = tid >> 6;
        float2 x0 = buf[tid]; float2 x1 = buf[tid + 1024];
        float2 x2 = buf[tid + 2048]; float2 x3 = buf[tid + 3072];
        { float a1 = TWO_PI_OVER_N * float(pos << 4); float s1, c1; s1 = sincos(a1, c1);
          apply_twiddle3(x1, x2, x3, float2(c1, s1)); }
        radix4(x0, x1, x2, x3);
        uint wr = grp * 256u + pos;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf[wr] = x0; buf[wr + 64] = x1; buf[wr + 128] = x2; buf[wr + 192] = x3;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Pass 4: stride=256
    {
        uint pos = tid & 255u; uint grp = tid >> 8;
        float2 x0 = buf[tid]; float2 x1 = buf[tid + 1024];
        float2 x2 = buf[tid + 2048]; float2 x3 = buf[tid + 3072];
        float a1 = TWO_PI_OVER_N * float(pos << 2); float s1, c1; s1 = sincos(a1, c1);
        apply_twiddle3(x1, x2, x3, float2(c1, s1));
        radix4(x0, x1, x2, x3);
        uint wr = grp * 1024u + pos;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf[wr] = x0; buf[wr + 256] = x1; buf[wr + 512] = x2; buf[wr + 768] = x3;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Pass 5: column-strided store (the corner turn)
    {
        float2 x0 = buf[tid]; float2 x1 = buf[tid + 1024];
        float2 x2 = buf[tid + 2048]; float2 x3 = buf[tid + 3072];
        float a1 = TWO_PI_OVER_N * float(tid); float s1, c1; s1 = sincos(a1, c1);
        apply_twiddle3(x1, x2, x3, float2(c1, s1));
        radix4(x0, x1, x2, x3);
        output[(tid)        * NR + col] = x0;
        output[(tid + 1024) * NR + col] = x1;
        output[(tid + 2048) * NR + col] = x2;
        output[(tid + 3072) * NR + col] = x3;
    }
}


// ============================================================================
// Multi-threadgroup kernels for N=8192 and N=16384
// Four-step FFT: two passes of column/row FFTs with twiddle+transpose between
// ============================================================================

// --- Twiddle + transpose kernel ---
// For N = N1 * N2, input is N1 rows x N2 cols after pass 1 (N1 row-FFTs of size N2).
// Element at position [row][col] = input[row * N2 + col], row in [0,N1), col in [0,N2).
// Apply twiddle W_N^{row*col} and transpose to N2 rows x N1 cols.

kernel void fft_twiddle_transpose(
    device const float2* input  [[buffer(0)]],
    device float2*       output [[buffer(1)]],
    constant uint&       N1     [[buffer(2)]],
    constant uint&       N2     [[buffer(3)]],
    uint tid        [[thread_index_in_threadgroup]],
    uint tg_id      [[threadgroup_position_in_grid]],
    uint tg_size    [[threads_per_threadgroup]]
) {
    uint gid = tg_id * tg_size + tid;
    uint N = N1 * N2;
    if (gid >= N) return;

    // Input layout: N1 rows x N2 cols
    uint row = gid / N2;  // n1 index, in [0, N1)
    uint col = gid % N2;  // k2 index, in [0, N2)

    float2 val = input[gid];

    // Apply twiddle: W_N^{row * col} = e^{-2*pi*i * row * col / N}
    float angle = -2.0f * M_PI_F * float(row * col) / float(N);
    float s, c;
    s = sincos(angle, c);
    val = cmul(val, float2(c, s));

    // Transpose: write to [col][row] in N2 x N1 layout
    output[col * N1 + row] = val;
}

// --- Simple transpose kernel ---
// Transposes from ROWS x COLS to COLS x ROWS

kernel void fft_transpose(
    device const float2* input  [[buffer(0)]],
    device float2*       output [[buffer(1)]],
    constant uint&       ROWS   [[buffer(2)]],
    constant uint&       COLS   [[buffer(3)]],
    uint tid        [[thread_index_in_threadgroup]],
    uint tg_id      [[threadgroup_position_in_grid]],
    uint tg_size    [[threads_per_threadgroup]]
) {
    uint gid = tg_id * tg_size + tid;
    uint N = ROWS * COLS;
    if (gid >= N) return;

    uint row = gid / COLS;
    uint col = gid % COLS;

    output[col * ROWS + row] = input[gid];
}


// ============================================================================
// Sub-FFT kernels for four-step decomposition
// These are identical to the main kernels but used by the host for sub-FFTs.
// N=64: 3 radix-4 passes, 16 threads, buf[64]
// N=128: 3 radix-4 + 1 radix-2, 32 threads, buf[128]
// ============================================================================

// ============================================================================
// N=32 = 4*4*2: 2 radix-4 passes + 1 radix-2 pass; T=8 threads
// (Added for paper13 Shazam read mapper batched-small-N workload.)
// ============================================================================
kernel void fft_32_stockham(
    device const float2* input  [[buffer(0)]],
    device float2*       output [[buffer(1)]],
    uint tid        [[thread_index_in_threadgroup]],
    uint tg_id      [[threadgroup_position_in_grid]]
) {
    const uint N = 32;
    const uint T = 8;
    const float TWO_PI_OVER_N = -2.0f * M_PI_F / float(N);
    const uint base = tg_id * N;
    threadgroup float2 buf[32];

    // Pass 0: radix-4, stride=1, no twiddles
    {
        float2 x0 = input[base + tid];
        float2 x1 = input[base + tid + T];
        float2 x2 = input[base + tid + 2*T];
        float2 x3 = input[base + tid + 3*T];
        radix4(x0, x1, x2, x3);
        uint wr = tid << 2;
        buf[wr]     = x0;
        buf[wr + 1] = x1;
        buf[wr + 2] = x2;
        buf[wr + 3] = x3;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Pass 1: radix-4, stride=4, tw_factor = N/(4*4) = 2
    {
        uint pos = tid & 3u;
        uint grp = tid >> 2;
        float2 x0 = buf[tid];
        float2 x1 = buf[tid + T];
        float2 x2 = buf[tid + 2*T];
        float2 x3 = buf[tid + 3*T];
        {
            float a1 = TWO_PI_OVER_N * float(pos * 2u);
            float s1, c1; s1 = sincos(a1, c1);
            apply_twiddle3(x1, x2, x3, float2(c1, s1));
        }
        radix4(x0, x1, x2, x3);
        uint wr = grp * 16u + pos;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf[wr]      = x0;
        buf[wr + 4]  = x1;
        buf[wr + 8]  = x2;
        buf[wr + 12] = x3;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Pass 2: radix-2, stride=16, tw_factor = 1 -> write to device
    // 16 butterflies, 8 threads -> 2 per thread
    {
        // Butterfly 1
        float2 x0 = buf[tid];
        float2 x1 = buf[tid + 16];
        float a1 = TWO_PI_OVER_N * float(tid);
        float s1, c1; s1 = sincos(a1, c1);
        x1 = cmul(x1, float2(c1, s1));
        radix2(x0, x1);
        output[base + tid]      = x0;
        output[base + tid + 16] = x1;

        // Butterfly 2
        float2 y0 = buf[tid + 8];
        float2 y1 = buf[tid + 8 + 16];
        float a2 = TWO_PI_OVER_N * float(tid + 8);
        float s2, c2; s2 = sincos(a2, c2);
        y1 = cmul(y1, float2(c2, s2));
        radix2(y0, y1);
        output[base + tid + 8]      = y0;
        output[base + tid + 8 + 16] = y1;
    }
}


kernel void fft_64_stockham(
    device const float2* input  [[buffer(0)]],
    device float2*       output [[buffer(1)]],
    uint tid        [[thread_index_in_threadgroup]],
    uint tg_id      [[threadgroup_position_in_grid]]
) {
    const uint N = 64;
    const uint T = 16;
    const float TWO_PI_OVER_N = -2.0f * M_PI_F / float(N);
    const uint base = tg_id * N;
    threadgroup float2 buf[64];

    // Pass 0: stride=1, no twiddles — read from device
    {
        float2 x0 = input[base + tid];
        float2 x1 = input[base + tid + T];
        float2 x2 = input[base + tid + 2*T];
        float2 x3 = input[base + tid + 3*T];
        radix4(x0, x1, x2, x3);
        uint wr = tid << 2;
        buf[wr]     = x0;
        buf[wr + 1] = x1;
        buf[wr + 2] = x2;
        buf[wr + 3] = x3;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Pass 1: stride=4, tw_factor = N/(4*4) = 4
    {
        uint pos = tid & 3u;
        uint grp = tid >> 2;
        float2 x0 = buf[tid];
        float2 x1 = buf[tid + T];
        float2 x2 = buf[tid + 2*T];
        float2 x3 = buf[tid + 3*T];
        {
            float a1 = TWO_PI_OVER_N * float(pos * 4u);
            float s1, c1; s1 = sincos(a1, c1);
            apply_twiddle3(x1, x2, x3, float2(c1, s1));
        }
        radix4(x0, x1, x2, x3);
        uint wr = grp * 16u + pos;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf[wr]      = x0;
        buf[wr + 4]  = x1;
        buf[wr + 8]  = x2;
        buf[wr + 12] = x3;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Pass 2: stride=16, tw_factor = 1 — write to device
    {
        float2 x0 = buf[tid];
        float2 x1 = buf[tid + T];
        float2 x2 = buf[tid + 2*T];
        float2 x3 = buf[tid + 3*T];
        float a1 = TWO_PI_OVER_N * float(tid);
        float s1, c1; s1 = sincos(a1, c1);
        apply_twiddle3(x1, x2, x3, float2(c1, s1));
        radix4(x0, x1, x2, x3);
        output[base + tid]      = x0;
        output[base + tid + T]  = x1;
        output[base + tid + 2*T] = x2;
        output[base + tid + 3*T] = x3;
    }
}


kernel void fft_128_stockham(
    device const float2* input  [[buffer(0)]],
    device float2*       output [[buffer(1)]],
    uint tid        [[thread_index_in_threadgroup]],
    uint tg_id      [[threadgroup_position_in_grid]]
) {
    const uint N = 128;
    const uint T = 32;
    const float TWO_PI_OVER_N = -2.0f * M_PI_F / float(N);
    const uint base = tg_id * N;
    threadgroup float2 buf[128];

    // Pass 0: radix-4, stride=1, no twiddles — read from device
    {
        float2 x0 = input[base + tid];
        float2 x1 = input[base + tid + T];
        float2 x2 = input[base + tid + 2*T];
        float2 x3 = input[base + tid + 3*T];
        radix4(x0, x1, x2, x3);
        uint wr = tid << 2;
        buf[wr]     = x0;
        buf[wr + 1] = x1;
        buf[wr + 2] = x2;
        buf[wr + 3] = x3;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Pass 1: radix-4, stride=4, tw_factor = N/(4*4) = 8
    {
        uint pos = tid & 3u;
        uint grp = tid >> 2;
        float2 x0 = buf[tid];
        float2 x1 = buf[tid + T];
        float2 x2 = buf[tid + 2*T];
        float2 x3 = buf[tid + 3*T];
        {
            float a1 = TWO_PI_OVER_N * float(pos * 8u);
            float s1, c1; s1 = sincos(a1, c1);
            apply_twiddle3(x1, x2, x3, float2(c1, s1));
        }
        radix4(x0, x1, x2, x3);
        uint wr = grp * 16u + pos;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf[wr]      = x0;
        buf[wr + 4]  = x1;
        buf[wr + 8]  = x2;
        buf[wr + 12] = x3;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Pass 2: radix-4, stride=16, tw_factor = N/(4*16) = 2
    {
        uint pos = tid & 15u;
        uint grp = tid >> 4;
        float2 x0 = buf[tid];
        float2 x1 = buf[tid + T];
        float2 x2 = buf[tid + 2*T];
        float2 x3 = buf[tid + 3*T];
        {
            float a1 = TWO_PI_OVER_N * float(pos * 2u);
            float s1, c1; s1 = sincos(a1, c1);
            apply_twiddle3(x1, x2, x3, float2(c1, s1));
        }
        radix4(x0, x1, x2, x3);
        uint wr = grp * 64u + pos;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf[wr]      = x0;
        buf[wr + 16] = x1;
        buf[wr + 32] = x2;
        buf[wr + 48] = x3;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Pass 3: radix-2, stride=64, tw_factor = 1 — write to device
    // 64 butterflies, 32 threads -> 2 per thread
    {
        // Butterfly 1
        float2 x0 = buf[tid];
        float2 x1 = buf[tid + 64];
        float a1 = TWO_PI_OVER_N * float(tid);
        float s1, c1; s1 = sincos(a1, c1);
        x1 = cmul(x1, float2(c1, s1));
        radix2(x0, x1);
        output[base + tid]      = x0;
        output[base + tid + 64] = x1;

        // Butterfly 2
        float2 y0 = buf[tid + 32];
        float2 y1 = buf[tid + 32 + 64];
        float a2 = TWO_PI_OVER_N * float(tid + 32);
        float s2, c2; s2 = sincos(a2, c2);
        y1 = cmul(y1, float2(c2, s2));
        radix2(y0, y1);
        output[base + tid + 32]      = y0;
        output[base + tid + 32 + 64] = y1;
    }
}
