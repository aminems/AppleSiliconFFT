// =============================================================================
// m6_tensor_ops.metal — paper_m6 Stage 4: Metal 4 tensor ops (Metal Performance
// Primitives matmul2d) on the M6 GPU. On M5-class GPUs and later these may run on
// the per-core matrix units ("neural accelerators").
//
// Part A: dense GEMM throughput probes C[M,N] = A[M,K] * B[K,N], row-major, with
//         inline tensors (tensor_inline) over plain device buffers. Tile
//         (TM x TN) per threadgroup, SG simdgroups per threadgroup.
// Compile with MTLLanguageVersion 4.0.
// =============================================================================
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// dextents are (innermost, outer): a row-major [rows x cols] matrix is (cols, rows).
#define GEMM_KERNEL(NAME, TIN, TOUT, TM, TN, SG)                                          \
kernel void NAME(device TIN*  A [[buffer(0)]], device TIN* B [[buffer(1)]],               \
                 device TOUT* C [[buffer(2)]], constant uint3& mnk [[buffer(3)]],         \
                 uint2 tg [[threadgroup_position_in_grid]]) {                             \
    const int M = mnk.x, N = mnk.y, K = mnk.z;                                            \
    auto tA = tensor<device TIN,  dextents<int32_t, 2>, tensor_inline>(A, dextents<int32_t, 2>(K, M)); \
    auto tB = tensor<device TIN,  dextents<int32_t, 2>, tensor_inline>(B, dextents<int32_t, 2>(N, K)); \
    auto tC = tensor<device TOUT, dextents<int32_t, 2>, tensor_inline>(C, dextents<int32_t, 2>(N, M)); \
    constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent),  \
                              false, false, false, matmul2d_descriptor::mode::multiply);  \
    matmul2d<desc, execution_simdgroups<SG>> op;                                          \
    auto mA = tA.slice(0, tg.y * TM);                                                     \
    auto mB = tB.slice(tg.x * TN, 0);                                                     \
    auto mC = tC.slice(tg.x * TN, tg.y * TM);                                             \
    op.run(mA, mB, mC);                                                                   \
}

GEMM_KERNEL(gemm_h_f_64x32_sg4,  half,  float, 64, 32, 4)
GEMM_KERNEL(gemm_h_f_64x64_sg4,  half,  float, 64, 64, 4)
GEMM_KERNEL(gemm_h_f_128x64_sg4, half,  float, 128, 64, 4)
GEMM_KERNEL(gemm_h_h_64x32_sg4,  half,  half,  64, 32, 4)
GEMM_KERNEL(gemm_f_f_64x32_sg4,  float, float, 64, 32, 4)
GEMM_KERNEL(gemm_f_f_64x64_sg4,  float, float, 64, 64, 4)

// =============================================================================
// Part B: FFT-4096 on the matrix units (fp16 storage), one threadgroup per FFT.
//
// 4096 = 16^3, radix-16 Stockham. In natural order the input of every stage is
// a 16 x 256 matrix X[t][j] = x[j + 256 t], so a stage's 256 DFT-16s are ONE
// real matmul  [Yr;Yi] (32x256) = W16 (32x32) * [Xr;Xi] (32x256),
// W16 = [[Fr, -Fi], [Fi, Fr]], F[t'][t] = exp(-2 pi i t t'/16).
// Threadgroup memory: A = [Xr(4096) | Xi(4096)] half (16 KiB), B = result (16 KiB).
// Between stages thread j (= column j) twiddles its 16 outputs by
// W_{16 LS}^{(j mod LS) t'} and scatters them to Stockham position
// (j/LS)*16*LS + (j mod LS) + t'*LS in A. Last stage (LS = 256) is natural order.
// W16 lives in device memory (2 KiB, cache-resident).
// =============================================================================
constant uint T4K_THREADS = 256u;          // 8 simdgroups

template<bool DO_MM, bool DO_SCATTER>
inline void fft4096_tensor_body(device const half2* input, device half2* output, device half* W16,
                                threadgroup half* A, threadgroup half* B, uint tid, uint tg_id) {
    const uint base = tg_id * 4096u;
    #pragma unroll
    for (uint k = 0; k < 16u; ++k) {
        const uint i = tid + k * T4K_THREADS;
        const half2 v = input[base + i];
        A[i] = v.x; A[4096u + i] = v.y;
    }
    auto tW = tensor<device half, dextents<int32_t, 2>, tensor_inline>(W16, dextents<int32_t, 2>(32, 32));
    auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(A, dextents<int32_t, 2>(256, 32));
    auto tB = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(B, dextents<int32_t, 2>(256, 32));
    constexpr auto desc = matmul2d_descriptor(32, 256, 32, false, false, false,
                                              matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<8>> op;

    const uint j = tid;                     // this thread's column
    uint LS = 1u;
    for (uint s = 0; s < 3u; ++s) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (DO_MM) op.run(tW, tA, tB);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (DO_SCATTER && s < 2u) {
            // scatter stage-s outputs to their Stockham positions p; p is then input
            // (t2 = p / 256, j2 = p mod 256) of stage s+1, whose DIT input twiddle is
            // W_{16*LS2}^{(j2 mod LS2) * t2} with LS2 = 16 * LS.
            const uint LS2 = LS * 16u;
            const uint k = j & (LS - 1u);
            const uint obase = (j / LS) * 16u * LS + k;
            #pragma unroll
            for (uint t = 0; t < 16u; ++t) {
                float2 y = float2(float(B[t * 256u + j]), float(B[(16u + t) * 256u + j]));
                const uint p = obase + t * LS;
                const uint t2 = p >> 8, k2 = (p & 255u) & (LS2 - 1u);
                float sn, cs;
                sn = sincos(-2.0f * M_PI_F * float(k2 * t2) / float(16u * LS2), cs);
                y = float2(y.x * cs - y.y * sn, y.x * sn + y.y * cs);
                A[p] = half(y.x); A[4096u + p] = half(y.y);
            }
            LS = LS2;
        }
    }
    // last stage (LS = 256): output position (j/256)*4096 + j + 256 t' = j + 256 t' (natural)
    #pragma unroll
    for (uint t = 0; t < 16u; ++t) {
        float2 y = float2(float(B[t * 256u + j]), float(B[(16u + t) * 256u + j]));
        output[base + j + 256u * t] = half2(y);
    }
}

#define TENSOR_FFT_KERNEL(NAME, MM, SC)                                                  \
kernel void NAME(device const half2* input [[buffer(0)]], device half2* output [[buffer(1)]], \
                 device half* W16 [[buffer(2)]],                                          \
                 uint tid [[thread_index_in_threadgroup]],                                \
                 uint tg_id [[threadgroup_position_in_grid]]) {                           \
    threadgroup half A[8192];                                                             \
    threadgroup half B[8192];                                                             \
    fft4096_tensor_body<MM, SC>(input, output, W16, A, B, tid, tg_id);                    \
}
TENSOR_FFT_KERNEL(fft4096_tensor_f16,          true,  true)
// Attribution probes (NOT FFTs; outputs are wrong by design):
TENSOR_FFT_KERNEL(fft4096_tensor_f16_mmonly,   true,  false)   // load + 3 matmuls + store
