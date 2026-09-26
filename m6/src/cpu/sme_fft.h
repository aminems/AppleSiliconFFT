/*
 * sme_fft.h
 *
 * Batched complex FFT on the Apple SME2 matrix unit (M4 and later; SVL = 512 bit).
 *
 * Radix-8 FFT in which every butterfly, twiddles included, is one 16x16
 * real matrix M = diag(w_post) F8 diag(w_pre) applied by FMOPA outer products into a ZA tile:
 *
  *     ZA[i][b] += M[i][k] * X[k][b]        k = 0..15 (8 complex inputs as re/im rows)
 *
 * The 16 lanes of a tile hold 16 independent FFTs, so the data layout is
 * "batch-minor" in blocks of 16 FFTs:
 *
 *     blk[n][0][0..15] = Re x_b[n],   blk[n][1][0..15] = Im x_b[n],   b = 0..15
 *
 * (32 floats = 128 B per point, N*128 B per block). sme_fft_pack / sme_fft_unpack
 * convert to/from vDSP's split-complex layout (one realp/imagp pair per FFT)
 * with 16x16 transposes through ZA.
 *
 * Supported sizes: N = 8^s (64, 512, 4096, 32768).
 * Build: clang -O3 -mcpu=apple-m4 (NOT -march=armv9: Apple cores have no
 * non-streaming SVE, and armv9 lets the compiler emit it outside streaming mode).
 */
#ifndef SME_FFT_H
#define SME_FFT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sme_fft_plan sme_fft_plan;

/* Forward transform (e^{-2 pi i nk/N}), N = 8^s. Returns NULL for unsupported N. */
sme_fft_plan *sme_fft_plan_create(int n);
/* dir = -1: forward e^{-2 pi i nk/N}; dir = +1: inverse e^{+2 pi i nk/N}, unnormalized. */
sme_fft_plan *sme_fft_plan_create_dir(int n, int dir);
void sme_fft_plan_destroy(sme_fft_plan *p);

/* Bytes of one 16-FFT block in batch-minor layout (N * 128). */
size_t sme_fft_block_bytes(const sme_fft_plan *p);

/*
 * In-place FFT of nblk blocks of 16 FFTs (batch-minor layout).
 * scratch: one block (sme_fft_block_bytes), 64-byte aligned.
 *
 * sme_fft_execute: constant-geometry (Pease) DIF, every stage reads in[J + r N/8] and
 * writes out[8J + q] contiguously. Output is in base-8 DIGIT-REVERSED order;
 * sme_fft_unpack undoes it (the transpose gathers rows anyway, so it is free).
 * sme_fft_execute_stockham: Stockham autosort, natural order, but its outputs are
 * strided by the sub-length Ns, which makes ZA stores ~3x slower (kept for comparison).
 */
void sme_fft_execute(const sme_fft_plan *p, float *blocks, int nblk, float *scratch);
void sme_fft_execute_stockham(const sme_fft_plan *p, float *blocks, int nblk, float *scratch);

/*
 * Column FFTs of a row-major split-complex matrix (N rows, row_stride floats per row):
 * transforms columns 0..ncols-1 (ncols multiple of 16) from in_re/in_im into out_re/out_im
 * (may alias in), natural order. No transpose: 16 adjacent columns are one SVL vector.
 * This is the SAR azimuth FFT / second pass of a 2-D FFT.
 * Columns go 64 at a time (4 tiles = one butterfly x 64 lanes, 256 B per row access),
 * the remainder 16 at a time. scratch: sme_fft_columns_scratch_bytes, 64-byte aligned.
 */
size_t sme_fft_columns_scratch_bytes(const sme_fft_plan *p);
void sme_fft_set_wide(sme_fft_plan *p, int wide); /* 0: 16-lane groups only (ablation) */
void sme_fft_columns(const sme_fft_plan *p, const float *in_re, const float *in_im, float *out_re,
                     float *out_im, size_t row_stride, int ncols, float *scratch);

/*
 * Layout conversion for one block of 16 FFTs. re[b], im[b] point to FFT b's
 * split-complex arrays (length N).
 */
void sme_fft_pack(const sme_fft_plan *p, float *blk, float *const re[16], float *const im[16]);
void sme_fft_unpack(const sme_fft_plan *p, const float *blk, float *re[16], float *im[16]); /* un-digit-reverses */

#ifdef __cplusplus
}
#endif

#endif
