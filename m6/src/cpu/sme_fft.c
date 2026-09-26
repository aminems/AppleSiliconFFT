/*
 * sme_fft.c — batched radix-8 Stockham FFT on the SME2 matrix unit. See sme_fft.h.
 *
 * Stage with sub-length Ns (1, 8, 64, ...), butterfly j = 0..N/8-1, t = j % Ns:
 *     x_r = in[j + r N/8],                    r = 0..7
 *     y_q = sum_r F8[q][r] w^{t r} x_r,       w = e^{-2 pi i / (8 Ns)}
 *     out[(j/Ns) 8 Ns + t + q Ns] = y_q
 * As a real 16x16 matrix M_t (rows 2q,2q+1 = Re,Im y_q; cols 2r,2r+1 = Re,Im x_r),
 * stored column-major so that column k is one SVL vector (the FMOPA left operand).
 *
 * ZA tile use: 4 fp32 tiles, each = one butterfly for 16 FFTs. Tiles share M_t when
 * a stage has >= 4 butterflies per twiddle index (all stages but the last);
 * in the last stage (one butterfly per t) each tile loads its own M_t.
 */
#include "sme_fft.h"

#include <arm_sme.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define VL 16 /* fp32 lanes at SVL = 512 */

struct sme_fft_plan {
    int n, stages;
    float *mats;      /* Stockham: all stages' matrices, 256 floats each, column-major */
    int mat_off[8];   /* first matrix of stage s (units of matrices) */
    float *cg_mats;   /* constant geometry: stage s has N/8^(s+1) matrices */
    int cg_off[8];
    int *rev;         /* base-8 digit reversal: natural index k lives at position rev[k] */
    int wide;         /* sme_fft_columns: 64-lane groups (default) or 16-lane only */
};

/* Real 16x16 column-major matrix of y_q = sum_r F8[q][r] e^{-2 pi i (pre_r r + post_q q)} x_r */
static void build_mat(float *M, double pre, double post, double sign) {
    for (int q = 0; q < 8; q++)
        for (int r = 0; r < 8; r++) {
            double ang = sign * 2.0 * M_PI * ((double)(q * r) / 8.0 + pre * r + post * q);
            float a = (float)cos(ang), b = (float)sin(ang);
            M[(2 * r) * 16 + 2 * q] = a;
            M[(2 * r + 1) * 16 + 2 * q] = -b;
            M[(2 * r) * 16 + 2 * q + 1] = b;
            M[(2 * r + 1) * 16 + 2 * q + 1] = a;
        }
}

sme_fft_plan *sme_fft_plan_create(int n) { return sme_fft_plan_create_dir(n, -1); }

sme_fft_plan *sme_fft_plan_create_dir(int n, int dir) {
    const double sign = dir < 0 ? -1.0 : 1.0;
    int s = 0;
    for (int m = 1; m < n; m *= 8) s++;
    int check = 1;
    for (int i = 0; i < s; i++) check *= 8;
    if (check != n || s < 1 || s > 7) return NULL;

    sme_fft_plan *p = calloc(1, sizeof *p);
    p->n = n;
    p->stages = s;
    p->wide = 1;
    int total = 0;
    for (int st = 0, ns = 1; st < s; st++, ns *= 8) { p->mat_off[st] = total; total += ns; }
    p->mats = aligned_alloc(64, (size_t)total * 256 * sizeof(float));

    for (int st = 0, ns = 1; st < s; st++, ns *= 8)
        for (int t = 0; t < ns; t++)
            build_mat(p->mats + (size_t)(p->mat_off[st] + t) * 256, (double)t / (8.0 * ns), 0, sign);

    /* constant geometry DIF, stage st: L = N/8^st, butterfly J, n1 = J / 8^st in [0, L/8):
       out[8J + q] = w_L^{n1 q} sum_r F8[q][r] in[J + r N/8] */
    total = 0;
    for (int st = 0, L = n; st < s; st++, L /= 8) { p->cg_off[st] = total; total += L / 8; }
    p->cg_mats = aligned_alloc(64, (size_t)total * 256 * sizeof(float));
    for (int st = 0, L = n; st < s; st++, L /= 8)
        for (int n1 = 0; n1 < L / 8; n1++)
            build_mat(p->cg_mats + (size_t)(p->cg_off[st] + n1) * 256, 0, (double)n1 / L, sign);
    p->rev = malloc((size_t)n * sizeof(int));
    for (int k = 0; k < n; k++) {
        int r = 0, v = k;
        for (int i = 0; i < s; i++) { r = r * 8 + v % 8; v /= 8; }
        p->rev[k] = r;
    }
    return p;
}

void sme_fft_plan_destroy(sme_fft_plan *p) {
    if (!p) return;
    free(p->mats);
    free(p->cg_mats);
    free(p->rev);
    free(p);
}

size_t sme_fft_block_bytes(const sme_fft_plan *p) { return (size_t)p->n * 32 * sizeof(float); }
size_t sme_fft_columns_scratch_bytes(const sme_fft_plan *p) { return 8 * sme_fft_block_bytes(p); }
void sme_fft_set_wide(sme_fft_plan *p, int wide) { p->wide = wide; }

/* ---- stage kernels ---------------------------------------------------------------- */

#define MOPA(tile, zn, zm) svmopa_za32_f32_m(tile, pg, pg, zn, zm)

/* Store tile rows to butterfly outputs: row i -> out[(d + (i>>1) ns) * 32 + (i&1) * 16]. */
#define STORE_TILE(tile, d)                                                          \
    do {                                                                             \
        float *o_ = out + (size_t)(d) * 32;                                          \
        for (int i_ = 0; i_ < 16; i_++)                                              \
            svst1_hor_za32(tile, i_, pg, o_ + (size_t)(i_ >> 1) * ns * 32 + (i_ & 1) * 16); \
    } while (0)

/* Stages with m = N/(8 ns) >= 4 butterflies per twiddle index: 4 tiles share M_t. */
static void stage_shared(const float *in, float *out, int n, int ns, const float *mats)
    __arm_streaming __arm_inout("za") {
    const svbool_t pg = svptrue_b32();
    const int m = n / (8 * ns);
    const size_t istr = (size_t)(n / 8) * 32; /* floats between inputs r and r+1 */
    for (int t = 0; t < ns; t++) {
        const float *M = mats + (size_t)t * 256;
        for (int u = 0; u < m; u += 4) {
            const int j0 = t + u * ns;
            const float *x0 = in + (size_t)j0 * 32;
            const size_t js = (size_t)ns * 32; /* floats between butterflies u and u+1 */
            svzero_za();
            for (int k = 0; k < 16; k++) {
                svfloat32_t zm = svld1_f32(pg, M + k * 16);
                const float *xk = x0 + (k >> 1) * istr + (k & 1) * 16;
                MOPA(0, zm, svld1_f32(pg, xk));
                MOPA(1, zm, svld1_f32(pg, xk + js));
                MOPA(2, zm, svld1_f32(pg, xk + 2 * js));
                MOPA(3, zm, svld1_f32(pg, xk + 3 * js));
            }
            /* j = t + (u+v) ns -> d = (u+v) 8 ns + t */
            const int d0 = u * 8 * ns + t;
            STORE_TILE(0, d0);
            STORE_TILE(1, d0 + 8 * ns);
            STORE_TILE(2, d0 + 16 * ns);
            STORE_TILE(3, d0 + 24 * ns);
        }
    }
}

/* Last stage (ns = N/8, one butterfly per t): 4 tiles = 4 consecutive t, own matrices. */
static void stage_last(const float *in, float *out, int n, int ns, const float *mats)
    __arm_streaming __arm_inout("za") {
    const svbool_t pg = svptrue_b32();
    const size_t istr = (size_t)(n / 8) * 32;
    for (int t = 0; t < ns; t += 4) {
        const float *M = mats + (size_t)t * 256;
        const float *x0 = in + (size_t)t * 32;
        svzero_za();
        for (int k = 0; k < 16; k++) {
            const float *xk = x0 + (k >> 1) * istr + (k & 1) * 16;
            const float *mk = M + k * 16;
            MOPA(0, svld1_f32(pg, mk), svld1_f32(pg, xk));
            MOPA(1, svld1_f32(pg, mk + 256), svld1_f32(pg, xk + 32));
            MOPA(2, svld1_f32(pg, mk + 512), svld1_f32(pg, xk + 64));
            MOPA(3, svld1_f32(pg, mk + 768), svld1_f32(pg, xk + 96));
        }
        STORE_TILE(0, t);
        STORE_TILE(1, t + 1);
        STORE_TILE(2, t + 2);
        STORE_TILE(3, t + 3);
    }
}

__arm_locally_streaming __arm_new("za")
void sme_fft_execute_stockham(const sme_fft_plan *p, float *blocks, int nblk, float *scratch) {
    const int n = p->n;
    const size_t bf = (size_t)n * 32;
    for (int b = 0; b < nblk; b++) {
        float *a = blocks + b * bf, *c = scratch;
        for (int st = 0, ns = 1; st < p->stages; st++, ns *= 8) {
            const float *mats = p->mats + (size_t)p->mat_off[st] * 256;
            if (8 * ns < n) stage_shared(a, c, n, ns, mats);
            else stage_last(a, c, n, ns, mats);
            float *tmp = a; a = c; c = tmp;
        }
        if (p->stages & 1) memcpy(blocks + b * bf, scratch, bf * sizeof(float));
    }
}

/* Constant-geometry stage: 4 tiles = butterflies J..J+3, inputs in[J + r N/8], outputs
   out[8J .. 8J+31] (4 KiB contiguous). Matrix n1 = J >> (3 st); tiles share it when 8^st >= 4. */
static void stage_cg(const float *in, float *out, int n, int st, const float *mats)
    __arm_streaming __arm_inout("za") {
    const svbool_t pg = svptrue_b32();
    const size_t istr = (size_t)(n / 8) * 32;
    const int sh = 3 * st;
    for (int J = 0; J < n / 8; J += 4) {
        const float *x0 = in + (size_t)J * 32;
        svzero_za();
        if (st > 0) {
            const float *M = mats + (size_t)(J >> sh) * 256;
            for (int k = 0; k < 16; k++) {
                svfloat32_t zm = svld1_f32(pg, M + k * 16);
                const float *xk = x0 + (k >> 1) * istr + (k & 1) * 16;
                MOPA(0, zm, svld1_f32(pg, xk));
                MOPA(1, zm, svld1_f32(pg, xk + 32));
                MOPA(2, zm, svld1_f32(pg, xk + 64));
                MOPA(3, zm, svld1_f32(pg, xk + 96));
            }
        } else {
            const float *M = mats + (size_t)J * 256;
            for (int k = 0; k < 16; k++) {
                const float *xk = x0 + (k >> 1) * istr + (k & 1) * 16;
                const float *mk = M + k * 16;
                MOPA(0, svld1_f32(pg, mk), svld1_f32(pg, xk));
                MOPA(1, svld1_f32(pg, mk + 256), svld1_f32(pg, xk + 32));
                MOPA(2, svld1_f32(pg, mk + 512), svld1_f32(pg, xk + 64));
                MOPA(3, svld1_f32(pg, mk + 768), svld1_f32(pg, xk + 96));
            }
        }
        float *o = out + (size_t)J * 8 * 32;
        for (int i = 0; i < 16; i++) svst1_hor_za32(0, i, pg, o + i * 16);
        for (int i = 0; i < 16; i++) svst1_hor_za32(1, i, pg, o + 256 + i * 16);
        for (int i = 0; i < 16; i++) svst1_hor_za32(2, i, pg, o + 512 + i * 16);
        for (int i = 0; i < 16; i++) svst1_hor_za32(3, i, pg, o + 768 + i * 16);
    }
}

/* Same stage with general split-complex addressing: point p of the 16 lanes is at
   ire/iim + p * is (floats); output position P goes to row rev ? rev[P] : P of ore/oim. */
static void stage_cg_io(const float *ire, const float *iim, size_t is, float *ore, float *oim, size_t os,
                        const int *rev, int n, int st, const float *mats) __arm_streaming __arm_inout("za") {
    const svbool_t pg = svptrue_b32();
    const size_t istr = (size_t)(n / 8) * is;
    const int sh = 3 * st;
    for (int J = 0; J < n / 8; J += 4) {
        svzero_za();
        const float *M = mats + (size_t)(J >> sh) * 256;
        const size_t mstep = st > 0 ? 0 : 256; /* stage 0: one matrix per butterfly */
        for (int k = 0; k < 16; k++) {
            const float *xk = ((k & 1) ? iim : ire) + (size_t)J * is + (k >> 1) * istr;
            const float *mk = M + k * 16;
            MOPA(0, svld1_f32(pg, mk), svld1_f32(pg, xk));
            MOPA(1, svld1_f32(pg, mk + mstep), svld1_f32(pg, xk + is));
            MOPA(2, svld1_f32(pg, mk + 2 * mstep), svld1_f32(pg, xk + 2 * is));
            MOPA(3, svld1_f32(pg, mk + 3 * mstep), svld1_f32(pg, xk + 3 * is));
        }
#define STORE_IO(tile)                                                                   \
        for (int i = 0; i < 16; i++) {                                                   \
            const int P = 8 * (J + tile) + (i >> 1);                                     \
            svst1_hor_za32(tile, i, pg, ((i & 1) ? oim : ore) + (size_t)(rev ? rev[P] : P) * os); \
        }
        STORE_IO(0) STORE_IO(1) STORE_IO(2) STORE_IO(3)
#undef STORE_IO
    }
}

/* 64-lane variant: the 4 tiles are ONE butterfly J for lanes 16v..16v+15 (v = tile), so all
   tiles share M and each row access is 256 B contiguous. Point p of lane v is at
   ire/iim + p * is + 16 v. Internal 64-lane layout: re = buf, im = buf + 64, is = 128. */
static void stage_cg64(const float *ire, const float *iim, size_t is, float *ore, float *oim, size_t os,
                       const int *rev, int n, int st, const float *mats) __arm_streaming __arm_inout("za") {
    const svbool_t pg = svptrue_b32();
    const size_t istr = (size_t)(n / 8) * is;
    const int sh = 3 * st;
    for (int J = 0; J < n / 8; J++) {
        svzero_za();
        const float *M = mats + (size_t)(J >> sh) * 256;
        for (int k = 0; k < 16; k++) {
            const float *xk = ((k & 1) ? iim : ire) + (size_t)J * is + (k >> 1) * istr;
            svfloat32_t zm = svld1_f32(pg, M + k * 16);
            MOPA(0, zm, svld1_f32(pg, xk));
            MOPA(1, zm, svld1_f32(pg, xk + 16));
            MOPA(2, zm, svld1_f32(pg, xk + 32));
            MOPA(3, zm, svld1_f32(pg, xk + 48));
        }
        for (int i = 0; i < 16; i++) {
            const int P = 8 * J + (i >> 1);
            float *o = ((i & 1) ? oim : ore) + (size_t)(rev ? rev[P] : P) * os;
            svst1_hor_za32(0, i, pg, o);
            svst1_hor_za32(1, i, pg, o + 16);
            svst1_hor_za32(2, i, pg, o + 32);
            svst1_hor_za32(3, i, pg, o + 48);
        }
    }
}

__arm_locally_streaming __arm_new("za")
void sme_fft_columns(const sme_fft_plan *p, const float *in_re, const float *in_im, float *out_re,
                     float *out_im, size_t row_stride, int ncols, float *scratch) {
    const int n = p->n, S = p->stages;
    int g = 0;
    if (p->wide) { /* 64 columns at a time; two 64-lane blocks of scratch */
        float *A = scratch, *B = scratch + (size_t)n * 128;
        for (; g + 64 <= ncols; g += 64) {
            const float *ir = in_re + g, *ii = in_im + g;
            float *orr = out_re + g, *oi = out_im + g;
            if (S == 1) { stage_cg64(ir, ii, row_stride, orr, oi, row_stride, NULL, n, 0, p->cg_mats); continue; }
            stage_cg64(ir, ii, row_stride, A, A + 64, 128, NULL, n, 0, p->cg_mats);
            for (int st = 1; st < S - 1; st++) {
                stage_cg64(A, A + 64, 128, B, B + 64, 128, NULL, n, st, p->cg_mats + (size_t)p->cg_off[st] * 256);
                float *t = A; A = B; B = t;
            }
            stage_cg64(A, A + 64, 128, orr, oi, row_stride, p->rev, n, S - 1,
                       p->cg_mats + (size_t)p->cg_off[S - 1] * 256);
        }
    }
    float *A = scratch, *B = scratch + (size_t)n * 32;
    for (; g < ncols; g += 16) {
        const float *ir = in_re + g, *ii = in_im + g;
        float *orr = out_re + g, *oi = out_im + g;
        const float *mats0 = p->cg_mats;
        if (S == 1) { stage_cg_io(ir, ii, row_stride, orr, oi, row_stride, NULL, n, 0, mats0); continue; }
        stage_cg_io(ir, ii, row_stride, A, A + 16, 32, NULL, n, 0, mats0);
        for (int st = 1; st < S - 1; st++) {
            stage_cg(A, B, n, st, p->cg_mats + (size_t)p->cg_off[st] * 256);
            float *t = A; A = B; B = t;
        }
        stage_cg_io(A, A + 16, 32, orr, oi, row_stride, p->rev, n, S - 1,
                    p->cg_mats + (size_t)p->cg_off[S - 1] * 256);
    }
}

__arm_locally_streaming __arm_new("za")
void sme_fft_execute(const sme_fft_plan *p, float *blocks, int nblk, float *scratch) {
    const int n = p->n;
    const size_t bf = (size_t)n * 32;
    for (int b = 0; b < nblk; b++) {
        float *a = blocks + b * bf, *c = scratch;
        for (int st = 0; st < p->stages; st++) {
            stage_cg(a, c, n, st, p->cg_mats + (size_t)p->cg_off[st] * 256);
            float *tmp = a; a = c; c = tmp;
        }
        if (p->stages & 1) memcpy(blocks + b * bf, scratch, bf * sizeof(float));
    }
}

/* ---- layout conversion: 16x16 transposes through ZA tiles 0 (re) and 1 (im) --------- */

__arm_locally_streaming __arm_new("za")
void sme_fft_pack(const sme_fft_plan *p, float *blk, float *const re[16], float *const im[16]) {
    const svbool_t pg = svptrue_b32();
    for (int n0 = 0; n0 < p->n; n0 += VL) {
        for (int b = 0; b < 16; b++) {
            svld1_hor_za32(0, b, pg, re[b] + n0);
            svld1_hor_za32(1, b, pg, im[b] + n0);
        }
        for (int i = 0; i < 16; i++) {
            svst1_ver_za32(0, i, pg, blk + (size_t)(n0 + i) * 32);
            svst1_ver_za32(1, i, pg, blk + (size_t)(n0 + i) * 32 + 16);
        }
    }
}

__arm_locally_streaming __arm_new("za")
void sme_fft_unpack(const sme_fft_plan *p, const float *blk, float *re[16], float *im[16]) {
    const svbool_t pg = svptrue_b32();
    for (int n0 = 0; n0 < p->n; n0 += VL) {
        for (int i = 0; i < 16; i++) {
            const float *row = blk + (size_t)p->rev[n0 + i] * 32;
            svld1_ver_za32(0, i, pg, row);
            svld1_ver_za32(1, i, pg, row + 16);
        }
        for (int b = 0; b < 16; b++) {
            svst1_hor_za32(0, b, pg, re[b] + n0);
            svst1_hor_za32(1, b, pg, im[b] + n0);
        }
    }
}
