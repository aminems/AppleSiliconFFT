/*
 * sme_stage5.c — paper_m6 Stage 5: SME2 on the M6 CPU.
 *
 *   ./sme_stage5 peak   FMOPA fp32 peak, 1..12 threads (UI QoS)
 *   ./sme_stage5 bw     streaming-mode load / ZA-store bandwidth vs working set
 *   ./sme_stage5 fft    radix-8 SME FFT vs vDSP_fft_zop: correctness, hot + cold GFLOPS
 *
 * Build (from repo root):
 *   clang -O3 -mcpu=apple-m4 -Isrc/cpu benchmarks/micro/sme/sme_stage5.c src/cpu/sme_fft.c \
 *         -framework Accelerate -o benchmarks/micro/sme/sme_stage5
 */
#include <Accelerate/Accelerate.h>
#include <arm_sme.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sme_fft.h"

static double now(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW) * 1e-9; }
static int cmpd(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
static double median(double *v, int n) { qsort(v, n, sizeof *v, cmpd); return v[n / 2]; }
static void *xalloc(size_t bytes) {
    void *p = aligned_alloc(16384, (bytes + 16383) & ~(size_t)16383);
    memset(p, 0, bytes);
    return p;
}

/* ---- peak ---------------------------------------------------------------------------- */

__arm_locally_streaming __arm_new("za")
static void fmopa_loop(long iters) {
    const svbool_t pg = svptrue_b32();
    svfloat32_t a = svdup_f32(1.0f), b = svdup_f32(0.5f), c = svdup_f32(0.25f), d = svdup_f32(2.0f);
    svzero_za();
    for (long i = 0; i < iters; i++) {
        svmopa_za32_f32_m(0, pg, pg, a, b); svmopa_za32_f32_m(1, pg, pg, a, c);
        svmopa_za32_f32_m(2, pg, pg, a, d); svmopa_za32_f32_m(3, pg, pg, b, c);
        svmopa_za32_f32_m(0, pg, pg, b, d); svmopa_za32_f32_m(1, pg, pg, c, d);
        svmopa_za32_f32_m(2, pg, pg, c, a); svmopa_za32_f32_m(3, pg, pg, d, a);
    }
    float sink[16];
    svst1_hor_za32(0, 0, pg, sink);
    __asm__ volatile("" ::"r"(sink) : "memory");
}

typedef struct {
    long iters;
    double secs;
    atomic_int *bar;
    int nt;
} peak_arg;

static void *peak_thread(void *vp) {
    peak_arg *a = vp;
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    fmopa_loop(40000000); /* ~0.25 s: let the clock ramp before timing */
    atomic_fetch_add(a->bar, 1);
    while (atomic_load(a->bar) < a->nt) { }
    double t0 = now();
    fmopa_loop(a->iters);
    a->secs = now() - t0;
    return NULL;
}

static void mode_peak(void) {
    printf("peak: fp32 FMOPA 16x16 (512 flop) into 4 tiles, UI QoS, median of 5\n");
    int threads[] = {1, 2, 4, 6, 8, 12};
    for (int ti = 0; ti < 6; ti++) {
        int nt = threads[ti];
        const long iters = 20000000 / nt; /* x8 fmopa */
        double tot[5];
        for (int rep = 0; rep < 5; rep++) {
            pthread_t th[12];
            peak_arg args[12];
            atomic_int bar = 0;
            for (int i = 0; i < nt; i++) {
                args[i] = (peak_arg){iters, 0, &bar, nt};
                pthread_create(&th[i], NULL, peak_thread, &args[i]);
            }
            double gf = 0;
            for (int i = 0; i < nt; i++) {
                pthread_join(th[i], NULL);
                gf += iters * 8.0 * 512 / args[i].secs * 1e-9;
            }
            tot[rep] = gf;
        }
        printf("  T=%2d  %7.0f GFLOPS total\n", nt, median(tot, 5));
    }
}

/* ---- bandwidth ------------------------------------------------------------------------- */

__arm_locally_streaming
static float ld_sweep(const float *buf, size_t nf, int reps) {
    const svbool_t pg = svptrue_b32();
    svfloat32_t a0 = svdup_f32(0), a1 = a0, a2 = a0, a3 = a0;
    for (int r = 0; r < reps; r++)
        for (size_t i = 0; i < nf; i += 64) {
            a0 = svadd_f32_x(pg, a0, svld1_f32(pg, buf + i));
            a1 = svadd_f32_x(pg, a1, svld1_f32(pg, buf + i + 16));
            a2 = svadd_f32_x(pg, a2, svld1_f32(pg, buf + i + 32));
            a3 = svadd_f32_x(pg, a3, svld1_f32(pg, buf + i + 48));
        }
    return svaddv_f32(pg, svadd_f32_x(pg, svadd_f32_x(pg, a0, a1), svadd_f32_x(pg, a2, a3)));
}

__arm_locally_streaming __arm_new("za")
static void st_sweep(float *buf, size_t nf, int reps) {
    const svbool_t pg = svptrue_b32();
    svzero_za();
    for (int r = 0; r < reps; r++)
        for (size_t i = 0; i < nf; i += 256)
            for (int s = 0; s < 16; s++) svst1_hor_za32(0, s, pg, buf + i + s * 16);
}

/* ld + fmopa + ZA store, the FFT's access pattern at 1 load : 1 fmopa : 1 store */
__arm_locally_streaming __arm_new("za")
static void ldst_sweep(const float *src, float *dst, size_t nf, int reps) {
    const svbool_t pg = svptrue_b32();
    svfloat32_t w = svdup_f32(1.0f);
    for (int r = 0; r < reps; r++)
        for (size_t i = 0; i < nf; i += 1024) {
            svzero_za();
            for (int k = 0; k < 16; k++) {
                const float *s = src + i + k * 16;
                svmopa_za32_f32_m(0, pg, pg, w, svld1_f32(pg, s));
                svmopa_za32_f32_m(1, pg, pg, w, svld1_f32(pg, s + 256));
                svmopa_za32_f32_m(2, pg, pg, w, svld1_f32(pg, s + 512));
                svmopa_za32_f32_m(3, pg, pg, w, svld1_f32(pg, s + 768));
            }
            for (int s = 0; s < 16; s++) {
                svst1_hor_za32(0, s, pg, dst + i + s * 16);
                svst1_hor_za32(1, s, pg, dst + i + 256 + s * 16);
                svst1_hor_za32(2, s, pg, dst + i + 512 + s * 16);
                svst1_hor_za32(3, s, pg, dst + i + 768 + s * 16);
            }
        }
}

static void mode_bw(void) {
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    printf("bw: streaming-mode SVL=512 access, 1 thread UI, median of 7 (GB/s; ldst counts ld+st bytes)\n");
    printf("  %10s %9s %9s %9s\n", "bytes", "load", "zastore", "ld+mopa+st");
    for (size_t bytes = 64 << 10; bytes <= (size_t)256 << 20; bytes *= 2) {
        size_t nf = bytes / 4;
        float *a = xalloc(bytes), *b = xalloc(bytes);
        for (size_t i = 0; i < nf; i++) a[i] = 1e-3f * (i & 1023);
        int reps = (int)fmax(1, (256 << 20) / bytes);
        double tl[7], ts[7], tls[7];
        volatile float sink = 0;
        ld_sweep(a, nf, 1); st_sweep(b, nf, 1); ldst_sweep(a, b, nf, 1);
        for (int k = 0; k < 7; k++) {
            double t0 = now(); sink += ld_sweep(a, nf, reps); tl[k] = now() - t0;
            t0 = now(); st_sweep(b, nf, reps); ts[k] = now() - t0;
            t0 = now(); ldst_sweep(a, b, nf, reps); tls[k] = now() - t0;
        }
        double gb = (double)bytes * reps * 1e-9;
        printf("  %10zu %9.1f %9.1f %9.1f\n", bytes, gb / median(tl, 7), gb / median(ts, 7),
               2 * gb / median(tls, 7));
        free(a); free(b);
    }
}

/* ---- FFT --------------------------------------------------------------------------------- */

static void fill(float *re, float *im, size_t n, unsigned seed) {
    srand(seed);
    for (size_t i = 0; i < n; i++) {
        re[i] = (float)rand() / RAND_MAX - 0.5f;
        im[i] = (float)rand() / RAND_MAX - 0.5f;
    }
}

/* Batch of B FFTs of size N in split-complex layout (vDSP) and blocked layout (SME). */
static void fft_case(int n, int batch, int trials) {
    int log2n = 0;
    while ((1 << log2n) < n) log2n++;
    const int nblk = batch / 16;
    sme_fft_plan *p = sme_fft_plan_create(n);
    FFTSetup vs = vDSP_create_fftsetup(log2n, kFFTRadix2);
    size_t nb = (size_t)n * batch;
    float *re = xalloc(nb * 4), *im = xalloc(nb * 4), *ore = xalloc(nb * 4), *oim = xalloc(nb * 4);
    float *blk = xalloc(nb * 8), *scratch = xalloc(sme_fft_block_bytes(p));
    fill(re, im, nb, 1234 + n);

    /* pack -> fft -> unpack, compare to vDSP */
    for (int b = 0; b < nblk; b++) {
        float *r[16], *i[16];
        for (int k = 0; k < 16; k++) { r[k] = re + (size_t)(b * 16 + k) * n; i[k] = im + (size_t)(b * 16 + k) * n; }
        sme_fft_pack(p, blk + (size_t)b * n * 32, r, i);
    }
    sme_fft_execute(p, blk, nblk, scratch);
    float *sre = xalloc(nb * 4), *sim = xalloc(nb * 4);
    for (int b = 0; b < nblk; b++) {
        float *r[16], *i[16];
        for (int k = 0; k < 16; k++) { r[k] = sre + (size_t)(b * 16 + k) * n; i[k] = sim + (size_t)(b * 16 + k) * n; }
        sme_fft_unpack(p, blk + (size_t)b * n * 32, r, i);
    }
    for (int f = 0; f < batch; f++) {
        DSPSplitComplex si = {re + (size_t)f * n, im + (size_t)f * n}, so = {ore + (size_t)f * n, oim + (size_t)f * n};
        vDSP_fft_zop(vs, &si, 1, &so, 1, log2n, kFFTDirection_Forward);
    }
    double num = 0, den = 0;
    for (size_t i = 0; i < nb; i++) {
        double dr = sre[i] - ore[i], di = sim[i] - oim[i];
        num += dr * dr + di * di;
        den += (double)ore[i] * ore[i] + (double)oim[i] * oim[i];
    }
    double relerr = sqrt(num / den);

    /* timing: each trial runs the whole batch `reps` times (hot = same buffers stay cached) */
    double flops = 5.0 * n * log2n * batch;
    const int reps = (int)fmax(1, 4e6 / nb);
    double t_sme[64], t_sk[64], t_pk[64], t_vd[64];
    for (int tr = -2; tr < trials; tr++) {
        double t0 = now();
        for (int r = 0; r < reps; r++) sme_fft_execute(p, blk, nblk, scratch);
        double t1 = now();
        for (int r = 0; r < reps; r++) sme_fft_execute_stockham(p, blk, nblk, scratch);
        double t2 = now();
        for (int r = 0; r < reps; r++)
            for (int b = 0; b < nblk; b++) {
                float *rr[16], *ii[16];
                for (int k = 0; k < 16; k++) { rr[k] = re + (size_t)(b * 16 + k) * n; ii[k] = im + (size_t)(b * 16 + k) * n; }
                sme_fft_pack(p, blk + (size_t)b * n * 32, rr, ii);
                sme_fft_execute(p, blk + (size_t)b * n * 32, 1, scratch);
                for (int k = 0; k < 16; k++) { rr[k] = sre + (size_t)(b * 16 + k) * n; ii[k] = sim + (size_t)(b * 16 + k) * n; }
                sme_fft_unpack(p, blk + (size_t)b * n * 32, rr, ii);
            }
        double t3 = now();
        for (int r = 0; r < reps; r++)
            for (int f = 0; f < batch; f++) {
                DSPSplitComplex si = {re + (size_t)f * n, im + (size_t)f * n}, so = {ore + (size_t)f * n, oim + (size_t)f * n};
                vDSP_fft_zop(vs, &si, 1, &so, 1, log2n, kFFTDirection_Forward);
            }
        double t4 = now();
        if (tr >= 0) {
            t_sme[tr] = (t1 - t0) / reps; t_sk[tr] = (t2 - t1) / reps;
            t_pk[tr] = (t3 - t2) / reps; t_vd[tr] = (t4 - t3) / reps;
        }
    }
    double ms = median(t_sme, trials), mk = median(t_sk, trials), mp = median(t_pk, trials), mv = median(t_vd, trials);
    double mib = nb * 8.0 / (1 << 20);
    printf("  N=%6d B=%6d (%6.1f MiB) err %.1e | SME-CG %6.1f | SME-Stockham %6.1f | SME-CG+pack %6.1f | vDSP %6.1f GF"
           " | CG/vDSP x%.2f, +pack x%.2f\n",
           n, batch, mib, relerr, flops / ms * 1e-9, flops / mk * 1e-9, flops / mp * 1e-9, flops / mv * 1e-9, mv / ms,
           mv / mp);
    fflush(stdout);
    vDSP_destroy_fftsetup(vs);
    sme_fft_plan_destroy(p);
    free(re); free(im); free(ore); free(oim); free(blk); free(scratch); free(sre); free(sim);
}

static void mode_fft(void) {
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    printf("fft: 1 thread UI; GFLOPS = 5 N log2 N / t; median of trials; hot = batch fits L2\n");
    int ns[] = {64, 512, 4096, 32768};
    for (int i = 0; i < 4; i++) {
        int n = ns[i];
        int hot = 16, cold = (int)(((size_t)256 << 20) / (8 * (size_t)n)); /* 256 MiB */
        fft_case(n, hot, 31);
        fft_case(n, cold, 7);
    }
}

/* ---- multi-threaded: T threads split the batch; SME (blocked layout) vs vDSP ------------ */

typedef struct {
    int which, n, log2n, nblk; /* which: 0 = SME, 1 = vDSP */
    const sme_fft_plan *p;
    FFTSetup vs;
    float *blk, *scratch, *re, *im, *ore, *oim;
    atomic_int *go;
    int nt, rounds;
    double secs;
} mt_arg;

static void *mt_thread(void *vp) {
    mt_arg *a = vp;
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    const size_t nb = (size_t)a->n * 16 * a->nblk;
    atomic_fetch_add(a->go, 1);
    while (atomic_load(a->go) < a->nt) { }
    double t0 = now();
    for (int r = 0; r < a->rounds; r++) {
        if (a->which == 0) sme_fft_execute(a->p, a->blk, a->nblk, a->scratch);
        else
            for (size_t f = 0; f < nb / a->n; f++) {
                DSPSplitComplex si = {a->re + f * a->n, a->im + f * a->n}, so = {a->ore + f * a->n, a->oim + f * a->n};
                vDSP_fft_zop(a->vs, &si, 1, &so, 1, a->log2n, kFFTDirection_Forward);
            }
    }
    a->secs = now() - t0;
    return NULL;
}

/* batch per thread = bpt FFTs; returns total GFLOPS (median of trials, A/B interleaved) */
static void fft_mt(int n, int bpt, int nt, int trials) {
    int log2n = 0;
    while ((1 << log2n) < n) log2n++;
    sme_fft_plan *p = sme_fft_plan_create(n);
    FFTSetup vs = vDSP_create_fftsetup(log2n, kFFTRadix2);
    mt_arg a[2][12];
    atomic_int go;
    const size_t nb = (size_t)n * bpt;
    const double work = (double)n * bpt * nt;
    int rounds = (int)fmax(1, 2e7 / work); /* ~20 M points per trial */
    for (int i = 0; i < nt; i++) {
        float *blk = xalloc(nb * 8), *re = xalloc(nb * 4), *im = xalloc(nb * 4);
        fill(re, im, nb, i);
        memcpy(blk, re, nb * 4); memcpy(blk + nb, im, nb * 4); /* values only matter for denormals */
        a[0][i] = (mt_arg){0, n, log2n, bpt / 16, p, vs, blk, xalloc(sme_fft_block_bytes(p)), 0, 0, 0, 0, &go, nt, rounds, 0};
        a[1][i] = (mt_arg){1, n, log2n, bpt / 16, p, vs, 0, 0, re, im, xalloc(nb * 4), xalloc(nb * 4), &go, nt, rounds, 0};
    }
    double g[2][64];
    for (int tr = -1; tr < trials; tr++)
        for (int w = 0; w < 2; w++) {
            pthread_t th[12];
            atomic_store(&go, 0);
            for (int i = 0; i < nt; i++) pthread_create(&th[i], NULL, mt_thread, &a[w][i]);
            double mx = 0;
            for (int i = 0; i < nt; i++) { pthread_join(th[i], NULL); mx = fmax(mx, a[w][i].secs); }
            if (tr >= 0) g[w][tr] = 5.0 * n * log2n * bpt * nt * rounds / mx * 1e-9;
        }
    double gs = median(g[0], trials), gv = median(g[1], trials);
    printf("  N=%6d T=%2d B/thr=%5d (%7.1f MiB tot) | SME %7.1f GF | vDSP %7.1f GF | x%.2f\n", n, nt, bpt,
           nb * 8.0 * nt / (1 << 20), gs, gv, gs / gv);
    fflush(stdout);
    for (int i = 0; i < nt; i++) {
        free(a[0][i].blk); free(a[0][i].scratch);
        free(a[1][i].re); free(a[1][i].im); free(a[1][i].ore); free(a[1][i].oim);
    }
    vDSP_destroy_fftsetup(vs);
    sme_fft_plan_destroy(p);
}

static void mode_mt(int n) {
    printf("mt: T threads (UI QoS) each with its own batch; hot = 16 FFTs/thread, cold = 256 MiB total\n");
    int ts[] = {1, 2, 3, 4, 6};
    for (int i = 0; i < 5; i++) fft_mt(n, 16, ts[i], 9);
    for (int i = 0; i < 5; i++) fft_mt(n, (int)(((size_t)256 << 20) / (8 * (size_t)n) / ts[i]) & ~15, ts[i], 5);
}

/* ---- column FFTs of an N x W split-complex matrix ------------------------------------------ */

static void col_case(int n, int w, int trials) {
    int log2n = 0;
    while ((1 << log2n) < n) log2n++;
    sme_fft_plan *p = sme_fft_plan_create(n);
    FFTSetup vs = vDSP_create_fftsetup(log2n, kFFTRadix2);
    const size_t nw = (size_t)n * w;
    float *re = xalloc(nw * 4), *im = xalloc(nw * 4), *sr = xalloc(nw * 4), *si = xalloc(nw * 4);
    float *vr = xalloc(nw * 4), *vi = xalloc(nw * 4), *tr_ = xalloc(nw * 4), *ti = xalloc(nw * 4);
    float *scratch = xalloc(sme_fft_columns_scratch_bytes(p));
    fill(re, im, nw, 99 + n + w);
    DSPSplitComplex A = {re, im}, V = {vr, vi}, T = {tr_, ti};

    sme_fft_columns(p, re, im, sr, si, w, w, scratch);
    vDSP_fftm_zop(vs, &A, w, 1, &V, w, 1, log2n, w, kFFTDirection_Forward);
    double num = 0, den = 0;
    for (size_t i = 0; i < nw; i++) {
        double dr = sr[i] - vr[i], di = si[i] - vi[i];
        num += dr * dr + di * di;
        den += (double)vr[i] * vr[i] + (double)vi[i] * vi[i];
    }

    double flops = 5.0 * n * log2n * w;
    int reps = (int)fmax(1, 4e6 / nw);
    double t[3][64];
    for (int k = -2; k < trials; k++) {
        double t0 = now();
        for (int r = 0; r < reps; r++) sme_fft_columns(p, re, im, sr, si, w, w, scratch);
        double t1 = now();
        for (int r = 0; r < reps; r++) vDSP_fftm_zop(vs, &A, w, 1, &V, w, 1, log2n, w, kFFTDirection_Forward);
        double t2 = now();
        for (int r = 0; r < reps; r++) { /* transpose -> contiguous fftm -> transpose back */
            vDSP_mtrans(re, 1, tr_, 1, w, n); vDSP_mtrans(im, 1, ti, 1, w, n);
            vDSP_fftm_zip(vs, &T, 1, n, log2n, w, kFFTDirection_Forward);
            vDSP_mtrans(tr_, 1, vr, 1, n, w); vDSP_mtrans(ti, 1, vi, 1, n, w);
        }
        double t3 = now();
        if (k >= 0) { t[0][k] = (t1 - t0) / reps; t[1][k] = (t2 - t1) / reps; t[2][k] = (t3 - t2) / reps; }
    }
    double a = median(t[0], trials), b = median(t[1], trials), c = median(t[2], trials);
    printf("  N=%6d W=%5d (%7.1f MiB) err %.1e | SME cols %7.1f GF | vDSP fftm strided %7.1f GF | vDSP T+fftm+T %7.1f GF | x%.2f x%.2f\n",
           n, w, nw * 8.0 / (1 << 20), sqrt(num / den), flops / a * 1e-9, flops / b * 1e-9, flops / c * 1e-9,
           b / a, c / a);
    fflush(stdout);
    vDSP_destroy_fftsetup(vs);
    sme_fft_plan_destroy(p);
    free(re); free(im); free(sr); free(si); free(vr); free(vi); free(tr_); free(ti); free(scratch);
}

static void mode_col(void) {
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    printf("col: column FFTs of an N x W row-major split-complex matrix, out-of-place, 1 thread UI\n");
    int ns[] = {64, 512, 4096, 32768};
    for (int i = 0; i < 4; i++) {
        int n = ns[i];
        col_case(n, 16, 15);
        col_case(n, 256, 9);
        col_case(n, (int)(((size_t)256 << 20) / (8 * (size_t)n)), 5);
    }
}

/* ---- multi-threaded column FFTs: T threads split the W columns (multiples of 64) ------------ */

typedef struct {
    int which, n, log2n, w, c0, nc;
    const sme_fft_plan *p;
    FFTSetup vs;
    float *re, *im, *orr, *oi, *scratch;
    atomic_int *go;
    int nt, reps;
    double secs;
} cmt_arg;

static void *cmt_thread(void *vp) {
    cmt_arg *a = vp;
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    atomic_fetch_add(a->go, 1);
    while (atomic_load(a->go) < a->nt) { }
    double t0 = now();
    for (int r = 0; r < a->reps; r++) {
        if (a->which == 0)
            sme_fft_columns(a->p, a->re + a->c0, a->im + a->c0, a->orr + a->c0, a->oi + a->c0, a->w, a->nc, a->scratch);
        else {
            DSPSplitComplex A = {a->re + a->c0, a->im + a->c0}, V = {a->orr + a->c0, a->oi + a->c0};
            vDSP_fftm_zop(a->vs, &A, a->w, 1, &V, a->w, 1, a->log2n, a->nc, kFFTDirection_Forward);
        }
    }
    a->secs = now() - t0;
    return NULL;
}

static int g_colmt_wide = 1;
static void colmt_case(int n, int w, int nt, int trials) {
    int log2n = 0;
    while ((1 << log2n) < n) log2n++;
    sme_fft_plan *p = sme_fft_plan_create(n);
    sme_fft_set_wide(p, g_colmt_wide);
    FFTSetup vss[12]; /* one vDSP setup per thread */
    for (int i = 0; i < nt; i++) vss[i] = vDSP_create_fftsetup(log2n, kFFTRadix2);
    const size_t nw = (size_t)n * w;
    float *re = xalloc(nw * 4), *im = xalloc(nw * 4), *orr = xalloc(nw * 4), *oi = xalloc(nw * 4);
    fill(re, im, nw, 7);
    int reps = (int)fmax(1, 8e6 / nw);
    atomic_int go;
    cmt_arg a[2][12];
    int per = (w / nt) & ~63;
    for (int i = 0; i < nt; i++)
        for (int k = 0; k < 2; k++)
            a[k][i] = (cmt_arg){k, n, log2n, w, i * per, i == nt - 1 ? w - i * per : per, p, vss[i], re, im, orr, oi,
                                k == 0 ? xalloc(sme_fft_columns_scratch_bytes(p)) : NULL, &go, nt, reps, 0};
    double g[2][32];
    for (int tr = -1; tr < trials; tr++)
        for (int k = 0; k < 2; k++) {
            pthread_t th[12];
            atomic_store(&go, 0);
            for (int i = 0; i < nt; i++) pthread_create(&th[i], NULL, cmt_thread, &a[k][i]);
            double mx = 0;
            for (int i = 0; i < nt; i++) { pthread_join(th[i], NULL); mx = fmax(mx, a[k][i].secs); }
            if (tr >= 0) g[k][tr] = 5.0 * n * log2n * w * reps / mx * 1e-9;
        }
    double gs = median(g[0], trials), gv = median(g[1], trials);
    printf("  N=%6d W=%6d (%6.1f MiB) T=%2d | SME cols %7.1f GF | vDSP fftm strided %7.1f GF | x%.2f\n", n, w,
           nw * 8.0 / (1 << 20), nt, gs, gv, gs / gv);
    fflush(stdout);
    for (int i = 0; i < nt; i++) free(a[0][i].scratch);
    free(re); free(im); free(orr); free(oi);
    for (int i = 0; i < nt; i++) vDSP_destroy_fftsetup(vss[i]);
    sme_fft_plan_destroy(p);
}

static void mode_colmt(int wide) {
    g_colmt_wide = wide;
    printf("colmt: column FFTs (%s), T threads (UI QoS) split the columns; A/B interleaved, median of 7\n",
           wide ? "64-lane groups" : "16-lane groups");
    int ts[] = {1, 2, 3, 4, 6};
    int cfg[][2] = {{4096, 256}, {4096, 8192}, {512, 2048}, {512, 65536}};
    for (int c = 0; c < 4; c++)
        for (int i = 0; i < 5; i++)
            if (cfg[c][1] / ts[i] >= 64) colmt_case(cfg[c][0], cfg[c][1], ts[i], 7);
}

int main(int argc, char **argv) {
    const char *m = argc > 1 ? argv[1] : "fft";
    if (!strcmp(m, "peak")) mode_peak();
    else if (!strcmp(m, "bw")) mode_bw();
    else if (!strcmp(m, "fft")) mode_fft();
    else if (!strcmp(m, "col")) mode_col();
    else if (!strcmp(m, "colmt")) mode_colmt(!(argc > 2 && !strcmp(argv[2], "narrow")));
    else if (!strcmp(m, "mt")) mode_mt(argc > 2 ? atoi(argv[2]) : 4096);
    else { fprintf(stderr, "usage: %s peak|bw|fft|col|colmt|mt [N]\n", argv[0]); return 1; }
    return 0;
}
