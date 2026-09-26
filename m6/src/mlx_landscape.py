"""paper_m6 Stage 2: MLX mx.fft.fft landscape on the M6 GPU.

Same grid as `FFTMicroBenchmarks m6-landscape`: N = 2^5..2^20, batch chosen so the
in+out footprint (2*B*N*8 bytes) targets L2 (1 MiB), SLC (8 MiB) or DRAM (256 MiB).

Timing is wall clock (MLX exposes no GPU timestamps), so small sizes include
Python + launch overhead. That is MLX's real user-facing cost, and the paper labels it that way.
  hot : K back-to-back fft+eval calls on the same input (>= 64 MiB of traffic), per-call time
  cold: a 512 MiB write (evaluated) before each single timed fft+eval
Validation: rel L2 against numpy.fft.fft on the first min(B,4) rows.

Usage: <venv>/bin/python mlx_landscape.py [out.csv]
"""
import sys
import time

import mlx.core as mx
import numpy as np

LEVELS = [("L2", 1 << 20), ("SLC", 8 << 20), ("DRAM", 256 << 20)]
out_path = sys.argv[1] if len(sys.argv) > 1 else "stage2_mlx.csv"

flush_src = mx.zeros((512 << 20) // 4, dtype=mx.float32)
mx.eval(flush_src)


def flush():
    f = flush_src + 1.0
    mx.eval(f)
    del f


def q(x, p):
    s = sorted(x)
    return s[int(p * (len(s) - 1) + 0.5)] if p != 0.5 else s[len(s) // 2]


rows = ["backend,variant,N,level,batch,footprint_MiB,mode,gflops_med,gflops_q1,gflops_q3,us_per_fft,gbps,relL2"]
print(f"MLX {mx.__version__}  device {mx.default_device()}")
rng = np.random.default_rng(0)
for e in range(5, 21):
    n = 1 << e
    flops = 5.0 * n * e
    for lvl, target in LEVELS:
        b = max(1, target // (2 * n * 8))
        fp = 2 * b * n * 8
        xn = (rng.uniform(-1, 1, (b, n)) + 1j * rng.uniform(-1, 1, (b, n))).astype(np.complex64)
        x = mx.array(xn)
        mx.eval(x)
        y = mx.fft.fft(x)
        mx.eval(y)
        vb = min(b, 4)
        ref = np.fft.fft(xn[:vb].astype(np.complex128), axis=1)
        err = float(np.linalg.norm(np.array(y[:vb]) - ref) / np.linalg.norm(ref))

        K = max(1, (64 << 20) // fp)

        def hot():
            t0 = time.perf_counter()
            for _ in range(K):
                y = mx.fft.fft(x)
                mx.eval(y)
            return (time.perf_counter() - t0) / K

        def cold():
            flush()
            t0 = time.perf_counter()
            y = mx.fft.fft(x)
            mx.eval(y)
            return time.perf_counter() - t0

        for _ in range(2):
            hot(); cold()
        h, c = [], []
        for _ in range(11):
            h.append(hot()); c.append(cold())
        for mode, t in (("hot", h), ("cold", c)):
            g = [flops * b / ti / 1e9 for ti in t]
            tm = q(t, 0.5)
            rows.append(f"mlx,mx.fft.fft,{n},{lvl},{b},{fp/1048576:.3f},{mode},{q(g,0.5):.2f},{q(g,0.25):.2f},"
                        f"{q(g,0.75):.2f},{tm*1e6/b:.4f},{fp/tm/1e9:.2f},{err:.2e}")
        print(f"mlx N={n:8d} {lvl:4s} B={b:6d} {fp/1048576:8.2f} MiB | hot {flops*b/q(h,0.5)/1e9:8.1f} GF"
              f" | cold {flops*b/q(c,0.5)/1e9:8.1f} GF | relL2 {err:.1e}", flush=True)
        del x, y

open(out_path, "w").write("\n".join(rows) + "\n")
print("CSV ->", out_path)
