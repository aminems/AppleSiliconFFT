# Bandwidth, Not FLOPS: FFT Kernels, Matrix Units and SAR Imaging on Apple M6

Code, raw measurements and paper for:

> M. A. Bergach, "Bandwidth, Not FLOPS: FFT Kernels, Matrix Units and SAR Imaging on Apple M6," preprint, 2026.

On Apple M6, FFT speed is set by data movement, not arithmetic. Large batches of every GPU library
run at or near the 150 GB/s main-memory limit; keeping 8192- and 16384-point transforms in registers
saves a pass through memory (2.2x and 1.85x MPSGraph, 4.4x with FP16 storage); the GPU's matrix
units cannot speed up the FFT, while the CPU's SME2 matrix unit can (up to 5.3x vDSP); and a
validated 4096 x 4096 range-Doppler SAR image takes 8.1 ms on the GPU (6.0 ms with FP16
intermediates). See `main.pdf`.

## Contents

| Path | What |
|---|---|
| `main.tex`, `references.bib`, `main.pdf` | The paper (IEEE two-column) |
| `figures/` | Figures, including the cold-cache landscape (`landscape_cold.pdf`) mentioned in Fig. 1 |
| `results/` | Raw logs and CSVs behind every number in the paper |
| `src/common/fft_device.h` | Device-side FFT primitive (`FFT<N,R>`), included by the kernels below |
| `src/metal/` | Metal kernels: one-pass N=8192/16384 (`fft_onepass_large.metal`), SAR pipeline (`m6_sar.metal`), GPU matrix-unit experiments (`m6_tensor_ops.metal`), and the M1-era kernels used for N <= 4096 |
| `src/cpu/sme_fft.{c,h}` | Radix-8 FFT on the CPU's SME2 matrix unit (batches and matrix columns) |
| `src/*.py` | MLX landscape script and plotting scripts |
| `benchmarks/micro/` | Swift benchmark harness (`M6Bench`) and the C SME benchmarks (`sme/`) |

## Requirements

- An Apple Silicon Mac. The paper's numbers come from an Apple M6 (macOS 27.0).
- The SME2 code needs an M4 or later CPU. The GPU matrix-unit modes (`m6-mma-peak`, `m6-tensor-fft`) need macOS 26 or later and an M5-class or later GPU.
- Swift 5.9+ and clang (Xcode or the Command Line Tools).
- Python 3 with pandas and matplotlib for the plots; MLX 0.29.3 and NumPy for the MLX landscape.

## Build

```sh
cd m6/benchmarks/micro
swift build -c release
.build/release/M6Bench            # lists the modes
```

The harness compiles the Metal kernels at run time from `m6/src`. It finds them by walking up from
the working directory and the executable, so run it from inside `m6/`, or set
`FFT_REPO_ROOT=/path/to/m6`.

The C SME benchmarks build with clang, from `m6/`:

```sh
cd m6
clang -O3 -mcpu=apple-m4 -Isrc/cpu benchmarks/micro/sme/sme_stage5.c src/cpu/sme_fft.c \
      -framework Accelerate -o sme_stage5
clang -O3 -mcpu=apple-m4 benchmarks/micro/sme/sme_fmopa_latency.c -o sme_fmopa_latency
clang -O3 -mcpu=apple-m4 benchmarks/micro/sme/sme_load_variants.c -o sme_load_variants
clang -O3 -mcpu=apple-m4 benchmarks/micro/sme/sme_stage_ablation.c -o sme_stage_ablation
```

Use `-mcpu=apple-m4`, not `-march=armv9.x`: an armv9 target lets the compiler emit non-streaming
SVE instructions, which Apple cores do not have (the program dies with SIGILL).

## Reproducing the paper

| Paper | Command (`M6Bench` = `benchmarks/micro/.build/release/M6Bench`) | Result file(s) |
|---|---|---|
| Table I (GPU peaks), Table II | `M6Bench m6-bw` | `results/stage1_bw.stdout` |
| Table III | `M6Bench m6-batch` | `results/stage1_batch.stdout` |
| Table IV | `M6Bench m6-cpu-scale`; thread placement: `M6Bench m6-cpu` | `results/stage1_cpu_scale.stdout`, `results/stage1_cpu.stdout` |
| Fig. 1, Section V | `M6Bench m6-landscape`; `python3 src/mlx_landscape.py results/stage2_mlx.csv`; plot: `python3 src/plot_landscape.py` | `results/stage2_landscape.{csv,stdout}`, `results/stage2_mlx.{csv,stdout}` |
| Table V | `M6Bench m6-onepass` (four-step comparison), `M6Bench m6-fp16` (one-pass FP32 and FP16), correctness: `M6Bench m6-onepass-check` | `results/stage3_onepass.{csv,stdout}`, `results/stage3_fp16.{csv,stdout}` |
| Table VI, Fig. 2 | `M6Bench m6-fp16`; plot: `python3 src/plot_onepass.py` | `results/stage3_fp16.{csv,stdout}` |
| Table VII | `M6Bench m6-mma-peak`, `M6Bench m6-tensor-fft` | `results/stage4_mma_peak.stdout`, `results/stage4_tensor_fft.{csv,stdout}` |
| Section IX, Table VIII | `./sme_stage5 peak`, `bw`, `fft`, `col`, `colmt`, `colmt narrow`, `mt 4096`; probes `./sme_fmopa_latency`, `./sme_load_variants`, `./sme_stage_ablation` | `results/stage5_*.stdout` |
| Section X, Table IX | `M6Bench m6-sar` (three recorded runs); per-stage errors: `SAR_DIAG=1 M6Bench m6-sar` | `results/stage6_sar.stdout`, `results/stage6_sar_run{2,3}.stdout`, `results/stage6_sar_diag.stdout` |

Two logs come from tools not included here: `results/m6_first_probe.stdout` (an initial smoke
test) and `results/stage1_microbench_1to5.stdout` (the M1-era microbenchmarks re-run on M6).

## Measurement notes

- Close other applications and use AC power. Each log starts with the date, the OS build and the
  busiest processes at the time of the run.
- *Hot* runs repeat the FFT on the same buffers, so data can stay in cache; *cold* runs flush the
  caches with a 512 MiB write first. On M6 the two differ by up to 3.7x, so always report both,
  with the footprint (Section III of the paper).
- The CPU's matrix unit is shared by the whole system. Before CPU benchmarks, check that it is idle:
  `./sme_stage5 peak` should report about 2.25 TFLOPS on one thread, and vDSP about 206 GFLOPS for
  N = 4096 (`M6Bench m6-cpu`).

## License

MIT (see `LICENSE` at the repository root).
