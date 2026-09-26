"""Stage 2 figure: FFT throughput vs N on M6, one panel per memory level.

Reads results/stage2_landscape.csv (ours, MPSGraph, vDSP) and results/stage2_mlx.csv.
'ours' = best hot variant per (N, level). Dashed line = single-pass bandwidth ceiling
for that level: BW * 5 log2 N / 16 GF (16 B moved per complex point, in + out).
Usage: python plot_landscape.py [--mode hot|cold]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
mode = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else "hot"

df = pd.concat([pd.read_csv(ROOT / "results/stage2_landscape.csv"),
                pd.read_csv(ROOT / "results/stage2_mlx.csv")])
df["log2N"] = df["N"].apply(lambda n: n.bit_length() - 1)

# fixed categorical order (reference palette slots 1-4) + distinct markers
SERIES = [("ours", "This work (Metal)", "#2a78d6", "o"),
          ("mpsgraph", "MPSGraph", "#eb6834", "s"),
          ("mlx", "MLX (wall clock)", "#1baf7a", "^"),
          ("vdsp", "vDSP, 1 thread", "#eda100", "D")]
# measured stage-1 streaming bandwidths (GB/s): L2 ~ copy at 1 MiB, SLC plateau, DRAM
LEVELS = [("L2", "L2-resident (1 MiB footprint)", 670),
          ("SLC", "SLC-resident (8 MiB)", 310),
          ("DRAM", "DRAM (256 MiB)", 150)]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6})
fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), sharey=True)
for ax, (lvl, title, bw) in zip(axes, LEVELS):
    xs = list(range(5, 21))
    ax.plot(xs, [bw * 5 * x / 16 for x in xs], ls=(0, (4, 3)), lw=1, color=INK2, zorder=1)
    xl = min(20, 1550 * 16 / (5 * bw))          # keep the label inside the y-range
    if xl < 20:
        ax.text(xl + 0.4, 1550, f"{bw} GB/s\nceiling", ha="left", va="center", fontsize=6.5, color=INK2)
    else:
        ax.text(19.7, bw * 5 * 20 / 16 + 30, f"{bw} GB/s ceiling", ha="right", va="bottom", fontsize=6.5, color=INK2)
    for key, label, color, marker in SERIES:
        d = df[(df.backend == key) & (df.level == lvl) & (df["mode"] == ("hot" if key == "vdsp" else mode))]
        if d.empty:
            continue
        # reindex so a missing size (no kernel) breaks the line instead of bridging it
        best = d.groupby("log2N")["gflops_med"].max().reindex(
            range(d.log2N.min(), d.log2N.max() + 1)).rename_axis("log2N").reset_index()
        ax.plot(best.log2N, best.gflops_med, color=color, lw=2 if key != "vdsp" else 1.5,
                marker=marker, ms=4, mec="#fcfcfb", mew=0.8, label=label, zorder=3)
    ax.set_title(title, fontsize=8, color=INK)
    ax.set_xticks(range(5, 21, 3))
    ax.set_xticklabels([f"$2^{{{x}}}$" for x in range(5, 21, 3)])
    ax.set_xlabel("FFT size N")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel(f"GFLOPS (5N log$_2$N), {mode}")
axes[0].set_ylim(0, 1700)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 0.995))
fig.subplots_adjust(left=0.08, right=0.99, bottom=0.15, top=0.83, wspace=0.08)
out = ROOT / f"figures/landscape_{mode}.pdf"
out.parent.mkdir(exist_ok=True)
fig.savefig(out)
fig.savefig(out.with_suffix(".png"), dpi=200)
print("->", out)
