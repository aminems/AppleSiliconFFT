"""Stage 3 figure: DRAM-footprint throughput, ours vs MPSGraph, fp32 vs fp16 storage.

Reads results/stage3_fp16.csv (hot, DRAM level). Ticks = single-pass DRAM ceilings
at 150 GB/s (16 B/point fp32, 8 B/point fp16).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "results/stage3_fp16.csv")
df = df[(df.level == "DRAM") & (df["mode"] == "hot")]

BLUE, ORANGE = "#2a78d6", "#eb6834"            # entity colours shared with the landscape figure
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
OURS = {4096: ("fft4096_dx", "fft4096_dx_f16"), 8192: ("onepass8192", "onepass8192_f16"),
        16384: ("onepass16384", "onepass16384_f16")}
BARS = [("MPSGraph fp32", ORANGE, None, lambda n: "MPSGraph"),
        ("MPSGraph fp16", ORANGE, "////", lambda n: "MPSGraph_f16"),
        ("This work fp32", BLUE, None, lambda n: OURS[n][0]),
        ("This work fp16", BLUE, "////", lambda n: OURS[n][1])]

plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.linewidth": 0.6, "hatch.linewidth": 0.8})
fig, ax = plt.subplots(figsize=(3.5, 2.6))
sizes, w = [4096, 8192, 16384], 0.19
for gi, n in enumerate(sizes):
    e = n.bit_length() - 1
    for bi, (label, color, hatch, var) in enumerate(BARS):
        v = df[(df.N == n) & (df.variant == var(n))].gflops_med.iloc[0]
        x = gi + (bi - 1.5) * (w + 0.02)
        ax.bar(x, v, w, color=color if hatch is None else SURF, edgecolor=color, hatch=hatch, lw=1.0,
               label=label if gi == 0 else None, zorder=3)
        if var(n).startswith(("fft", "onepass")) and n != 4096:
            ax.text(x, v + 20, f"{v:.0f}", ha="center", va="bottom", fontsize=6.5, color=INK)
    c32, c16 = 150 * 5 * e / 16, 150 * 5 * e / 8
    ax.hlines(c32, gi - 2 * (w + 0.02), gi + 2 * (w + 0.02), color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=4)
    ax.hlines(c16, gi - 2 * (w + 0.02), gi + 2 * (w + 0.02), color=INK2, lw=0.8, ls=(0, (1, 1.5)), zorder=4)
ax.text(2.42, 150 * 5 * 14 / 16, "fp32\nceiling", fontsize=6, color=INK2, va="center")
ax.text(2.42, 150 * 5 * 14 / 8, "fp16\nceiling", fontsize=6, color=INK2, va="center")
ax.set_xticks(range(3))
ax.set_xticklabels([f"N = {n}" for n in sizes])
ax.set_xlim(-0.55, 2.75)
ax.set_ylabel("GFLOPS (5N log$_2$N)")
ax.set_ylim(0, 1750)
ax.grid(axis="y", color=GRID, lw=0.6)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(ncol=2, frameon=False, fontsize=6.5, loc="upper left", handlelength=1.4, columnspacing=1.0)
fig.subplots_adjust(left=0.17, right=0.99, bottom=0.1, top=0.98)
out = ROOT / "figures/onepass_dram.pdf"
fig.savefig(out)
fig.savefig(out.with_suffix(".png"), dpi=220)
print("->", out)
