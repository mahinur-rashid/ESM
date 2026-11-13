"""Concept figures and the coverage and trade-off figures, drawn at print size (131 mm text block, 9 pt type).

    python scripts/make_extra_figures.py paper_out/tables paper_out/figures

Writes fig_predsets.pdf, fig_problems.pdf (illustrative), fig_coverage.pdf and fig_tradeoff.pdf (from the tables).
"""
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

TABLES, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
W = 5.1
INK, MUTED, GRID = "#222222", "#555555", "#e4e3de"
BLUE, ORANGE, GREY = "#2a78d6", "#d9541e", "#c9c8c2"
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False, "text.color": INK, "axes.labelcolor": INK})
PAIRS = ["LV→EE", "EE→LV", "LV W→E", "PT N→S", "PT→LV", "LV→PT", "LV+EE→PT"]


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf")
    fig.savefig(f"{OUT}/{name}.png", dpi=200)
    plt.close(fig)


def predsets():
    fields = [("Field A: an easy field", ["wheat", "barley", "rye", "oats"], [0.82, 0.10, 0.05, 0.03]),
              ("Field B: an ambiguous field", ["maize", "soybean", "sunflower", "sorghum"], [0.46, 0.33, 0.13, 0.08])]
    thr = 0.20
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.7), sharey=True, layout="constrained")
    for ax, (title, crops, p) in zip(axs, fields):
        ax.bar(range(4), p, color=[BLUE if v >= thr else GREY for v in p], width=0.65)
        ax.axhline(thr, ls="--", color=INK, lw=1)
        ax.text(3.4, thr + 0.025, "threshold", ha="right", va="bottom", fontsize=8.5)
        ax.set_xticks(range(4)); ax.set_xticklabels(crops)
        ax.set_ylim(0, 1); ax.set_title(title)
        kept = [c for c, v in zip(crops, p) if v >= thr]
        ax.set_xlabel("set = {" + ", ".join(kept) + "}", color=BLUE, fontweight="bold", labelpad=6)
    axs[0].set_ylabel("Classifier probability")
    save(fig, "fig_predsets")


def problems():
    rows = [("1  Unnamed crops", "A model trained in Latvia meets an olive grove in Portugal. It has no 'olive' label, so its set misses at any threshold.",
             "70% of Portuguese parcels"),
            ("2  Under-coverage", "For crops it knows, target fields look different (other climate, other crop calendar), so sets contain the true crop less often. Frozen Presto even becomes more confident.",
             "coverage 0.17 instead of 0.90"),
            ("3  Valid or informative", "Under severe shift the classifier is close to guessing. A set that keeps the 90% promise must list most crops: true, but not useful.",
             "27 of 34 crops per set")]
    fig = plt.figure(figsize=(W, 3.3))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    import textwrap
    for i, (t, body, num) in enumerate(rows):
        y0 = 0.68 - i * 0.33
        ax.add_patch(FancyBboxPatch((0.01, y0), 0.98, 0.29, boxstyle="round,pad=0,rounding_size=0.02",
                                    fc="#f6f5f1", ec="#bdbcb5", lw=0.8))
        ax.text(0.03, y0 + 0.215, t, fontsize=9.5, fontweight="bold", va="center")
        ax.text(0.03, y0 + 0.095, "\n".join(textwrap.wrap(body, 78)), fontsize=8.8, va="center", linespacing=1.25)
        ax.text(0.97, y0 + 0.215, num, fontsize=9, ha="right", va="center", color=ORANGE, fontweight="bold")
    save(fig, "fig_problems")


def coverage():
    lines = open(f"{TABLES}/models.tex").read().split("\n")
    i = [k for k, l in enumerate(lines) if "LAC coverage in target" in l][0]
    data = {}
    for l in lines[i + 1:i + 6]:
        c = [x.strip() for x in l.rstrip("\\").split("&")]
        data[c[0].replace("\\quad", "").strip()] = [tuple(map(float, x.split(" / "))) for x in c[1:]]
    col = {"Transformer": "#2a78d6", "TempCNN": "#1baf7a", "Presto, frozen": "#eb6834", "Presto, fine-tuned": "#b04a9e", "Random forest": "#7a7a7a"}
    mk = {"Transformer": "o", "TempCNN": "s", "Presto, frozen": "D", "Presto, fine-tuned": "^", "Random forest": "v"}
    fig, axs = plt.subplots(2, 1, figsize=(W, 5.4), sharex=True, layout="constrained")
    for ax, j, t in ((axs[0], 0, "(a) Class-balanced samples"), (axs[1], 1, "(b) Natural crop frequencies")):
        ax.axhspan(0, 0.88, color="#f3e6df", zorder=0, lw=0)
        ax.axhline(0.90, ls="--", color=MUTED, lw=1)
        for k, (m, v) in enumerate(data.items()):
            x = [p + (k - 2) * 0.08 for p in range(7)]
            ax.plot(x, [a[j] for a in v], marker=mk[m], ms=6, lw=1.5, color=col[m], mec="white", mew=0.8, label=m)
        ax.set_ylim(0, 1); ax.set_title(t, loc="left"); ax.set_ylabel("LAC coverage")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.text(6.35, 0.915, "target 0.90", ha="right", va="bottom", fontsize=8.5, color=MUTED)
    axs[1].set_xticks(range(7)); axs[1].set_xticklabels(PAIRS)
    axs[1].set_xlabel("Transfer (mild to severe); shaded: below 0.88")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=3, frameon=False)
    save(fig, "fig_coverage")


def tradeoff():
    rows = []
    for line in open(f"{TABLES}/lac_methods.tex"):
        cells = [c.strip() for c in line.strip().rstrip("\\").split("&")]
        if len(cells) != 8 or cells[0] == "":
            continue
        name = re.sub(r"\\quad\s*", "", cells[0]).strip()

        def parse(c):
            m = re.search(r"([\d.]+) / ([\d.]+) \(([\d.]+)\)", c)
            return float(m.group(1)), float(m.group(3))
        rows.append((name, parse(cells[1]), parse(cells[6])))
    labels = {"Standard split CP": "Standard", "Region-robust": "Region-robust", "Class-region-robust": "Class-region-robust",
              "Cov.-shift weighted, $w\\in[0.01,100]$": "Cov.-shift weighted", "clipped to $w\\in[0.1,10]$": "Cov.-shift, clipped",
              "$k$-NN localized": "k-NN", "BBSE weighted": "BBSE", "EM-adjusted, label-shift weighted": "EM", "QTC": "QTC",
              "source confidences at natural frequencies": "QTC, natural-freq. source", "Oracle, one threshold": "ORACLE"}
    pts = [(labels[n], a, b) for n, a, b in rows if n in labels]
    cat = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#b04a9e", "#008300", "#8a5cd1", "#7a7a7a", "#0b6e79", "#7b4a2e"]
    mks = ["o"] * 8 + ["s", "s"]
    fig, axs = plt.subplots(2, 1, figsize=(W, 6.2), layout="constrained")
    for ax, idx, title in ((axs[0], 1, "(a) Mild shift: LV→EE"), (axs[1], 2, "(b) Severe shift: LV→PT")):
        ax.axhline(0.90, ls="--", color=MUTED, lw=1); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
        meth = [p for p in pts if p[0] != "ORACLE"]
        orc = [p for p in pts if p[0] == "ORACLE"][0]
        for j, (lab, *v) in enumerate(meth):
            cov, size = v[idx - 1]
            ax.scatter(size, cov, s=50, color=cat[j], marker=mks[j], edgecolor="white", lw=1, zorder=3, label=lab)
        cov, size = orc[idx]
        ax.scatter(size, cov, s=260, marker="*", facecolor="none", edgecolor=INK, lw=1.1, zorder=2, label="Oracle (target labels)")
        ax.set_title(title, loc="left"); ax.set_ylabel("Coverage (balanced)"); ax.set_ylim(0, 1.03)
    axs[1].set_xlabel("Average set size (crops)")
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=3, frameon=False)
    save(fig, "fig_tradeoff")


predsets(); problems(); coverage(); tradeoff()
print("written to", OUT)
