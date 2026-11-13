"""Dataset figures of the paper from the output of figure_data.py.

    python scripts/plot_data_figures.py figure_data out_dir

Writes fig_map.pdf (parcel locations and transfers), fig_cropmix.pdf (most common crops), fig_timeseries.pdf (NDVI
curves) and table_sets.tex (example prediction sets, as a LaTeX table). Figures are drawn at print size for a 131 mm
text block, with 9 pt type.
"""
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import ConnectionPatch, Patch  # noqa: E402

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 9, "axes.titlesize": 9.5, "xtick.labelsize": 8.5,
                     "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
W = 5.1
INK, MUTED, GRID = "#222222", "#666666", "#e4e3de"
CTRY = {"EE": ("Estonia", "#2a78d6"), "LV": ("Latvia", "#1baf7a"), "PT": ("Portugal", "#eb6834")}
HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = pd.read_csv(os.path.join(HERE, "..", "data", "hcat_names.csv"), dtype=str).set_index("code")["name"].to_dict()
SHORT = {"pasture_meadow_grassland_grass": "permanent grassland", "not_known_and_other": "not known and other",
         "olive_plantations": "olive groves", "olives_for_oil_production": "olives (oil)",
         "vineyards_wine_vine_rebland_grapes": "vineyards", "winter_common_soft_wheat": "winter wheat",
         "spring_common_soft_wheat": "spring wheat", "grain_maize_corn_popcorn": "grain maize",
         "green_silo_maize": "silage maize", "legumes_harvested_green": "legumes harvested green",
         "winter_rapeseed_rape": "winter rapeseed", "temporary_grass": "temporary grass",
         "sweet_chestnuts": "sweet chestnuts", "fallow_land_not_crop": "fallow land"}


def name(code):
    n = NAMES.get(str(code), str(code))
    n = SHORT.get(n, n).replace("_", " ")
    return n if len(n) <= 26 else n[:25] + "."


def known_by_latvia(counts):
    """Latvian model's label space: crops with >= 300 sampled Latvian parcels (>= 100 kept, at most 1,500 sampled)."""
    lv = counts["LV"].where(counts["LV"] >= 100, 0).clip(upper=1500)
    return set(lv.index[lv >= 300].astype(str))


def outlines():
    try:
        import geopandas as gpd
        w = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        return w[w.name.isin(["Latvia", "Estonia", "Portugal", "Lithuania", "Spain", "Russia", "Belarus", "Finland"])]
    except Exception:
        return None


def save(fig, out, name):
    fig.savefig(f"{out}/{name}.pdf", dpi=300)
    fig.savefig(f"{out}/{name}.png", dpi=200)
    plt.close(fig)


def study_map(fd, out):
    nat = pd.read_csv(f"{fd}/natural_sample.csv", dtype={"cls": str})
    world = outlines()
    fig = plt.figure(figsize=(W, 3.5))
    axB = fig.add_axes([0.0, 0.02, 0.56, 0.88])
    axP = fig.add_axes([0.70, 0.02, 0.27, 0.88])
    for ax, ext, cs in ((axB, (20.8, 28.6, 55.5, 59.9), ["EE", "LV"]), (axP, (-9.7, -6.0, 36.8, 42.3), ["PT"])):
        if world is not None:
            world.plot(ax=ax, color="#f4f3ef", edgecolor="#bdbcb5", lw=0.6)
        for c in cs:
            d = nat[nat.country == c]
            ax.scatter(d.lon, d.lat, s=1.0, color=CTRY[c][1], alpha=0.5, lw=0, rasterized=True)
            ax.text(d.lon.median(), d.lat.median() - (1.3 if c == "PT" else 0), CTRY[c][0], ha="center", va="center",
                    fontsize=10, fontweight="bold", color=INK, bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect(1 / np.cos(np.deg2rad(np.mean(ext[2:]))))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color("#bdbcb5")
    lv, pt = nat[nat.country == "LV"], nat[nat.country == "PT"]
    axB.plot([lv.lon.median()] * 2, [55.6, 58.15], ls="--", color=INK, lw=1)
    axB.text(lv.lon.median() + 0.12, 55.75, "LV west/east split", fontsize=8.5)
    axP.plot([-9.6, -6.1], [pt.lat.median()] * 2, ls="--", color=INK, lw=1)
    axP.text(-6.15, pt.lat.median() + 0.1, "PT north/\nsouth split", fontsize=8.5, ha="right")
    axB.annotate("", xy=(27.0, 58.3), xytext=(27.0, 57.2), arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=1.3))
    axB.text(27.15, 57.75, "LV→EE\nEE→LV", fontsize=8.5, va="center")
    con = ConnectionPatch(xyA=(28.5, 57.0), coordsA=axB.transData, xyB=(-9.6, 40.3), coordsB=axP.transData,
                          arrowstyle="<|-|>", color=INK, lw=1.3, connectionstyle="arc3,rad=-0.3")
    fig.add_artist(con)
    fig.text(0.635, 0.30, "PT→LV\nLV→PT\nLV+EE→PT", fontsize=8.5, ha="center", va="center")
    axB.set_title("Baltic countries", fontsize=9.5, loc="left")
    axP.set_title("Portugal", fontsize=9.5, loc="left")
    save(fig, out, "fig_map")


def crop_mix(fd, out):
    counts = pd.read_csv(f"{fd}/counts.csv", dtype={"cls": str}).set_index("cls")
    known = known_by_latvia(counts)
    fig, axs = plt.subplots(3, 1, figsize=(W, 6.3), layout="constrained")
    for ax, c in zip(axs, ["EE", "LV", "PT"]):
        share = (counts[c] / counts[c].sum()).sort_values(ascending=False).head(8)[::-1]
        unk = [i not in known for i in share.index.astype(str)]
        ax.barh(range(len(share)), share.values, color=CTRY[c][1], height=0.72,
                hatch=["////" if u else "" for u in unk], edgecolor="white", lw=0)
        for i, v in enumerate(share.values):
            ax.text(v + 0.006, i, f"{v:.0%}", va="center", fontsize=8.5)
        ax.set_yticks(range(len(share))); ax.set_yticklabels([name(i) for i in share.index])
        ax.set_xlim(0, 0.56); ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0))
        ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.set_title(CTRY[c][0], loc="left")
    axs[-1].set_xlabel("Share of parcels")
    fig.legend(handles=[Patch(fc="#999", hatch="////", ec="white", label="crop code a model trained in Latvia cannot output")],
               loc="outside lower center", frameon=False)
    save(fig, out, "fig_cropmix")


def timeseries(fd, out):
    nat = pd.read_csv(f"{fd}/natural_sample.csv", dtype={"cls": str})
    ex = pd.read_csv(f"{fd}/examples.csv", dtype={"cls": str})
    mcols = [c for c in nat.columns if c.startswith("ndvi_m")]
    pairs = ex.groupby("cls").country.unique()
    multi = [c for c in pairs.index if len(pairs[c]) >= 2]
    single = [c for c in pairs.index if len(pairs[c]) == 1]
    size = nat.groupby("cls").size()
    crops = sorted(multi, key=lambda c: -size.get(c, 0))[:4]
    for want in ("olive", "vine"):
        crops += [c for c in single if want in NAMES.get(c, "")][:1]
    crops += [c for c in sorted(single, key=lambda c: -size.get(c, 0)) if c not in crops]
    crops = crops[:6]
    fig, axs = plt.subplots(3, 2, figsize=(W, 6.6), sharex=True, sharey=True, layout="constrained")
    mid = np.array([15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349])
    for ax, crop in zip(axs.flat, crops):
        for c in ["EE", "LV", "PT"]:
            d = nat[(nat.cls == crop) & (nat.country == c)][mcols]
            if len(d) >= 15:
                q = d.quantile([0.25, 0.5, 0.75]).values
                ax.fill_between(mid, q[0], q[2], color=CTRY[c][1], alpha=0.18, lw=0)
                ax.plot(mid, q[1], color=CTRY[c][1], lw=2, label=CTRY[c][0])
            e = ex[(ex.cls == crop) & (ex.country == c) & (ex.parcel == 0)]
            if len(e) and len(d) >= 15:
                ax.scatter(e.doy, e.ndvi, s=8, color=CTRY[c][1], edgecolor="white", lw=0.4, zorder=3)
        ax.set_title(name(crop))
        ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True); ax.set_ylim(-0.1, 1.0)
    for ax in axs[-1]:
        ax.set_xticks(mid[::2]); ax.set_xticklabels(["Jan", "Mar", "May", "Jul", "Sep", "Nov"])
    for ax in axs[:, 0]:
        ax.set_ylabel("NDVI")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=CTRY[c][1], lw=2, label=CTRY[c][0]) for c in CTRY],
               loc="outside lower center", ncol=3, frameon=False)
    save(fig, out, "fig_timeseries")


def sets_table(fd, out, n_rows=8):
    """Example prediction sets as LaTeX table rows (true crop, set size, first crops of the set, result)."""
    def tex(t):
        return t.replace("&", "\\&").replace("_", " ")
    lines = []
    for pname, title in (("LV_to_EE", "Latvia$\\to$Estonia"), ("LV_to_PT", "Latvia$\\to$Portugal")):
        d = pd.read_csv(f"{fd}/sets_{pname}.csv", dtype={"true": str, "set": str}, keep_default_na=False)
        rng = np.random.RandomState(3)
        d = d.iloc[rng.permutation(len(d))[:n_rows]].reset_index(drop=True)
        lines.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{title}}}}}\\\\")
        for _, r in d.iterrows():
            known = str(r.known) == "True"; covered = str(r.covered) == "True"
            mark = "unknown crop" if not known else ("covered" if covered else "missed")
            members = [name(x) for x in r.set.split("|") if x]
            txt = ", ".join(members[:3]) + (f", \\dots\\ (+{len(members) - 3})" if len(members) > 3 else "")
            lines.append(f"{tex(name(r.true))} & {len(members)} & \\{{{tex(txt)}\\}} & {mark}\\\\")
        lines.append("\\midrule" if pname == "LV_to_EE" else "")
    open(f"{out}/table_sets.tex", "w").write("\n".join(l for l in lines if l) + "\n")


if __name__ == "__main__":
    fd, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    study_map(fd, out); crop_mix(fd, out); timeseries(fd, out); sets_table(fd, out)
    print("written to", out)
