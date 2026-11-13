"""Every paper table except the taxonomy table, Figure 1, and the numbers quoted in the text.

    python scripts/make_paper_tables.py results/main results/controls paper_out

The main results come from the `main` experiment (the five models, frozen Presto with a constant location and the
OTHER-class transformer); the monthly transformer, the frozen-Presto probe controls and the merged-code transformer
come from the `controls` experiment. Label-free methods and warning signs use the unlabeled pool of all target
parcels at natural frequencies. Writes LaTeX table bodies to <out>/tables/, the figure to <out>/figures/fig_main.pdf
and the numbers quoted in the text to <out>/paper_numbers.json.
"""
import json
import math
import os
import re
import sys

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from scipy.stats import rankdata, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TM, TC, OUT = sys.argv[1:4]
for d in ("figures", "tables"):
    os.makedirs(f"{OUT}/{d}", exist_ok=True)
rd = lambda t, f: pd.read_csv(f"{t}/{f}.csv")                                         # noqa: E731
RI, CB, SPL, EXT = rd(TM, "run_info"), rd(TM, "conformal_by_run"), rd(TM, "splits"), rd(TM, "extra_by_run")
SYN, ADP, WRS = rd(TM, "synthetic_label_shift"), rd(TM, "adapt_by_run"), rd(TM, "worst_region_by_split")
FLO, FEWA = rd(TM, "region_floor"), rd(TM, "few_by_arch")
FULL = True                                             # all diagnostics present


def with_controls(main, other):
    """Rows of `main`, plus the models only `other` (the controls experiment) has."""
    return pd.concat([main, other[~other.arch.isin(main.arch.unique())]], ignore_index=True)


RIc = with_controls(RI, rd(TC, "run_info")); CBc = with_controls(CB, rd(TC, "conformal_by_run"))
EXTc = with_controls(EXT, rd(TC, "extra_by_run"))

P = ["LV_to_EE", "EE_to_LV", "LVwest_to_LVeast", "PTnorth_to_PTsouth", "PT_to_LV", "LV_to_PT", "LVEE_to_PT"]
PT3 = ["PT_to_LV", "LV_to_PT", "LVEE_to_PT"]; PT2 = ["LV_to_PT", "LVEE_to_PT"]
NAME = {"LV_to_EE": "LV$\\to$EE", "EE_to_LV": "EE$\\to$LV", "LVwest_to_LVeast": "LV W$\\to$E", "PTnorth_to_PTsouth": "PT N$\\to$S",
        "PT_to_LV": "PT$\\to$LV", "LV_to_PT": "LV$\\to$PT", "LVEE_to_PT": "LV+EE$\\to$PT"}
PLAIN = {"LV_to_EE": "LV→EE", "EE_to_LV": "EE→LV", "LVwest_to_LVeast": "LV W→E", "PTnorth_to_PTsouth": "PT N→S",
         "PT_to_LV": "PT→LV", "LV_to_PT": "LV→PT", "LVEE_to_PT": "LV+EE→PT"}
MAIN = ["transformer", "tempcnn", "presto_frozen", "presto_finetune", "random_forest"]
ANAME = {"transformer": "Transformer", "tempcnn": "TempCNN", "presto_frozen": "Presto, frozen",
         "presto_finetune": "Presto, fine-tuned", "random_forest": "Random forest"}
HEAD = " & " + " & ".join(NAME[p] for p in P) + "\\\\"
TH = 0.88                                    # one threshold for "under-covers": the success rule's
N = {}
neg = lambda s: re.sub(r"(^|[ (/])-(\d)", r"\1$-$\2", s, flags=re.M)                          # noqa: E731


def srho(a, b):
    return float(spearmanr(a, b).correlation)


def frho(x, y):
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


ri = RI[RI.arch.isin(MAIN)]
cb = CB[CB.arch.isin(MAIN)]
ex = cb[cb.design == "exchangeable"]
src = ri[ri.space == "source"]
if not (ri.groupby(["pair", "space", "arch"]).size().eq(3).all() and len(ri) == 210):
    print("warning: expected 210 main runs with three seeds each, found", len(ri))


def pairmean(df, col):
    return df.groupby("pair")[col].mean().reindex(P)


# ================= Table 1: the transfer pairs =================
L = [HEAD, "\\midrule"]
row = lambda lab, vals: L.append(lab + " & " + " & ".join(vals) + "\\\\")              # noqa: E731
row("Crops known / present in target", [f"{int(pairmean(src,'K')[p])} / {int(pairmean(src,'n_eval_classes')[p])}" for p in P])
row("Target parcels of unknown crops", [f"{pairmean(src,'unknown_frac_nat')[p]:.2f}" for p in P])
row("Accuracy, source / target", [f"{pairmean(src,'acc_src')[p]:.2f} / {pairmean(src,'acc_tgt')[p]:.2f}" for p in P])
row("Domain-classifier AUC, all target parcels", [f"{pairmean(src,'domain_auc_all')[p]:.2f}" for p in P])
if FULL:
    row("\\quad both sides at natural frequencies", [f"{pairmean(src,'domain_auc_natsrc_all')[p]:.2f}" for p in P])
    N["auc_natsrc"] = {p: float(pairmean(src, "domain_auc_natsrc_all")[p]) for p in P}
row("LAC in source: coverage / set size", [f"{pairmean(src,'src_cov_LAC')[p]:.2f} / {pairmean(src,'src_size_LAC')[p]:.1f}" for p in P])
open(f"{OUT}/tables/pairs.tex", "w").write("\n".join(L) + "\n")
N["unknown"] = {p: float(pairmean(src, "unknown_frac_nat")[p]) for p in P}
N["eval_frac"] = {p: float(pairmean(src, "eval_frac_nat")[p]) for p in P}

# ================= Table 2: per model =================
pm = ex[(ex.space == "source") & (ex.method == "standard")].groupby(["score", "arch", "pair"]).agg(
    cov=("coverage", "mean"), nat=("coverage_nat", "mean"), size=("size", "mean")).reset_index()
pr = src.groupby(["arch", "pair"]).mean(numeric_only=True)
g2 = lambda s, a, p, c: pm[(pm.score == s) & (pm.arch == a) & (pm.pair == p)][c].iloc[0]   # noqa: E731
L = [HEAD, "\\midrule"]


def block(title, fn):
    L.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{title}}}}}\\\\")
    for a in MAIN:
        L.append("\\quad " + ANAME[a] + " & " + " & ".join(fn(a, p) for p in P) + "\\\\")


block("Accuracy, source / target", lambda a, p: f"{pr.loc[(a,p),'acc_src']:.2f} / {pr.loc[(a,p),'acc_tgt']:.2f}")
block("Mean top-class confidence, source / all target parcels (natural frequencies)",
      lambda a, p: f"{pr.loc[(a,p),'conf_src_nat']:.2f} / {pr.loc[(a,p),'conf_tgt_all']:.2f}")
block("LAC set size, in source / in target", lambda a, p: f"{pr.loc[(a,p),'src_size_LAC']:.1f} / {g2('LAC',a,p,'size'):.1f}")
block("LAC coverage in target, balanced / natural frequencies (in source: 0.89--0.91)",
      lambda a, p: f"{g2('LAC',a,p,'cov'):.2f} / {g2('LAC',a,p,'nat'):.2f}")
block("Non-randomized APS coverage in target (set size)", lambda a, p: f"{g2('APS',a,p,'cov'):.2f} ({g2('APS',a,p,'size'):.1f})")
open(f"{OUT}/tables/models.tex", "w").write("\n".join(L) + "\n")
N["models"] = {f"{a}|{p}": dict(acc_src=float(pr.loc[(a, p), "acc_src"]), acc_tgt=float(pr.loc[(a, p), "acc_tgt"]),
                                conf_src=float(pr.loc[(a, p), "conf_src_nat"]), conf_tgt_all=float(pr.loc[(a, p), "conf_tgt_all"]),
                                conf_tgt_known=float(pr.loc[(a, p), "conf_tgt"]), conf_src_bal=float(pr.loc[(a, p), "conf_src"]),
                                size_src=float(pr.loc[(a, p), "src_size_LAC"]), size_tgt=float(g2("LAC", a, p, "size")),
                                lac=float(g2("LAC", a, p, "cov")), lac_nat=float(g2("LAC", a, p, "nat")),
                                aps=float(g2("APS", a, p, "cov")), aps_size=float(g2("APS", a, p, "size")),
                                src_size_APS=float(pr.loc[(a, p), "src_size_APS"]), src_size_APS_rand=float(pr.loc[(a, p), "src_size_APS_rand"]))
               for a in MAIN for p in P}
N["src_cov_range"] = [float(ri.groupby(["pair", "space", "arch"]).src_cov_LAC.mean().min()),
                      float(ri.groupby(["pair", "space", "arch"]).src_cov_LAC.mean().max())]

# ================= Table 3: warning signs per model =================
std = ex[(ex.score == "LAC") & (ex.method == "standard")].merge(RI, on=["pair", "space", "arch", "seed"])
std["shortfall"] = 0.9 - std.coverage
std["conf_drop_all"] = std.conf_src_nat - std.conf_tgt_all
std["acc_drop"] = std.acc_src - std.acc_tgt
cond = std.groupby(["arch", "pair", "space"]).mean(numeric_only=True).reset_index()
condnat = ex[(ex.score == "LAC") & (ex.method == "standard")].groupby(["arch", "pair", "space"]).coverage_nat.mean()
INPUT = {"transformer": "full series", "tempcnn": "full series", "presto_frozen": "monthly", "presto_finetune": "monthly",
         "random_forest": "monthly"}
rng = np.random.default_rng(0); BOOT = rng.integers(0, 7, size=(2000, 7))


def uncertainty(c, sig):
    X = np.array([[c.set_index(["pair", "space"]).loc[(p, s), sig] for s in ("source", "shared")] for p in P])
    Y = np.array([[c.set_index(["pair", "space"]).loc[(p, s), "shortfall"] for s in ("source", "shared")] for p in P])
    lopo = [frho(np.delete(X, i, 0).ravel(), np.delete(Y, i, 0).ravel()) for i in range(7)]
    bs = [frho(X[b].ravel(), Y[b].ravel()) for b in BOOT if np.unique(X[b]).size > 1 and np.unique(Y[b]).size > 1]
    return float(min(lopo)), float(max(lopo)), float(np.percentile(bs, 5)), float(np.percentile(bs, 95))


L = ["Model & Input & In-source acc. & Below 0.88, balanced / natural & $\\rho$ conf.\\ drop & $\\rho$ label shift & "
     "$\\rho$ domain AUC & $\\rho$ acc.\\ drop & Within-pair $\\rho$ label shift\\\\", "\\midrule"]
LU = ["Model & Signal & $\\rho$ & Leave one pair out & Pair bootstrap, 90\\%\\\\", "\\midrule"]
N["diag"] = {}
for a in MAIN:
    c = cond[cond.arch == a]
    within = float(np.nanmean([srho(x.pred_tv_all, x.shortfall) for _, x in std[std.arch == a].groupby(["pair", "space"])
                               if x.pred_tv_all.nunique() > 1 and x.shortfall.nunique() > 1]))
    within_c = float(np.nanmean([srho(x.conf_drop_all, x.shortfall) for _, x in std[std.arch == a].groupby(["pair", "space"])
                                 if x.conf_drop_all.nunique() > 1 and x.shortfall.nunique() > 1]))
    d = dict(acc=[float(c.acc_src.min()), float(c.acc_src.max())],
             below_bal=int((c.coverage < TH).sum()), below_nat=int((condnat.loc[a] < TH).sum()),
             conf=srho(c.conf_drop_all, c.shortfall), tv=srho(c.pred_tv_all, c.shortfall), auc=srho(c.domain_auc_all, c.shortfall),
             acc_rho=srho(c.acc_drop, c.shortfall), within_tv=within, within_conf=within_c,
             conf_known=srho(c.conf_src - c.conf_tgt, c.shortfall), tv_known=srho(c.pred_tv, c.shortfall),
             unc_tv=uncertainty(c, "pred_tv_all"), unc_conf=uncertainty(c, "conf_drop_all"))
    N["diag"][a] = d
    L.append(f"{ANAME[a]} & {INPUT[a]} & {d['acc'][0]:.2f}--{d['acc'][1]:.2f} & {d['below_bal']} / {d['below_nat']} & "
             f"{d['conf']:+.2f} & {d['tv']:+.2f} & {d['auc']:+.2f} & {d['acc_rho']:+.2f} & {d['within_tv']:+.2f}\\\\")
    for sig, lab, u in (("conf", "confidence drop", d["unc_conf"]), ("tv", "label shift", d["unc_tv"])):
        LU.append(f"{ANAME[a] if sig == 'conf' else ''} & {lab} & {d[sig]:+.2f} & {u[0]:+.2f} to {u[1]:+.2f} & {u[2]:+.2f} to {u[3]:+.2f}\\\\")
allc = cond[cond.arch.isin(MAIN)]
nb_bal, nb_nat = int((allc.coverage < TH).sum()), int((condnat.loc[MAIN] < TH).sum())
L.append("\\midrule")
L.append(f"All, 70 conditions & & & {nb_bal} / {nb_nat} & {srho(allc.conf_drop_all, allc.shortfall):+.2f} & "
         f"{srho(allc.pred_tv_all, allc.shortfall):+.2f} & {srho(allc.domain_auc_all, allc.shortfall):+.2f} & "
         f"{srho(allc.acc_drop, allc.shortfall):+.2f} & \\\\")
open(f"{OUT}/tables/diagnostics.tex", "w").write(neg("\n".join(L)) + "\n")
open(f"{OUT}/tables/diag_uncertainty.tex", "w").write(neg("\n".join(LU)) + "\n")
pt = allc.pair.isin(PT3)
N.update(n_below_bal=nb_bal, n_below_nat=nb_nat, rho70_tv=srho(allc.pred_tv_all, allc.shortfall),
         rho70_conf=srho(allc.conf_drop_all, allc.shortfall), rho70_auc=srho(allc.domain_auc_all, allc.shortfall),
         rho70_acc=srho(allc.acc_drop, allc.shortfall),
         rho_tv_within_cluster=[srho(allc[~pt].pred_tv_all, allc[~pt].shortfall), srho(allc[pt].pred_tv_all, allc[pt].shortfall)],
         rho_acc_within_cluster=[srho(allc[~pt].acc_drop, allc[~pt].shortfall), srho(allc[pt].acc_drop, allc[pt].shortfall)])
N["cov_by_cond"] = {f"{a}|{p}|{s}": [float(cond[(cond.arch == a) & (cond.pair == p) & (cond.space == s)].coverage.iloc[0]),
                                      float(condnat.loc[(a, p, s)])] for a in MAIN for p in P for s in ("source", "shared")}

# ================= Table 4: controls (source label space) =================
CR = [("transformer", "Transformer", "full"), ("transformer_monthly", "Transformer", "monthly"), ("tempcnn", "TempCNN", "full"),
      ("presto_finetune", "Presto, fine-tuned", "monthly"), ("random_forest", "Random forest", "monthly"),
      ("presto_frozen", "Presto, frozen", "monthly"), ("presto_frozen_l2", "\\quad L2 chosen on validation", "monthly"),
      ("presto_frozen_l2s1", "\\quad L2 fixed at 0.1", "monthly"), ("presto_frozen_l2s2", "\\quad L2 fixed at 1", "monthly"),
      ("presto_frozen_raw", "\\quad unstandardised embeddings", "monthly"), ("presto_frozen_clip", "\\quad clipped at $\\pm$3 SD", "monthly"),
      ("presto_frozen_ts", "\\quad temperature scaling", "monthly"), ("presto_frozen_mlp", "\\quad MLP probe", "monthly"),
      ("presto_frozen_noloc", "\\quad constant location", "monthly")]
csrc = CBc[(CBc.design == "exchangeable") & (CBc.space == "source") & (CBc.method == "standard")]
cv = csrc.groupby(["arch", "score", "pair"]).coverage.mean()
csz = csrc.groupby(["arch", "score", "pair"])["size"].mean()
rs = RIc[RIc.space == "source"].groupby(["arch", "pair"]).mean(numeric_only=True)
es = EXTc[EXTc.space == "source"].groupby(["arch", "pair"]).mean(numeric_only=True)
L = ["Model & Input & Acc. & Confidence & LAC (size) & APS & $\\rho$ conf.\\ drop & Beyond 99\\%, std.\\ / raw\\\\", "\\midrule"]
N["controls"] = {}
for arch, name, inp in CR:
    acc = float(rs.loc[arch].acc_src.mean())
    cs, ct = float(rs.loc[[(arch, p) for p in PT3]].conf_src_nat.mean()), float(rs.loc[[(arch, p) for p in PT3]].conf_tgt_all.mean())
    lac = float(np.mean([cv[(arch, "LAC", p)] for p in PT3])); lsz = float(np.mean([csz[(arch, "LAC", p)] for p in PT3]))
    aps = [float(cv[(arch, "APS", p)]) for p in PT2]
    sf = pd.Series({p: 0.9 - cv[(arch, "LAC", p)] for p in P})
    cd = pd.Series({p: rs.loc[(arch, p)].conf_src_nat - rs.loc[(arch, p)].conf_tgt_all for p in P})
    rc = srho(cd[P], sf[P])
    if arch.startswith("presto_frozen"):
        b = float(es.loc[[(arch, p) for p in PT3]].tgt_beyond_src_q99_all.mean()); br = float(es.loc[[(arch, p) for p in PT3]].tgt_beyond_src_q99_raw_all.mean())
        btxt = f"{b:.2f} / {br:.2f}"
    else:
        b = br = None; btxt = "--"
    N["controls"][arch] = dict(acc=acc, conf=[cs, ct], lac=lac, lac_size=lsz, aps=aps, rho7=rc, beyond=b, beyond_raw=br,
                               lac_by_pair={p: float(cv[(arch, "LAC", p)]) for p in P}, aps_by_pair={p: float(cv[(arch, "APS", p)]) for p in P},
                               acc_by_pair={p: float(rs.loc[(arch, p)].acc_src) for p in P})
    if arch == "presto_frozen":
        L.append("\\midrule")
    L.append(neg(f"{name} & {inp} & {acc:.2f} & {cs:.2f} / {ct:.2f} & {lac:.2f} ({lsz:.1f}) & {aps[0]:.2f}, {aps[1]:.2f} & {rc:+.2f} & {btxt}\\\\"))
open(f"{OUT}/tables/controls.tex", "w").write("\n".join(L) + "\n")

# per-pair embedding distance and domain classifier, frozen Presto with and without location
L = [HEAD, "\\midrule"]
for arch, lab in (("presto_frozen", "Real location"), ("presto_frozen_noloc", "Constant location")):
    L.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{lab}}}}}\\\\")
    L.append("\\quad confidence, source / target & " + " & ".join(f"{rs.loc[(arch,p)].conf_src_nat:.2f} / {rs.loc[(arch,p)].conf_tgt_all:.2f}" for p in P) + "\\\\")
    L.append("\\quad beyond source 99\\%, standardised / raw & " + " & ".join(f"{es.loc[(arch,p)].tgt_beyond_src_q99_all:.2f} / {es.loc[(arch,p)].tgt_beyond_src_q99_raw_all:.2f}" for p in P) + "\\\\")
    L.append("\\quad domain-classifier AUC & " + " & ".join(f"{rs.loc[(arch,p)].domain_auc_all:.3f}" for p in P) + "\\\\")
    if "src_q99_z" in es.columns:
        L.append("\\quad source 99th percentile, standardised & " + " & ".join(f"{es.loc[(arch,p)].src_q99_z:.1f}" for p in P) + "\\\\")
        L.append("\\quad mean distance, source / target & " + " & ".join(f"{es.loc[(arch,p)].z_norm_src:.1f} / {es.loc[(arch,p)].z_norm_tgt_all:.1f}" for p in P) + "\\\\")
    fs = CBc[(CBc.arch == arch) & (CBc.space == "source") & (CBc.design == "exchangeable") & (CBc.score == "LAC") & (CBc.method == "covshift_weighted_all")].groupby("pair").full.mean()
    L.append("\\quad cov.-shift weighting, full sets & " + " & ".join(f"{fs[p]:.2f}" for p in P) + "\\\\")
open(f"{OUT}/tables/presto_pairs.tex", "w").write("\n".join(L) + "\n")
N["presto_pairs"] = {f"{arch}|{p}": dict(conf=[float(rs.loc[(arch, p)].conf_src_nat), float(rs.loc[(arch, p)].conf_tgt_all)],
                                         conf_known=[float(rs.loc[(arch, p)].conf_src), float(rs.loc[(arch, p)].conf_tgt)],
                                         beyond=float(es.loc[(arch, p)].tgt_beyond_src_q99_all), beyond_raw=float(es.loc[(arch, p)].tgt_beyond_src_q99_raw_all),
                                         beyond_known=float(es.loc[(arch, p)].tgt_beyond_src_q99), beyond_raw_known=float(es.loc[(arch, p)].tgt_beyond_src_q99_raw),
                                         z_src=float(es.loc[(arch, p)].z_norm_src), z_tgt=float(es.loc[(arch, p)].z_norm_tgt),
                                         raw_src=float(es.loc[(arch, p)].raw_norm_src), raw_tgt=float(es.loc[(arch, p)].raw_norm_tgt),
                                         ml=[float(es.loc[(arch, p)].max_logit_src), float(es.loc[(arch, p)].max_logit_tgt)],
                                         auc=float(rs.loc[(arch, p)].domain_auc_all))
                     for arch in ("presto_frozen", "presto_frozen_noloc") for p in P}
if "src_q99_z" in es.columns:        # the thresholds themselves, overall and per source country
    N["q99"] = {f"{arch}|{p}": {c: float(es.loc[(arch, p)][c]) for c in es.columns
                                if c.startswith(("src_q99", "src_mean_z", "tgt_beyond_src_q99_all")) and np.isfinite(es.loc[(arch, p)][c])}
                for arch in ("presto_frozen", "presto_frozen_noloc") for p in P}
for arch in ("presto_frozen", "presto_frozen_noloc", "presto_finetune", "transformer"):
    fs = CBc[(CBc.arch == arch) & (CBc.space == "source") & (CBc.design == "exchangeable") & (CBc.score == "LAC") & (CBc.method == "covshift_weighted_all")].groupby("pair").full.mean()
    N[f"covshift_full|{arch}"] = {p: float(fs[p]) for p in P}
N["temperature_ts"] = float(EXTc[EXTc.arch == "presto_frozen_ts"].temperature.mean())

# ================= method tables =================
METHODS = [("standard", "Standard split CP"),
           ("standard_natcal", "\\quad calibrated at natural frequencies"),
           ("region_robust", "Region-robust"),
           ("class_region_robust", "Class-region-robust"),
           ("covshift_weighted_all", "Cov.-shift weighted, $w\\in[0.01,100]$"),
           ("covshift_clip10_all", "\\quad clipped to $w\\in[0.1,10]$"),
           ("knn_local", "$k$-NN localized"),
           ("label_shift_bbse_all", "BBSE weighted"),
           ("em_label_shift_all", "EM-adjusted, label-shift weighted"),
           ("qtc_all", "QTC"),
           *([("qtc_natsrc_all", "\\quad source confidences at natural frequencies")] if FULL else []),
           ("oracle_target_labels", "Oracle, one threshold"),
           ("oracle_natcal", "\\quad calibrated at natural frequencies"),
           ("mondrian_oracle", "Oracle, per region")]
GROUP = {"standard": "\\multicolumn{8}{l}{\\textit{Calibrated on source labels only}}\\\\",
         "covshift_weighted_all": "\\midrule\n\\multicolumn{8}{l}{\\textit{Also use unlabeled target parcels (all crops, natural frequencies)}}\\\\",
         "oracle_target_labels": "\\midrule\n\\multicolumn{8}{l}{\\textit{References calibrated on target labels}}\\\\"}
IS_REF = lambda m: m.startswith("oracle") or m == "mondrian_oracle"               # noqa: E731
A_ALL = {}


def method_table(space, score, fname, methods=METHODS):
    d = ex[(ex.space == space) & (ex.score == score)]
    a = d.groupby(["pair", "method"]).agg(cov=("coverage", "mean"), nat=("coverage_nat", "mean"), size=("size", "mean"),
                                          full=("full", "mean"), res=("restores", "mean"), wr=("worst_region_cov", "mean")).reset_index()
    nm = d.groupby(["pair", "method", "arch"]).restores.mean().ge(0.5).groupby(["pair", "method"]).sum()
    L = [HEAD, "\\midrule"]
    for m, lab in methods:
        if m in GROUP:
            L.append(GROUP[m])
        cells = []
        for p in P:
            r = a[(a.pair == p) & (a.method == m)].iloc[0]
            t = f"{r['cov']:.2f} / {r['nat']:.2f} ({r['size']:.1f})"
            if r.full > 0.5:
                t += "$^\\dagger$"
            if r.res >= 0.5 and not IS_REF(m):
                t = "\\textbf{" + t + "}"
            k = int(nm[(p, m)])
            if not IS_REF(m) and k > 0:
                t += f"$^{{{k}}}$"
            cells.append(t)
        L.append(lab + " & " + " & ".join(cells) + "\\\\")
    open(f"{OUT}/tables/{fname}.tex", "w").write("\n".join(L) + "\n")
    A_ALL[(space, score)] = (a, nm)
    return a, nm


for space, score, fname in (("source", "LAC", "lac_methods"), ("shared", "LAC", "lac_methods_shared"), ("source", "APS", "aps_methods"),
                            ("source", "APS_rand", "apsrand_methods"), ("source", "RAPS", "raps_methods")):
    method_table(space, score, fname)
A_LAC, NM_LAC = A_ALL[("source", "LAC")]
for out_name, parts in (("methods_other_a", (("lac_methods_shared", "LAC, shared label space"), ("apsrand_methods", "Randomized APS"))),
                        ("methods_other_b", (("raps_methods", "RAPS"), ("aps_methods", "Non-randomized APS")))):
    L = [HEAD, "\\midrule"]
    for k, (fname, lab) in enumerate(parts):
        body = open(f"{OUT}/tables/{fname}.tex").read().split("\n")[2:]
        L += (["\\midrule\\midrule"] if k else []) + [f"\\multicolumn{{8}}{{c}}{{\\textbf{{{lab}}}}}\\\\", "\\midrule"] + [x for x in body if x]
    open(f"{OUT}/tables/{out_name}.tex", "w").write("\n".join(L) + "\n")
N["methods"] = {f"{sp}|{sc}|{m}|{p}": dict(cov=float(r["cov"]), nat=float(r.nat), size=float(r["size"]), full=float(r.full),
                                         res=float(r.res), wr=float(r.wr), n_models=int(A_ALL[(sp, sc)][1][(p, m)]))
                for (sp, sc), (a, _) in A_ALL.items() for _, r in a.iterrows() for m, p in [(r.method, r.pair)]}
# pool of evaluated crops (needs target labels) vs pool of all target parcels
OLD_NEW = [("covshift_weighted", "covshift_weighted_all"), ("covshift_clip10", "covshift_clip10_all"),
           ("label_shift_bbse", "label_shift_bbse_all"), ("em_label_shift", "em_label_shift_all"), ("qtc", "qtc_all")]
L = [HEAD, "\\midrule"]
for o, n in OLD_NEW:
    lab = dict(METHODS)[n].replace("\\quad ", "")
    L.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{lab}}}}}\\\\")
    for m, tag in ((o, "pool of evaluated crops (needs target labels)"), (n, "pool of all target parcels")):
        cells = []
        for p in P:
            r = A_LAC[(A_LAC.pair == p) & (A_LAC.method == m)].iloc[0]
            cells.append(f"{r['cov']:.2f} ({r['size']:.1f}){'$^\\dagger$' if r.full > 0.5 else ''}" + (f"$^{{{int(NM_LAC[(p, m)])}}}$" if NM_LAC[(p, m)] else ""))
        L.append(f"\\quad {tag} & " + " & ".join(cells) + "\\\\")
open(f"{OUT}/tables/pool_comparison.tex", "w").write("\n".join(L) + "\n")

# success rule applied to natural-frequency coverage (LAC, source label space): models restored, balanced / natural
lx = ex[(ex.space == "source") & (ex.score == "LAC")].copy()
lx["restores_nat"] = (lx.coverage_nat >= 0.88) & (lx.full <= 0.10) & (lx["size"] <= 1.25 * lx.oracle_size)
nb_ = lx.groupby(["method", "pair", "arch"]).restores.mean().ge(0.5).groupby(["method", "pair"]).sum()
nn_ = lx.groupby(["method", "pair", "arch"]).restores_nat.mean().ge(0.5).groupby(["method", "pair"]).sum()
L = [HEAD, "\\midrule"]
N["restores_nat"] = {}
for m, lab in METHODS:
    if IS_REF(m):
        continue
    L.append(lab + " & " + " & ".join(f"{int(nb_[(m, p)])} / {int(nn_[(m, p)])}" for p in P) + "\\\\")
    N["restores_nat"][m] = {p: [int(nb_[(m, p)]), int(nn_[(m, p)])] for p in P}
open(f"{OUT}/tables/restores_nat.tex", "w").write("\n".join(L) + "\n")

# ================= scores =================
L = [HEAD, "\\midrule"]
agg_s = ex[ex.space == "source"].groupby(["score", "method", "pair"]).agg(cov=("coverage", "mean"), size=("size", "mean"),
                                                                       res=("restores", "mean"), absent=("absent_in_set", "mean")).reset_index()
for s, lab in (("LAC", "LAC"), ("APS_rand", "APS, randomized"), ("RAPS", "RAPS"), ("APS", "APS, non-randomized")):
    gs = lambda m, p, c: agg_s[(agg_s.score == s) & (agg_s.method == m) & (agg_s.pair == p)][c].iloc[0]   # noqa: E731
    L.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{lab}}}}}\\\\")
    L.append("\\quad in source: set size & " + " & ".join(f"{pairmean(src, 'src_size_' + s)[p]:.1f}" for p in P) + "\\\\")
    L.append("\\quad in target: source-calibrated & " + " & ".join(
        ("\\textbf{%s}" if gs("standard", p, "res") >= 0.5 else "%s") % f"{gs('standard',p,'cov'):.2f} ({gs('standard',p,'size'):.1f})" for p in P) + "\\\\")
    L.append("\\quad in target: oracle & " + " & ".join(f"{gs('oracle_target_labels',p,'cov'):.2f} ({gs('oracle_target_labels',p,'size'):.1f})" for p in P) + "\\\\")
L.append("\\midrule")
gs = lambda m, p, c: agg_s[(agg_s.score == "LAC") & (agg_s.method == m) & (agg_s.pair == p)][c].iloc[0]   # noqa: E731
L.append("LAC oracle: crops in set never seen in target & " + " & ".join(f"{gs('oracle_target_labels',p,'absent'):.1f}" for p in P) + "\\\\")
open(f"{OUT}/tables/scores.tex", "w").write("\n".join(L) + "\n")
N["oracle_size"] = {p: float(gs("oracle_target_labels", p, "size")) for p in P}
N["oracle_absent"] = {p: float(gs("oracle_target_labels", p, "absent")) for p in P}

# ================= target labels (few-label summaries per model group) =================
def k_of(m):
    return math.ceil((m + 1) * 0.9)


def bp10(m):
    return beta_dist.ppf(0.10, k_of(m), m + 1 - k_of(m))


GKEY = ["pair", "space", "design", "sampling", "score", "m"]
few2 = FEWA[FEWA.arch.isin(["transformer", "tempcnn"])].groupby(GKEY).mean(numeric_only=True).reset_index()
few3 = FEWA[FEWA.arch.isin(["presto_frozen", "presto_finetune", "random_forest"])].groupby(GKEY).mean(numeric_only=True).reset_index()
f2 = few2[(few2.space == "source") & (few2.score == "LAC")]; f3 = few3[(few3.space == "source") & (few3.score == "LAC")]
fg = lambda f, d, s, m, c, p: f[(f.design == d) & (f.sampling == s) & (f.m == m) & (f.pair == p)][c].iloc[0]   # noqa: E731
fpool = lambda d, s, m, c, p: (6 * fg(f2, d, s, m, c, p) + 9 * fg(f3, d, s, m, c, p)) / 15                      # noqa: E731
L = [" & theory & " + " & ".join(NAME[p] for p in P) + "\\\\", "\\midrule",
     "\\multicolumn{9}{l}{\\textit{10th-percentile coverage, $m$ random labels from the deployment regions}}\\\\"]
for m in (10, 50, 200):
    L.append(f"$m={m}$, transformer, TempCNN & {bp10(m):.2f} & " + " & ".join(f"{fg(f2,'exchangeable','random',m,'cov_p10',p):.2f}" for p in P) + "\\\\")
    L.append(f"$m={m}$, Presto, random forest & {bp10(m):.2f} & " + " & ".join(f"{fg(f3,'exchangeable','random',m,'cov_p10',p):.2f}" for p in P) + "\\\\")
L.append("\\multicolumn{9}{l}{\\textit{Mean coverage, $m$ labels from other regions of the target (spatially disjoint); theory: $k/(m+1)$}}\\\\")
for m in (25, 50, 200):
    L.append(f"$m={m}$ & {k_of(m)/(m+1):.2f} & " + " & ".join(f"{fpool('disjoint','other_regions',m,'cov_mean',p):.2f}" for p in P) + "\\\\")
orc_size = lambda p: A_ALL[("source", "LAC")][0].query("pair == @p and method == 'oracle_target_labels'")["size"].iloc[0]   # noqa: E731
L.append("\\multicolumn{9}{l}{\\textit{Set size relative to the oracle, $m$ labels from other regions of the target}}\\\\")
for m in (25, 200):
    L.append(f"$m={m}$ & & " + " & ".join(f"{fpool('disjoint','other_regions',m,'size_mean',p) / orc_size(p):.2f}" for p in P) + "\\\\")
open(f"{OUT}/tables/labels.tex", "w").write("\n".join(L) + "\n")
for tag, f in (("seq", f2), ("new", f3)):
    fx = f[(f.design == "exchangeable") & (f.sampling == "random")]
    N[f"beta_dev_{tag}"] = float(max(abs(r.cov_p10 - bp10(int(r.m))) for r in fx.itertuples()))
    fd = f[(f.design == "disjoint") & (f.sampling == "other_regions")]
    N[f"disjoint_mean_range_{tag}"] = [float(fd.cov_mean.min()), float(fd.cov_mean.max())]
    N[f"disjoint_p10_m50_{tag}"] = {p: float(fg(f, "disjoint", "other_regions", 50, "cov_p10", p)) for p in P}
for tag, ff in (("seq", few2), ("new", few3)):
    fx = ff[(ff.design == "exchangeable") & (ff.sampling == "random")]
    N[f"beta_dev_all_{tag}"] = float(max(abs(r.cov_p10 - bp10(int(r.m))) for r in fx.itertuples()))
N["disjoint_pooled"] = {f"{p}|{m}": float(fpool("disjoint", "other_regions", m, "cov_mean", p)) for p in P for m in (25, 50, 100, 200)}
N["disjoint_minus_theory"] = {f"{p}|{m}": float(fpool("disjoint", "other_regions", m, "cov_mean", p) - k_of(m) / (m + 1))
                              for p in P for m in (25, 50, 100, 200)}
N["size_ratio_disjoint"] = {p: {m: float(fpool("disjoint", "other_regions", m, "size_mean", p) /
                                         A_LAC[(A_LAC.pair == p) & (A_LAC.method == "oracle_target_labels")]["size"].iloc[0]) for m in (25, 200)} for p in P}

# ================= regional guarantees: Mondrian vs PAC per region (five main models) =================
fr = FEWA[FEWA.arch.isin(MAIN) & (FEWA.space == "source") & (FEWA.score == "LAC") & FEWA.sampling.str.startswith("mondrian")]
fr = fr.groupby(["pair", "sampling", "m"]).mean(numeric_only=True).reset_index()
sim = np.random.default_rng(0)


def mondrian_theory(m):
    k = k_of(m)
    return float((sim.binomial(500, beta_dist.rvs(k, m + 1 - k, size=(20000, 10), random_state=m)) / 500).min(1).mean()) if k <= m else 1.0


L = ["Labels per region & \\multicolumn{3}{c}{Split conformal per region} & \\multicolumn{3}{c}{PAC per region ($\\delta=0.1$)}\\\\",
     "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
     "$m$ & worst region (theory) & all regions $\\ge 0.90$ & set size & worst region & all regions $\\ge 0.90$ & set size\\\\", "\\midrule"]
N["regional"] = {}
for m in (25, 50, 100, 200):
    s1 = fr[(fr.sampling == ("mondrian_per_region" if m <= 50 else "mondrian_per_region_more")) & (fr.m == m)]
    s1b = fr[(fr.sampling == "mondrian_per_region_more") & (fr.m == m)]
    s2 = fr[(fr.sampling == "mondrian_per_region_pac") & (fr.m == m)]
    th = mondrian_theory(m)
    N["regional"][m] = dict(std_worst=float(s1.worst_region_mean.mean()), std_all90=float(s1b.worst_ge_nominal.mean()), std_size=float(s1.size_mean.mean()),
                            pac_worst=float(s2.worst_region_mean.mean()), pac_all90=float(s2.worst_ge_nominal.mean()), pac_size=float(s2.size_mean.mean()),
                            theory=th)
    r = N["regional"][m]
    L.append(f"{m} & {r['std_worst']:.2f} ({th:.2f}) & {r['std_all90']:.2f} & {r['std_size']:.1f} & {r['pac_worst']:.2f} & {r['pac_all90']:.2f} & {r['pac_size']:.1f}\\\\")
L.append("\\midrule")
L.append("10 & " + f"{fr[(fr.sampling=='mondrian_per_region')&(fr.m==10)].worst_region_mean.mean():.2f} ({mondrian_theory(10):.2f}) & -- & "
         f"{fr[(fr.sampling=='mondrian_per_region')&(fr.m==10)].size_mean.mean():.1f} & \\multicolumn{{3}}{{c}}{{no PAC threshold exists for $m<22$}}\\\\")
open(f"{OUT}/tables/regional.tex", "w").write("\n".join(L) + "\n")
N["regional_m10"] = dict(worst=float(fr[(fr.sampling == "mondrian_per_region") & (fr.m == 10)].worst_region_mean.mean()), theory=mondrian_theory(10))

# ================= recalibration vs adaptation =================
ad = ADP[ADP.arch.isin(MAIN)].copy()
ad["grp"] = np.where(ad.pair.isin(PT3), "pt", np.where(ad.pair == "PTnorth_to_PTsouth", "ptns", "other"))
if "design" not in ad.columns:
    ad["design"] = "exchangeable"
VAR = [("source_calibration", "Source calibration, no target labels"), ("recalibrate_m", "Recalibrate with $m$"),
       ("recalibrate_half", "Recalibrate with $m/2$"), ("adapt_m_source_calibration", "Refit head with $m$, source calibration"),
       ("adapt_half_recalibrate_half", "Refit head with $m/2$, recalibrate with $m/2$")]
GRP = [("other", "Baltic and within Latvia"), ("ptns", "PT north$\\to$south"), ("pt", "Cross-country Portugal")]
L = [" & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{g}}}" for _, g in GRP) + "\\\\",
     "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
     "Use of $m$ target labels" + " & acc. & cov. & size" * 3 + "\\\\", "\\midrule"]
N["adapt"] = {}
DES = [("exchangeable", "labels from the deployment regions")] + ([("disjoint", "labels from other regions of the target")] if (ad.design == "disjoint").any() else [])
for des, dlab in DES:
    for m in (50, 200):
        L.append(f"\\multicolumn{{10}}{{l}}{{\\textit{{$m={m}$, {dlab}}}}}\\\\")
        for v, lab in VAR:
            if not ((ad.design == des) & (ad.variant == v)).any():
                continue
            cells = []
            for grp, _ in GRP:
                x = ad[(ad.design == des) & (ad.m == m) & (ad.variant == v) & (ad.grp == grp)]
                N["adapt"][f"{des}|{m}|{v}|{grp}"] = dict(acc=float(x.acc.mean()), cov=float(x.coverage.mean()), nat=float(x.coverage_nat.mean()),
                                                          size=float(x["size"].mean()), wr=float(x.worst_region_cov.mean()))
                cells.append(f"{x.acc.mean():.2f} & {x.coverage.mean():.2f} & {x['size'].mean():.1f}")
            L.append(f"\\quad {lab} & " + " & ".join(cells) + "\\\\")
open(f"{OUT}/tables/adapt.tex", "w").write("\n".join(L) + "\n")
N["adapt_by_arch_pt200"] = {a: {v: float(ad[(ad.design == "exchangeable") & (ad.arch == a) & (ad.m == 200) & (ad.grp == "pt") & (ad.variant == v)]["size"].mean())
                                for v, _ in VAR} for a in MAIN}

# ================= label-space remedies: exact codes, merged codes, OTHER class (transformer, source) =================
LV = [("transformer", "Exact HCAT codes"), ("transformer_merged6", "Codes merged to crop groups"), ("transformer_openset", "Exact codes + OTHER class")]
cl = CBc[(CBc.space == "source") & (CBc.design == "exchangeable") & (CBc.score == "LAC")]
rl = RIc[RIc.space == "source"].groupby(["arch", "pair"]).mean(numeric_only=True)
gl = lambda a, p, m, c: cl[(cl.arch == a) & (cl.pair == p) & (cl.method == m)][c].mean()   # noqa: E731
L = [HEAD, "\\midrule"]
N["labelspace"] = {}
for a, lab in LV:
    L.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{lab}}}}}\\\\")
    have = [p for p in P if (a, p) in rl.index]
    cell = lambda f: [f(p) if p in have else "--" for p in P]                               # noqa: E731
    L.append("\\quad crops known / unknown parcels & " + " & ".join(cell(lambda p: f"{rl.loc[(a,p)].K:.0f} / {rl.loc[(a,p)].unknown_frac_nat:.2f}")) + "\\\\")
    L.append("\\quad deployment coverage, standard (set size) & " + " & ".join(cell(lambda p: f"{gl(a,p,'standard','coverage_nat')*rl.loc[(a,p)].eval_frac_nat:.2f} ({gl(a,p,'standard','size'):.1f})")) + "\\\\")
    L.append("\\quad deployment coverage, oracle (set size) & " + " & ".join(cell(lambda p: f"{gl(a,p,'oracle_target_labels','coverage_nat')*rl.loc[(a,p)].eval_frac_nat:.2f} ({gl(a,p,'oracle_target_labels','size'):.1f})")) + "\\\\")
    if a == "transformer_openset" and "other_in_set_known_standard" in rl.columns:
        L.append("\\quad sets with OTHER, known / unknown crops & " + " & ".join(cell(lambda p: f"{rl.loc[(a,p)].other_in_set_known_standard:.2f} / {rl.loc[(a,p)].other_in_set_unknown_standard:.2f}")) + "\\\\")
        L.append("\\quad sets with OTHER alone, unknown crops & " + " & ".join(cell(lambda p: f"{rl.loc[(a,p)].other_only_unknown_standard:.2f}")) + "\\\\")
        N["other_in_set"] = {p: {c: float(rl.loc[(a, p)][c]) for c in rl.columns if c.startswith(("other_", "size_known", "size_unknown", "unknown_share_eval"))}
                             for p in have}
    for p in have:
        N["labelspace"][f"{a}|{p}"] = dict(K=float(rl.loc[(a, p)].K), unknown=float(rl.loc[(a, p)].unknown_frac_nat), eval=float(rl.loc[(a, p)].eval_frac_nat),
                                           acc_tgt=float(rl.loc[(a, p)].acc_tgt), cov_nat=float(gl(a, p, "standard", "coverage_nat")),
                                           deploy=float(gl(a, p, "standard", "coverage_nat") * rl.loc[(a, p)].eval_frac_nat),
                                           deploy_oracle=float(gl(a, p, "oracle_target_labels", "coverage_nat") * rl.loc[(a, p)].eval_frac_nat),
                                           size=float(gl(a, p, "standard", "size")), size_oracle=float(gl(a, p, "oracle_target_labels", "size")))
open(f"{OUT}/tables/labelspace.tex", "w").write("\n".join(L) + "\n")

# ================= EM / BBSE sanity =================
L = [HEAD, "\\midrule", "\\multicolumn{8}{l}{\\textit{Prior error (total variation), unlabeled pool of evaluated crops}}\\\\"]
for col, lab in (("noadj_tv", "no adjustment (source prior)"), ("em_prior_tv", "EM"), ("bbse_prior_tv", "BBSE")):
    L.append(f"\\quad {lab} & " + " & ".join(f"{pairmean(src, col)[p]:.2f}" for p in P) + "\\\\")
L.append("\\multicolumn{8}{l}{\\textit{Prior error, unlabeled pool of all target parcels (natural frequencies)}}\\\\")
for col, lab in (("noadj_tv_all", "no adjustment (source prior)"), ("em_prior_tv_all", "EM"), ("bbse_prior_tv_all", "BBSE")):
    L.append(f"\\quad {lab} & " + " & ".join(f"{pairmean(src, col)[p]:.2f}" for p in P) + "\\\\")
# an exactly singular confusion matrix has condition number inf, which the run files store as missing
cond_ = src.bbse_cond.fillna(np.inf)
log_med = lambda p: float(np.log10(cond_[src.pair == p].median()))                # noqa: E731
fmt_l = lambda v: "$\\infty$" if not np.isfinite(v) else f"{v:.1f}"             # noqa: E731
L.append("BBSE confusion matrix: median $\\log_{10}$ condition number & " + " & ".join(fmt_l(log_med(p)) for p in P) + "\\\\")
L.append("\\quad share of runs singular or above $10^4$ & " + " & ".join(f"{(cond_[src.pair == p] > 1e4).mean():.2f}" for p in P) + "\\\\")
L.append("\\quad share of runs exactly singular & " + " & ".join(f"{(~np.isfinite(cond_[src.pair == p])).mean():.2f}" for p in P) + "\\\\")
SEQ = src.arch.isin(["transformer", "tempcnn"])
for msk, lab in ((SEQ, "transformer, TempCNN"), (~SEQ, "Presto, random forest")):
    L.append(f"\\quad singular or above $10^4$, {lab} & " + " & ".join(f"{(cond_[(src.pair == p) & msk] > 1e4).mean():.2f}" for p in P) + "\\\\")
N["bbse_group_above_1e4"] = {"seq": float((cond_[SEQ] > 1e4).mean()), "presto_rf": float((cond_[~SEQ] > 1e4).mean())}
open(f"{OUT}/tables/em_bbse.tex", "w").write("\n".join(L) + "\n")
sy = SYN[SYN.arch.isin(MAIN) & (SYN.space == "source")]
L = ["Model & TV, no adj. & TV, BBSE & TV, EM & standard & BBSE & EM & true weights\\\\", "\\midrule"]
for a in MAIN:
    x = sy[sy.arch == a].mean(numeric_only=True)
    L.append(f"{ANAME[a]} & {x.tv_noadj:.2f} & {x.tv_bbse:.2f} & {x.tv_em:.2f} & {x.cov_standard:.2f} ({x.size_standard:.1f}) & "
             f"{x.cov_bbse_weighted:.2f} ({x.size_bbse_weighted:.1f}) & {x.cov_em_label_shift:.2f} ({x.size_em_label_shift:.1f}) & "
             f"{x.cov_true_weights:.2f} ({x.size_true_weights:.1f})\\\\")
open(f"{OUT}/tables/synthetic.tex", "w").write("\n".join(L) + "\n")
EMD = [("standard", "Standard (no adjustment)"), ("em_label_shift", "EM-adjusted scores, EM weights"),
       ("em_weights_plain", "Plain scores, EM weights"), ("true_weights", "Plain scores, true weights"),
       ("true_weights_emadj", "EM-adjusted scores, true weights"), ("true_weights_trueadj", "Truly adjusted scores, true weights")]
if "cov_em_weights_plain" in sy.columns:
    L = ["Scores and weights & " + " & ".join(ANAME[a] for a in MAIN) + "\\\\", "\\midrule",
         "\\multicolumn{6}{l}{\\textit{Synthetic pure label shift}}\\\\"]
    for k, lab in EMD:
        L.append(f"\\quad {lab} & " + " & ".join(f"{sy[sy.arch == a]['cov_' + k].mean():.2f} ({sy[sy.arch == a]['size_' + k].mean():.1f})" for a in MAIN) + "\\\\")
    REAL = [("standard", "Standard (no adjustment)"), ("em_label_shift_all", "EM-adjusted scores, EM weights"),
            ("em_weights_plain_all", "Plain scores, EM weights"), ("true_label_weights", "Plain scores, true weights"),
            ("true_weights_emadj_all", "EM-adjusted scores, true weights")]
    rl_ = ex[(ex.space == "source") & (ex.score == "LAC")]
    for grp, glab in ((P[:3], "Real transfers: Baltic and within Latvia"), (P[3:4], "Real transfers: PT north$\\to$south"), (PT3, "Real transfers: cross-country Portugal")):
        L.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{glab}}}}}\\\\")
        for k, lab in REAL:
            L.append(f"\\quad {lab} & " + " & ".join(
                f"{rl_[(rl_.arch == a) & rl_.pair.isin(grp) & (rl_.method == k)].coverage.mean():.2f} ({rl_[(rl_.arch == a) & rl_.pair.isin(grp) & (rl_.method == k)]['size'].mean():.1f})"
                for a in MAIN) + "\\\\")
    open(f"{OUT}/tables/em_decomp.tex", "w").write("\n".join(L) + "\n")
    N["em_decomp_syn"] = {a: {k: float(sy[sy.arch == a]["cov_" + k].mean()) for k, _ in EMD} for a in MAIN}
    N["em_decomp_real"] = {f"{a}|{p}": {k: float(rl_[(rl_.arch == a) & (rl_.pair == p) & (rl_.method == k)].coverage.mean()) for k, _ in REAL}
                           for a in MAIN for p in P}
N["em"] = {p: dict(noadj=float(pairmean(src, "noadj_tv")[p]), em=float(pairmean(src, "em_prior_tv")[p]), bbse=float(pairmean(src, "bbse_prior_tv")[p]),
                   noadj_all=float(pairmean(src, "noadj_tv_all")[p]), em_all=float(pairmean(src, "em_prior_tv_all")[p]),
                   bbse_all=float(pairmean(src, "bbse_prior_tv_all")[p]), log10_cond_median=log_med(p),
                   share_above_1e4=float((cond_[src.pair == p] > 1e4).mean()), share_singular=float((~np.isfinite(cond_[src.pair == p])).mean())) for p in P}
N["synthetic"] = {a: {k: float(v) for k, v in sy[sy.arch == a].mean(numeric_only=True).items() if k != "seed"} for a in MAIN}
N["bbse_by_arch_above_1e4"] = {a: float((cond_[src.arch == a] > 1e4).mean()) for a in MAIN}

# ================= appendix: per condition, deployment, bootstrap, clip =================
pc = ex[(ex.score == "LAC") & (ex.method == "standard")].groupby(["pair", "space", "arch"]).agg(
    m=("coverage", "mean"), s=("coverage", "std"), n=("coverage_nat", "mean")).reset_index()
L = ["Pair & Space & " + " & ".join(ANAME[a] for a in MAIN) + "\\\\", "\\midrule"]
for p in P:
    for spc in ("source", "shared"):
        cells = [(lambda r: f"{r['m']:.3f} $\\pm$ {r['s']:.3f} / {r['n']:.3f}")(pc[(pc.pair == p) & (pc.space == spc) & (pc.arch == a)].iloc[0]) for a in MAIN]
        L.append((NAME[p] if spc == "source" else "") + f" & {spc} & " + " & ".join(cells) + "\\\\")
open(f"{OUT}/tables/per_condition.tex", "w").write("\n".join(L) + "\n")
N["shared_minus_source_by_model"] = {f"{a}|{p}": float(pc[(pc.pair == p) & (pc.space == "shared") & (pc.arch == a)].m.iloc[0] -
                                                       pc[(pc.pair == p) & (pc.space == "source") & (pc.arch == a)].m.iloc[0]) for a in MAIN for p in P}

dep = ex[(ex.space == "source") & (ex.score == "LAC")].merge(ri[["pair", "space", "arch", "seed", "eval_frac_nat"]], on=["pair", "space", "arch", "seed"])
dep["deploy"] = dep.coverage_nat * dep.eval_frac_nat
L = [HEAD, "\\midrule"]
for m, lab in (("standard", "Standard"), ("standard_natcal", "Standard, calibrated at natural frequencies"), ("oracle_target_labels", "Oracle")):
    d = dep[dep.method == m]
    for col, clab in (("coverage", "balanced crops"), ("coverage_nat", "natural frequencies"), ("deploy", "all target parcels")):
        L.append(f"{lab}, {clab} & " + " & ".join(f"{d[d.pair == p][col].mean():.2f}" for p in P) + "\\\\")
    L.append("\\midrule")
wc = lambda m, p: ex[(ex.space == "source") & (ex.score == "LAC") & (ex.method == m) & (ex.pair == p)].worst_class_cov.mean()  # noqa: E731
L.append("Worst crop, standard & " + " & ".join(f"{wc('standard', p):.2f}" for p in P) + "\\\\")
L.append("Worst crop, oracle & " + " & ".join(f"{wc('oracle_target_labels', p):.2f}" for p in P) + "\\\\")
L.append("Worst region in source, leave one out & " + " & ".join(f"{pairmean(src,'src_lro_worst_LAC')[p]:.2f}" for p in P) + "\\\\")
if FULL:
    L.append("\\midrule")
    L.append("In source, natural frequencies: standard / natural-frequency calibration & " + " & ".join(
        f"{pairmean(src,'src_cov_nat_LAC')[p]:.2f} / {pairmean(src,'src_cov_nat_LAC_natcal')[p]:.2f}" for p in P) + "\\\\")
    N["src_nat"] = {p: dict(std=float(pairmean(src, "src_cov_nat_LAC")[p]), natcal=float(pairmean(src, "src_cov_nat_LAC_natcal")[p]),
                            std_bal=float(pairmean(src, "src_cov_LAC")[p]), natcal_bal=float(pairmean(src, "src_cov_LAC_natcal")[p]),
                            size_std=float(pairmean(src, "src_size_nat_LAC")[p]), size_natcal=float(pairmean(src, "src_size_nat_LAC_natcal")[p])) for p in P}
open(f"{OUT}/tables/deployment.tex", "w").write("\n".join(L) + "\n")
N["deploy"] = {f"{m}|{p}": float(dep[(dep.method == m) & (dep.pair == p)].deploy.mean()) for m in ("standard", "standard_natcal", "oracle_target_labels") for p in P}
N["natfreq"] = {f"{m}|{p}": float(dep[(dep.method == m) & (dep.pair == p)].coverage_nat.mean()) for m in ("standard", "standard_natcal", "oracle_target_labels", "oracle_natcal") for p in P}
N["worst_crop_std"] = {p: float(wc("standard", p)) for p in P}

bt = ex[(ex.space == "source") & (ex.score == "LAC")].groupby(["pair", "method"]).agg(
    wr=("worst_region_cov", "mean"), lo=("wr_lo", "mean"), hi=("wr_hi", "mean")).reset_index()
wrs = WRS[WRS.arch.isin(MAIN) & (WRS.space == "source")]
fz = FLO[(FLO.space == "source") & (FLO.design == "exchangeable")].set_index("pair")
L = [HEAD, "\\midrule"]
N["wr"] = {}
for m, lab in (("standard", "Standard"), ("oracle_target_labels", "Oracle, one threshold"), ("mondrian_oracle", "Oracle, per region")):
    b = lambda p, c: bt[(bt.pair == p) & (bt.method == m)][c].iloc[0]                    # noqa: E731
    q = lambda p: np.percentile(wrs[(wrs.pair == p) & (wrs.method == m)].worst_region_cov, [5, 95])   # noqa: E731
    L.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{lab}}}}}\\\\")
    L.append("\\quad mean [parametric 95\\%] & " + " & ".join(f"{b(p,'wr'):.2f} [{b(p,'lo'):.2f}, {b(p,'hi'):.2f}]" for p in P) + "\\\\")
    L.append("\\quad 5--95\\% over runs and splits & " + " & ".join(f"{q(p)[0]:.2f}--{q(p)[1]:.2f}" for p in P) + "\\\\")
    for p in P:
        N["wr"][f"{m}|{p}"] = dict(mean=float(b(p, "wr")), lo=float(b(p, "lo")), hi=float(b(p, "hi")), q05=float(q(p)[0]), q95=float(q(p)[1]))
L.append("Chance level & " + " & ".join(f"{fz.loc[p,'floor_mean']:.2f}" for p in P) + "\\\\")
open(f"{OUT}/tables/bootstrap.tex", "w").write("\n".join(L) + "\n")
N["floor"] = {p: float(fz.loc[p, "floor_mean"]) for p in P}

L = [HEAD, "\\midrule"]
for c, lab in (("covshift_clip10_all", "[0.1, 10]"), ("covshift_weighted_all", "[0.01, 100]"), ("covshift_clip1000_all", "[0.001, 1000]")):
    a = ex[(ex.space == "source") & (ex.score == "LAC") & (ex.method == c)].groupby("pair").agg(
        cov=("coverage", "mean"), full=("full", "mean"), wn=("w_test_norm_median", "median")).reindex(P)
    L.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{$w\\in{lab}$}}}}\\\\")
    L.append("\\quad coverage & " + " & ".join(f"{a.loc[p,'cov']:.2f}" for p in P) + "\\\\")
    L.append("\\quad share of full sets & " + " & ".join(f"{a.loc[p,'full']:.2f}" for p in P) + "\\\\")
    L.append("\\quad median normalised test weight & " + " & ".join(f"{a.loc[p,'wn']:.4f}" for p in P) + "\\\\")
    N[f"clip|{c}"] = {p: dict(cov=float(a.loc[p, "cov"]), full=float(a.loc[p, "full"]), wn=float(a.loc[p, "wn"])) for p in P}
open(f"{OUT}/tables/clip.tex", "w").write("\n".join(L) + "\n")

# ================= more numbers for the text =================
N["qtc_beta_all"] = {f"{a}|{p}": float(src[(src.arch == a) & (src.pair == p)].qtc_beta_all.mean()) for a in MAIN for p in P}
if FULL:
    N["qtc_beta_natsrc_all"] = {f"{a}|{p}": float(src[(src.arch == a) & (src.pair == p)].qtc_beta_natsrc_all.mean()) for a in MAIN for p in P}
N["domain_auc_all"] = {f"{a}|{p}": float(src[(src.arch == a) & (src.pair == p)].domain_auc_all.mean()) for a in MAIN for p in P}
N["pred_tv_all"] = {f"{a}|{p}": float(src[(src.arch == a) & (src.pair == p)].pred_tv_all.mean()) for a in MAIN for p in P}
N["bcts_T"] = [float(RI.bcts_temperature.min()), float(RI.bcts_temperature.max())]
N["seq_gap"] = float(std[std.arch.isin(["transformer", "tempcnn"])].groupby(["pair", "space", "seed", "arch"]).coverage.mean()
                     .unstack().diff(axis=1).abs().iloc[:, 1].mean())
N["mondrian_by_arch"] = {a: [float(ex[(ex.arch == a) & (ex.space == "source") & (ex.score == "LAC") & (ex.method == "mondrian_oracle")].groupby("pair").worst_region_cov.mean().min()),
                             float(ex[(ex.arch == a) & (ex.space == "source") & (ex.score == "LAC") & (ex.method == "mondrian_oracle")].groupby("pair").worst_region_cov.mean().max())] for a in MAIN}
json.dump(N, open(f"{OUT}/paper_numbers.json", "w"), indent=1, default=float)

# ================= Figures (one analysis per figure, drawn at print size for a 131 mm text block) =================
COL = {"transformer": "#2a78d6", "tempcnn": "#1baf7a", "presto_frozen": "#eb6834", "presto_finetune": "#b04a9e", "random_forest": "#8a8a85"}
MK = {"transformer": "o", "tempcnn": "s", "presto_frozen": "D", "presto_finetune": "^", "random_forest": "v"}
INK, MUTED, GRID = "#2b2b2b", "#6b6b6b", "#e4e4e1"
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 9,
                     "axes.labelsize": 9, "axes.titlesize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                     "legend.fontsize": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "text.color": INK, "axes.linewidth": 0.7, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
W = 5.1  # inches: the figures are included at their natural size
for x, xl, name in (("conf_drop_all", "Drop in mean top-class confidence, source to target", "fig_confdrop"),
                    ("pred_tv_all", "Shift in predicted crop frequencies (TV distance)", "fig_predshift")):
    fig, ax = plt.subplots(figsize=(W, 3.0), layout="constrained")
    for a in MAIN:
        c = cond[cond.arch == a]
        ax.scatter(c[x], c.shortfall, s=30, marker=MK[a], color=COL[a], edgecolor="white", linewidth=0.6, label=ANAME[a], zorder=3)
    ax.axhline(0, color=MUTED, lw=0.8, ls=":")
    if x == "conf_drop_all":
        ax.axvline(0, color=MUTED, lw=0.8, ls=":")
    else:
        ax.set_xlim(left=0)
    ax.set_xlabel(xl); ax.set_ylabel("LAC coverage shortfall")
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, handletextpad=0.3)
    fig.savefig(f"{OUT}/figures/{name}.pdf"); fig.savefig(f"{OUT}/figures/{name}.png", dpi=200); plt.close(fig)
fig, ax = plt.subplots(figsize=(W, 2.9), layout="constrained")
wr = A_LAC.set_index(["pair", "method"]).wr; fls = fz.floor_mean
y = np.arange(len(P))[::-1]
for i, p in zip(y, P):
    ax.plot([wr[(p, "oracle_target_labels")], wr[(p, "mondrian_oracle")]], [i, i], color=GRID, lw=2.5, zorder=1)
    ax.scatter(wr[(p, "oracle_target_labels")], i, s=36, marker="o", facecolor="white", edgecolor=INK, linewidth=1.1, zorder=3,
               label="Oracle, one threshold" if i == y[0] else None)
    ax.scatter(wr[(p, "mondrian_oracle")], i, s=36, marker="o", color=INK, zorder=3, label="Oracle, one threshold per region" if i == y[0] else None)
    ax.plot([fls[p]] * 2, [i - 0.32, i + 0.32], color="#eb6834", lw=1.6, zorder=2, label="Chance level" if i == y[0] else None)
ax.set_yticks(y); ax.set_yticklabels([PLAIN[p] for p in P]); ax.set_xlim(0.7, 0.92)
ax.set_xlabel("Worst-region coverage")
ax.grid(True, axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
fig.savefig(f"{OUT}/figures/fig_worstregion.pdf"); fig.savefig(f"{OUT}/figures/fig_worstregion.png", dpi=200); plt.close(fig)
print("done:", len(os.listdir(f"{OUT}/tables")), "tables")
