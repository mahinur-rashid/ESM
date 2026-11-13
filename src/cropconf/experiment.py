"""One run = one (transfer pair, label space, model, seed): train on the source country, calibrate with every
method, evaluate on the target country under two split designs, and study few target labels.

With cfg["full_diagnostics"] (the default) the run also computes, each with its own random stream: label-free
methods and warning signs on the unlabeled pool of *all* target parcels (any crop) at natural frequencies (suffix
_all); calibration at natural crop frequencies (_natcal); QTC with natural-frequency source confidences; EM/BBSE checks,
EM decomposed into its weights and its score adjustment, and a synthetic pure label-shift test; training-conditional
(PAC) regional thresholds; recalibration versus refitting a linear head with the same target labels (from the
deployment regions or from other regions); embedding-distance thresholds; OTHER-class diagnostics. Without it the
run computes only the methods on the pool of evaluated crops, which is faster. Model names ending in _merged6 train on HCAT codes merged
to six digits (crop groups); names ending in _openset add an explicit "OTHER" class (all source crops below the
known-crop threshold), and a set then covers a target parcel of an unknown crop if it contains OTHER."""
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conformal import (adjust_probs, bbse_weights, bbse_weights_w, class_region_robust_q, cp_quantile, em_priors,
                        em_priors_w, eval_sets, fast_eval, few_summary, fit_bcts, fit_domain, fit_domain_w, fit_domain_ww, floor_stats,
                        knn_q, label_weighted_sets, make_regions, qtc_beta, qtc_beta_w, qtc_beta_ww, region_cov, region_q,
                        region_q_pac, region_robust_q, score_functions, stratified_pick, tv, weighted_q)

OTHER = "OTHER"


def parse_arch(arch):
    """Experiment variant encoded in the model name: label level (10 = exact HCAT code) and open-set class."""
    return dict(level=6 if arch.endswith("_merged6") else 10, open_set=arch.endswith("_openset"))


def run_key(pname, space, arch, seed):
    return f"{pname}__{space}__{arch}__s{seed}"


def pair_ids(data, src, tgt, split):
    """Indices of the source and target parcels of a pair, before crop filtering."""
    if split is None:
        return np.where(np.isin(data.country, src))[0], np.where(data.country == tgt)[0]
    ids = np.where(data.country == tgt)[0]
    coord = data.center[ids, 1] if split["axis"] == "lat" else data.center[ids, 0]
    hi = coord >= np.nanmedian(coord)
    s_side = hi if split["src"] == "high" else ~hi
    return ids[s_side], ids[~s_side]


def classes_for(codes, src_ids, tgt_ids, space, cfg):
    """Crops the model knows (source: >= min_src_per_class source parcels; shared: also >= min_tgt_per_class in the
    target), and the crops present in the target."""
    n_src = pd.Series(codes[src_ids]).value_counts(); n_tgt = pd.Series(codes[tgt_ids]).value_counts()
    src_ok = set(n_src.index[n_src >= cfg["min_src_per_class"]])
    tgt_ok = set(n_tgt.index[n_tgt >= cfg["min_tgt_per_class"]])
    return (sorted(src_ok & tgt_ok) if space == "shared" else sorted(src_ok)), tgt_ok


def summarize_extra(extra, n_cal, n_test, n_tgt, P_u, w_all=None, src_country=None):
    """Per-run diagnostics: confidence and, for frozen-Presto probes, embedding distance from the source mean.
    Target = parcels of evaluated crops; with w_all, also all target parcels at natural frequencies (suffix _all)."""
    a = n_cal + n_test
    sl_src, sl_tgt, sl_all = slice(n_cal, a), slice(a, a + n_tgt), slice(a, None)
    rec = dict(conf_src=float(P_u[sl_src].max(1).mean()), conf_tgt=float(P_u[sl_tgt].max(1).mean()))
    for k in ("probe_l2", "probe_val_nll", "probe_val_nll_l2_0", "temperature"):
        if k in extra:
            rec[k] = float(extra[k])
    if "z_norm" in extra:
        z, ml = extra["z_norm"], extra["max_logit"]
        q99 = float(np.percentile(z[sl_src], 99))
        rec.update(z_norm_src=float(z[sl_src].mean()), z_norm_tgt=float(z[sl_tgt].mean()),
                   tgt_beyond_src_q99=float((z[sl_tgt] > q99).mean()),
                   max_logit_src=float(ml[sl_src].mean()), max_logit_tgt=float(ml[sl_tgt].mean()))
        if "raw_norm" in extra:
            rz = extra["raw_norm"]; rq99 = float(np.percentile(rz[sl_src], 99))
            rec.update(raw_norm_src=float(rz[sl_src].mean()), raw_norm_tgt=float(rz[sl_tgt].mean()),
                       tgt_beyond_src_q99_raw=float((rz[sl_tgt] > rq99).mean()))
        if w_all is not None:
            w = w_all / w_all.sum()
            rec.update(z_norm_tgt_all=float((w * z[sl_all]).sum()), tgt_beyond_src_q99_all=float((w * (z[sl_all] > q99)).sum()))
            if "raw_norm" in extra:
                rec["tgt_beyond_src_q99_raw_all"] = float((w * (extra["raw_norm"][sl_all] > rq99)).sum())
        if src_country is not None:          # the thresholds themselves, overall and per source country
            rec.update(src_q99_z=q99, src_q99_raw=rq99 if "raw_norm" in extra else np.nan)
            zs = z[sl_src]
            for c in np.unique(src_country):
                qc = float(np.percentile(zs[src_country == c], 99))
                rec[f"src_q99_z_{c}"] = qc; rec[f"src_mean_z_{c}"] = float(zs[src_country == c].mean())
                if w_all is not None:
                    rec[f"tgt_beyond_src_q99_all_{c}"] = float((w * (z[sl_all] > qc)).sum())
    return rec


def label_weighted_sets_pw(s_cal_true, w_cal, S_test, w_test_class, alpha):
    """Weighted split conformal with per-calibration-point weights and a class-dependent test weight."""
    o = np.argsort(s_cal_true); s_sorted = s_cal_true[o]; cw = np.cumsum(w_cal[o]); tot = cw[-1]
    q = np.empty(S_test.shape[1])
    for c in range(len(q)):
        j = np.searchsorted(cw, (1 - alpha) * (tot + w_test_class[c]), side="left")
        q[c] = s_sorted[j] if (tot > 0 and j < len(s_sorted)) else np.inf
    return S_test <= q[None, :]


def natural_weights(codes, ids, nat_counts):
    """Per-parcel weight = natural (full-archive) count of its crop code / loaded count among `ids`."""
    c = codes[ids]
    loaded = pd.Series(c).value_counts()
    return nat_counts.reindex(c).fillna(0).values.astype(float) / loaded.reindex(c).values.astype(float)


def train_head(Fs, ys, Ft, yt, K, seed, device, steps=300):
    """Linear head on standardised features: source fields and target labels, each set with half the total weight."""
    torch.manual_seed(seed)
    X = torch.as_tensor(np.vstack([Fs, Ft]), dtype=torch.float32, device=device)
    Y = torch.as_tensor(np.r_[ys, yt], dtype=torch.long, device=device)
    w = torch.as_tensor(np.r_[np.full(len(ys), 0.5 / len(ys)), np.full(len(yt), 0.5 / max(len(yt), 1))],
                        dtype=torch.float32, device=device)
    head = nn.Linear(X.shape[1], K).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(steps):
        loss = (F.cross_entropy(head(X), Y, reduction="none") * w).sum()
        opt.zero_grad(); loss.backward(); opt.step()

    def predict(Z):
        with torch.no_grad():
            return F.softmax(head(torch.as_tensor(Z, dtype=torch.float32, device=device)), -1).cpu().numpy().astype(np.float64)
    return predict


def do_run(data, predictor, cfg, pname, src, tgt, split, space, arch, seed, device):
    t_run = time.time()
    alpha, nominal = cfg["alpha"], 1.0 - cfg["alpha"]
    min_group, n_boot, min_cell = cfg["min_group_eval"], cfg["n_boot"], cfg["min_cell"]
    cpq = lambda s: cp_quantile(s, alpha)                                  # noqa: E731
    SCORE_FNS = score_functions(cfg)

    full = cfg.get("full_diagnostics", True)
    variant = parse_arch(arch)
    codes, counts = data.codes(variant["level"]), data.counts_at(variant["level"])

    # ---------------- crops, parcels, natural frequencies ----------------
    src_all, tgt_all = pair_ids(data, src, tgt, split)
    classes, tgt_ok = classes_for(codes, src_all, tgt_all, space, cfg)
    if variant["open_set"]:
        if (~np.isin(codes[src_all], classes)).sum() < cfg["min_tgt_per_class"]:
            return {"skipped": "too few source parcels for the OTHER class"}
        if (~np.isin(codes[tgt_all], classes)).sum() >= cfg["min_tgt_per_class"]:
            tgt_ok = tgt_ok | {OTHER}
        classes = classes + [OTHER]
    eval_classes = [c for c in classes if c in tgt_ok]
    if len(eval_classes) < 3:
        return {"skipped": f"only {len(eval_classes)} evaluable classes"}
    cmap = {c: i for i, c in enumerate(classes)}; K = len(classes)
    y = np.array([cmap.get(c, cmap[OTHER] if variant["open_set"] else -1) for c in codes])
    src_ids = src_all[y[src_all] >= 0]
    eval_idx = [cmap[c] for c in eval_classes]
    tgt_ids = tgt_all[np.isin(y[tgt_all], eval_idx)]
    present = np.zeros(K, dtype=bool); present[eval_idx] = True

    nat_all = counts[tgt]; nat_tot = nat_all.sum()                         # full-archive (uncapped) target counts
    if variant["open_set"]:                                                # OTHER = every crop the model cannot name
        known_nat = nat_all.reindex(classes[:-1]).fillna(0)
        nat_all = pd.concat([known_nat, pd.Series({OTHER: nat_tot - known_nat.sum()})])
    unknown_frac = 1.0 - float(nat_all.reindex(classes).fillna(0).sum() / nat_tot)
    eval_frac = float(nat_all.reindex(eval_classes).fillna(0).sum() / nat_tot)
    nat_eval = nat_all.reindex(eval_classes).fillna(0).values.astype(float)
    nat_share = {cmap[c]: v / nat_eval.sum() for c, v in zip(eval_classes, nat_eval)}

    # ---------------- source split, model, predictions ----------------
    rs = np.random.RandomState(seed)
    perm = rs.permutation(src_ids)
    n_cal, n_test = int(cfg["src_cal_frac"] * len(perm)), int(cfg["src_test_frac"] * len(perm))
    cal, test, train = np.sort(perm[:n_cal]), np.sort(perm[n_cal:n_cal + n_test]), np.sort(perm[n_cal + n_test:])
    unk = tgt_all[~np.isin(y[tgt_all], eval_idx)]                         # target parcels of crops not evaluated
    used = np.concatenate([cal, test, tgt_ids] + ([unk] if full else []))
    P_u, H_u, extra = predictor.fit_predict(arch, train, used, y, K, seed)
    if full:
        tgt_every = np.concatenate([tgt_ids, unk])                          # every loaded target parcel, any crop
        w_every = natural_weights(codes, tgt_every, counts[tgt])
        w_of = np.zeros(data.n); w_of[tgt_every] = w_every                  # natural-frequency weight per target parcel
        nat_src = counts[src].sum(1) if isinstance(src, list) else counts[src]
        w_src_of = np.zeros(data.n); w_src_of[src_ids] = natural_weights(codes, src_ids, nat_src)
        extra = summarize_extra(extra, len(cal), len(test), len(tgt_ids), P_u, w_all=w_every,
                                src_country=data.country[test] if full else None)
    else:
        extra = summarize_extra(extra, len(cal), len(test), len(tgt_ids), P_u)
    pos = -np.ones(data.n, dtype=np.int64); pos[used] = np.arange(len(used))
    P = lambda ids: P_u[pos[ids]]                                          # noqa: E731
    H = lambda ids: H_u[pos[ids]]                                          # noqa: E731
    acc = lambda ids: float((P(ids).argmax(1) == y[ids]).mean())            # noqa: E731
    ent = lambda p: -(p * np.log(p + 1e-12)).sum(1)                        # noqa: E731
    bcts, bcts_t = fit_bcts(P(test), y[test])
    prior_cal = np.maximum(np.bincount(y[test], minlength=K) / len(test), 1e-12)
    pred_src = np.bincount(P(test).argmax(1), minlength=K) / len(test)
    pred_tgt = np.bincount(P(tgt_ids).argmax(1), minlength=K) / len(tgt_ids)
    info = dict(pair=pname, space=space, arch=arch, seed=seed, K=K, n_eval_classes=len(eval_classes),
                n_train=len(train), n_cal=len(cal), n_target=len(tgt_ids), within_country=split is not None,
                acc_src=acc(test), acc_tgt=acc(tgt_ids),
                conf_src=float(P(test).max(1).mean()), conf_tgt=float(P(tgt_ids).max(1).mean()),
                perplexity_src=float(np.exp(ent(P(test))).mean()), perplexity_tgt=float(np.exp(ent(P(tgt_ids))).mean()),
                pred_tv=float(0.5 * np.abs(pred_src - pred_tgt).sum()),
                unknown_frac_nat=unknown_frac, eval_frac_nat=eval_frac)
    if full:
        we = w_every / w_every.sum(); ws = w_src_of[test] / w_src_of[test].sum()
        pred_src_nat = np.bincount(P(test).argmax(1), weights=ws, minlength=K)
        pred_tgt_all = np.bincount(P(tgt_every).argmax(1), weights=we, minlength=K)
        info.update(n_unknown_loaded=len(unk), unknown_share_loaded=float(we[len(tgt_ids):].sum()),
                    conf_src_nat=float((ws * P(test).max(1)).sum()), conf_tgt_all=float((we * P(tgt_every).max(1)).sum()),
                    pred_tv_all=float(0.5 * np.abs(pred_src_nat - pred_tgt_all).sum()),
                    acc_tgt_nat=float(np.average(P(tgt_ids).argmax(1) == y[tgt_ids], weights=w_of[tgt_ids])),
                    bbse_cond=bbse_weights_w(P(cal), y[cal], P(cal), np.ones(len(cal)))[1])   # confusion-matrix condition no.
        if full:                             # an exactly singular matrix has cond = inf, stored as null: flag it
            info["bbse_singular"] = bool(not np.isfinite(info["bbse_cond"]))
        truth_nat = nat_all.reindex(classes).fillna(0).values.astype(float)      # target prior over known classes
        # class weights for calibration at natural frequencies: source (natcal) and target (oracle_natcal)
        cw_src = np.array([w_src_of[cal][y[cal] == c].mean() if (y[cal] == c).any() else 0.0 for c in range(K)])
        srng2 = np.random.default_rng(10 ** 6 + seed)                     # randomised scores of the new methods
        synthetic, adapt = [], []

    nuts3 = data.meta.nuts3.values
    src_reg = make_regions(data.center[src_ids], nuts3[src_ids], cfg["n_regions"], cfg["min_region_size"])
    reg_of = -np.ones(data.n, dtype=int); reg_of[src_ids] = src_reg
    reg_cal, reg_test = reg_of[cal], reg_of[test]
    tgt_reg = make_regions(data.center[tgt_ids], nuts3[tgt_ids], cfg["n_regions"], cfg["min_region_size"])
    srng = np.random.default_rng(seed)        # randomised scores; shared across the whole run, in a fixed order

    # ---------------- in-source check: coverage, size, leave-one-region-out, for every score ----------------
    for sname, fn in SCORE_FNS.items():
        S_cal, S_te = fn(P(cal), srng), fn(P(test), srng)
        s_true = S_cal[np.arange(len(cal)), y[cal]]
        sets = S_te <= cpq(s_true)
        cov = sets[np.arange(len(test)), y[test]].astype(float)
        lro = []
        for g in np.unique(reg_test[reg_test >= 0]):
            mk = reg_test == g
            if mk.sum() >= min_group:
                qg = cpq(s_true[reg_cal != g])
                lro.append(float((S_te[mk][np.arange(mk.sum()), y[test][mk]] <= qg).mean()))
        info[f"src_cov_{sname}"] = float(cov.mean()); info[f"src_size_{sname}"] = float(sets.sum(1).mean())
        info[f"src_lro_worst_{sname}"] = min(lro) if lro else np.nan
        if full and sname == "LAC":
            wt_src = w_src_of[test] / w_src_of[test].sum()
            sets_nc = label_weighted_sets_pw(s_true, w_src_of[cal], S_te, cw_src, alpha)
            cov_nc = sets_nc[np.arange(len(test)), y[test]].astype(float)
            info.update(src_cov_nat_LAC=float((wt_src * cov).sum()), src_cov_LAC_natcal=float(cov_nc.mean()),
                        src_cov_nat_LAC_natcal=float((wt_src * cov_nc).sum()), src_size_LAC_natcal=float(sets_nc.sum(1).mean()),
                        src_size_nat_LAC=float((wt_src * sets.sum(1)).sum()),
                        src_size_nat_LAC_natcal=float((wt_src * sets_nc.sum(1)).sum()))

    rows, few, splits = [], [], []
    for r in range(cfg["n_splits"]):
        rsr = np.random.RandomState(10000 * seed + r); rng_s = np.random.default_rng(10000 * seed + r)

        # ================= design A: exchangeable pool / evaluation halves of the target =================
        pt = rsr.permutation(len(tgt_ids)); half = len(tgt_ids) // 2
        tc, tt = tgt_ids[pt[:half]], tgt_ids[pt[half:]]                    # labelled pool, evaluation
        rc_, rt_ = tgt_reg[pt[:half]], tgt_reg[pt[half:]]
        yt = y[tt]
        w_par = np.array([nat_share[c] for c in yt]) / (np.bincount(yt, minlength=K)[yt] / len(yt))
        _, rn_eval = region_cov(np.ones(len(tt)), rt_, min_group)
        fl_mean, fl_p05 = floor_stats(rn_eval, nominal, seed=r)
        raw_w, auc = fit_domain(H(cal), H(tc), seed=r)
        wc_raw, wt_raw = raw_w(H(cal)), raw_w(H(tt))
        bw = bbse_weights(P(cal), y[cal], P(tc))
        pi_em = em_priors(bcts(P(tc)), prior_cal)
        ratio_em = pi_em / prior_cal
        beta_hat = qtc_beta(P(cal).max(1), P(tc).max(1), alpha)
        sp = dict(design="exchangeable", split=r, domain_auc=auc, qtc_beta=beta_hat, floor_mean=fl_mean, floor_p05=fl_p05,
                  em_prior_tv=float(0.5 * np.abs(pi_em - np.bincount(yt, minlength=K) / len(yt)).sum()),
                  em_absent_classes=int((pi_em[~present] > 1e-3).sum()))
        for cname, (lo, hi) in cfg["w_clips"].items():
            wc = np.clip(wc_raw, lo, hi)
            sp[f"ess_{cname}"] = float(wc.sum() ** 2 / (wc ** 2).sum())
        if full:
            # unlabeled pool of ALL target parcels: the pool half of evaluated crops + half of every other target parcel
            rsu = np.random.RandomState(20000 * seed + r)
            U = np.concatenate([tc, unk[rsu.permutation(len(unk))[: len(unk) // 2]]])
            wU = w_of[U]
            raw_w2, auc2 = fit_domain_w(H(cal), H(U), wU, seed=r)
            wc2_raw, wt2_raw = raw_w2(H(cal)), raw_w2(H(tt))
            bw2, cond2 = bbse_weights_w(P(cal), y[cal], P(U), wU)
            pi_em2 = em_priors_w(bcts(P(U)), prior_cal, wU)
            ratio_em2 = pi_em2 / prior_cal
            beta2 = qtc_beta_w(P(cal).max(1), P(U).max(1), wU, alpha)
            truth_bal = np.bincount(yt, minlength=K) / len(yt)
            sp.update(domain_auc_all=auc2, qtc_beta_all=beta2, n_pool_all=len(U),
                      pool_unknown_share=float(wU[len(tc):].sum() / wU.sum()),
                      noadj_tv=tv(prior_cal, truth_bal), bbse_prior_tv=tv(np.maximum(bw * prior_cal, 1e-12), truth_bal),
                      em_prior_tv_all=tv(pi_em2, truth_nat), noadj_tv_all=tv(prior_cal, truth_nat),
                      bbse_prior_tv_all=tv(np.maximum(bw2 * prior_cal, 1e-12), truth_nat))
            cw_tgt = np.array([w_of[tc][y[tc] == c].mean() if (y[tc] == c).any() else 0.0 for c in range(K)])
            if full:
                beta_ns = qtc_beta_ww(P(cal).max(1), w_src_of[cal], P(U).max(1), wU, alpha)
                sp.update(domain_auc_natsrc_all=fit_domain_ww(H(cal), w_src_of[cal], H(U), wU, seed=r),
                          qtc_beta_natsrc_all=beta_ns)
                r_true_bal = np.where(prior_cal > 1e-11, truth_bal / prior_cal, 0.0)   # oracle label-shift ratio
        splits.append(sp)
        for sname, fn in SCORE_FNS.items():
            S_cal, S_tc, S_tt = fn(P(cal), srng), fn(P(tc), srng), fn(P(tt), srng)
            s_true = S_cal[np.arange(len(cal)), y[cal]]
            res, extra_m = {}, {}
            # calibrated on source labels
            res["standard"] = S_tt <= cpq(s_true)
            res["region_robust"] = S_tt <= region_robust_q(s_true, reg_cal, alpha, min_cell)
            res["class_region_robust"] = S_tt <= class_region_robust_q(s_true, y[cal], reg_cal, K, alpha, min_cell)[None, :]
            # also use unlabelled target parcels (the pool half, without its labels)
            for cname, (lo, hi) in cfg["w_clips"].items():
                qw = weighted_q(s_true, np.clip(wc_raw, lo, hi), np.clip(wt_raw, lo, hi), alpha)
                res[cname] = S_tt <= qw[:, None]; extra_m[cname] = {"inf_frac": float(np.isinf(qw).mean())}
            res["knn_local"] = S_tt <= knn_q(H(cal), s_true, H(tt), cfg["knn_k"], alpha, device)[:, None]
            res["label_shift_bbse"] = label_weighted_sets(s_true, y[cal], S_tt, bw, alpha)
            S_cal_em, S_tt_em = fn(adjust_probs(bcts(P(cal)), ratio_em), srng), fn(adjust_probs(bcts(P(tt)), ratio_em), srng)
            res["em_label_shift"] = label_weighted_sets(S_cal_em[np.arange(len(cal)), y[cal]], y[cal], S_tt_em, ratio_em, alpha)
            res["qtc"] = S_tt <= cp_quantile(s_true, max(beta_hat, 1e-9))     # recalibrate at level 1 - beta_hat
            # references calibrated on target labels from the pool
            s_tc_true = S_tc[np.arange(len(tc)), y[tc]]
            q_orc = cpq(s_tc_true)
            res["oracle_target_labels"] = S_tt <= q_orc
            res["mondrian_oracle"] = S_tt <= region_q(s_tc_true, rc_, rt_, q_orc, alpha)[:, None]
            S_tc_em = fn(adjust_probs(bcts(P(tc)), ratio_em), srng)
            res["em_oracle"] = S_tt_em <= cpq(S_tc_em[np.arange(len(tc)), y[tc]])
            for cname, (lo, hi) in cfg["w_clips"].items():
                wcl, wtl = np.clip(wc_raw, lo, hi), np.clip(wt_raw, lo, hi)
                extra_m[cname]["w_test_norm_median"] = float(np.median(wtl / (wcl.sum() + wtl)))
            if full:
                for cname, (lo, hi) in cfg["w_clips"].items():
                    wcl, wtl = np.clip(wc2_raw, lo, hi), np.clip(wt2_raw, lo, hi)
                    qw = weighted_q(s_true, wcl, wtl, alpha)
                    res[cname + "_all"] = S_tt <= qw[:, None]
                    extra_m[cname + "_all"] = {"inf_frac": float(np.isinf(qw).mean()),
                                               "w_test_norm_median": float(np.median(wtl / (wcl.sum() + wtl)))}
                res["label_shift_bbse_all"] = label_weighted_sets(s_true, y[cal], S_tt, bw2, alpha)
                S_cal_e2, S_tt_e2 = fn(adjust_probs(bcts(P(cal)), ratio_em2), srng2), fn(adjust_probs(bcts(P(tt)), ratio_em2), srng2)
                res["em_label_shift_all"] = label_weighted_sets(S_cal_e2[np.arange(len(cal)), y[cal]], y[cal], S_tt_e2, ratio_em2, alpha)
                res["qtc_all"] = S_tt <= cp_quantile(s_true, max(beta2, 1e-9))
                res["standard_natcal"] = label_weighted_sets_pw(s_true, w_src_of[cal], S_tt, cw_src, alpha)
                res["oracle_natcal"] = label_weighted_sets_pw(s_tc_true, w_of[tc], S_tt, cw_tgt, alpha)
                if full:
                    # QTC with source confidences at natural frequencies; threshold from balanced / natural calibration
                    res["qtc_natsrc_all"] = S_tt <= cp_quantile(s_true, max(beta_ns, 1e-9))
                    res["qtc_nat_all"] = label_weighted_sets_pw(s_true, w_src_of[cal], S_tt, cw_src, max(beta_ns, 1e-9))
                    # EM decomposed: EM weights with unadjusted scores; true weights with unadjusted / EM-adjusted scores
                    res["em_weights_plain_all"] = label_weighted_sets(s_true, y[cal], S_tt, ratio_em2, alpha)
                    res["true_label_weights"] = label_weighted_sets(s_true, y[cal], S_tt, r_true_bal, alpha)
                    res["true_weights_emadj_all"] = label_weighted_sets(S_cal_e2[np.arange(len(cal)), y[cal]], y[cal], S_tt_e2,
                                                                        r_true_bal, alpha)
                    if variant["open_set"] and sname == "LAC":     # how often the OTHER class enters a set
                        o = cmap[OTHER]; unk_t = yt == o
                        for mn in ("standard", "oracle_target_labels"):
                            st = res[mn]
                            sp.update({f"other_in_set_known_{mn}": float(st[~unk_t, o].mean()),
                                       f"other_in_set_unknown_{mn}": float(st[unk_t, o].mean()) if unk_t.any() else np.nan,
                                       f"other_only_unknown_{mn}": float((st[unk_t, o] & (st[unk_t].sum(1) == 1)).mean()) if unk_t.any() else np.nan,
                                       f"other_only_known_{mn}": float((st[~unk_t, o] & (st[~unk_t].sum(1) == 1)).mean()),
                                       f"size_known_{mn}": float(st[~unk_t].sum(1).mean()),
                                       f"size_unknown_{mn}": float(st[unk_t].sum(1).mean()) if unk_t.any() else np.nan,
                                       f"unknown_share_eval_{mn}": float(unk_t.mean())})
            for mname, sets in res.items():
                e = eval_sets(sets, yt, rt_, w_par, present, min_group, n_boot,
                              boot=(sname == "LAC" and mname in ("standard", "oracle_target_labels", "mondrian_oracle")))
                e.update(design="exchangeable", score=sname, method=mname, split=r, **extra_m.get(mname, {}))
                rows.append(e)
            # few target labels: random or region-stratified draws of m labels, and m labels per region (Mondrian)
            if sname in cfg["few_scores"]:
                s_tt_true = S_tt[np.arange(len(tt)), yt]
                for m in cfg["few_label_m"]:
                    for sampling in ("random", "stratified"):
                        vals = []
                        for _ in range(cfg["few_label_draws"]):
                            pick = (rsr.choice(len(tc), size=min(m, len(tc)), replace=False) if sampling == "random"
                                    else stratified_pick(rc_, min(m, len(tc)), rsr))
                            vals.append(fast_eval(s_tt_true, S_tt, cpq(s_tc_true[pick]), rt_, min_group))
                        few.append(dict(design="exchangeable", sampling=sampling, score=sname, m=m, split=r, **few_summary(vals)))
                for m_r in cfg["few_mondrian_m"]:
                    vals = []
                    for _ in range(cfg["few_label_draws"]):
                        pick = []
                        for g in np.unique(rc_[rc_ >= 0]):
                            idx = np.where(rc_ == g)[0]
                            pick += list(rsr.choice(idx, size=min(m_r, len(idx)), replace=False))
                        pick = np.array(pick, dtype=int)
                        q_r = region_q(s_tc_true[pick], rc_[pick], rt_, cpq(s_tc_true[pick]), alpha, min_n=1)
                        vals.append(fast_eval(s_tt_true, S_tt, q_r, rt_, min_group))
                    few.append(dict(design="exchangeable", sampling="mondrian_per_region", score=sname, m=m_r, split=r,
                                    **few_summary(vals)))
                if full and sname == "LAC":
                    rs_pac = np.random.RandomState(50000 * seed + r)
                    for m_r in cfg["pac_m"]:
                        v_std, v_pac = [], []
                        for _ in range(cfg["few_label_draws"]):
                            pick = []
                            for g in np.unique(rc_[rc_ >= 0]):
                                idx = np.where(rc_ == g)[0]
                                pick += list(rs_pac.choice(idx, size=min(m_r, len(idx)), replace=False))
                            pick = np.array(pick, dtype=int)
                            q_r = region_q(s_tc_true[pick], rc_[pick], rt_, cpq(s_tc_true[pick]), alpha, min_n=1)
                            v_std.append(fast_eval(s_tt_true, S_tt, q_r, rt_, min_group))
                            q_p = region_q_pac(s_tc_true[pick], rc_[pick], rt_, alpha, cfg["pac_delta"])
                            v_pac.append(fast_eval(s_tt_true, S_tt, q_p, rt_, min_group))
                        for sampling, vals in (("mondrian_per_region_more", v_std), ("mondrian_per_region_pac", v_pac)):
                            few.append(dict(design="exchangeable", sampling=sampling, score=sname, m=m_r, split=r,
                                            **few_summary(vals), worst_ge_nominal=float(np.mean([v[2] >= nominal for v in vals]))))

        # ---------- recalibration vs refitting a linear head with the same m target labels (LAC) ----------
        if full and space == "source" and r < cfg["adapt_splits"] and arch in cfg["adapt_archs"]:
            rs_ad = np.random.RandomState(60000 * seed + r)
            mu_h, sd_h = H(test).mean(0), H(test).std(0) + 1e-6
            Z = lambda ids: (H(ids) - mu_h) / sd_h                         # noqa: E731
            s_cal0 = 1.0 - P(cal)[np.arange(len(cal)), y[cal]]
            S_tt0 = 1.0 - P(tt)
            for m in cfg["adapt_m"]:
                for d in range(cfg["adapt_draws"]):
                    lab = tc[rs_ad.choice(len(tc), size=min(m, len(tc)), replace=False)]; h2 = len(lab) // 2
                    s_lab = 1.0 - P(lab)[np.arange(len(lab)), y[lab]]
                    pA = train_head(Z(test), y[test], Z(lab), y[lab], K, 70000 + 1000 * seed + 100 * r + 10 * d, device)
                    pB = train_head(Z(test), y[test], Z(lab[:h2]), y[lab[:h2]], K, 80000 + 1000 * seed + 100 * r + 10 * d, device)
                    PA_tt, PB_tt = pA(Z(tt)), pB(Z(tt))
                    variants = {
                        "recalibrate_m": (S_tt0, cpq(s_lab), P(tt)),
                        "recalibrate_half": (S_tt0, cpq(s_lab[h2:]), P(tt)),
                        "source_calibration": (S_tt0, cpq(s_cal0), P(tt)),
                        "adapt_m_source_calibration": (1.0 - PA_tt, cpq(1.0 - pA(Z(cal))[np.arange(len(cal)), y[cal]]), PA_tt),
                        "adapt_half_recalibrate_half": (1.0 - PB_tt, cpq(1.0 - pB(Z(lab[h2:]))[np.arange(len(lab) - h2), y[lab[h2:]]]), PB_tt),
                    }
                    for vname, (S_, q_, P_) in variants.items():
                        sets = S_ <= q_
                        e = eval_sets(sets, yt, rt_, w_par, present, min_group, n_boot)
                        adapt.append(dict(variant=vname, m=m, split=r, draw=d, acc=float((P_.argmax(1) == yt).mean()),
                                          **({"design": "exchangeable"} if full else {}),
                                          **{k: e[k] for k in ("coverage", "coverage_nat", "size", "full", "worst_region_cov")}))

        # ================= design B: spatially disjoint, labels from other regions of the target =================
        regs = np.unique(tgt_reg[tgt_reg >= 0])
        if len(regs) >= 4:
            regs = regs[rng_s.permutation(len(regs))]
            pool_regs, dep_regs = regs[: len(regs) // 2], regs[len(regs) // 2:]
            p_idx = np.where(np.isin(tgt_reg, pool_regs))[0]
            d_idx = np.where(np.isin(tgt_reg, dep_regs))[0]
            d_idx = d_idx[rsr.permutation(len(d_idx))]
            d_lab, d_ev = d_idx[: len(d_idx) // 2], d_idx[len(d_idx) // 2:]
            P_lab, D_lab, D_ev = tgt_ids[p_idx], tgt_ids[d_lab], tgt_ids[d_ev]
            r_dlab, r_dev = tgt_reg[d_lab], tgt_reg[d_ev]
            yd = y[D_ev]
            wd = np.array([nat_share[c] for c in yd]) / (np.bincount(yd, minlength=K)[yd] / len(yd))
            _, rn_d = region_cov(np.ones(len(D_ev)), r_dev, min_group)
            fl_d, fl_d05 = floor_stats(rn_d, nominal, seed=100 + r)
            splits.append(dict(design="disjoint", split=r, floor_mean=fl_d, floor_p05=fl_d05,
                               n_pool_other=len(P_lab), n_pool_same=len(D_lab), n_eval=len(D_ev)))
            for sname, fn in SCORE_FNS.items():
                S_cal, S_pl, S_dl, S_dv = fn(P(cal), srng), fn(P(P_lab), srng), fn(P(D_lab), srng), fn(P(D_ev), srng)
                s_true = S_cal[np.arange(len(cal)), y[cal]]
                s_pl = S_pl[np.arange(len(P_lab)), y[P_lab]]; s_dl = S_dl[np.arange(len(D_lab)), y[D_lab]]
                q_same = cpq(s_dl)
                res = {"standard": S_dv <= cpq(s_true),
                       "oracle_other_regions": S_dv <= cpq(s_pl),
                       "oracle_same_regions": S_dv <= q_same,
                       "mondrian_same_regions": S_dv <= region_q(s_dl, r_dlab, r_dev, q_same, alpha)[:, None]}
                for mname, sets in res.items():
                    e = eval_sets(sets, yd, r_dev, wd, present, min_group, n_boot, boot=(sname == "LAC"))
                    e.update(design="disjoint", score=sname, method=mname, split=r)
                    rows.append(e)
                if sname in cfg["few_scores"]:
                    s_dv_true = S_dv[np.arange(len(D_ev)), yd]
                    for m in cfg["disjoint_m"]:
                        for sampling, pool_s in (("other_regions", s_pl), ("same_regions", s_dl)):
                            vals = []
                            for _ in range(cfg["few_label_draws"]):
                                pick = rsr.choice(len(pool_s), size=min(m, len(pool_s)), replace=False)
                                vals.append(fast_eval(s_dv_true, S_dv, cpq(pool_s[pick]), r_dev, min_group))
                            few.append(dict(design="disjoint", sampling=sampling, score=sname, m=m, split=r, **few_summary(vals)))
            # recalibration vs head refit with m labels from OTHER regions of the target (LAC)
            if full and space == "source" and r < cfg["adapt_splits"] and arch in cfg["adapt_archs"]:
                rs_dj = np.random.RandomState(90000 * seed + r)
                mu_h, sd_h = H(test).mean(0), H(test).std(0) + 1e-6
                Zd = lambda ids: (H(ids) - mu_h) / sd_h                     # noqa: E731
                s_cal0 = 1.0 - P(cal)[np.arange(len(cal)), y[cal]]
                S_dv0 = 1.0 - P(D_ev)
                for m in cfg["adapt_m"]:
                    for d in range(cfg["adapt_draws"]):
                        lab = P_lab[rs_dj.choice(len(P_lab), size=min(m, len(P_lab)), replace=False)]; h2 = len(lab) // 2
                        s_lab = 1.0 - P(lab)[np.arange(len(lab)), y[lab]]
                        pA = train_head(Zd(test), y[test], Zd(lab), y[lab], K, 91000 + 1000 * seed + 100 * r + 10 * d, device)
                        pB = train_head(Zd(test), y[test], Zd(lab[:h2]), y[lab[:h2]], K, 92000 + 1000 * seed + 100 * r + 10 * d, device)
                        PA, PB = pA(Zd(D_ev)), pB(Zd(D_ev))
                        variants = {
                            "recalibrate_m": (S_dv0, cpq(s_lab), P(D_ev)),
                            "source_calibration": (S_dv0, cpq(s_cal0), P(D_ev)),
                            "adapt_m_source_calibration": (1.0 - PA, cpq(1.0 - pA(Zd(cal))[np.arange(len(cal)), y[cal]]), PA),
                            "adapt_half_recalibrate_half": (1.0 - PB, cpq(1.0 - pB(Zd(lab[h2:]))[np.arange(len(lab) - h2), y[lab[h2:]]]), PB),
                        }
                        for vname, (S_, q_, P_) in variants.items():
                            sets = S_ <= q_
                            e = eval_sets(sets, yd, r_dev, wd, present, min_group, n_boot)
                            adapt.append(dict(variant=vname, m=m, split=r, draw=d, design="disjoint",
                                              acc=float((P_.argmax(1) == yd).mean()),
                                              **{k: e[k] for k in ("coverage", "coverage_nat", "size", "full", "worst_region_cov")}))

    if full:
        # ---------- synthetic pure label shift: the source test set resampled to random crop priors ----------
        rs_syn = np.random.RandomState(40000 + seed)
        bcts_c, _ = fit_bcts(P(cal), y[cal])
        prior_c = np.maximum(np.bincount(y[cal], minlength=K) / len(cal), 1e-12)
        s_cal_lac = 1.0 - P(cal)[np.arange(len(cal)), y[cal]]
        n_y = np.bincount(y[test], minlength=K)
        for d in range(cfg["synthetic_draws"]):
            q = rs_syn.dirichlet(np.ones(K))
            pw = q[y[test]] / n_y[y[test]]
            T = test[rs_syn.choice(len(test), size=len(test), replace=True, p=pw / pw.sum())]
            truth = np.bincount(y[T], minlength=K) / len(T)
            bw_s, _ = bbse_weights_w(P(cal), y[cal], P(T), np.ones(len(T)))
            pi_s = em_priors(bcts_c(P(T)), prior_c)
            r_em = pi_s / prior_c; r_true = truth / prior_c
            S_T = 1.0 - P(T)
            S_cal_em = 1.0 - adjust_probs(bcts_c(P(cal)), r_em); S_T_em = 1.0 - adjust_probs(bcts_c(P(T)), r_em)
            sets = {"standard": S_T <= cpq(s_cal_lac),
                    "bbse_weighted": label_weighted_sets(s_cal_lac, y[cal], S_T, bw_s, alpha),
                    "em_label_shift": label_weighted_sets(S_cal_em[np.arange(len(cal)), y[cal]], y[cal], S_T_em, r_em, alpha),
                    "true_weights": label_weighted_sets(s_cal_lac, y[cal], S_T, r_true, alpha)}
            if full:          # EM decomposed: estimated weights on plain scores; true weights on EM- / truly adjusted scores
                S_cal_tr = 1.0 - adjust_probs(bcts_c(P(cal)), r_true); S_T_tr = 1.0 - adjust_probs(bcts_c(P(T)), r_true)
                sets.update({"em_weights_plain": label_weighted_sets(s_cal_lac, y[cal], S_T, r_em, alpha),
                             "true_weights_emadj": label_weighted_sets(S_cal_em[np.arange(len(cal)), y[cal]], y[cal], S_T_em, r_true, alpha),
                             "true_weights_trueadj": label_weighted_sets(S_cal_tr[np.arange(len(cal)), y[cal]], y[cal], S_T_tr, r_true, alpha),
                             "bcts_true_weights": label_weighted_sets(1.0 - bcts_c(P(cal))[np.arange(len(cal)), y[cal]], y[cal],
                                                                      1.0 - bcts_c(P(T)), r_true, alpha)})
            rec = dict(draw=d, tv_noadj=tv(prior_c, truth), tv_bbse=tv(np.maximum(bw_s * prior_c, 1e-12), truth),
                       tv_em=tv(pi_s, truth))
            for k, st in sets.items():
                rec[f"cov_{k}"] = float(st[np.arange(len(T)), y[T]].mean()); rec[f"size_{k}"] = float(st.sum(1).mean())
            synthetic.append(rec)

    sp_df = pd.DataFrame(splits)
    ex = sp_df[sp_df.design == "exchangeable"]
    info.update(domain_auc=float(ex.domain_auc.mean()), qtc_beta=float(ex.qtc_beta.mean()),
                em_prior_tv=float(ex.em_prior_tv.mean()), em_absent_classes=float(ex.em_absent_classes.mean()),
                bcts_temperature=bcts_t, minutes=round((time.time() - t_run) / 60, 2))
    out = {"info": info, "rows": rows, "few": few, "splits": splits, "extra": extra}
    if full:
        for k in ("domain_auc_all", "qtc_beta_all", "em_prior_tv_all", "noadj_tv", "noadj_tv_all", "bbse_prior_tv",
                  "bbse_prior_tv_all", "pool_unknown_share"):
            info[k] = float(ex[k].mean())
        if full:
            for k in [c for c in ex.columns if c.startswith(("domain_auc_natsrc", "qtc_beta_natsrc", "other_", "size_known",
                                                             "size_unknown", "unknown_share_eval"))]:
                info[k] = float(ex[k].mean())
        out.update(synthetic=synthetic, adapt=adapt)
    return out
