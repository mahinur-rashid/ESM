"""Conformity scores, split-conformal calibration methods and evaluation metrics."""
import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


# ----------------------------------------------------------------------------------------------------------------
# scores (lower = more conforming); a set contains every label whose score is <= the calibrated threshold
# ----------------------------------------------------------------------------------------------------------------
def score_lac(p, rng):
    """LAC (Sadinle et al., 2019): one minus the label's probability."""
    return 1.0 - p


def score_aps(p, rng, randomized, lam=0.0, kreg=0):
    """APS (Romano et al., 2020): probability mass of the labels ranked at or above the label; optionally randomised
    (mass strictly above + u * own mass) and with the RAPS rank penalty (Angelopoulos et al., 2021)."""
    order = np.argsort(-p, axis=1)
    ps = np.take_along_axis(p, order, 1)
    cum = np.cumsum(ps, 1)
    if randomized:
        u = rng.random(len(p))[:, None]
        cum = cum - (1.0 - u) * ps
    if lam > 0:
        rank = np.arange(1, p.shape[1] + 1)[None, :]
        cum = cum + lam * np.maximum(0, rank - kreg)
    s = np.empty_like(p); np.put_along_axis(s, order, cum, 1)
    return s


def score_functions(cfg):
    return {
        "LAC": score_lac,
        "APS": lambda p, rng: score_aps(p, rng, False),
        "APS_rand": lambda p, rng: score_aps(p, rng, True),
        "RAPS": lambda p, rng: score_aps(p, rng, True, cfg["raps_lambda"], cfg["raps_kreg"]),
    }


# ----------------------------------------------------------------------------------------------------------------
# thresholds
# ----------------------------------------------------------------------------------------------------------------
def cp_quantile(s, alpha):
    """Split-conformal threshold: the ceil((n+1)(1-alpha))-th smallest calibration score (inf if n is too small)."""
    n = len(s); k = math.ceil((n + 1) * (1 - alpha))
    return np.inf if (n == 0 or k > n or k < 1) else float(np.sort(s)[k - 1])


def weighted_q(s_cal, w_cal, w_test, alpha):
    """Weighted conformal threshold per test point (Tibshirani et al., 2019)."""
    o = np.argsort(s_cal); s_sorted, cw = s_cal[o], np.cumsum(w_cal[o])
    j = np.searchsorted(cw, (1 - alpha) * (cw[-1] + w_test), side="left")
    return np.where(j < len(s_sorted), s_sorted[np.minimum(j, len(s_sorted) - 1)], np.inf)


def label_weighted_sets(s_cal_true, y_cal, S_test, w, alpha):
    """Label-shift conformal sets (Podkopaev & Ramdas, 2021) with class weights w."""
    o = np.argsort(s_cal_true); s_sorted = s_cal_true[o]; cw = np.cumsum(w[y_cal][o]); tot = cw[-1]
    q = np.empty(S_test.shape[1])
    for c in range(len(q)):
        j = np.searchsorted(cw, (1 - alpha) * (tot + w[c]), side="left")
        q[c] = s_sorted[j] if (tot > 0 and j < len(s_sorted)) else np.inf
    return S_test <= q[None, :]


def knn_q(F_cal, s_cal, F_test, k, alpha, device):
    """Localised threshold from the k calibration points nearest (cosine) to each test point."""
    A = F.normalize(torch.as_tensor(F_cal, device=device, dtype=torch.float32), dim=1)
    B = F.normalize(torch.as_tensor(F_test, device=device, dtype=torch.float32), dim=1)
    sc = torch.as_tensor(s_cal, device=device, dtype=torch.float64)
    k = min(k, len(F_cal)); kq = math.ceil((k + 1) * (1 - alpha))
    out = []
    for i in range(0, len(B), 2048):
        idx = (B[i:i + 2048] @ A.T).topk(k, dim=1).indices
        nb = sc[idx].sort(dim=1).values
        out.append(nb[:, kq - 1] if kq <= k else torch.full((len(idx),), float("inf"), device=device, dtype=torch.float64))
    return torch.cat(out).cpu().numpy()


def region_robust_q(s_true, reg, alpha, min_cell):
    """Most conservative per-region threshold over source regions (Ali et al., 2026)."""
    qs = [cp_quantile(s_true[reg == g], alpha) for g in np.unique(reg[reg >= 0]) if (reg == g).sum() >= min_cell]
    return max(qs) if qs else cp_quantile(s_true, alpha)


def class_region_robust_q(s_true, y_cal, reg, n_cls, alpha, min_cell):
    """Per crop, the most conservative threshold over source regions (Ali et al., 2026)."""
    q = np.full(n_cls, cp_quantile(s_true, alpha))
    for c in range(n_cls):
        mc = y_cal == c
        cells = [cp_quantile(s_true[mc & (reg == g)], alpha) for g in np.unique(reg[mc & (reg >= 0)])
                 if (mc & (reg == g)).sum() >= min_cell]
        if cells:
            q[c] = max(cells)
        elif mc.sum() >= min_cell:
            q[c] = cp_quantile(s_true[mc], alpha)
    return q


def region_q(s_true, reg_lab, reg_eval, q_fallback, alpha, min_n=10):
    """Mondrian calibration: one threshold per region from the labelled points in that region."""
    q = np.full(len(reg_eval), q_fallback)
    for g in np.unique(reg_eval[reg_eval >= 0]):
        mk = reg_lab == g
        if mk.sum() >= min_n:
            q[reg_eval == g] = cp_quantile(s_true[mk], alpha)
    return q


# ----------------------------------------------------------------------------------------------------------------
# shift estimates
# ----------------------------------------------------------------------------------------------------------------
def fit_domain(F_s, F_t, seed):
    """Logistic domain classifier on standardised features; returns the density-ratio function and its held-out AUC."""
    Xd = np.vstack([F_s, F_t]); d = np.r_[np.zeros(len(F_s)), np.ones(len(F_t))]
    perm = np.random.RandomState(seed).permutation(len(Xd)); cut = int(0.8 * len(Xd))
    sc = StandardScaler().fit(Xd[perm[:cut]])
    clf = LogisticRegression(C=0.1, max_iter=3000).fit(sc.transform(Xd[perm[:cut]]), d[perm[:cut]])
    auc = roc_auc_score(d[perm[cut:]], clf.predict_proba(sc.transform(Xd[perm[cut:]]))[:, 1])
    ratio = len(F_s) / len(F_t)

    def raw_w(Fx):
        p = clf.predict_proba(sc.transform(Fx))[:, 1].clip(1e-9, 1 - 1e-9)
        return p / (1 - p) * ratio
    return raw_w, float(auc)


def bbse_weights(p_cal, y_cal, p_tgt):
    """Black-box shift estimation (Lipton et al., 2018): confusion-matrix system solved by least squares."""
    K = p_cal.shape[1]
    C = np.zeros((K, K)); np.add.at(C, (p_cal.argmax(1), y_cal), 1.0 / len(y_cal))
    mu = np.bincount(p_tgt.argmax(1), minlength=K) / len(p_tgt)
    w, *_ = np.linalg.lstsq(C, mu, rcond=None)
    return np.clip(w, 0.0, None)


def em_priors(p_tgt, prior_src, iters=200, tol=1e-7):
    """EM re-estimation of target class priors (Saerens et al., 2002)."""
    pi = prior_src.copy()
    for _ in range(iters):
        q = p_tgt * (pi / np.maximum(prior_src, 1e-12))[None, :]
        q /= q.sum(1, keepdims=True)
        new = q.mean(0)
        if np.abs(new - pi).max() < tol:
            pi = new; break
        pi = new
    return pi


def fit_bcts(p, y):
    """Bias-corrected temperature scaling (Alexandari et al., 2020), fitted by NLL; returns (map, temperature)."""
    lp = np.log(np.clip(p, 1e-12, 1.0)); n, K = p.shape

    def nll_grad(theta):
        t = np.exp(theta[0]); b = theta[1:]
        z = lp / t + b; z = z - z.max(1, keepdims=True); e = np.exp(z); q = e / e.sum(1, keepdims=True)
        nll = -np.log(q[np.arange(n), y] + 1e-12).mean()
        d = q.copy(); d[np.arange(n), y] -= 1.0; d /= n
        g_b = d.sum(0); g_t = (d * (-lp / t)).sum() * t
        return nll, np.r_[g_t, g_b]
    res = minimize(nll_grad, np.zeros(K + 1), jac=True, method="L-BFGS-B")
    t, b = float(np.exp(res.x[0])), res.x[1:]

    def apply(q):
        z = np.log(np.clip(q, 1e-12, 1.0)) / t + b; z = z - z.max(1, keepdims=True); e = np.exp(z)
        return e / e.sum(1, keepdims=True)
    return apply, t


def adjust_probs(p, ratio):
    q = p * ratio[None, :]
    return q / np.maximum(q.sum(1, keepdims=True), 1e-12)


def qtc_beta(conf_src, conf_tgt, alpha):
    """QTC (Yilmaz & Heckel, 2022): beta = min(QTC-T, QTC-S) from top-class confidences."""
    b_t = float(np.mean(conf_src < np.quantile(conf_tgt, alpha)))
    b_s = 1.0 - float(np.mean(conf_tgt < np.quantile(conf_src, 1 - alpha)))
    return min(b_t, b_s)


# ----------------------------------------------------------------------------------------------------------------
# regions and metrics
# ----------------------------------------------------------------------------------------------------------------
def make_regions(center, nuts3, k, min_size):
    """k-means regions of parcel centroids (NUTS3 if centroids are missing); regions below min_size get -1."""
    if np.isfinite(center).all() and len(center) >= k * min_size:
        lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(center)
    else:
        lab = pd.factorize(nuts3)[0]
    sizes = np.bincount(lab); good = np.where(sizes >= min_size)[0]
    remap = -np.ones(lab.max() + 1, dtype=int); remap[good] = np.arange(len(good))
    return remap[lab]


def region_cov(cov, reg, min_n):
    gs = [g for g in np.unique(reg[reg >= 0]) if (reg == g).sum() >= min_n]
    rc = np.array([cov[reg == g].mean() for g in gs]); rn = np.array([(reg == g).sum() for g in gs])
    return rc, rn


def floor_stats(rn, nominal, sims=4000, seed=0):
    """Chance level of the worst region: expected minimum coverage if every region had exactly nominal coverage."""
    if len(rn) == 0:
        return np.nan, np.nan
    r = np.random.default_rng(seed)
    mins = (r.binomial(rn[None, :], nominal, size=(sims, len(rn))) / rn[None, :]).min(1)
    return float(mins.mean()), float(np.percentile(mins, 5))


def boot_worst(rc, rn, n_boot, seed=0):
    """95% bootstrap interval of the worst-region coverage (binomial resampling of each region)."""
    if len(rn) == 0:
        return np.nan, np.nan
    r = np.random.default_rng(seed)
    mins = (r.binomial(rn[None, :], np.clip(rc, 0, 1)[None, :], size=(n_boot, len(rn))) / rn[None, :]).min(1)
    return float(np.percentile(mins, 2.5)), float(np.percentile(mins, 97.5))


def eval_sets(sets, y, reg, wnat, present, min_group, n_boot, boot=False):
    """Coverage (balanced and at natural frequencies), size, absent crops in sets, full/empty sets, worst crop/region."""
    n = len(y); cov = sets[np.arange(n), y].astype(float); size = sets.sum(1)
    out = dict(coverage=cov.mean(), coverage_nat=float((wnat * cov).sum() / wnat.sum()), size=size.mean(),
               absent_in_set=float(sets[:, ~present].sum(1).mean()), full=(size == sets.shape[1]).mean(),
               empty=(size == 0).mean())
    cls = [cov[y == c].mean() for c in np.unique(y) if (y == c).sum() >= min_group]
    out["worst_class_cov"] = float(min(cls)) if cls else np.nan
    rc, rn = region_cov(cov, reg, min_group)
    out["worst_region_cov"] = float(rc.min()) if len(rc) else np.nan
    if boot:
        out["wr_lo"], out["wr_hi"] = boot_worst(rc, rn, n_boot)
    return out


def fast_eval(s_true_eval, S_eval, q, reg, min_group):
    """Coverage, size and worst region for a scalar or per-parcel threshold (few-label loops)."""
    qq = q if np.ndim(q) else np.full(len(s_true_eval), q)
    cov = (s_true_eval <= qq).astype(float)
    rc, _ = region_cov(cov, reg, min_group)
    return cov.mean(), float((S_eval <= qq[:, None]).sum(1).mean()), (float(rc.min()) if len(rc) else np.nan)


def stratified_pick(reg, m, rs):
    groups = [np.where(reg == g)[0] for g in np.unique(reg[reg >= 0])]
    per = m // max(len(groups), 1); pick = []
    for g in groups:
        pick += list(rs.choice(g, size=min(per, len(g)), replace=False))
    rest = np.setdiff1d(np.arange(len(reg)), pick)
    if len(pick) < m:
        pick += list(rs.choice(rest, size=min(m - len(pick), len(rest)), replace=False))
    return np.array(pick[:m], dtype=int)


def few_summary(vals):
    covs, sizes, worst = map(np.array, zip(*vals))
    return dict(cov_mean=float(covs.mean()), cov_p10=float(np.percentile(covs, 10)), cov_p90=float(np.percentile(covs, 90)),
                size_mean=float(sizes.mean()),
                worst_region_mean=float(np.nanmean(worst)) if np.isfinite(worst).any() else np.nan)


# ----------------------------------------------------------------------------------------------------------------
# weighted unlabeled target pools, label-shift checks, training-conditional thresholds
# ----------------------------------------------------------------------------------------------------------------
def weighted_quantile(x, w, q):
    o = np.argsort(x); cw = np.cumsum(w[o]); cw = cw / cw[-1]
    return float(x[o][min(np.searchsorted(cw, q, side="left"), len(x) - 1)])


def fit_domain_w(F_s, F_t, w_t, seed):
    """As fit_domain, with target points weighted (e.g. to natural crop frequencies); weights rescaled to mean 1."""
    w_t = w_t / w_t.mean()
    Xd = np.vstack([F_s, F_t]); d = np.r_[np.zeros(len(F_s)), np.ones(len(F_t))]; sw = np.r_[np.ones(len(F_s)), w_t]
    perm = np.random.RandomState(seed).permutation(len(Xd)); cut = int(0.8 * len(Xd))
    tr, te = perm[:cut], perm[cut:]
    sc = StandardScaler().fit(Xd[tr])
    clf = LogisticRegression(C=0.1, max_iter=3000).fit(sc.transform(Xd[tr]), d[tr], sample_weight=sw[tr])
    auc = roc_auc_score(d[te], clf.predict_proba(sc.transform(Xd[te]))[:, 1], sample_weight=sw[te])
    ratio = len(F_s) / len(F_t)

    def raw_w(Fx):
        p = clf.predict_proba(sc.transform(Fx))[:, 1].clip(1e-9, 1 - 1e-9)
        return p / (1 - p) * ratio
    return raw_w, float(auc)


def fit_domain_ww(F_s, w_s, F_t, w_t, seed):
    """As fit_domain_w, with source points weighted too (e.g. both sides at natural crop frequencies); returns the AUC."""
    w_s, w_t = w_s / w_s.mean(), w_t / w_t.mean()
    Xd = np.vstack([F_s, F_t]); d = np.r_[np.zeros(len(F_s)), np.ones(len(F_t))]; sw = np.r_[w_s, w_t]
    perm = np.random.RandomState(seed).permutation(len(Xd)); cut = int(0.8 * len(Xd))
    tr, te = perm[:cut], perm[cut:]
    sc = StandardScaler().fit(Xd[tr])
    clf = LogisticRegression(C=0.1, max_iter=3000).fit(sc.transform(Xd[tr]), d[tr], sample_weight=sw[tr])
    return float(roc_auc_score(d[te], clf.predict_proba(sc.transform(Xd[te]))[:, 1], sample_weight=sw[te]))


def qtc_beta_ww(conf_src, w_src, conf_tgt, w_tgt, alpha):
    """QTC beta with both source and target confidences weighted (natural crop frequencies on both sides)."""
    b_t = float(np.sum(w_src * (conf_src < weighted_quantile(conf_tgt, w_tgt, alpha))) / w_src.sum())
    b_s = 1.0 - float(np.sum(w_tgt * (conf_tgt < weighted_quantile(conf_src, w_src, 1 - alpha))) / w_tgt.sum())
    return min(b_t, b_s)


def bbse_weights_w(p_cal, y_cal, p_tgt, w_tgt):
    K = p_cal.shape[1]
    C = np.zeros((K, K)); np.add.at(C, (p_cal.argmax(1), y_cal), 1.0 / len(y_cal))
    mu = np.bincount(p_tgt.argmax(1), weights=w_tgt, minlength=K) / w_tgt.sum()
    w, *_ = np.linalg.lstsq(C, mu, rcond=None)
    return np.clip(w, 0.0, None), float(np.linalg.cond(C))


def em_priors_w(p_tgt, prior_src, w, iters=200, tol=1e-7):
    pi = prior_src.copy(); w = w / w.sum()
    for _ in range(iters):
        q = p_tgt * (pi / np.maximum(prior_src, 1e-12))[None, :]
        q /= q.sum(1, keepdims=True)
        new = (q * w[:, None]).sum(0)
        if np.abs(new - pi).max() < tol:
            pi = new; break
        pi = new
    return pi


def qtc_beta_w(conf_src, conf_tgt, w_tgt, alpha):
    b_t = float(np.mean(conf_src < weighted_quantile(conf_tgt, w_tgt, alpha)))
    b_s = 1.0 - float(np.sum(w_tgt * (conf_tgt < np.quantile(conf_src, 1 - alpha))) / w_tgt.sum())
    return min(b_t, b_s)


def tv(p, q):
    return float(0.5 * np.abs(np.asarray(p) / np.sum(p) - np.asarray(q) / np.sum(q)).sum())


def pac_order(m, alpha, delta):
    """Training-conditional (PAC) split conformal (Vovk, 2012): the smallest order statistic k of m calibration
    scores whose coverage, distributed Beta(k, m+1-k), falls below 1-alpha with probability at most delta."""
    from scipy.stats import beta as beta_dist
    if m < 1:
        return None
    k = np.arange(1, m + 1)
    ok = np.nonzero(beta_dist.cdf(1 - alpha, k, m + 1 - k) <= delta)[0]
    return int(k[ok[0]]) if len(ok) else None


def pac_quantile(s, alpha, delta):
    k = pac_order(len(s), alpha, delta)
    return np.inf if k is None else float(np.sort(s)[k - 1])


def region_q_pac(s_true, reg_lab, reg_eval, alpha, delta):
    """Mondrian calibration with a PAC threshold per region; parcels outside labelled regions use the pooled one."""
    q = np.full(len(reg_eval), pac_quantile(s_true, alpha, delta))
    for g in np.unique(reg_eval[reg_eval >= 0]):
        mk = reg_lab == g
        if mk.sum() >= 1:
            q[reg_eval == g] = pac_quantile(s_true[mk], alpha, delta)
    return q
