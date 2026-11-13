"""Run the experiment grid (resumable: one JSON file per run) and aggregate the runs into tables."""
import glob
import json
import math
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

from .experiment import do_run, run_key


def to_py(o):
    """JSON-safe copy (numpy scalars -> Python, non-finite floats -> null)."""
    if isinstance(o, dict):
        return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    return o


def plan(cfg):
    """Seed-major order, so that an interrupted run still covers every model and pair."""
    return [(p, src, tgt, spl, sp, a, s) for s in cfg["seeds"] for archs, spaces in cfg["grids"] for a in archs
            for sp in spaces for (p, src, tgt, spl) in cfg["pairs"]]


def run_grid(data, predictor, cfg, run_dir, device, log=print):
    os.makedirs(run_dir, exist_ok=True)
    t_start, todo = time.time(), plan(cfg)
    for i, (p, src, tgt, spl, sp, a, s) in enumerate(todo):
        key = run_key(p, sp, a, s); path = os.path.join(run_dir, key + ".json")
        if os.path.exists(path):
            continue
        if (time.time() - t_start) / 3600 > cfg["max_hours"]:
            log(f"time limit of {cfg['max_hours']} h reached; re-run to resume"); break
        out = do_run(data, predictor, cfg, p, src, tgt, spl, sp, a, s, device)
        with open(path, "w") as f:
            json.dump(to_py(out), f)
        if "info" in out:
            inf = out["info"]
            log(f"[{i + 1}/{len(todo)}] {key}: {inf['K']} crops, accuracy source {inf['acc_src']:.3f} "
                f"target {inf['acc_tgt']:.3f} ({inf['minutes']:.1f} min)")
        else:
            log(f"[{i + 1}/{len(todo)}] {key}: skipped ({out['skipped']})")


def aggregate(cfg, run_dir, table_dir):
    """Per-run JSON files -> CSV tables: run_info, conformal_by_run, by_pair, by_condition, few_by_pair, splits,
    region_floor, info_by_pair, extra_by_run. Means are taken over target splits within a run, then over runs."""
    os.makedirs(table_dir, exist_ok=True)
    nominal = 1.0 - cfg["alpha"]
    INFO, ROWS, FEW, SPL, EXTRA, SYN, ADAPT = [], [], [], [], [], [], []
    for f in sorted(glob.glob(os.path.join(run_dir, "*.json"))):
        d = json.load(open(f))
        if "info" not in d:
            continue
        inf = d["info"]; INFO.append(inf)
        base = {k: inf[k] for k in ("pair", "space", "arch", "seed")}
        ROWS += [{**base, **r} for r in d["rows"]]
        FEW += [{**base, **r} for r in d["few"]]
        SPL += [{**base, **r} for r in d["splits"]]
        if d.get("extra"):
            EXTRA.append({**base, **d["extra"]})
        SYN += [{**base, **r} for r in d.get("synthetic", [])]
        ADAPT += [{**base, **r} for r in d.get("adapt", [])]
    INFO, ROWS, FEW, SPL = map(pd.DataFrame, (INFO, ROWS, FEW, SPL))
    text = {"pair", "space", "arch", "score", "method", "sampling", "design"}
    for df in (INFO, ROWS, FEW, SPL):
        for c in df.columns:
            if c not in text and df[c].dtype == object:
                df[c] = pd.to_numeric(df[c], errors="coerce")
    keys = ["pair", "space", "arch"]

    per_run = ROWS.groupby(keys + ["seed", "design", "score", "method"]).mean(numeric_only=True).drop(columns="split").reset_index()
    orc = per_run[(per_run.design == "exchangeable") & (per_run.method == "oracle_target_labels")][keys + ["seed", "score", "size"]]
    per_run = per_run.merge(orc.rename(columns={"size": "oracle_size"}), on=keys + ["seed", "score"], how="left")
    S = cfg["success"]
    per_run["restores"] = ((per_run.coverage >= S["min_cov"]) & (per_run.full <= S["max_full"]) &
                           (per_run["size"] <= S["max_size_ratio"] * per_run.oracle_size))
    per_run = per_run.merge(INFO[keys + ["seed", "K", "n_eval_classes"]], on=keys + ["seed"], how="left")
    per_run["size_rel_present"] = per_run["size"] / per_run.n_eval_classes

    def agg(df, by):
        return df.groupby(by).agg(coverage=("coverage", "mean"), coverage_sd=("coverage", "std"),
                                  coverage_nat=("coverage_nat", "mean"), size=("size", "mean"),
                                  size_rel_present=("size_rel_present", "mean"), absent_in_set=("absent_in_set", "mean"),
                                  full=("full", "mean"), worst_region=("worst_region_cov", "mean"),
                                  wr_lo=("wr_lo", "mean"), wr_hi=("wr_hi", "mean"), worst_class=("worst_class_cov", "mean"),
                                  restores=("restores", "mean"), inf_frac=("inf_frac", "mean"), n_runs=("seed", "size")).reset_index()

    by_pair = agg(per_run, ["pair", "space", "design", "score", "method"])
    by_cond = agg(per_run, keys + ["design", "score", "method"])
    spl_run = SPL.groupby(keys + ["seed", "design"]).mean(numeric_only=True).drop(columns="split").reset_index()
    floor = spl_run.groupby(["pair", "space", "design"])[["floor_mean", "floor_p05"]].mean().reset_index()
    info_pair = INFO.groupby(["pair", "space"]).mean(numeric_only=True).reset_index()
    few_run = FEW.groupby(keys + ["seed", "design", "sampling", "score", "m"]).mean(numeric_only=True).drop(columns="split").reset_index()
    few_pair = few_run.groupby(["pair", "space", "design", "sampling", "score", "m"]).mean(numeric_only=True).reset_index().drop(columns="seed")
    k_of = lambda m: math.ceil((m + 1) * nominal)                                        # noqa: E731
    few_pair["beta_p10"] = [beta_dist.ppf(0.10, k_of(m), m + 1 - k_of(m)) if k_of(m) <= m else np.nan for m in few_pair.m]

    tables = [("run_info", INFO), ("conformal_by_run", per_run), ("by_pair", by_pair), ("by_condition", by_cond),
              ("few_by_pair", few_pair), ("splits", SPL), ("region_floor", floor), ("info_by_pair", info_pair)]
    if EXTRA:
        tables.append(("extra_by_run", pd.DataFrame(EXTRA)))
    if SYN:            # synthetic pure label shift, mean over draws
        tables.append(("synthetic_label_shift", pd.DataFrame(SYN).groupby(keys + ["seed"]).mean(numeric_only=True)
                       .drop(columns="draw").reset_index()))
    if ADAPT:          # recalibration vs adaptation, mean over draws and splits
        AD = pd.DataFrame(ADAPT)
        by = keys + ["seed", "variant", "m"] + (["design"] if "design" in AD.columns else [])
        tables.append(("adapt_by_run", AD.groupby(by).mean(numeric_only=True).drop(columns=["split", "draw"]).reset_index()))
    if cfg.get("full_diagnostics"):  # few-label summaries per model as well
        tables.append(("few_by_arch", few_run.groupby(keys + ["design", "sampling", "score", "m"]).mean(numeric_only=True)
                       .reset_index().drop(columns="seed")))
    if cfg.get("full_diagnostics"):  # worst region per evaluation split, for intervals that include calibration randomness
        wr = ROWS[(ROWS.design == "exchangeable") & (ROWS.score == "LAC") &
                  ROWS.method.isin(["standard", "oracle_target_labels", "mondrian_oracle"])]
        tables.append(("worst_region_by_split", wr[keys + ["seed", "split", "method", "worst_region_cov", "coverage"]]))
    for name, df in tables:
        df.to_csv(os.path.join(table_dir, f"{name}.csv"), index=False)
    return dict(tables)
