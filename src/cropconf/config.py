"""Experiment configuration.

One dictionary holds every setting. `make_config` returns the settings used in the paper for one of its two
experiments (`main`: the five models and the location / OTHER-class variants; `controls`: the input, probe and
label-space controls); `fast_dev=True` shrinks everything for a quick smoke test on synthetic data.
"""
import copy

# Transfer pairs: (name, source countries, target country, within-country split or None)
PAIRS = [
    ("LV_to_EE", ["LV"], "EE", None),
    ("EE_to_LV", ["EE"], "LV", None),
    ("PT_to_LV", ["PT"], "LV", None),
    ("LV_to_PT", ["LV"], "PT", None),
    ("LVEE_to_PT", ["LV", "EE"], "PT", None),
    ("PTnorth_to_PTsouth", ["PT"], "PT", {"axis": "lat", "src": "high"}),   # source: northern half of Portugal
    ("LVwest_to_LVeast", ["LV"], "LV", {"axis": "lon", "src": "low"}),      # source: western half of Latvia
]

MAIN_MODELS = ["transformer", "tempcnn", "presto_frozen", "presto_finetune", "random_forest"]
PROBE_CONTROLS = ["presto_frozen_l2", "presto_frozen_raw", "presto_frozen_clip", "presto_frozen_l2s1", "presto_frozen_l2s2",
                  "presto_frozen_ts", "presto_frozen_mlp"]

EXPERIMENTS = {
    # name -> grids of (models, label spaces); tag names the output folders
    "main": dict(tag="main", adapt_archs=MAIN_MODELS, grids=[
        (MAIN_MODELS, ["source", "shared"]),
        (["presto_frozen_noloc", "transformer_openset"], ["source"]),          # constant location; OTHER class
    ]),
    "controls": dict(tag="controls", adapt_archs=[], grids=[
        (["transformer_monthly"], ["source", "shared"]),                       # input control
        (PROBE_CONTROLS, ["source"]),                                          # frozen-Presto probe controls
        (["transformer_merged6"], ["source"]),                                 # codes merged to crop groups
    ]),
}

DIAGNOSTICS = dict(
    full_diagnostics=True,                             # unlabeled pool of all target parcels, natural frequencies,
                                                       # EM/BBSE checks, PAC thresholds, head refit, ...
    pac_m=[25, 50, 100, 200], pac_delta=0.1,          # labels per region; PAC: P(coverage < 1 - alpha) <= delta
    adapt_m=[50, 200], adapt_draws=5, adapt_splits=2,  # recalibration vs refitting a linear head
    synthetic_draws=5,                                 # synthetic pure label shift per run
)
DIAGNOSTICS_FAST_DEV = dict(pac_m=[10, 25], adapt_m=[10, 20], adapt_draws=2, adapt_splits=1, synthetic_draws=2)

DEFAULTS = dict(
    pairs=PAIRS,
    label_spaces=["source", "shared"],
    seeds=[0, 1, 2],
    # data
    zip_md5="66d04b70399bdcd522ecd8ab0ea275ca",            # EuroCropsML preprocess.zip, Zenodo record 15095445
    min_src_per_class=300,      # a crop is known to the model if it has >= 300 source parcels
    min_tgt_per_class=100,      # crops with < 100 parcels in a country are dropped; also the "present in target" rule
    max_per_class=1500,         # at most 1,500 parcels per crop and country
    seed_data=0,
    min_obs=3,                  # parcels with fewer valid observations are dropped
    t_store=96,                 # at most 96 observations per parcel are stored (evenly spaced)
    # sequence models
    t_max=64, d_model=128, n_layers=2, n_heads=4, dropout=0.1,
    epochs=25, batch_size=256, lr=1e-3, weight_decay=1e-4,
    # Presto and random forest
    presto_epochs=15, presto_lr_encoder=1e-4, presto_lr_head=1e-3, probe_steps=500, probe_lr=1e-2,
    rf_trees=200, rf_max_samples=0.5,
    # controls
    probe_l2_grid=[0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1], probe_val_frac=0.15,
    # source split
    src_cal_frac=0.15, src_test_frac=0.15,
    # conformal prediction
    alpha=0.10, n_splits=5, knn_k=200,
    w_clips={"covshift_weighted": (0.01, 100.0), "covshift_clip10": (0.1, 10.0), "covshift_clip1000": (0.001, 1000.0)},
    raps_lambda=0.01, raps_kreg=5,
    # few target labels
    few_label_m=[10, 25, 50, 100, 200], few_label_draws=40, few_mondrian_m=[10, 25, 50],
    disjoint_m=[25, 50, 100, 200], few_scores=["LAC", "APS_rand"],
    # regions and group metrics
    n_regions=10, min_region_size=200, min_group_eval=30, min_cell=20, n_boot=500,
    # success rule, fixed before the experiments
    success=dict(min_cov=0.88, max_full=0.10, max_size_ratio=1.25),
    max_hours=11.0,             # stop starting new runs after this many hours (Kaggle sessions end at 12 h)
)

FAST_DEV = dict(min_src_per_class=40, min_tgt_per_class=20, max_per_class=150, epochs=1, seeds=[0],
                presto_epochs=1, probe_steps=50, rf_trees=20, n_splits=2, knn_k=50, few_label_m=[10, 25],
                few_label_draws=4, few_mondrian_m=[10], disjoint_m=[25], n_regions=4, min_region_size=30, n_boot=50)


def make_config(experiment, fast_dev=False, **overrides):
    if experiment not in EXPERIMENTS:
        raise ValueError(f"unknown experiment {experiment!r}; choose from {sorted(EXPERIMENTS)}")
    cfg = copy.deepcopy(DEFAULTS)
    spec = EXPERIMENTS[experiment]
    grids = copy.deepcopy(spec["grids"])
    archs = [a for g, _ in grids for a in g]
    cfg.update(experiment=experiment, archs=archs, grids=grids, tag=spec["tag"], fast_dev=fast_dev)
    cfg.update(copy.deepcopy(DIAGNOSTICS), adapt_archs=list(spec["adapt_archs"]))
    if fast_dev:
        cfg.update(copy.deepcopy(FAST_DEV)); cfg.update(copy.deepcopy(DIAGNOSTICS_FAST_DEV))
    cfg.update(overrides)
    cfg["countries"] = sorted({c for _, src, tgt, _ in cfg["pairs"] for c in list(src) + [tgt]})
    return cfg
