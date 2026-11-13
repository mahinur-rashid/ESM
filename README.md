# Conformal crop maps across countries

Code for *Unnamed, Under-Covered or Uninformative? Conformal Crop Maps Across Countries*. It trains crop-type
classifiers in one country, calibrates conformal prediction sets with eight methods and target-label oracles, and
measures coverage, set size and regional coverage when the maps are applied to another country.

## Contents

| Path | What it is |
|---|---|
| `src/cropconf/config.py` | all settings; the two experiments (`main`, `controls`) |
| `src/cropconf/data.py` | EuroCropsML loading and preprocessing (see below) |
| `src/cropconf/models.py` | transformer, TempCNN, Presto (frozen probe, fine-tuned, controls), random forest |
| `src/cropconf/conformal.py` | conformity scores, calibration methods, shift estimates, metrics |
| `src/cropconf/experiment.py` | one run: train, calibrate with every method, evaluate on the target, few target labels |
| `src/cropconf/runner.py` | the run grid (resumable) and aggregation into CSV tables |
| `scripts/run_experiment.py` | runs one experiment end to end |
| `scripts/unknown_crops.py` | unknown-crop shares by HCAT taxonomy level (taxonomy table) |
| `scripts/make_paper_tables.py` | every other table, Figure 1 and the numbers quoted in the text, from `results/` |
| `scripts/make_synthetic_data.py` | a small synthetic archive in EuroCropsML's format, for a smoke test |
| `scripts/figure_data_standalone.py` | data for the dataset figures (map, crop mix, NDVI curves, example sets); self-contained |
| `scripts/plot_data_figures.py` | draws the map, crop-mix and NDVI figures and the example-sets table from its output |
| `kaggle/run_on_kaggle.ipynb` | launcher for a free Kaggle GPU |
| `kaggle/figure_data.ipynb` | the figure-data script as a Kaggle notebook (CPU) |
| `results/main` | per-run results of the `main` experiment, as used in the paper |
| `results/controls` | per-run results of the `controls` experiment, as used in the paper |
| `results/figure_data` | output of `figure_data_standalone.py` used for the dataset figures |
| `scripts/make_extra_figures.py` | the two concept figures and the coverage and trade-off figures, from the tables |
| `figures_src/fig_pipeline.tex` | the method diagram (TikZ; compile with pdflatex) |
| `data/hcat_names.csv` | names of the HCAT crop codes, from the EuroCrops taxonomy (Schneider et al., 2023) |

## Data

EuroCropsML v11 (Reuss & Macdonald, Zenodo, doi:[10.5281/zenodo.15095445](https://doi.org/10.5281/zenodo.15095445),
CC BY-SA 4.0; Reuss et al., 2025), file `preprocess.zip` (1.5 GB, md5 `66d04b70399bdcd522ecd8ab0ea275ca`). It
contains Sentinel-2 L1C time series for every declared parcel of Estonia, Latvia and Portugal in 2021, one `.npz` file
per parcel, named `<NUTS3>_<parcelID>_<HCAT code>.npz`, with `data` (observations × 13 bands, B1–B12 and B8A; each
value is the median over the parcel's pixels; observations with cloud probability above 0.5 removed by the dataset
authors), `dates` and `center` (parcel centroid, longitude and latitude). We do not use the dataset's own train/test
splits (`split.zip`).

## Preprocessing

All steps are in `src/cropconf/data.py` and `models.py`; settings are in `config.py`.

1. **Index.** Every file is assigned a country (the first two letters of its NUTS3 code) and a crop (its HCAT code).
   Natural crop frequencies, used for natural-frequency coverage and calibration, the unlabeled target pool and
   deployment coverage, are counted over all files.
2. **Crop selection.** Per country, crop codes with at least 100 parcels are kept, and at most 1,500 parcels are sampled
   per crop and country (random, seed 0).
3. **Observations.** Per parcel, observations with non-finite or no-data values (≤ −998, i.e. the −999 fill value) are dropped and the rest sorted
   by date. Parcels with fewer than 3 observations are dropped; parcels with more than 96 keep 96 evenly spaced ones.
   Reflectances are divided by 10,000 and clipped to [−5, 5]. Day of year is counted from 1 January 2021 (= 1).
4. **Labels.** In the *source* label space a model knows every crop with at least 300 source parcels; in the *shared*
   space, only crops that also have at least 100 target parcels. A target crop is *present* if it has at least 100
   target parcels; coverage is measured on target parcels of crops that are both known and present.
5. **Source split.** 70% training, 15% conformal calibration, 15% in-source test (per seed).
6. **Sequence-model input.** Bands are standardised with source-training statistics. During training each parcel
   contributes a random subset of up to 64 observations; at evaluation, up to 64 evenly spaced ones.
7. **Monthly composites** (Presto, random forest, monthly-transformer control). Per-band median of each parcel's
   observations within each calendar month of 2021; empty months are filled by linear interpolation between months,
   edge months take the nearest observed month. NDVI = (B8 − B4) / (B8 + B4) per month.
8. **Presto input.** Bands B2–B8A, B11, B12 (reflectance / 10,000) and NDVI in Presto's 17-channel layout; the
   Sentinel-1, ERA5 and SRTM channels are masked, the Dynamic World input is set to its missing class, and the parcel
   centroid is passed as latitude and longitude (one constant location, the mean centroid, in the location-free
   control).
9. **Random-forest input.** The 12 monthly values of the 10 bands and NDVI (132 features).
10. **Regions.** k-means (k = 10) on parcel centroids, separately for source and target parcels of each pair; regions
    with fewer than 200 parcels are not used for regional metrics. Within-country transfers split a country at the
    median latitude (Portugal, north → south) or longitude (Latvia, west → east).

## Running

```bash
pip install -r requirements.txt

# smoke test on synthetic data (a few minutes on a CPU)
python scripts/make_synthetic_data.py synthetic.zip
python scripts/run_experiment.py --experiment main --data synthetic.zip --out smoke --fast-dev

# the paper's two experiments (one GPU; on an NVIDIA T4 about 5.5 and 2.5 hours)
python scripts/run_experiment.py --experiment main     --data preprocess.zip --out outputs
python scripts/run_experiment.py --experiment controls --data preprocess.zip --out outputs
python scripts/unknown_crops.py --data preprocess.zip
```

Each run is saved as one JSON file under `outputs/runs_<experiment>/`; an interrupted experiment resumes where it
stopped. The tables are written to `outputs/tables_<experiment>/`. Presto's code and weights (`single_file_presto.py`,
`data/default_model.pt`, MIT license, Tseng et al., 2023) are downloaded from
[github.com/nasaharvest/presto](https://github.com/nasaharvest/presto) on first use, or read from `--presto-dir`.
On Kaggle, use `kaggle/run_on_kaggle.ipynb`.

## Reproducing the paper's tables and figures

```bash
python scripts/make_paper_tables.py results/main results/controls paper_out
```

This writes the LaTeX tables to `paper_out/tables/`, the three warning-sign and worst-region figures to `paper_out/figures/`, and the numbers quoted in the
text to `paper_out/paper_numbers.json`. The taxonomy table is the output of `scripts/unknown_crops.py`. With the
tables in `results/` this reproduces the paper's tables exactly; to use your own runs, pass `outputs/tables_main` and
`outputs/tables_controls`.

The results in `results/` were produced with this code on Kaggle (one NVIDIA T4). Re-running with the same seeds
reproduces every model exactly except TempCNN, whose GPU convolutions are not deterministic (target coverage changes
by about 0.008 on average). `results/controls` holds the tables that the paper uses for the control models
(`run_info`, `conformal_by_run`, `by_condition`, `extra_by_run`, `splits`, `synthetic_label_shift`,
`worst_region_by_split`); a few diagnostic columns that the paper reports only for the main models are absent there.


The other figures:

```
python scripts/make_extra_figures.py paper_out/tables paper_out/figures      # concept, coverage, trade-off
python scripts/plot_data_figures.py results/figure_data paper_out/figures     # map, crop mix, NDVI, example sets
```

## Experiments

Every run covers one of seven transfer pairs (LV→EE, EE→LV, PT→LV, LV→PT, LV+EE→PT, PT north→south, LV west→east),
one label space and one of three seeds, with four conformity scores (LAC, APS, randomised APS, RAPS), the calibration
methods and target-label oracles, an exchangeable and a spatially disjoint design (five target splits each), and
few-label calibration (random, from other regions, per region; split conformal and PAC thresholds).

| Experiment | Models | Runs |
|---|---|---|
| `main` | transformer, TempCNN (full time series); Presto frozen + linear probe, Presto fine-tuned, random forest (monthly composites), both label spaces; frozen Presto with one constant location and the transformer with an OTHER class (source label space) | 252 |
| `controls` | transformer on monthly composites (both label spaces); frozen Presto with seven probe variants (validation-tuned L2, unstandardised, clipped, L2 0.1 and 1, temperature scaling, MLP) and the transformer on codes merged to crop groups (source label space) | 210 |
