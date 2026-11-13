"""A small synthetic archive in EuroCropsML's format (12 crops, three countries with shifted phenology), for a quick
smoke test of the whole pipeline without downloading the dataset.

    python scripts/make_synthetic_data.py synthetic.zip
    python scripts/run_experiment.py --experiment sequence --data synthetic.zip --out smoke --fast-dev
"""
import io
import sys
import zipfile

import numpy as np

out = sys.argv[1] if len(sys.argv) > 1 else "synthetic.zip"
rng = np.random.RandomState(1)
n_cls = 12
codes = [f"33{rng.randint(10**7, 10**8)}" for _ in range(n_cls)]
peak = rng.uniform(110, 250, n_cls)
width = rng.uniform(25, 55, n_cls)
base = rng.uniform(500, 2500, 13)
amp = rng.uniform(-800, 2500, 13)
cls_off = rng.normal(0, 60, (n_cls, 13))          # weak spectral differences; timing carries most of the signal
all_days = np.arange(1, 366, 5)


def series(c, shift_fn, stretch, lat):
    keep = rng.rand(len(all_days)) > 0.4
    t = all_days[keep]
    tau = 180 + stretch * (t - 180) - shift_fn(lat)      # source-calendar time seen at date t
    curve = np.exp(-0.5 * ((tau - peak[c]) / width[c]) ** 2)
    x = base + cls_off[c] + np.outer(curve, amp) + rng.normal(0, 120, (len(t), 13))
    x = np.clip(x, 0, 10000).astype(np.float64)
    if rng.rand() < 0.1:
        x[0] = -999                                     # a no-data observation
    dates = (np.datetime64("2021-01-01") + (t - 1).astype("timedelta64[D]")).astype("datetime64[D]")
    return x, dates


countries = {
    "LV": dict(n=400, lon=(21, 28), lat=(56, 58), shift=lambda lat: 0.0, stretch=1.0, regions=["LV003", "LV005", "LV006"]),
    "EE": dict(n=150, lon=(22, 28), lat=(57.5, 59.6), shift=lambda lat: 8.0, stretch=1.0, regions=["EE001", "EE004"]),
    "PT": dict(n=150, lon=(-9, -6.5), lat=(37, 42), shift=lambda lat: -35 + 6 * (lat - 39.5), stretch=1.1,
               regions=["PT111", "PT16B", "PT181"]),
}
pid = 100000
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
    for cc, spec in countries.items():
        for c in range(n_cls):
            for _ in range(spec["n"]):
                lat = rng.uniform(*spec["lat"]); lon = rng.uniform(*spec["lon"])
                x, d = series(c, spec["shift"], spec["stretch"], lat)
                buf = io.BytesIO()
                np.savez(buf, data=x, dates=d, center=np.array([lon, lat]))
                pid += 1
                zf.writestr(f"preprocess/{rng.choice(spec['regions'])}_{pid}_{codes[c]}.npz", buf.getvalue())
print("wrote", out)
