# Data for the paper's dataset figures. Self-contained: needs only EuroCropsML's preprocess zip (and, optionally, the
# parcel cache of an earlier run, which makes step 3 fast). Output: figure_data/ and figure_data.zip.
import glob, io, os, shutil, time, zipfile
import numpy as np
import pandas as pd

ZIP_PATH = "/kaggle/input/eurocropsml/preprocess_zip.bin"  # searched for under /kaggle/input if not here
CACHE_PATH = None  # optional parcel cache (cache_ext_*) of an earlier run; searched for automatically
OUT = "/kaggle/working/figure_data"

COUNTRIES = ["EE", "LV", "PT"]
MIN_PER_CLASS, MAX_PER_CLASS, SEED_DATA, MIN_OBS, T_STORE, MIN_SRC = 100, 1500, 0, 3, 96, 300
N_NATURAL, N_EXAMPLES = 6000, 3
REF_DATE = np.datetime64("2021-01-01")
MONTH_START = np.array([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335])
S2_KEEP = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
CACHE_KEYS = ("X", "DOY", "NOBS", "CENTER", "KEEP_ROWS")


def find(path, pattern):
    if path and os.path.exists(path):
        return path
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    return hits[0] if hits else None


def index_archive(archive):
    rows = []
    for name in archive.namelist():
        if name.endswith(".npz"):
            parts = os.path.basename(name)[:-4].rsplit("_", 2)
            if len(parts) == 3:
                rows.append((name, parts[0], parts[0][:2].upper(), parts[2]))
    return pd.DataFrame(rows, columns=["member", "nuts3", "country", "cls"])


def npz_keys(z):
    keys = list(z.keys())
    d = "data" if "data" in keys else max((k for k in keys if z[k].ndim == 2), key=lambda k: z[k].size)
    t = "dates" if "dates" in keys else next(k for k in keys if "date" in k.lower())
    c = "center" if "center" in keys else next((k for k in keys if "cent" in k.lower()), None)
    return d, t, c


def to_doy(dates):
    d = np.asarray(dates)
    if d.dtype.kind == "M":
        d = d.astype("datetime64[D]")
    elif d.dtype.kind in "USO":
        d = pd.to_datetime(d.astype(str)).values.astype("datetime64[D]")
    elif d.dtype.kind in "iuf" and np.nanmax(d) <= 400:
        return d.astype(np.float32)
    elif d.dtype.kind in "iuf" and np.nanmin(d) > 19000000:
        d = pd.to_datetime(d.astype(np.int64).astype(str), format="%Y%m%d").values.astype("datetime64[D]")
    return ((d - REF_DATE).astype(np.int64) + 1).astype(np.float32)


def read_raw(archive, member, keys):
    """Cleaned observations of one parcel: day of year, reflectance (obs x 13, /10,000), centroid."""
    with np.load(io.BytesIO(archive.read(member)), allow_pickle=False) as z:
        x = np.asarray(z[keys[0]], dtype=np.float64)
        doy = to_doy(z[keys[1]])
        c = np.asarray(z[keys[2]], dtype=np.float64).reshape(-1)[:2] if keys[2] else np.array([np.nan, np.nan])
    if x.shape[0] != len(doy) and x.shape[1] == len(doy):
        x = x.T
    ok = np.isfinite(x).all(1) & (x > -998).all(1) & np.isfinite(doy)
    x, doy = x[ok], doy[ok]
    o = np.argsort(doy, kind="stable")
    return doy[o], x[o] / (10000.0 if np.nanmedian(x) > 2 else 1.0), c


def ndvi(x):
    red, nir = x[..., 3], x[..., 7]
    return np.where(nir + red > 0, (nir - red) / np.maximum(nir + red, 1e-6), np.nan)


def monthly_series(doy, v):
    m = np.searchsorted(MONTH_START, doy, side="right") - 1
    return pd.Series(v).groupby(m).median().reindex(range(12)).interpolate(limit_direction="both").values


def select_parcels(index):
    rng = np.random.RandomState(SEED_DATA)
    parts = []
    for _, g in index[index.country.isin(COUNTRIES)].groupby(["cls", "country"]):
        if len(g) >= MIN_PER_CLASS:
            parts.append(g.sample(min(len(g), MAX_PER_CLASS), random_state=rng))
    return pd.concat(parts).reset_index(drop=True)


def load_cache(path, n_sel):
    cands = [path] if path else []
    cands += sorted(glob.glob("/kaggle/input/**/cache_ext_*", recursive=True))
    for c in cands:
        try:
            if os.path.isfile(c):
                with np.load(c, allow_pickle=False) as z:
                    a = {k: z[k] for k in CACHE_KEYS}
            elif os.path.isdir(c):
                sub = [r for r, _, f in os.walk(c) if "X.npy" in f]
                if not sub:
                    continue
                a = {k: np.load(os.path.join(sub[0], k + ".npy"), allow_pickle=False) for k in CACHE_KEYS}
            else:
                continue
            if a["X"].shape[0] == len(a["KEEP_ROWS"]) and a["KEEP_ROWS"].max() < n_sel and a["X"].shape[1] == T_STORE:
                print("cache loaded:", c)
                return a
        except Exception as e:
            print("cache not usable:", c, e)
    return None


def read_selected(archive, sel, keys):
    n0 = len(sel)
    X = np.zeros((n0, T_STORE, 13), np.float16); DOY = np.zeros((n0, T_STORE), np.float32)
    NOBS = np.zeros(n0, np.int32); CENTER = np.full((n0, 2), np.nan); keep = np.zeros(n0, bool)
    t0 = time.time()
    for i, member in enumerate(sel.member.values):
        doy, x, c = read_raw(archive, member, keys)
        if len(doy) >= MIN_OBS and x.shape[1] == 13:
            if len(doy) > T_STORE:
                k = np.round(np.linspace(0, len(doy) - 1, T_STORE)).astype(int); doy, x = doy[k], x[k]
            X[i, :len(doy)] = np.clip(x, -5, 5).astype(np.float16); DOY[i, :len(doy)] = doy
            NOBS[i] = len(doy); CENTER[i] = c; keep[i] = True
        if (i + 1) % 20000 == 0:
            print(f"  {i + 1}/{n0} parcels read ({time.time() - t0:.0f}s)")
    r = np.where(keep)[0]
    return dict(X=X[r], DOY=DOY[r], NOBS=NOBS[r], CENTER=CENTER[r], KEEP_ROWS=r)


def rf_features(a):
    X, D, N = a["X"], a["DOY"], a["NOBS"]
    n, T, C = X.shape
    valid = np.arange(T)[None, :] < N[:, None]
    rows, cols = np.nonzero(valid)
    month = np.searchsorted(MONTH_START, D[rows, cols], side="right") - 1
    obs = pd.DataFrame(X[rows, cols].astype(np.float32)); obs["p"], obs["m"] = rows, month
    med = obs.groupby(["p", "m"]).median()
    M = np.full((n, 12, C), np.nan, np.float32)
    M[med.index.get_level_values(0).values, med.index.get_level_values(1).values] = med.values
    flat = pd.DataFrame(M.transpose(0, 2, 1).reshape(-1, 12)).interpolate(axis=1, limit_direction="both")
    M = np.nan_to_num(flat.values.reshape(n, C, 12).transpose(0, 2, 1).astype(np.float32))
    nd = np.nan_to_num(ndvi(M)).astype(np.float32)
    return np.concatenate([M[:, :, S2_KEEP].reshape(n, -1), nd], 1)


def lac(model, Xc, yc, Xt, alpha=0.1):
    col = {c: i for i, c in enumerate(model.classes_)}
    pc = model.predict_proba(Xc)
    s = np.sort(1 - pc[np.arange(len(yc)), [col[c] for c in yc]])
    q = s[min(len(s) - 1, int(np.ceil((len(s) + 1) * (1 - alpha))) - 1)]
    pt = model.predict_proba(Xt)
    return pt, pt >= 1 - q - 1e-12


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.RandomState(0)
    zpath = find(ZIP_PATH, "*preprocess*")
    print("archive:", zpath)
    archive = zipfile.ZipFile(zpath)
    index = index_archive(archive)
    index = index[index.country.isin(COUNTRIES)]
    print("parcels per country:", index.country.value_counts().to_dict())
    counts = index.groupby(["cls", "country"]).size().unstack(fill_value=0)
    counts.to_csv(f"{OUT}/counts.csv")
    with np.load(io.BytesIO(archive.read(index.member.iloc[0])), allow_pickle=False) as z:
        keys = npz_keys(z)

    # 1. natural-frequency sample
    rows, t0 = [], time.time()
    for ctry, g in index.groupby("country"):
        for member, cls in g.sample(min(N_NATURAL, len(g)), random_state=rng)[["member", "cls"]].values:
            try:
                doy, x, c = read_raw(archive, member, keys)
            except Exception:
                continue
            if len(doy) >= MIN_OBS:
                rows.append([ctry, cls, c[0], c[1], len(doy), doy.min(), doy.max()] + list(monthly_series(doy, ndvi(x))))
        print(f"1. natural sample {ctry} done ({time.time() - t0:.0f}s)")
    pd.DataFrame(rows, columns=["country", "cls", "lon", "lat", "nobs", "first_doy", "last_doy"]
                 + [f"ndvi_m{m + 1:02d}" for m in range(12)]).to_csv(f"{OUT}/natural_sample.csv", index=False, float_format="%.4f")

    # 2. example parcels: six most common crops per country, plus olives and vineyards
    picks = [(c, k) for c in COUNTRIES for k in counts[c].sort_values(ascending=False).head(6).index]
    picks += [("PT", k) for k in ("3303050000", "3303050100", "3303060000") if k in counts.index and counts.loc[k, "PT"] > 0]
    ex = []
    for ctry, cls in dict.fromkeys(picks):
        g = index[(index.country == ctry) & (index.cls == cls)]
        for j, member in enumerate(g.sample(min(N_EXAMPLES, len(g)), random_state=rng).member.values):
            doy, x, c = read_raw(archive, member, keys)
            ex += [[ctry, cls, j, d, r[3], r[7], v] for d, r, v in zip(doy, x, ndvi(x))]
    pd.DataFrame(ex, columns=["country", "cls", "parcel", "doy", "red", "nir", "ndvi"]).to_csv(
        f"{OUT}/examples.csv", index=False, float_format="%.4f")
    print("2. examples written")

    # 3. real LAC sets of a random forest on the paper's monthly features
    from sklearn.ensemble import RandomForestClassifier
    sel = select_parcels(index)
    print("selected parcels:", len(sel))
    a = load_cache(find(CACHE_PATH, "cache_ext_*"), len(sel))
    if a is None:
        print("no usable cache: reading the selected parcels (slower)")
        a = read_selected(archive, sel, keys)
    meta = sel.iloc[a["KEEP_ROWS"]].reset_index(drop=True)
    y, ctry = meta.cls.values.astype(str), meta.country.values.astype(str)
    feat = rf_features(a)
    summary = []
    for pname, src, tgt in (("LV_to_EE", "LV", "EE"), ("LV_to_PT", "LV", "PT")):
        s_idx = np.where(ctry == src)[0]
        vc = pd.Series(y[s_idx]).value_counts()
        s_idx = s_idx[np.isin(y[s_idx], vc.index[vc >= MIN_SRC])]
        perm = rng.permutation(s_idx)
        n_tr, n_cal = int(0.7 * len(perm)), int(0.15 * len(perm))
        tr, cal, te = perm[:n_tr], perm[n_tr:n_tr + n_cal], perm[n_tr + n_cal:]
        rf = RandomForestClassifier(n_estimators=200, max_samples=0.5, n_jobs=-1, random_state=0).fit(feat[tr], y[tr])
        col = {c: i for i, c in enumerate(rf.classes_)}
        _, S_te = lac(rf, feat[cal], y[cal], feat[te])
        cov_src = S_te[np.arange(len(te)), [col[c] for c in y[te]]].mean()
        t_idx = np.where(ctry == tgt)[0]
        pt, S = lac(rf, feat[cal], y[cal], feat[t_idx])
        yt = y[t_idx]; kn = np.isin(yt, rf.classes_)
        cov = np.mean([S[i, col[yt[i]]] for i in np.where(kn)[0]])
        summary.append(f"{pname}: {len(rf.classes_)} source crops | in-source coverage {cov_src:.3f} (size {S_te.sum(1).mean():.1f}) | "
                       f"target coverage on known crops {cov:.3f}, size {S.sum(1).mean():.1f}, accuracy "
                       f"{np.mean(rf.classes_[pt.argmax(1)][kn] == yt[kn]):.3f}")
        print("3.", summary[-1])
        out = []
        for i in rng.choice(len(t_idx), size=min(300, len(t_idx)), replace=False):
            o = np.argsort(-pt[i])
            out.append(dict(true=yt[i], known=bool(kn[i]), set_size=int(S[i].sum()), covered=bool(kn[i] and S[i, col[yt[i]]]),
                            set="|".join(rf.classes_[o][S[i][o]][:40]),
                            top5="|".join(f"{rf.classes_[k]}:{pt[i, k]:.3f}" for k in o[:5])))
        pd.DataFrame(out).to_csv(f"{OUT}/sets_{pname}.csv", index=False)
    open(f"{OUT}/summary.txt", "w").write("\n".join(summary) + "\n")
    shutil.make_archive(OUT, "zip", OUT)
    print("DONE: download figure_data.zip from the Output tab")


main()
