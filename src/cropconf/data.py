"""EuroCropsML loading and preprocessing.

EuroCropsML's `preprocess.zip` holds one .npz file per parcel, named `<NUTS3>_<parcelID>_<HCAT code>.npz`, with
`data` (observations x 13 Sentinel-2 L1C bands, the per-date median over the parcel's pixels, cloudy observations
already removed by the dataset authors), `dates` and `center` (parcel centroid, [lon, lat]).

Our preprocessing, in order:
  1. index all files by country (first two letters of NUTS3) and crop code;
  2. per country, keep crop codes with >= `min_tgt_per_class` parcels and sample at most `max_per_class` of them
     (random, seed `seed_data`);
  3. per parcel, drop observations with non-finite or no-data (<= -998, the -999 fill value) values, sort by date, drop parcels with
     fewer than `min_obs` observations, keep at most `t_store` evenly spaced observations, divide by 10,000 and clip
     to [-5, 5];
  4. (for Presto, the random forest and the monthly control) per-band median per calendar month, empty months
     filled by linear interpolation between months (edge months take the nearest observed month), plus NDVI.
Day of year is counted from 1 January 2021 (= day 1).
"""
import glob
import hashlib
import io
import os
import time
import zipfile

import numpy as np
import pandas as pd

REF_DATE = np.datetime64("2021-01-01")
CACHE_KEYS = ("X", "DOY", "NOBS", "CENTER", "KEEP_ROWS")
MONTH_START = np.array([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335])
S2_KEEP = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]   # B2..B8A, B11, B12 of B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B10 B11 B12


# ----------------------------------------------------------------------------------------------------------------
# archive access
# ----------------------------------------------------------------------------------------------------------------
class FolderArchive:
    """Same interface as zipfile.ZipFile (namelist/read) for an unpacked folder of .npz files."""

    def __init__(self, root):
        self.root = root
        self.names = []
        for dirpath, _, filenames in os.walk(root):
            rel = os.path.relpath(dirpath, root)
            self.names += [os.path.join(rel, f) for f in filenames if f.endswith(".npz")]

    def namelist(self):
        return self.names

    def read(self, name):
        with open(os.path.join(self.root, name), "rb") as f:
            return f.read()


def find_archive(path=None, roots=("/kaggle/input",)):
    """`path`, $ECML_ZIP, or a copy anywhere under `roots`: preprocess.zip, a renamed copy whose name contains
    "preprocess" (e.g. preprocess_zip.bin, which Kaggle keeps as one file), or an unpacked folder of .npz files."""
    for p in (path, os.environ.get("ECML_ZIP")):
        if p and os.path.exists(p):
            return p
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames.sort()
            for f in sorted(filenames):
                full = os.path.join(dirpath, f)
                if "preprocess" in f.lower() and os.path.isfile(full) and zipfile.is_zipfile(full):
                    return full
            if any(f.endswith(".npz") for f in filenames[:50]):          # unpacked archive: stop descending here
                return os.path.dirname(dirpath) if os.path.basename(dirpath) == "2021" else dirpath
    raise FileNotFoundError("EuroCropsML preprocess.zip not found: pass --data or set ECML_ZIP "
                            f"(searched {list(roots)}; download: https://zenodo.org/records/15095445)")


def open_archive(path):
    return FolderArchive(path) if os.path.isdir(path) else zipfile.ZipFile(path)


def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def index_archive(archive):
    """One row per parcel file: member name, NUTS3 region, country, HCAT crop code."""
    rows = []
    for name in archive.namelist():
        if not name.endswith(".npz"):
            continue
        parts = os.path.basename(name)[:-4].rsplit("_", 2)    # <NUTS3>_<parcelID>_<HCAT code>
        if len(parts) != 3:
            continue
        nuts3, _, cls = parts
        rows.append((name, nuts3, nuts3[:2].upper(), cls))
    return pd.DataFrame(rows, columns=["member", "nuts3", "country", "cls"])


def npz_keys(z):
    keys = list(z.keys())
    data_key = "data" if "data" in keys else max((k for k in keys if z[k].ndim == 2), key=lambda k: z[k].size, default=None)
    date_key = "dates" if "dates" in keys else next((k for k in keys if "date" in k.lower()), None)
    cen_key = "center" if "center" in keys else next((k for k in keys if "cent" in k.lower()), None)
    if data_key is None or date_key is None:
        raise KeyError(f"could not find data/dates arrays; keys present: {keys}")
    return data_key, date_key, cen_key


def to_doy(dates):
    d = np.asarray(dates)
    if d.dtype.kind == "M":
        d = d.astype("datetime64[D]")
    elif d.dtype.kind in "USO":
        d = pd.to_datetime(d.astype(str)).values.astype("datetime64[D]")
    elif d.dtype.kind in "iuf" and np.nanmax(d) <= 400:
        return d.astype(np.float32)                 # already day of year
    elif d.dtype.kind in "iuf" and np.nanmin(d) > 19000000:
        d = pd.to_datetime(d.astype(np.int64).astype(str), format="%Y%m%d").values.astype("datetime64[D]")
    else:
        raise ValueError(f"unrecognised date format: dtype={d.dtype}, sample={d[:3]}")
    return ((d - REF_DATE).astype(np.int64) + 1).astype(np.float32)


# ----------------------------------------------------------------------------------------------------------------
# parcel selection and time series
# ----------------------------------------------------------------------------------------------------------------
def merge_code(code, level):
    """Truncate a 10-digit HCAT code to its first `level` digits (each pair of digits is one taxonomy level)."""
    return code[:level].ljust(10, "0")


def select_parcels(index, cfg):
    """Per country, crop codes with >= min_tgt_per_class parcels, at most max_per_class parcels each."""
    rng = np.random.RandomState(cfg["seed_data"])
    parts = []
    for (_, _), g in index[index.country.isin(cfg["countries"])].groupby(["cls", "country"]):
        if len(g) >= cfg["min_tgt_per_class"]:
            parts.append(g.sample(min(len(g), cfg["max_per_class"]), random_state=rng))
    return pd.concat(parts).reset_index(drop=True)


def _load_member(archive, member, keys, cfg):
    with np.load(io.BytesIO(archive.read(member)), allow_pickle=False) as z:
        data = np.asarray(z[keys[0]], dtype=np.float32)
        dates = z[keys[1]]
        center = np.asarray(z[keys[2]], dtype=np.float64).reshape(-1)[:2] if keys[2] else np.array([np.nan, np.nan])
    if data.ndim != 2:
        return None
    if data.shape[0] != len(dates) and data.shape[1] == len(dates):
        data = data.T
    doy = to_doy(dates)
    ok = np.isfinite(data).all(1) & (data > -998).all(1) & np.isfinite(doy)
    data, doy = data[ok], doy[ok]
    order = np.argsort(doy, kind="stable")
    data, doy = data[order], doy[order]
    if len(doy) < cfg["min_obs"]:
        return None
    if len(doy) > cfg["t_store"]:
        keep = np.round(np.linspace(0, len(doy) - 1, cfg["t_store"])).astype(int)
        data, doy = data[keep], doy[keep]
    return data, doy, center


def _cache_name(cfg, n_selected):
    return f"cache_ext_{'-'.join(cfg['countries'])}_{n_selected}_{cfg['max_per_class']}{'_dev' if cfg['fast_dev'] else ''}.npz"


def _find_cache(path, n_selected, t_store, search_dirs):
    """A cache file under `path`, or a copy in `search_dirs` (also renamed "... (1)" or unpacked into .npy files)."""
    stem = os.path.basename(path)[:-4]
    cands = [path] + [c for r in search_dirs if os.path.isdir(r)
                      for c in sorted(glob.glob(os.path.join(glob.escape(r), "**", glob.escape(stem) + "*"), recursive=True))]
    for c in cands:
        try:
            if os.path.isfile(c):
                with np.load(c, allow_pickle=False) as z:
                    arrs = {k: z[k] for k in CACHE_KEYS}
            elif os.path.isdir(c) and all(os.path.isfile(os.path.join(c, k + ".npy")) for k in CACHE_KEYS):
                arrs = {k: np.load(os.path.join(c, k + ".npy"), allow_pickle=False) for k in CACHE_KEYS}
            else:
                continue
        except Exception:
            continue
        if (arrs["X"].shape[0] == len(arrs["KEEP_ROWS"]) and arrs["KEEP_ROWS"].max() < n_selected
                and arrs["X"].shape[1] == t_store):
            return arrs, c
    return None, None


class Data:
    """All parcels of all countries in the experiment, with their time series and metadata.

    Attributes: meta (DataFrame: member, nuts3, country, cls), cls_code, country, center ([lon, lat]),
    X (parcels x t_store x 13, float16, reflectance), doy, nobs, counts (full-archive parcel counts per crop and
    country, used for natural frequencies), n, c_in.
    """

    def __init__(self, archive_path, cfg, cache_dir, cache_search_dirs=("/kaggle/input",), log=print):
        self.cfg = cfg
        archive = open_archive(archive_path)
        index = index_archive(archive)
        log(f"parcel files: {len(index)} | " + ", ".join(f"{k}: {v}" for k, v in index.country.value_counts().items()))
        self.counts = index.groupby(["cls", "country"]).size().unstack(fill_value=0).reindex(
            columns=cfg["countries"], fill_value=0)
        sel = select_parcels(index, cfg)
        log(f"parcels selected: {len(sel)}")
        cache = os.path.join(cache_dir, _cache_name(cfg, len(sel)))
        arrs, found = _find_cache(cache, len(sel), cfg["t_store"], list(cache_search_dirs))
        if arrs is not None:
            log(f"loaded cache: {found}")
        else:
            arrs = self._read_series(archive, sel, cfg, log)
            os.makedirs(cache_dir, exist_ok=True)
            np.savez(cache, **arrs)
        self.X, self.doy, self.nobs, self.center = arrs["X"], arrs["DOY"], arrs["NOBS"], arrs["CENTER"]
        self.meta = sel.iloc[arrs["KEEP_ROWS"]].reset_index(drop=True)
        self.cls_code = self.meta.cls.values.astype(str)
        self.country = self.meta.country.values.astype(str)
        self.n, self.c_in = len(self.meta), self.X.shape[2]
        self._monthly = None

    @staticmethod
    def _read_series(archive, sel, cfg, log):
        with np.load(io.BytesIO(archive.read(sel.member.iloc[0])), allow_pickle=False) as z0:
            keys = npz_keys(z0)
            n_bands = z0[keys[0]].shape[1] if z0[keys[0]].shape[0] == len(z0[keys[1]]) else z0[keys[0]].shape[0]
            scale = 10000.0 if np.nanmedian(z0[keys[0]]) > 2.0 else 1.0
        n0, T = len(sel), cfg["t_store"]
        X = np.zeros((n0, T, n_bands), dtype=np.float16)
        DOY = np.zeros((n0, T), dtype=np.float32)
        NOBS = np.zeros(n0, dtype=np.int32)
        CENTER = np.full((n0, 2), np.nan)
        keep = np.zeros(n0, dtype=bool)
        t0 = time.time()
        for i, member in enumerate(sel.member.values):
            r = _load_member(archive, member, keys, cfg)
            if r is not None and r[0].shape[1] == n_bands:
                d, t, c = r
                X[i, :len(t)] = np.clip(d / scale, -5, 5).astype(np.float16)
                DOY[i, :len(t)] = t; NOBS[i] = len(t); CENTER[i] = c; keep[i] = True
            if (i + 1) % 25000 == 0:
                log(f"  {i + 1}/{n0} parcels read ({time.time() - t0:.0f}s)")
        rows = np.where(keep)[0]
        return dict(X=X[rows], DOY=DOY[rows], NOBS=NOBS[rows], CENTER=CENTER[rows], KEEP_ROWS=rows)

    # ------------------------------------------------------------------------------------------------------------
    def codes(self, level=10):
        """Crop code of every parcel, merged to its first `level` digits of the HCAT code (10 = exact code)."""
        return self.cls_code if level == 10 else np.array([merge_code(c, level) for c in self.cls_code])

    def counts_at(self, level=10):
        """Full-archive parcel counts per crop and country, with crop codes merged to `level` digits."""
        if level == 10:
            return self.counts
        return self.counts.groupby([merge_code(c, level) for c in self.counts.index]).sum()

    def monthly(self):
        """Twelve monthly per-band medians (parcels x 12 x 13) and NDVI (parcels x 12), computed once."""
        if self._monthly is None:
            T = self.X.shape[1]
            valid = np.arange(T)[None, :] < self.nobs[:, None]
            rows, cols = np.nonzero(valid)
            month = np.searchsorted(MONTH_START, self.doy[rows, cols], side="right") - 1
            obs = pd.DataFrame(self.X[rows, cols].astype(np.float32))
            obs["parcel"], obs["month"] = rows, month
            med = obs.groupby(["parcel", "month"]).median()
            M = np.full((self.n, 12, self.c_in), np.nan, dtype=np.float32)
            M[med.index.get_level_values(0).values, med.index.get_level_values(1).values] = med.values
            flat = pd.DataFrame(M.transpose(0, 2, 1).reshape(-1, 12)).interpolate(axis=1, limit_direction="both")
            M = np.nan_to_num(flat.values.reshape(self.n, self.c_in, 12).transpose(0, 2, 1).astype(np.float32), nan=0.0)
            b4, b8 = M[..., 3], M[..., 7]
            ndvi = np.where((b8 + b4) > 0, (b8 - b4) / np.maximum(b8 + b4, 1e-6), 0.0).astype(np.float32)
            self._monthly = (M, ndvi)
        return self._monthly

    def rf_features(self):
        """132 random-forest features: 12 months x (10 bands + NDVI); also their standardised copy."""
        M, ndvi = self.monthly()
        feat = np.concatenate([M[:, :, S2_KEEP].reshape(self.n, -1), ndvi], 1).astype(np.float32)
        return feat, (feat - feat.mean(0)) / (feat.std(0) + 1e-6)
