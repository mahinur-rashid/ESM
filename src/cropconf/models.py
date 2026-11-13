"""Parcel classifiers. Every model returns class probabilities and features for a set of parcels.

Models (name used in the configuration):
  transformer, tempcnn         trained from scratch on the full irregular time series
  transformer_monthly          the transformer on the 12 monthly composites (control)
  presto_frozen                frozen Presto encoder + linear probe
  presto_frozen_l2             same, probe L2 penalty chosen on a validation split of the source training fields
  presto_frozen_noloc(_l2)     same, with one constant location for every parcel
  presto_frozen_raw/_clip/_l2s1/_l2s2/_ts/_mlp   probe controls (see FROZEN)
  presto_finetune              Presto encoder and linear head fine-tuned end to end
  random_forest                random forest on the flattened monthly composites and NDVI
"""
import importlib.util
import math
import os
import random
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier

from .data import S2_KEEP

SEQUENCE = ("transformer", "tempcnn")
# frozen-Presto probes: location input, L2 chosen on validation, standardise, clip, fixed L2, temperature, MLP
_P = dict(location="loc", tune=False, standardize=True, clip=None, l2=0.0, ts=False, mlp=False)
FROZEN = {
    "presto_frozen": dict(_P),
    "presto_frozen_l2": dict(_P, tune=True),
    "presto_frozen_noloc": dict(_P, location="noloc"),
    "presto_frozen_noloc_l2": dict(_P, location="noloc", tune=True),
    # probe controls
    "presto_frozen_raw": dict(_P, standardize=False),          # probe on unstandardised embeddings
    "presto_frozen_clip": dict(_P, clip=3.0),                  # standardised, clipped to +-3 source SDs
    "presto_frozen_l2s1": dict(_P, l2=1e-1),                   # strong fixed L2
    "presto_frozen_l2s2": dict(_P, l2=1.0),                    # very strong fixed L2
    "presto_frozen_ts": dict(_P, ts=True),                     # temperature scaling on held-out source fields
    "presto_frozen_mlp": dict(_P, mlp=True),                   # one-hidden-layer MLP probe
}
VARIANTS = ("_openset", "_merged6")      # experiment variants of a base model (see experiment.parse_arch)


def base_arch(arch):
    for v in VARIANTS:
        if arch.endswith(v):
            return arch[: -len(v)]
    return arch
MONTH_MID = [16, 46, 75, 106, 136, 167, 197, 228, 259, 289, 320, 350]
PRESTO_URL = "https://raw.githubusercontent.com/nasaharvest/presto/main/"


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ----------------------------------------------------------------------------------------------------------------
# sequence models
# ----------------------------------------------------------------------------------------------------------------
class DOYEncoding(nn.Module):
    def __init__(self, d, base=1000.0):
        super().__init__()
        self.register_buffer("inv", 1.0 / (base ** (torch.arange(0, d, 2).float() / d)))

    def forward(self, t):
        ang = t[..., None] * self.inv
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class TSTransformer(nn.Module):
    def __init__(self, c_in, n_cls, d=128, layers=2, heads=4, drop=0.1):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(c_in, d), nn.LayerNorm(d))
        self.pe = DOYEncoding(d)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, drop, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d); self.head = nn.Linear(d, n_cls)

    def forward(self, x, doy, mask, return_feat=False):
        h = self.enc(self.inp(x) + self.pe(doy), src_key_padding_mask=~mask)
        m = mask.float()[..., None]
        h = self.norm((h * m).sum(1) / m.sum(1).clamp(min=1.0))
        logits = self.head(h)
        return (logits, h) if return_feat else logits


class TempCNN(nn.Module):
    def __init__(self, c_in, n_cls, d=128, drop=0.2):
        super().__init__()
        self.pe = DOYEncoding(16)
        chans = [c_in + 16, d, d, d]
        self.convs = nn.ModuleList([nn.Conv1d(chans[i], chans[i + 1], 5, padding=2) for i in range(3)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(d) for _ in range(3)])
        self.drop = nn.Dropout(drop); self.norm = nn.LayerNorm(d); self.head = nn.Linear(d, n_cls)

    def forward(self, x, doy, mask, return_feat=False):
        h = torch.cat([x, self.pe(doy)], -1).transpose(1, 2)
        m = mask.float()[:, None, :]
        for conv, bn in zip(self.convs, self.bns):
            h = self.drop(F.relu(bn(conv(h * m))))
        h = self.norm((h * m).sum(2) / m.sum(2).clamp(min=1.0))
        logits = self.head(h)
        return (logits, h) if return_feat else logits


class SeriesInput:
    """Full irregular series: random subsets of up to t_max observations in training, evenly spaced at evaluation."""

    def __init__(self, data, cfg, device):
        self.data, self.cfg, self.device = data, cfg, device
        self.T = data.X.shape[1]
        self.X = torch.from_numpy(data.X).to(device)
        self.DOY = torch.from_numpy(data.doy).to(device)
        self.NOBS = torch.from_numpy(data.nobs.astype(np.int64)).to(device)
        self.pos = torch.arange(self.T, device=device)
        self.mean = torch.zeros(data.c_in, device=device); self.std = torch.ones(data.c_in, device=device)

    def set_norm(self, train_ids):
        valid = np.arange(self.T)[None, :] < self.data.nobs[train_ids][:, None]
        obs = self.data.X[train_ids][valid].astype(np.float32)
        self.mean = torch.as_tensor(obs.mean(0), device=self.device)
        self.std = torch.as_tensor(obs.std(0) + 1e-6, device=self.device)

    def batch(self, ids, train):
        dev = self.device
        ids_t = torch.as_tensor(ids, device=dev, dtype=torch.long)
        n = self.NOBS[ids_t]; tm = self.cfg["t_max"]; B = len(ids_t)
        if train:
            r = torch.rand(B, self.T, device=dev).masked_fill(self.pos[None, :] >= n[:, None], 2.0)
            vals, sel = torch.topk(r, tm, dim=1, largest=False)
            order = torch.argsort(sel, dim=1)
            sel, vals = sel.gather(1, order), vals.gather(1, order)
            mask = vals < 2.0
        else:
            pos = torch.arange(tm, device=dev, dtype=torch.float32)[None, :].expand(B, tm)
            nf = n[:, None].float()
            spread = torch.round(pos * (nf - 1) / (tm - 1)).long()
            long_seq = n[:, None] > tm
            sel = torch.where(long_seq, spread, pos.long()).clamp(max=self.T - 1)
            mask = (pos < nf) | long_seq
        x = ((self.X[ids_t[:, None], sel].float() - self.mean) / self.std).clamp(-10, 10)
        x = x.masked_fill(~mask[..., None], 0.0)
        doy = self.DOY[ids_t[:, None], sel].masked_fill(~mask, 0.0)
        return x, doy, mask


class MonthlyInput:
    """Twelve monthly composites of all 13 bands, dated at mid-month, no subsampling."""

    def __init__(self, data, device):
        self.data, self.device = data, device
        self.M = torch.from_numpy(data.monthly()[0]).to(device)
        self.mid = torch.tensor(MONTH_MID, dtype=torch.float32, device=device)

    def set_norm(self, train_ids):
        flat = self.data.monthly()[0][train_ids].reshape(-1, self.data.c_in)
        self.mean = torch.as_tensor(flat.mean(0), device=self.device)
        self.std = torch.as_tensor(flat.std(0) + 1e-6, device=self.device)

    def batch(self, ids, train=False):
        ids_t = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        x = ((self.M[ids_t] - self.mean) / self.std).clamp(-10, 10)
        B = len(ids_t)
        return x, self.mid[None, :].expand(B, 12), torch.ones(B, 12, dtype=torch.bool, device=self.device)


def train_network(mdl, inp, train, y_t, cfg):
    """AdamW, one-cycle schedule, gradient clipping at 1; batches from `inp.batch(ids, train=True)`."""
    opt = torch.optim.AdamW(mdl.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total = cfg["epochs"] * math.ceil(len(train) / cfg["batch_size"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=total, pct_start=0.1) if total >= 20 else None
    for _ in range(cfg["epochs"]):
        mdl.train()
        perm = np.random.permutation(train)
        for i in range(0, len(perm), cfg["batch_size"]):
            ids = perm[i:i + cfg["batch_size"]]
            if len(ids) < 2:
                continue
            x, doy, m = inp.batch(ids, True)
            loss = F.cross_entropy(mdl(x, doy, m), y_t[ids])
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
            opt.step()
            if sched is not None:
                sched.step()
    mdl.eval()
    return mdl


@torch.no_grad()
def predict_network(mdl, inp, ids, bs=4096):
    mdl.eval()
    P, H = [], []
    for i in range(0, len(ids), bs):
        x, doy, m = inp.batch(ids[i:i + bs], False)
        logits, h = mdl(x, doy, m, return_feat=True)
        P.append(F.softmax(logits.float(), -1).cpu()); H.append(h.float().cpu())
    return torch.cat(P).numpy().astype(np.float64), torch.cat(H).numpy()


# ----------------------------------------------------------------------------------------------------------------
# Presto
# ----------------------------------------------------------------------------------------------------------------
def presto_files(work_dir, presto_dir=None):
    """single_file_presto.py and default_model.pt: from `presto_dir`, /kaggle/input, or downloaded from GitHub."""
    import glob
    if presto_dir:
        return os.path.join(presto_dir, "single_file_presto.py"), os.path.join(presto_dir, "default_model.pt")
    py = sorted(glob.glob("/kaggle/input/**/single_file_presto.py", recursive=True))
    pt = sorted(glob.glob("/kaggle/input/**/default_model.pt", recursive=True))
    if py and pt:
        return py[0], pt[0]
    dst = os.path.join(work_dir, "presto"); os.makedirs(dst, exist_ok=True)
    for name, rel in (("single_file_presto.py", "single_file_presto.py"), ("default_model.pt", "data/default_model.pt")):
        path = os.path.join(dst, name)
        if not os.path.isfile(path):
            urllib.request.urlretrieve(PRESTO_URL + rel, path)
    return os.path.join(dst, "single_file_presto.py"), os.path.join(dst, "default_model.pt")


class Presto:
    """Presto inputs (17 channels x 12 months) and encoder. Only Sentinel-2, NDVI and location are given; the
    Sentinel-1, ERA5 and SRTM channels are masked, and the land-cover input is set to its missing class."""

    def __init__(self, data, device, work_dir, presto_dir=None):
        self.device = device
        py, self.weights = presto_files(work_dir, presto_dir)
        spec = importlib.util.spec_from_file_location("single_file_presto", py)
        self.module = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.module)
        M, ndvi = data.monthly()
        P17 = np.zeros((data.n, 12, 17), dtype=np.float32)
        P17[:, :, 2:12] = M[:, :, S2_KEEP]      # reflectance / 10,000, Presto's own scaling
        P17[:, :, 16] = ndvi
        self.x = torch.from_numpy(P17).to(device)
        self.mask = torch.zeros(1, 12, 17, device=device); self.mask[:, :, [0, 1, 12, 13, 14, 15]] = 1.0
        latlon = torch.from_numpy(np.stack([data.center[:, 1], data.center[:, 0]], 1).astype(np.float32)).to(device)
        self.latlon = torch.nan_to_num(latlon)
        self._emb = {}

    def encoder(self):
        m = self.module.Presto.construct()
        missing, unexpected = m.load_state_dict(torch.load(self.weights, map_location="cpu"), strict=False)
        assert not missing and not unexpected, f"Presto weights do not match: {missing[:3]} {unexpected[:3]}"
        return m.encoder.to(self.device)

    def encode(self, enc, ids, latlons):
        B = len(ids)
        return enc(self.x[ids], dynamic_world=torch.full((B, 12), 9, dtype=torch.long, device=self.device),
                   latlons=latlons, mask=self.mask.expand(B, -1, -1), month=0, eval_task=True)

    @torch.no_grad()
    def embeddings(self, location="loc", bs=2048):
        """Frozen embeddings of all parcels, with real locations or one constant location (the mean centroid)."""
        if location not in self._emb:
            enc = self.encoder().eval()
            const = self.latlon.mean(0, keepdim=True)
            out = []
            for i in range(0, len(self.x), bs):
                ids = torch.arange(i, min(i + bs, len(self.x)), device=self.device)
                ll = self.latlon[ids] if location == "loc" else const.expand(len(ids), -1)
                out.append(self.encode(enc, ids, ll).float())
            self._emb[location] = torch.cat(out)
        return self._emb[location]


class PrestoClassifier(nn.Module):
    def __init__(self, presto, n_cls):
        super().__init__()
        self.presto = presto                      # inputs and encode(); not a submodule
        self.encoder = presto.encoder()           # pre-trained weights, loaded before the head is initialised
        self.head = nn.Linear(self.encoder.embedding_size, n_cls)

    def forward(self, ids, return_feat=False):
        h = self.presto.encode(self.encoder, ids, self.presto.latlon[ids])
        logits = self.head(h)
        return (logits, h) if return_feat else logits


def train_probe(Z, yt, K, l2, cfg, device):
    """Linear probe: full-batch Adam; l2 adds l2 * ||W||^2 to the mean cross-entropy."""
    head = nn.Linear(Z.shape[1], K).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=cfg["probe_lr"], weight_decay=1e-4)
    for _ in range(cfg["probe_steps"]):
        loss = F.cross_entropy(head(Z), yt) + l2 * head.weight.pow(2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return head


def _mlp_probe(Z, yt, K, cfg, device, steps=1000):
    head = nn.Sequential(nn.Linear(Z.shape[1], 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, K)).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(steps):
        head.train()
        loss = F.cross_entropy(head(Z), yt)
        opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    return head


def _fit_temperature(logits, yt):
    """Scalar temperature minimising NLL on held-out fields."""
    from scipy.optimize import minimize_scalar
    lg, yy = logits.detach(), yt

    def nll(log_t):
        with torch.no_grad():
            return float(F.cross_entropy(lg / float(np.exp(log_t)), yy))
    return float(np.exp(minimize_scalar(nll, bounds=(-4, 4), method="bounded").x))


def fit_probe(emb, train, used, y, K, seed, opts, cfg, device):
    seed_all(seed)
    Etr = emb[torch.as_tensor(train, device=device)]
    mu, sd = Etr.mean(0), Etr.std(0) + 1e-6

    def rep(E):                               # the representation the probe sees
        if not opts["standardize"]:
            return E
        Z_ = (E - mu) / sd
        return Z_.clamp(-opts["clip"], opts["clip"]) if opts["clip"] else Z_
    Z = rep(Etr)
    yt = torch.as_tensor(y[train], device=device)
    extra = {}
    l2 = opts["l2"]
    if opts["tune"]:
        rs = np.random.RandomState(1000 + seed); pm = rs.permutation(len(train)); nv = int(cfg["probe_val_frac"] * len(train))
        va, fi = torch.as_tensor(pm[:nv], device=device), torch.as_tensor(pm[nv:], device=device)
        nll = {}
        for lam in cfg["probe_l2_grid"]:
            h = train_probe(Z[fi], yt[fi], K, lam, cfg, device)
            with torch.no_grad():
                nll[lam] = float(F.cross_entropy(h(Z[va]), yt[va]))
        l2 = min(nll, key=nll.get)
        extra = {"probe_l2": l2, "probe_val_nll": nll[l2], "probe_val_nll_l2_0": nll[0.0]}
    temperature = 1.0
    if opts["ts"]:                            # probe on 85% of the training fields, temperature on the other 15%
        rs = np.random.RandomState(1000 + seed); pm = rs.permutation(len(train)); nv = int(cfg["probe_val_frac"] * len(train))
        va, fi = torch.as_tensor(pm[:nv], device=device), torch.as_tensor(pm[nv:], device=device)
        head = train_probe(Z[fi], yt[fi], K, l2, cfg, device)
        temperature = _fit_temperature(head(Z[va]), yt[va])
        extra["temperature"] = temperature
    elif opts["mlp"]:
        head = _mlp_probe(Z, yt, K, cfg, device)
    else:
        head = train_probe(Z, yt, K, l2, cfg, device)
    with torch.no_grad():
        Eu = emb[torch.as_tensor(used, device=device)]
        logits = head(rep(Eu)) / temperature
        P = F.softmax(logits, -1).cpu().numpy().astype(np.float64)
        extra["z_norm"] = ((Eu - mu) / sd).norm(dim=1).cpu().numpy()   # distance from the source-training mean, in source SDs
        extra["raw_norm"] = (Eu - mu).norm(dim=1).cpu().numpy()         # the same, unstandardised
        extra["max_logit"] = logits.max(1).values.cpu().numpy()
    return P, Eu.cpu().numpy(), extra


def finetune_presto(presto, train, used, y, K, seed, cfg, device):
    seed_all(seed)
    mdl = PrestoClassifier(presto, K).to(device)
    mdl.encoder.pos_embed.requires_grad_(False); mdl.encoder.month_embed.requires_grad_(False)
    enc_params = [p for p in mdl.encoder.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": cfg["presto_lr_encoder"]},
                             {"params": mdl.head.parameters(), "lr": cfg["presto_lr_head"]}], weight_decay=1e-4)
    steps = cfg["presto_epochs"] * math.ceil(len(train) / cfg["batch_size"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))
    y_all = torch.as_tensor(y, device=device)
    for _ in range(cfg["presto_epochs"]):
        mdl.train(); perm = np.random.permutation(train)
        for i in range(0, len(perm), cfg["batch_size"]):
            ids = torch.as_tensor(perm[i:i + cfg["batch_size"]], device=device)
            if len(ids) < 2:
                continue
            loss = F.cross_entropy(mdl(ids), y_all[ids])
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0); opt.step(); sched.step()
    mdl.eval(); Ps, Hs = [], []
    with torch.no_grad():
        for i in range(0, len(used), 4096):
            logits, h = mdl(torch.as_tensor(used[i:i + 4096], device=device), return_feat=True)
            Ps.append(F.softmax(logits.float(), -1).cpu()); Hs.append(h.float().cpu())
    return torch.cat(Ps).numpy().astype(np.float64), torch.cat(Hs).numpy()


# ----------------------------------------------------------------------------------------------------------------
# one entry point for every model
# ----------------------------------------------------------------------------------------------------------------
class Predictor:
    """fit_predict(arch, train, used, y, K, seed) -> (probabilities, features, extra diagnostics) for `used`."""

    def __init__(self, data, cfg, device, work_dir, presto_dir=None):
        self.data, self.cfg, self.device = data, cfg, device
        archs = {base_arch(a) for a in cfg["archs"]}
        self.series = SeriesInput(data, cfg, device) if archs & set(SEQUENCE) else None
        self.monthly = MonthlyInput(data, device) if "transformer_monthly" in archs else None
        self.presto = Presto(data, device, work_dir, presto_dir) if any(a.startswith("presto") for a in archs) else None
        if self.presto is not None:
            for a in archs:
                if a in FROZEN:
                    self.presto.embeddings(FROZEN[a]["location"])
        self.rf = data.rf_features() if "random_forest" in archs else None

    def fit_predict(self, arch, train, used, y, K, seed):
        cfg, dev = self.cfg, self.device
        arch = base_arch(arch)
        if arch in SEQUENCE:
            self.series.set_norm(train)
            y_t = torch.as_tensor(np.maximum(y, 0), device=dev)
            seed_all(seed)
            cls = TSTransformer if arch == "transformer" else TempCNN
            mdl = (cls(self.data.c_in, K, cfg["d_model"], cfg["n_layers"], cfg["n_heads"], cfg["dropout"]) if arch == "transformer"
                   else cls(self.data.c_in, K, cfg["d_model"])).to(dev)
            train_network(mdl, self.series, train, y_t, cfg)
            P, H = predict_network(mdl, self.series, used)
            return P, H, {}
        if arch == "transformer_monthly":
            seed_all(seed)
            self.monthly.set_norm(train)
            y_t = torch.as_tensor(np.maximum(y, 0), device=dev)
            mdl = TSTransformer(self.data.c_in, K, cfg["d_model"], cfg["n_layers"], cfg["n_heads"], cfg["dropout"]).to(dev)
            train_network(mdl, self.monthly, train, y_t, cfg)
            P, H = predict_network(mdl, self.monthly, used)
            return P, H, {}
        if arch in FROZEN:
            opts = FROZEN[arch]
            return fit_probe(self.presto.embeddings(opts["location"]), train, used, y, K, seed, opts, cfg, dev)
        if arch == "presto_finetune":
            P, H = finetune_presto(self.presto, train, used, y, K, seed, cfg, dev)
            return P, H, {}
        if arch == "random_forest":
            seed_all(seed)
            feat, feat_std = self.rf
            rf = RandomForestClassifier(n_estimators=cfg["rf_trees"], max_features="sqrt", max_samples=cfg["rf_max_samples"],
                                        n_jobs=-1, random_state=seed)
            rf.fit(feat[train], y[train])
            P = np.zeros((len(used), K)); P[:, rf.classes_] = rf.predict_proba(feat[used])
            return P.astype(np.float64), feat_std[used], {}
        raise ValueError(f"unknown model {arch!r}")
