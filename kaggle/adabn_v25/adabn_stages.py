"""UMUD v25 AdaBN: re-estimate ONLY the BatchNorm running statistics of the KG0 map networks per image domain.

PREREG: artifacts/candidates_v25/PREREG_v25_adabn.json (written before this kernel was pushed; its sha256 is embedded
in the launcher tables and copied into manifest.json). User approval 2026-11-05: "ok go ahead, you can use the test BN
stats". What changes: BatchNorm running_mean / running_var, recomputed from the UNLABELED images of one domain at a time
(a test scanner group, or one benchmark domain for the gates). No labels, no pseudo-labels, no gradient, no other weight.

Networks (one subprocess stage each; production code embedded unchanged by build_kernel.py):
  apo     apo_v2fix band/upper/lower   apo_fix_src.py = kaggle/apo_v2_fix/ext/infer_fix_ext.py
          13 BN (decoder 10, stem 2, head 1); letterbox 512, no flip, fp32; folds 0-4 test/static, 0-2 dynamic/VL
  ours    ours_fasc384_noflip          pipeline.py    = kaggle/infer_maskbank/pipeline.py
          58 BN (ResNet-34 encoder 36, U-Net++ decoder 22); full frame squashed to 384, identity pass, fp32
  orient  fasc_orient_v1 seg/cos2/sin2 fo_src.py      = kaggle/fasc_orient_v1/train.py
          10 BN (decoder); viewport + letterbox 512, hflip TTA, fp16 autocast; folds 0-4
  ours_apo384 / dlt_apo (fallbacks, never reached on test) are NOT adapted: the intake copies the frozen files.
Arms (maps written for every image of test / static / dynamic / vl):
  B0    production BN statistics (apo: fix_main's training-image repair of the fp16-inf layer, reproduced here).
        Reproduced against the frozen production banks on a fixed subset FIRST; a mismatch aborts the network (exit 4).
  B1    full replacement by the domain statistics.   orient: v1's non-finite fold-2/4 channels are restored after
        their layer is estimated (keep-inf), so the only change is domain adaptation.
  B2    0.5 * B0 + 0.5 * domain statistics (mean and variance), each layer estimated under the blended upstream layers.
  B1fx  orient only: B1 that also replaces the non-finite channels (separates the inf repair from adaptation).
Statistics: exact per-channel population moments of each BN layer's input over all canvases of the domain, with the
production views (orient: both hflip views), layers in forward execution order, each layer set before the next one is
estimated (forward cut by an exception after the hooked layer); eval mode, fp32 forward (autocast and TF32 off),
float64 shifted sums, stored fp32. Equivalent to the momentum=None cumulative average with the whole domain as one
batch and upstream layers already adapted, but independent of batch size and order. Encoders without BN (ConvNeXt,
LayerNorm) are run once per batch and their outputs reused (self-checked against the plain forward; off if it fails).
Domains: test SIE / LUM / T12 / T10 / LEG (embedded artifacts/final/test_layout.csv), VL (249 raw .tiff only),
DYN (168), S660 / S800 (static by image size); the other 6 static images keep B0 (unadapted).
In-kernel scanner-matched OOF checks: orient (mask layouts 1152x864, 996x556, 1340x810, 1088x644) and apo (1152x864):
fold k's own validation images of one layout adapt fold k's model, scored with the training-time metric.
Outputs: /kaggle/working/v25/{maps/<arm>/<net>/<ds>__<model>.npz + manifests, bn_stats/<net>/<domain>.npz,
reports/*.json, domain_manifest.csv, bn_shift.json, repro.json, oof_check.json, manifest.json, logs/}.
GPL-3.0-only.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback
import types
import warnings
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

MODE = os.environ.get("V25_MODE", "full")  # "full" (GPU) | "cpusmoke" (CPU dry run: 2 images per domain)
if MODE not in ("full", "cpusmoke"):
    raise SystemExit(f"V25_MODE must be 'full' or 'cpusmoke', got {MODE!r}")
SMOKE = MODE == "cpusmoke"
WORK = Path(os.environ.get("V25_WORK", "/kaggle/working"))
OUT = WORK / "v25"
MAPS = OUT / "maps"
BNDIR = OUT / "bn_stats"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
TMP = Path(os.environ.get("V25_TMP", "/tmp/v25"))
INPUT_ROOT = Path(os.environ.get("V25_INPUT", "/kaggle/input"))
TABLES_PATH = TMP / "tables.json"
COMP_SLUG = "umud-challenge-muscle-architecture-in-ultrasound-data"
T0 = time.time()
LAUNCH_T0 = float(os.environ.get("V25_LAUNCH_T0", str(T0)))
DEADLINE_S = float(os.environ.get("V25_DEADLINE_S", str(10.5 * 3600)))  # Kaggle stops a session at 12 h

SMOKE_PER_DOMAIN = 2
N_OOF = 3 if SMOKE else 40  # validation images per (fold, layout) in the OOF checks
OOF_SEED = 20261105
BLEND_W = 0.5  # B2: weight of the B0 (source) statistics
ARM_CFG = {"B1": {"mode": "replace", "keep_bad": True}, "B2": {"mode": "blend", "keep_bad": True},
           "B1fx": {"mode": "replace", "keep_bad": False}}
ARMS = {"apo": ["B1", "B2"], "ours": ["B1", "B2"], "orient": ["B1", "B2", "B1fx"]}
EXPECTED_BN = {"apo": 13, "ours": 58, "orient": 10}
EXPECTED_N = {"test": 309, "static": 35, "dynamic": 168, "vl": 249}
DOMAIN_ORDER = ["VL", "DYN", "S660", "S800", "SIE", "LUM", "T12", "T10", "LEG"]
STATIC_DOMAIN = {"660x556": "S660", "800x600": "S800"}  # other static sizes (6 images) keep B0
REPRO_TOL = {"mean_abs": 0.05, "frac_gt2": 1e-3, "block_mean_max_abs": 0.25}
FIX_TOL = {"dice": 0.003, "score": 0.002}
APO_OOF_LAYOUTS = ["1152x864"]
ORIENT_OOF_LAYOUTS = ["1152x864", "996x556", "1340x810", "1088x644"]
OOF_RULE = {"min_n": 10, "min_folds_le0": 4, "mean_le": 0.0, "dice_drop_max": 0.005}
STAGE = "?"
TABLES: dict = {}


def log(msg: str) -> None:
    print(f"[v25 {STAGE} {(time.time() - T0) / 60:6.1f}m] {msg}", flush=True)


def past_deadline() -> bool:
    return time.time() - LAUNCH_T0 > DEADLINE_S


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def arr_sha1(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha1()
    h.update(f"{a.dtype.str}{a.shape}".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def block_means(a: np.ndarray, n: int = 16) -> np.ndarray:
    """n x n block means of a 2-D array (block edges at round(linspace)); used for the B0 reproduction digests."""
    a = np.asarray(a, dtype=np.float64)
    h, w = a.shape
    ys = np.linspace(0, h, n + 1).round().astype(int)
    xs = np.linspace(0, w, n + 1).round().astype(int)
    s = np.zeros((h + 1, w + 1))
    s[1:, 1:] = a.cumsum(0).cumsum(1)
    tot = (s[np.ix_(ys[1:], xs[1:])] - s[np.ix_(ys[:-1], xs[1:])] - s[np.ix_(ys[1:], xs[:-1])]
           + s[np.ix_(ys[:-1], xs[:-1])])
    return tot / np.maximum(np.outer(np.diff(ys), np.diff(xs)), 1)


def tail3(p: str) -> str:
    """Last three path parts, lower case (the eval_orient VL key rule)."""
    return "/".join(Path(str(p).replace("\\", "/")).parts[-3:]).lower()


def mem_available() -> int:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


class InputIndex:
    """One walk of /kaggle/input: file name -> paths (attached kernel outputs are found by their slug folder)."""

    def __init__(self) -> None:
        self.by_name: dict | None = None

    def _build(self) -> None:
        self.by_name = defaultdict(list)
        for root, _dirs, files in os.walk(INPUT_ROOT):
            for f in files:
                self.by_name[f].append(Path(root) / f)

    def find(self, name: str, slug: str | None = None, required: bool = True) -> Path | None:
        if self.by_name is None:
            self._build()
        hits = sorted(p for p in self.by_name.get(name, []) if slug is None or slug in p.parts)
        if not hits:
            if required:
                raise FileNotFoundError(f"{name} (slug {slug}) not under {INPUT_ROOT}")
            return None
        if len(hits) > 1:
            log(f"several {name} for {slug}: {[str(h) for h in hits]}; using the first")
        return hits[0]


INDEX = InputIndex()


class NpzZip:
    """Incremental np.savez_compressed-compatible writer (same as fasc_orient_v1 NpzZip)."""

    def __init__(self, path: Path, level: int = 6) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.zf = zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=level, allowZip64=True)
        self.keys: set[str] = set()

    def add(self, key: str, arr: np.ndarray) -> None:
        if key in self.keys:
            raise KeyError(f"duplicate key {key} in {self.path}")
        self.keys.add(key)
        with self.zf.open(key + ".npy", "w", force_zip64=True) as fh:
            np.lib.format.write_array(fh, np.ascontiguousarray(arr), allow_pickle=False)

    def close(self) -> None:
        self.zf.close()


def domain_of(ds: str, key: str, w: int, h: int) -> str | None:
    if ds == "test":
        g = TABLES["test_group"].get(key)
        if g is None:
            raise KeyError(f"test image {key} is not in the embedded layout table")
        return g
    if ds == "static":
        return STATIC_DOMAIN.get(f"{w}x{h}")
    if ds == "dynamic":
        return "DYN"
    if ds == "vl":
        return "VL"
    raise ValueError(ds)


def smoke_select(items: list[dict]) -> list[dict]:
    if not SMOKE:
        return items
    keep, seen = [], Counter()
    for it in items:
        d = it["domain"] or "UNADAPTED"
        if seen[d] < (SMOKE_PER_DOMAIN if it["domain"] else 1):
            keep.append(it)
            seen[d] += 1
    return keep


# ============================================================================================ torch / BatchNorm tools
def torch_setup():
    import torch

    warnings.filterwarnings("ignore", message=".*autocast.*")
    dev = torch.device("cuda" if torch.cuda.is_available() and not SMOKE else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if dev.type == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    return torch, dev


def autocast_off(torch, dev):
    return torch.autocast(device_type=dev.type, enabled=False)


def _fp32_autocast(torch, dev):
    """Replaces apo_fix_src._ac: the production apo_v2fix maps were made in fp32 (CPU kernels), never fp16."""
    return torch.autocast(dev.type, enabled=False)


class StopForward(Exception):
    """Raised by a hook to cut the forward pass once the statistic it needs has been taken."""


def bn_modules(torch, model) -> dict:
    return {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)}


def get_state(torch, mods: dict) -> dict:
    with torch.no_grad():
        return {n: (m.running_mean.detach().clone(), m.running_var.detach().clone()) for n, m in mods.items()}


def set_state(torch, mods: dict, state: dict) -> None:
    with torch.no_grad():
        for n, (mu, var) in state.items():
            mods[n].running_mean.copy_(mu)
            mods[n].running_var.copy_(var)


def bad_channels(torch, state: dict) -> dict:
    out = {}
    for n, (mu, var) in state.items():
        bad = ~(torch.isfinite(mu) & torch.isfinite(var))
        if bool(bad.any()):
            out[n] = torch.nonzero(bad).flatten().tolist()
    return out


def exec_order(torch, model, dev, run_once) -> list[str]:
    """BN layer names in forward execution order (forward pre-hooks on one input); each must run exactly once."""
    mods = bn_modules(torch, model)
    calls: list[str] = []
    hooks = [m.register_forward_pre_hook(lambda mod, inp, n=n: calls.append(n)) for n, m in mods.items()]
    try:
        with torch.no_grad(), autocast_off(torch, dev):
            run_once()
    finally:
        for h in hooks:
            h.remove()
    if len(calls) != len(mods) or set(calls) != set(mods):
        raise RuntimeError(f"BN execution: {len(calls)} calls for {len(mods)} layers (a layer runs twice or never)")
    return calls


class Moments:
    """Per-channel population mean / unbiased variance of (B, C, H, W) inputs; float64, shifted sums."""

    def __init__(self) -> None:
        self.n, self.k, self.s1, self.s2 = 0, None, None, None

    def add(self, x) -> None:
        if x.dim() != 4:
            raise RuntimeError(f"BatchNorm input of rank {x.dim()} (expected 4)")
        xd = x.detach().double()
        if self.k is None:
            self.k = xd.mean(dim=(0, 2, 3))
        xc = xd - self.k.view(1, -1, 1, 1)
        s1, s2 = xc.sum(dim=(0, 2, 3)), (xc * xc).sum(dim=(0, 2, 3))
        self.s1 = s1 if self.s1 is None else self.s1 + s1
        self.s2 = s2 if self.s2 is None else self.s2 + s2
        self.n += int(xd.shape[0] * xd.shape[2] * xd.shape[3])

    def result(self):
        m = self.s1 / self.n
        var = (self.s2 / self.n - m * m).clamp_min(0.0) * (self.n / max(1, self.n - 1))
        return self.k + m, var, self.n


def cache_device(torch, dev, nbytes: int):
    if dev.type == "cuda":
        free, _total = torch.cuda.mem_get_info()
        if nbytes <= 0.35 * free:
            return dev
    if nbytes <= 0.4 * mem_available():
        return torch.device("cpu")
    return None


class Feeder:
    """Feeds the canvases of one domain (calibration views) through a model, catching StopForward.

    Optional encoder cache: when the encoder has no BatchNorm, its outputs do not depend on any BN statistic, so they are
    computed once per batch and replayed (the encoder's forward is temporarily replaced). A self-check compares the
    replayed forward with the plain forward on batch 0; any difference > 1e-4 relative disables the cache."""

    def __init__(self, torch, dev, canvases: list, batch: int, make_x, enc_getter=None) -> None:
        self.torch, self.dev, self.make_x, self.enc_getter = torch, dev, make_x, enc_getter
        self.batches = [np.stack(canvases[i:i + batch]) for i in range(0, len(canvases), batch)]
        self.cache, self.enc = None, None
        self.info: dict = {"cache": "off"}

    def build_cache(self, model) -> dict:
        torch = self.torch
        if self.enc_getter is None:
            return self.info
        enc = self.enc_getter(model)
        if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in enc.modules()):
            self.info = {"cache": "off (BatchNorm inside the encoder)"}
            return self.info
        holder: dict = {}

        def grab(mod, inp, out):
            holder["out"] = out
            raise StopForward

        feats, where = [], None
        h = enc.register_forward_hook(grab)
        try:
            with torch.no_grad(), autocast_off(torch, self.dev):
                for b in self.batches:
                    holder.clear()
                    try:
                        model(self.make_x(b))
                    except StopForward:
                        pass
                    out = holder.get("out")
                    if not isinstance(out, (list, tuple)) or not out or not all(torch.is_tensor(t) for t in out):
                        self.info = {"cache": f"off (encoder output {type(out).__name__})"}
                        return self.info
                    if where is None:
                        nbytes = sum(t.numel() * t.element_size() for t in out) * len(self.batches)
                        where = cache_device(torch, self.dev, nbytes)
                        self.info = {"cache_bytes": int(nbytes)}
                        if where is None:
                            self.info["cache"] = "off (memory budget)"
                            return self.info
                    feats.append([t.detach().to(where) for t in out])
        finally:
            h.remove()
        self.cache, self.enc = feats, enc
        with torch.no_grad(), autocast_off(torch, self.dev):
            x = self.make_x(self.batches[0])
            y0 = model(x).float()
            y1 = self._cached_forward(model, 0, x).float()
            y2 = self._cached_forward(model, 0, x).float()  # a second replay: the cache must not be modified in place
            err = max(float((y0 - y1).abs().max()), float((y0 - y2).abs().max()))
            scale = float(y0.abs().max())
        ok = err <= 1e-4 * max(1.0, scale)
        self.info.update(cache=str(where) if ok else "off (self-check failed)", selfcheck_max_abs=err,
                         selfcheck_scale=scale)
        if not ok:
            self.cache, self.enc = None, None
        return self.info

    def _cached_forward(self, model, bi: int, x):
        feats = [t.clone() if t.device.type == self.dev.type else t.to(self.dev, non_blocking=True)
                 for t in self.cache[bi]]
        self.enc.forward = lambda *a, **k: list(feats)
        try:
            return model(x)
        finally:
            del self.enc.forward

    def run(self, model) -> None:
        torch = self.torch
        with torch.no_grad(), autocast_off(torch, self.dev):
            for bi, b in enumerate(self.batches):
                x = self.make_x(b)
                try:
                    if self.cache is not None:
                        self._cached_forward(model, bi, x)
                    else:
                        model(x)
                except StopForward:
                    pass

    def free(self) -> None:
        self.cache, self.enc, self.batches = None, None, []


def adapt_sequential(torch, model, mods: dict, order: list, feeder: Feeder, src_state: dict, arm: str,
                     bad: dict) -> tuple[dict, dict]:
    """Sequential AdaBN of every BN layer (execution order). Returns (new state, {layer: {n_px, C}}).

    replace: running stats := domain moments.  blend: := BLEND_W * source + (1 - BLEND_W) * domain moments.
    keep_bad: channels whose SOURCE stats are non-finite (v1 folds 2/4) are restored to the source values right after
    their layer is estimated, i.e. before any downstream layer is estimated."""
    cfg = ARM_CFG[arm]
    keep = bad if cfg["keep_bad"] else {}
    set_state(torch, mods, src_state)
    info = {}
    for name in order:
        bn = mods[name]
        acc = Moments()

        def pre(mod, inp, acc=acc):
            acc.add(inp[0])
            raise StopForward

        h = bn.register_forward_pre_hook(pre)
        try:
            feeder.run(model)
        finally:
            h.remove()
        mean, var, n = acc.result()
        if not (bool(torch.isfinite(mean).all()) and bool(torch.isfinite(var).all())):
            raise FloatingPointError(f"non-finite domain moments in {name}")
        mu_s, var_s = (t.double() for t in src_state[name])
        if cfg["mode"] == "replace":
            new_mu, new_var = mean.clone(), var.clone()
        else:
            new_mu = BLEND_W * mu_s + (1.0 - BLEND_W) * mean
            new_var = BLEND_W * var_s + (1.0 - BLEND_W) * var
        fin = torch.isfinite(new_mu) & torch.isfinite(new_var)
        if keep.get(name):
            ii = torch.tensor(keep[name], dtype=torch.long, device=new_mu.device)
            new_mu[ii], new_var[ii] = mu_s[ii], var_s[ii]
            fin[ii] = True
        if not bool(fin.all()):
            raise FloatingPointError(f"{name}: non-finite statistics outside the kept channels ({arm})")
        with torch.no_grad():
            bn.running_mean.copy_(new_mu.to(bn.running_mean.dtype))
            bn.running_var.copy_(new_var.to(bn.running_var.dtype))
        info[name] = {"n_px": int(n), "C": int(mean.numel())}
    return get_state(torch, mods), info


def shift_summary(torch, src: dict, new: dict, eps: float = 1e-5) -> dict:
    out = {}
    for n, (mu_n, var_n) in new.items():
        mu_s, var_s = (t.double().cpu() for t in src[n])
        mu_n, var_n = mu_n.double().cpu(), var_n.double().cpu()
        ok = torch.isfinite(mu_s) & torch.isfinite(var_s) & (var_s > 0) & torch.isfinite(var_n)
        if not bool(ok.any()):
            continue
        dmu = ((mu_n - mu_s).abs() / torch.sqrt(var_s + eps))[ok]
        lv = torch.log((var_n + eps) / (var_s + eps))[ok]
        out[n] = {"dmu_over_sigma_med": float(dmu.median()), "dmu_over_sigma_max": float(dmu.max()),
                  "logvar_ratio_med": float(lv.median()), "logvar_ratio_absmax": float(lv.abs().max())}
    if out:
        out["_summary"] = {"dmu_over_sigma_med_of_layers": float(np.median([v["dmu_over_sigma_med"] for v in out.values()])),
                           "logvar_ratio_med_of_layers": float(np.median([v["logvar_ratio_med"] for v in out.values()]))}
    return out


def save_bn(path: Path, folds: list, src: dict, states: dict, infos: dict) -> None:
    arrays = {}
    for k in folds:
        for n, (mu, var) in src[k].items():
            arrays[f"B0|f{k}|{n}|mean"] = mu.float().cpu().numpy()
            arrays[f"B0|f{k}|{n}|var"] = var.float().cpu().numpy()
    for (arm, k), st in states.items():
        for n, (mu, var) in st.items():
            arrays[f"{arm}|f{k}|{n}|mean"] = mu.float().cpu().numpy()
            arrays[f"{arm}|f{k}|{n}|var"] = var.float().cpu().numpy()
            if n in infos.get((arm, k), {}):
                arrays[f"{arm}|f{k}|{n}|n_px"] = np.int64(infos[(arm, k)][n]["n_px"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


# ============================================================================================ map banks
class BankWriter:
    """<MAPS>/<arm>/<net>/<ds>__<model>.npz (uint8, native resolution, production keys) + production manifests."""

    def __init__(self, arm: str, st) -> None:
        self.dir = MAPS / arm / st.name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.st, self.w, self.rows = st, {}, []

    def add(self, it: dict, arrays: dict) -> None:
        for mn, a in arrays.items():
            if a.shape != (it["H"], it["W"]) or a.dtype != np.uint8:
                raise ValueError(f"{it['ds']}/{it['key']} {mn}: {a.dtype}{a.shape} != uint8{(it['H'], it['W'])}")
            k = (it["ds"], mn)
            if k not in self.w:
                self.w[k] = NpzZip(self.dir / f"{it['ds']}__{mn}.npz")
            self.w[k].add(it["key"], a)
        self.rows.append([it["ds"], it["key"], it["src"], it["H"], it["W"]])

    def close(self) -> None:
        for w in self.w.values():
            w.close()
        for fname, keep in self.st.manifests().items():
            with open(self.dir / fname, "w", newline="", encoding="utf-8") as fh:
                wr = csv.writer(fh)
                wr.writerow(["dataset", "image_id", "source_filename", "height", "width"])
                wr.writerows(r for r in self.rows if keep(r[0]))


class B0Reader:
    def __init__(self, st) -> None:
        self.dir, self.files = MAPS / "B0" / st.name, {}

    def get(self, it: dict, model_names: list) -> dict:
        out = {}
        for mn in model_names:
            k = (it["ds"], mn)
            if k not in self.files:
                self.files[k] = np.load(self.dir / f"{it['ds']}__{mn}.npz")
            out[mn] = self.files[k][it["key"]]
        return out


# ============================================================================================ network stages
class Stage:
    name = ""
    model_names: list = []
    calib_batch = 8
    enc_getter = None
    torch = None
    dev = None

    def manifests(self) -> dict:
        return {"manifest.csv": lambda ds: True}

    def folds_for(self, ds: str) -> list:
        return [0]

    def repro_folds(self, ds: str) -> list:
        return self.folds_for(ds)

    def frozen_path(self, ds: str, mn: str) -> Path | None:
        return None


class ApoStage(Stage):
    """apo_v2fix through kaggle/apo_v2_fix/ext/infer_fix_ext.py (functions used unchanged; fp32, no flip)."""

    name = "apo"
    CH = ("band", "upper", "lower")
    model_names = [f"apo_v2fix_{c}" for c in CH]
    calib_batch = 8

    def __init__(self) -> None:
        m = load_module("apo_fix_src", TMP / "apo_fix_src.py")
        self.m = m
        m.OUT = LOGS / "apo_fix"
        m.OUT.mkdir(parents=True, exist_ok=True)
        if not m.COMP_ROOT.exists():
            for cand in (INPUT_ROOT / "competitions" / COMP_SLUG, INPUT_ROOT / COMP_SLUG):
                if cand.exists():
                    m.COMP_ROOT = cand
                    break
        m.ENC_EXTRA = {"drop_path_rate": 0.1}  # fix_main sets this before building the fold models
        m._ac = _fp32_autocast
        self.folds_all = [0] if SMOKE else list(range(m.N_FOLDS))
        self.ext_folds = [0] if SMOKE else list(m.EXT_FOLDS)
        self.oof_folds = [0] if SMOKE else list(range(m.N_FOLDS))
        self.oof_sel: dict = {}

    def manifests(self) -> dict:
        return {"manifest_test.csv": lambda ds: ds == "test", "manifest_ext.csv": lambda ds: ds != "test"}

    def folds_for(self, ds: str) -> list:
        return self.folds_all if ds in ("test", "static") else self.ext_folds

    def enc_getter(self, model):
        return model.base.encoder

    def make_x(self, b: np.ndarray):
        torch = self.torch
        x = torch.from_numpy(b.astype(np.float32)[:, None] / 255.0).to(self.dev)
        return x.to(memory_format=torch.channels_last)

    def build_items(self) -> list[dict]:
        m = self.m
        raw = []
        for p in m.list_images(m.find_dir(m.COMP_ROOT, "test_set_v2")):
            raw.append(dict(ds="test", key=p.stem, path=p, src=p.name))
        ext = m.external_datasets()
        for name in ("static", "dynamic", "vl"):
            lst = ext.get(name, [])
            if name == "vl":  # raw (unannotated) .tiff frames only; the .tif twins carry the drawn lines
                lst = [(s, p) for s, p in lst if p.suffix.lower() == ".tiff"]
            for sub, p in lst:
                raw.append(dict(ds=name, key=p.stem, path=p, src=f"{sub}/{p.relative_to(m.EXT_DIR / sub)}"))
        items, used = [], defaultdict(set)
        for it in raw:  # run_inference_fixed.run_set: same skips, same key de-duplication
            try:
                g = m.read_gray(it["path"])
            except Exception as e:  # noqa: BLE001
                if it["ds"] == "test":
                    raise
                log(f"skip {it['path']}: {e!r}")
                continue
            if it["ds"] != "test" and min(g.shape) < 64:
                continue
            k, j = it["key"], 1
            while k in used[it["ds"]]:
                j += 1
                k = f"{it['key']}__{j}"
            used[it["ds"]].add(k)
            vp = m.detect_viewport(g)
            canvas, tf, _ = m.letterbox(g, vp, m.SIZE)
            it.update(key=k, H=int(g.shape[0]), W=int(g.shape[1]), canvas=canvas, tf=tf, vp=[int(v) for v in vp])
            items.append(it)
        return items

    def load_models(self, torch, dev):
        m = self.m
        smp = m.import_smp()
        if smp.__version__ != "0.5.0":
            raise RuntimeError(f"smp {smp.__version__} != 0.5.0")
        _t, _nn, _F, _DL, _TrDS, _EvDS, ApoNet = m.torch_parts()
        ck0 = INDEX.find("apo_v2_fold0.pt", "umud-apo-v2")
        fcsv = ck0.parent.parent / "folds.csv"  # apo_v2/folds.csv next to apo_v2/ckpt/ (never rglob: v1 has one too)
        if not fcsv.exists():
            raise FileNotFoundError(fcsv)
        items, n_pairs, dups = m.load_training_items()
        fold_of = {r["image_id"]: int(r["fold"]) for r in read_csv(fcsv)}
        miss = [it["id"] for it in items if it["id"] not in fold_of]
        if miss:
            raise RuntimeError(f"{len(miss)} apo training items missing from {fcsv}")
        rep = {"folds_csv": str(fcsv), "n_train_items": len(items), "n_pairs": n_pairs, "dups": dups,
               "checkpoints": {}, "fix_check": {}}
        models = {}
        for k in sorted(set(self.folds_all) | set(self.ext_folds)):
            t = time.time()
            ck = ck0.parent / f"{m.MODEL_NAME}_fold{k}.pt"
            state = torch.load(ck, map_location="cpu", weights_only=False)
            sd = state["model"]
            nonfin = [n for n, v in sd.items()
                      if torch.is_tensor(v) and v.dtype.is_floating_point and not bool(torch.isfinite(v).all())]
            base = m.build_model(smp, ApoNet, state.get("encoder", m.ENCODER), pretrained=False)
            base.load_state_dict({n: v.float() if v.dtype.is_floating_point else v for n, v in sd.items()})
            base = base.to(dev).eval().to(memory_format=torch.channels_last)
            tr = [it for it in items if fold_of[it["id"]] != k]
            va = [it for it in items if fold_of[it["id"]] == k][: m.N_CHECK]
            nb = m._recalibrate(torch, base, tr, True, dev) if nonfin else 0  # = production fix (bad layer only)
            chk = m._val(torch, base, va, dev)
            exp = TABLES.get("apo_fix_expected", {}).get(str(k))
            ok = exp is not None and abs(chk["dice"] - exp["dice"]) <= FIX_TOL["dice"] \
                and abs(chk["score"] - exp["score"]) <= FIX_TOL["score"]
            rep["fix_check"][str(k)] = {"n_recalibrated_layers": nb, "check": chk, "expected": exp, "pass": bool(ok)}
            rep["checkpoints"][str(k)] = {"path": str(ck), "sha256": sha256_file(ck), "epoch": state.get("epoch"),
                                          "nonfinite": nonfin}
            log(f"apo fold {k}: nonfinite {nonfin}, recalibrated {nb}; check dice {chk['dice']:.4f} score "
                f"{chk['score']:.4f} vs production {exp}; pass={ok} ({time.time() - t:.0f}s)")
            models[k] = base
        rep["fix_check_pass"] = all(v["pass"] for v in rep["fix_check"].values())
        self.oof_sel = self._select_oof(items, fold_of)
        del items
        gc.collect()
        return models, rep

    def _select_oof(self, items: list, fold_of: dict) -> dict:
        from PIL import Image

        m = self.m
        msk_dir = m.find_dir(m.COMP_ROOT, "apo_masks_new_model_v1")
        masks = {p.stem: p for p in msk_dir.iterdir() if p.suffix.lower() in m.IMAGE_SUFFIXES}
        size = {}
        for it in items:
            with Image.open(masks[it["id"]]) as im:
                size[it["id"]] = "{}x{}".format(*im.size)  # native mask size = scanner layout (images are resized)
        out = {}
        for lay in APO_OOF_LAYOUTS:
            for k in self.oof_folds:
                cand = sorted((it for it in items if fold_of[it["id"]] == k and size[it["id"]] == lay),
                              key=lambda it: it["id"])
                pick = np.sort(np.random.default_rng(OOF_SEED + k).permutation(len(cand))[:N_OOF])
                out[(lay, k)] = [cand[int(i)] for i in pick]
        return out

    def infer(self, models: list, it: dict) -> dict:
        torch, m = self.torch, self.m
        x = torch.from_numpy(it["canvas"].astype(np.float32)[None, None] / 255.0).to(self.dev)
        x = x.to(memory_format=torch.channels_last)
        with torch.inference_mode():
            pr = m._predict(torch, models, x, self.dev, False)[0].cpu().numpy()
        pn = m.to_native(pr, it["tf"])
        return {f"apo_v2fix_{c}": np.clip(np.rint(pn[i] * 255.0), 0, 255).astype(np.uint8)
                for i, c in enumerate(self.CH)}

    def frozen_path(self, ds: str, mn: str) -> Path | None:
        slug = "umud-apo-v2-fix-test" if ds == "test" else "umud-apo-v2-fix-ext"
        return INDEX.find(f"{ds}__{mn}.npz", slug, required=False)

    def run_oof(self, models, mods, order, src, bad) -> dict:
        """Siemens (1152x864 masks): fold k's validation images adapt fold k; production check metric (_val)."""
        torch, dev, m = self.torch, self.dev, self.m
        res: dict = {}
        for (lay, k), va in self.oof_sel.items():
            if past_deadline():
                res.setdefault(lay, {})[str(k)] = {"skipped": "deadline"}
                continue
            rec = {"n": len(va), "ids": [it["id"] for it in va]}
            if not va or (len(va) < OOF_RULE["min_n"] and not SMOKE):
                res.setdefault(lay, {})[str(k)] = {**rec, "skipped": "too few images"}
                continue
            canv = [m.letterbox(it["img"], it["vp"], m.SIZE)[0] for it in va]
            fd = Feeder(torch, dev, canv, self.calib_batch, self.make_x, self.enc_getter)
            set_state(torch, mods[k], src[k])
            rec["cache"] = fd.build_cache(models[k])
            rec["B0"] = m._val(torch, models[k], va, dev)
            for arm in ARMS["apo"]:
                adapt_sequential(torch, models[k], mods[k], order, fd, src[k], arm, bad[k])
                rec[arm] = m._val(torch, models[k], va, dev)
                set_state(torch, mods[k], src[k])
            fd.free()
            res.setdefault(lay, {})[str(k)] = rec
            log(f"apo OOF {lay} fold {k} n={len(va)}: " + ", ".join(
                f"{a} score {rec[a]['score']:.4f} dice {rec[a]['dice']:.4f}" for a in ["B0"] + ARMS["apo"]))
        return {"metric": "apo fix check (_val): score = mean relative MT error (band, edge), dice",
                "layouts": res, "verdict": oof_verdict(res, ARMS["apo"], "score", "dice")}


class OursStage(Stage):
    """ours_fasc384 through the torch stage of kaggle/infer_maskbank/pipeline.py (noflip = pass 0 of [x, flip x])."""

    name = "ours"
    model_names = ["ours_fasc384_noflip"]
    calib_batch = 16
    CKPT = "fasc_unetpp_r34_384.pt"

    def __init__(self) -> None:
        P = load_module("pipeline", TMP / "pipeline.py")
        P.TMP = TMP / "ours"
        P.DATA = P.TMP / "data"
        P.MASKBANK = P.TMP / "mb"  # the data collectors write listings here: keep them out of /kaggle/working
        P.VL_OUT = P.TMP / "vl_data"  # raw frames + (unused) annotated / label-mask exports stay in /tmp
        P.INTERNAL_MANIFEST = P.TMP / "manifest_internal.csv"
        for d in (P.DATA, P.MASKBANK, P.VL_OUT):
            d.mkdir(parents=True, exist_ok=True)
        self.P = P
        self.ck = None

    def make_x(self, b: np.ndarray):
        return self.torch.from_numpy(b.astype(np.float32)[:, None] / 255.0).to(self.dev)

    def _checkpoint(self):
        if self.ck is None:
            path = self.P.find_input_file(self.CKPT)
            self.ck = (path, self.torch.load(path, map_location="cpu", weights_only=False))
        return self.ck

    def build_items(self) -> list[dict]:
        import cv2
        from PIL import Image

        P = self.P
        recs = P.collect_test()
        try:
            recs += P.collect_benchmarks()
        except Exception:  # noqa: BLE001
            log(f"ERROR benchmarks:\n{traceback.format_exc()}")
        try:
            recs += P.collect_vl()
        except Exception:  # noqa: BLE001
            log(f"ERROR vl:\n{traceback.format_exc()}")
        vl_keys = set(TABLES["vl_index_keys"])
        _path, ck = self._checkpoint()
        self.use_viewport = bool(ck.get("viewport_preprocessing", False))
        self.size = int(ck.get("size", 384))
        items, dropped = [], []
        for r in recs:
            if r["dataset"] == "vl" and r["image_id"] not in vl_keys:
                dropped.append(r["image_id"])  # unpaired annotated copies (local patch_log_vl_pairs.json)
                continue
            with Image.open(r["path"]) as image:
                gray = np.asarray(image.convert("L"), dtype=np.uint8)
            h, w = gray.shape
            if (h, w) != (int(r["height"]), int(r["width"])):
                raise RuntimeError(f"size mismatch vs manifest: {r['image_id']}")
            box = self.P.training_viewport(gray) if self.use_viewport else (0, 0, w, h)
            left, top, right, bottom = box
            canvas = cv2.resize(gray[top:bottom, left:right], (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            items.append(dict(ds=r["dataset"], key=r["image_id"], path=Path(r["path"]), src=r["source_filename"],
                              H=h, W=w, canvas=canvas, box=[int(v) for v in box]))
        if dropped:
            log(f"ours: {len(dropped)} VL records outside the 249 raw frames dropped: {dropped}")
        self.dropped_vl = dropped
        return items

    def load_models(self, torch, dev):
        P = self.P
        smp = P.ensure_smp()
        if smp.__version__ != "0.5.0":
            raise RuntimeError(f"smp {smp.__version__} != 0.5.0")
        path, ck = self._checkpoint()
        model = smp.UnetPlusPlus(encoder_name=ck.get("encoder", "resnet34"), encoder_weights=None, in_channels=1,
                                 classes=1, activation=None)
        model.load_state_dict({k.removeprefix("module."): v for k, v in ck["model"].items()}, strict=True)
        model.to(dev).eval()
        sha = sha256_file(path)
        exp = TABLES.get("ckpt_sha256", {}).get(self.CKPT)
        rep = {"checkpoints": {"0": {"path": str(path), "sha256": sha, "expected_sha256": exp,
                                     "sha_match": sha == exp}},
               "viewport_crop": self.use_viewport, "size": self.size, "dropped_vl": self.dropped_vl}
        log(f"ours: {path} sha256 match {sha == exp}; viewport_crop={self.use_viewport} size={self.size}")
        return {0: model}, rep

    def infer(self, models: list, it: dict) -> dict:
        import cv2

        torch, P = self.torch, self.P
        x = it["canvas"].astype(np.float32) / 255.0
        batch = np.ascontiguousarray(np.stack([x, x[:, ::-1]])[:, None])  # production batch; pass 0 = noflip
        with torch.inference_mode():
            logits = models[0](torch.from_numpy(batch).to(self.dev))
            probability = torch.sigmoid(logits).float().cpu().numpy()[:, 0]
        left, top, right, bottom = it["box"]
        full = np.zeros((it["H"], it["W"]), dtype=np.float32)
        full[top:bottom, left:right] = cv2.resize(np.ascontiguousarray(probability[0]), (right - left, bottom - top),
                                                  interpolation=cv2.INTER_LINEAR)
        return {"ours_fasc384_noflip": P.to_uint8(full)}

    def frozen_path(self, ds: str, mn: str) -> Path | None:
        return INDEX.find(f"{ds}__{mn}.npz", "umud-infer-maskbank", required=False)


class _SerialPool:
    def __init__(self, *a, **k) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a) -> bool:
        return False

    def imap(self, f, it, chunksize: int = 1):
        return map(f, it)


class OrientStage(Stage):
    """fasc_orient_v1 through kaggle/fasc_orient_v1/train.py (viewport, letterbox, forward_maps hflip TTA, fp16)."""

    name = "orient"
    CH = ("seg", "cos2", "sin2")
    model_names = [f"fasc_orient_v1_{c}" for c in CH]
    calib_batch = 4  # images per calibration batch (x 2 hflip views)

    def __init__(self) -> None:
        fo = load_module("fo_src", TMP / "fo_src.py")
        self.fo = fo
        self.folds_all = list(range(fo.N_FOLDS))
        self.adapt_folds = [0, 2] if SMOKE else list(range(fo.N_FOLDS))  # smoke: fold 2 carries the inf channels
        self.oof_folds = [2] if SMOKE else list(range(fo.N_FOLDS))
        self.oof_layouts = ORIENT_OOF_LAYOUTS[:2] if SMOKE else ORIENT_OOF_LAYOUTS
        self.oof_sel: dict = {}

    def folds_for(self, ds: str) -> list:
        return self.adapt_folds

    def repro_folds(self, ds: str) -> list:
        return self.folds_all

    def enc_getter(self, model):
        return model.encoder

    def make_x(self, b: np.ndarray):
        torch = self.torch
        x = self.fo.to_input(torch.from_numpy(np.ascontiguousarray(b)).to(self.dev), torch)
        return torch.cat([x, torch.flip(x, dims=[3])], 0)  # forward_maps(tta=True) feeds both views

    def prepare_oof(self) -> dict:
        """Before torch / any cv2 call in this process: pick fold k's validation representatives per native mask layout
        (v1 metrics/folds.csv), then run v1's prepare_items on them and their aliases serially (no fork)."""
        from PIL import Image

        fo = self.fo
        ck0 = INDEX.find("fold0.pt", "umud-fasc-orient-v1")
        fcsv = ck0.parent.parent / "metrics" / "folds.csv"
        rows = read_csv(fcsv)
        rep_fold, aliases = {}, defaultdict(list)
        for r in rows:
            rep_fold[r["representative"]] = int(r["fold"])
            aliases[r["representative"]].append(r["stem"])
        pairs = fo.find_pairs(fo.find_dir("fasc_images_new_model_v1"), fo.find_dir("fasc_masks_new_model_v1"))
        pair_of = {Path(a).stem: (a, b) for a, b in pairs}
        lay = {}
        for rep in rep_fold:
            if rep in pair_of:
                with Image.open(pair_of[rep][1]) as im:
                    lay[rep] = "{}x{}".format(*im.size)
        sel = {}
        for L in self.oof_layouts:
            for k in self.oof_folds:
                cand = sorted(r for r, f in rep_fold.items() if f == k and lay.get(r) == L)
                pick = np.sort(np.random.default_rng(OOF_SEED + k).permutation(len(cand))[:N_OOF])
                sel[(L, k)] = [cand[int(i)] for i in pick]
        chosen = set().union(*sel.values()) if sel else set()
        sub = sorted({pair_of[s] for r in chosen for s in aliases[r] if s in pair_of}, key=lambda t: Path(t[0]).stem)
        fo.mp = types.SimpleNamespace(get_context=lambda *a, **k: types.SimpleNamespace(Pool=_SerialPool))
        t = time.time()
        items, _alias_rows, stats = fo.prepare_items(sub) if sub else ([], [], {})
        by_stem = {it["stem"]: it for it in items}
        missing = []
        for key, reps in sel.items():
            lst = []
            for r in reps:
                it = by_stem.get(r)
                if it is None:
                    missing.append(r)
                    continue
                it["fold_v1"] = rep_fold[r]
                lst.append(it)
            self.oof_sel[key] = lst
        info = {"folds_csv": str(fcsv), "n_pairs_prepared": len(sub), "seconds": round(time.time() - t, 1),
                "n_selected": {f"{L}/f{k}": len(v) for (L, k), v in self.oof_sel.items()},
                "representatives_missing": missing,
                "layout_counts_all_folds": dict(Counter(lay.values()).most_common(20))}
        log(f"orient OOF prep: {info['n_selected']} ({info['seconds']}s); missing reps {len(missing)}")
        return info

    def build_items(self) -> list[dict]:
        fo = self.fo
        test_dir = fo.find_dir("test_set_v2")
        files = sorted([p for p in test_dir.rglob("*") if p.suffix.lower() in fo.IMAGE_SUFFIXES], key=lambda p: p.name)
        entries = [("test", p.stem, p, p.name) for p in files]
        for name in ("static", "dynamic", "vl"):
            for key, path, src in fo.gather_extra(name):  # keys computed on the FULL list, as production
                if name == "vl" and path.suffix.lower() != ".tiff":
                    continue  # annotated .tif twins: never a model input here
                entries.append((name, key, path, src))
        items = []
        for ds, key, path, src in entries:
            gray = fo.read_gray(path)
            if max(gray.shape) > 4000 or min(gray.shape) < 64:
                log(f"skip {ds}/{src}: shape {gray.shape}")
                continue
            box, kind = fo.viewport(gray, path.suffix.lower())
            x1, y1, x2, y2 = box
            canvas, mtx = fo.make_input(gray[y1:y2, x1:x2])
            items.append(dict(ds=ds, key=key, path=path, src=src, H=int(gray.shape[0]), W=int(gray.shape[1]),
                              canvas=canvas, mtx=mtx, box=[int(v) for v in box], layout=kind))
        return items

    def load_models(self, torch, dev):
        fo = self.fo
        smp = fo.ensure_smp()
        if smp.__version__ != "0.5.0":
            raise RuntimeError(f"smp {smp.__version__} != 0.5.0")
        ck0 = INDEX.find("fold0.pt", "umud-fasc-orient-v1")
        models, rep = {}, {"checkpoints": {}}
        for k in self.folds_all:
            cp = ck0.parent / f"fold{k}.pt"
            st = torch.load(cp, map_location="cpu", weights_only=False)
            mdl = fo.build_model(pretrained=False)
            mdl.load_state_dict({n: v.float() if v.is_floating_point() else v for n, v in st["state_dict"].items()})
            models[k] = mdl.to(dev).eval().to(memory_format=torch.channels_last)
            rep["checkpoints"][str(k)] = {"path": str(cp), "sha256": sha256_file(cp), "epoch": st.get("epoch"),
                                          "dtypes": sorted({str(v.dtype) for v in st["state_dict"].values()
                                                            if torch.is_tensor(v)})}
        return models, rep

    def infer(self, models: list, it: dict) -> dict:
        """train.py predict_native from the stored canvas (same make_input), device as a parameter."""
        torch, fo = self.torch, self.fo
        x1, y1, x2, y2 = it["box"]
        x = fo.to_input(torch.from_numpy(it["canvas"][None]).to(self.dev), torch)
        acc = None
        with torch.no_grad():
            for model in models:
                p, c, s = fo.forward_maps(model, x, torch, True)
                cur = torch.stack([p[0], c[0], s[0]], -1)
                acc = cur if acc is None else acc + cur
        maps = (acc / len(models)).cpu().numpy().astype(np.float32)
        back = fo.unletterbox(maps, it["mtx"], x2 - x1, y2 - y1)
        full = np.zeros((it["H"], it["W"], 3), np.float32)
        full[y1:y2, x1:x2] = back
        enc = fo.encode_channels(full[..., 0], full[..., 1], full[..., 2])
        return {f"fasc_orient_v1_{c}": enc[c] for c in self.CH}

    def frozen_path(self, ds: str, mn: str) -> Path | None:
        return INDEX.find(f"{ds}__{mn}.npz", "umud-fasc-orient-v1", required=False)

    def run_oof(self, models, mods, order, src, bad) -> dict:
        """Per native mask layout: fold k's validation images adapt fold k's model; v1 evaluate() (hflip TTA)."""
        torch, dev, fo = self.torch, self.dev, self.fo
        res: dict = {}
        for (lay, k), its in self.oof_sel.items():
            if past_deadline():
                res.setdefault(lay, {})[str(k)] = {"skipped": "deadline"}
                continue
            rec = {"n": len(its), "stems": [it["stem"] for it in its]}
            if not its or (len(its) < OOF_RULE["min_n"] and not SMOKE):
                res.setdefault(lay, {})[str(k)] = {**rec, "skipped": "too few images"}
                continue
            cache = fo.build_val_cache(its, torch)
            canv = [fo.make_input(it["crop"])[0] for it in its]
            fd = Feeder(torch, dev, canv, self.calib_batch, self.make_x, self.enc_getter)
            set_state(torch, mods[k], src[k])
            rec["cache"] = fd.build_cache(models[k])
            rec["B0"] = evaluate_dev(fo, models[k], cache, torch, dev)
            for arm in ARMS["orient"]:
                if arm == "B1fx" and not bad[k]:
                    rec[arm] = rec["B1"]  # identical to B1 when the fold has no non-finite channel
                    continue
                adapt_sequential(torch, models[k], mods[k], order, fd, src[k], arm, bad[k])
                rec[arm] = evaluate_dev(fo, models[k], cache, torch, dev)
                set_state(torch, mods[k], src[k])
            fd.free()
            del cache
            res.setdefault(lay, {})[str(k)] = rec
            log(f"orient OOF {lay} fold {k} n={len(its)}: " + ", ".join(
                f"{a} ang {rec[a]['ang_mae']:.3f} dice50 {rec[a]['dice50']:.3f}" for a in ["B0"] + ARMS["orient"]))
        return {"metric": "fasc_orient_v1 evaluate() with hflip TTA: ang_mae (deg), dice50, med_err",
                "layouts": res, "verdict": oof_verdict(res, ARMS["orient"], "ang_mae", "dice50")}


def evaluate_dev(fo, model, cache, torch, dev, tta: bool = True) -> dict:
    """kaggle/fasc_orient_v1/train.py evaluate() with the device as a parameter (the original calls .cuda())."""
    import math

    model.eval()
    dice_sum = {t: 0.0 for t in fo.SEG_THRESHOLDS}
    n_img, ang_img_sum, ang_img_n, ang_pix_sum, ang_pix_n, med_sum, med_n = 0, 0.0, 0, 0.0, 0, 0.0, 0
    with torch.no_grad():
        for b in cache:
            x = fo.to_input(b["img"].to(dev), torch)
            p, c, s = fo.forward_maps(model, x, torch, tta)
            seg_t = b["seg"].to(dev) > 0
            skel = b["skel"].to(dev) > 0
            om = b["om"].to(dev) > 0
            valid = b["valid"].to(dev) > 0
            ct = b["cos2"].to(dev).float()
            st = b["sin2"].to(dev).float()
            for t in fo.SEG_THRESHOLDS:
                pred = p > t
                inter = (pred & seg_t).sum((1, 2)).float()
                den = pred.sum((1, 2)).float() + seg_t.sum((1, 2)).float()
                dice_sum[t] += float(torch.where(den > 0, 2 * inter / den.clamp_min(1), torch.ones_like(den)).sum())
            th_p = 0.5 * torch.atan2(s, c)
            th_t = 0.5 * torch.atan2(st, ct)
            d = torch.remainder(th_p - th_t, math.pi)
            d = torch.minimum(d, math.pi - d) * (180.0 / math.pi)
            m = skel & om
            for i in range(x.shape[0]):
                n_img += 1
                mi = m[i]
                cnt = int(mi.sum())
                if cnt < 10:
                    continue
                e = d[i][mi]
                ang_img_sum += float(e.mean())
                ang_img_n += 1
                ang_pix_sum += float(e.sum())
                ang_pix_n += cnt
                gt_med = float(torch.median(th_t[i][mi]))
                pr = (p[i] > 0.5) & valid[i]
                pm = float(torch.median(th_p[i][pr])) if int(pr.sum()) >= 50 else float(torch.median(th_p[i][mi]))
                dm = abs(pm - gt_med) % math.pi
                med_sum += min(dm, math.pi - dm) * 180.0 / math.pi
                med_n += 1
    dices = {t: dice_sum[t] / max(1, n_img) for t in fo.SEG_THRESHOLDS}
    best_t = max(dices, key=dices.get)
    return {"dice50": dices[0.5], "dice_best": dices[best_t], "thr_best": best_t,
            "ang_mae": ang_img_sum / max(1, ang_img_n), "ang_mae_pix": ang_pix_sum / max(1, ang_pix_n),
            "med_err": med_sum / max(1, med_n), "n_val": n_img, "n_ang": ang_img_n}


def oof_verdict(res: dict, arms: list, metric: str, dice_key: str) -> dict:
    """PREREG rule per (layout, arm): metric(arm) - metric(B0) <= 0 on >= 4 evaluable folds (n >= 10 each), mean <= 0,
    and the mean dice drop <= 0.005. Fewer than 4 evaluable folds -> 'n/a' (never a pass)."""
    out = {}
    for lay, folds in res.items():
        for arm in arms:
            d, dd = [], []
            for rec in folds.values():
                if arm in rec and "B0" in rec:
                    d.append(rec[arm][metric] - rec["B0"][metric])
                    dd.append(rec[arm][dice_key] - rec["B0"][dice_key])
            if len(d) < OOF_RULE["min_folds_le0"]:
                v = "n/a"
            else:
                v = "pass" if (sum(x <= 0 for x in d) >= OOF_RULE["min_folds_le0"] and np.mean(d) <= OOF_RULE["mean_le"]
                               and -np.mean(dd) <= OOF_RULE["dice_drop_max"]) else "fail"
            out[f"{lay}/{arm}"] = {"d_per_fold": [float(x) for x in d], "mean_d": float(np.mean(d)) if d else None,
                                   "mean_d_dice": float(np.mean(dd)) if dd else None, "verdict": v}
    return out


# ============================================================================================ generic network run
def finish_items(st, items: list[dict]) -> tuple[list[dict], dict]:
    tails = set(TABLES["vl_raw_tails"])
    out, dropped = [], []
    for it in items:
        if it["ds"] == "vl" and tail3(it["src"]) not in tails:
            dropped.append(it["src"])
            continue
        it["domain"] = domain_of(it["ds"], it["key"], it["W"], it["H"])
        out.append(it)
    n_ds = Counter(it["ds"] for it in out)
    n_dom = Counter(it["domain"] or "UNADAPTED" for it in out)
    dup = [k for k, c in Counter((it["ds"], it["key"]) for it in out).items() if c > 1]
    info = {"n_dataset": dict(n_ds), "n_domain": dict(n_dom), "vl_dropped_not_raw": dropped, "duplicate_keys": dup}
    log(f"{st.name} data: {dict(n_ds)} domains {dict(n_dom)}; VL dropped {len(dropped)}")
    if dup or any(n_ds.get(k, 0) != v for k, v in EXPECTED_N.items()):
        msg = f"data mismatch vs production: {dict(n_ds)} (expected {EXPECTED_N}), duplicate keys {dup[:5]}"
        if not SMOKE:
            raise RuntimeError(msg)
        log("WARNING " + msg)
    return out, info


def repro_check(st, items: list[dict], models: dict) -> dict:
    """B0 maps vs the production banks on the fixed subset (embedded digests of the local frozen arrays; full-resolution
    diff when the attached frozen kernel output holds the identical array)."""
    subset = TABLES["repro"][st.name]
    if SMOKE:  # first entry of each dataset
        seen, sub2 = set(), []
        for e in subset:
            if e["ds"] not in seen:
                seen.add(e["ds"])
                sub2.append(e)
        subset = sub2
    idx = {(it["ds"], it["key"]): it for it in items}
    rows, ok_all, frozen_files = [], True, {}
    for e in subset:
        it = idx.get((e["ds"], e["key"]))
        if it is None:
            rows.append({"ds": e["ds"], "key": e["key"], "pass": False, "error": "key not produced by this kernel"})
            ok_all = False
            continue
        arrays = st.infer([models[k] for k in st.repro_folds(e["ds"])], it)
        for mn, ref in e["maps"].items():
            a = arrays[mn]
            r = {"ds": e["ds"], "key": e["key"], "model": mn, "shape_ok": list(a.shape) == list(ref["shape"]),
                 "self_sha1_equal": arr_sha1(a) == ref["sha1"]}
            if r["shape_ok"]:
                r["block_max_abs"] = float(np.abs(block_means(a) - np.asarray(ref["bm"])).max())
                fp = st.frozen_path(e["ds"], mn)
                frozen = None
                if fp is not None:
                    if fp not in frozen_files:
                        frozen_files[fp] = np.load(fp)
                    if e["key"] in frozen_files[fp].files:
                        frozen = frozen_files[fp][e["key"]]
                if frozen is not None and arr_sha1(frozen) == ref["sha1"]:
                    d = np.abs(a.astype(np.int16) - frozen.astype(np.int16))
                    r.update(basis="full resolution vs the attached production array", max_abs=int(d.max()),
                             mean_abs=float(d.mean()), frac_gt2=float((d > 2).mean()))
                    r["pass"] = r["mean_abs"] <= REPRO_TOL["mean_abs"] and r["frac_gt2"] <= REPRO_TOL["frac_gt2"]
                else:
                    r["basis"] = "16x16 block means vs the local production array (attached copy absent or different)"
                    r["pass"] = r["block_max_abs"] <= REPRO_TOL["block_mean_max_abs"]
            else:
                r["pass"] = False
            ok_all = ok_all and r["pass"]
            rows.append(r)
    for f in frozen_files.values():
        f.close()
    n_fail = sum(not r["pass"] for r in rows)
    log(f"{st.name} B0 reproduction: {len(rows) - n_fail}/{len(rows)} maps pass -> {'PASS' if ok_all else 'FAIL'}")
    for r in rows:
        if not r["pass"]:
            log(f"  repro FAIL {r}")
    return {"pass": bool(ok_all), "tolerance": REPRO_TOL, "n_maps": len(rows), "n_fail": n_fail, "rows": rows,
            "folds": {ds: st.repro_folds(ds) for ds in EXPECTED_N}}


def write_bank(arm: str, st, items: list[dict], models: dict) -> dict:
    bw = BankWriter(arm, st)
    t = time.time()
    for i, it in enumerate(items):
        bw.add(it, st.infer([models[k] for k in st.folds_for(it["ds"])], it))
        if (i + 1) % 100 == 0:
            log(f"  {arm} {st.name}: {i + 1}/{len(items)} maps ({time.time() - t:.0f}s)")
    bw.close()
    return {"n": len(items), "seconds": round(time.time() - t, 1)}


def run_arms(st, items, models, mods, order, src, bad, rep) -> None:
    torch, dev = st.torch, st.dev
    arms = ARMS[st.name]
    writers = {arm: BankWriter(arm, st) for arm in arms}
    by_dom = defaultdict(list)
    for it in items:
        by_dom[it["domain"]].append(it)
    rep["arms_run"] = {"arms": arms, "domains": {}, "complete": True}
    shifts: dict = {}
    for dom in DOMAIN_ORDER:
        its = by_dom.get(dom, [])
        if not its:
            continue
        if past_deadline():
            log(f"deadline: domain {dom} and later not adapted")
            rep["arms_run"]["complete"] = False
            break
        t = time.time()
        folds = st.folds_for(its[0]["ds"])
        states, infos, drec = {}, {}, {"n": len(its), "folds": folds, "cache": {}}
        for k in folds:
            fd = Feeder(torch, dev, [it["canvas"] for it in its], st.calib_batch, st.make_x, st.enc_getter)
            set_state(torch, mods[k], src[k])
            drec["cache"][str(k)] = fd.build_cache(models[k])
            for arm in arms:
                if arm == "B1fx" and not bad[k]:
                    states[(arm, k)] = states[("B1", k)]
                    continue
                stt, inf = adapt_sequential(torch, models[k], mods[k], order, fd, src[k], arm, bad[k])
                states[(arm, k)], infos[(arm, k)] = stt, inf
                shifts.setdefault(dom, {}).setdefault(arm, {})[f"f{k}"] = shift_summary(torch, src[k], stt)
            set_state(torch, mods[k], src[k])
            fd.free()
            del fd
            gc.collect()
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        save_bn(BNDIR / st.name / f"{dom}.npz", folds, src, states, infos)
        drec["calib_s"] = round(time.time() - t, 1)
        t2 = time.time()
        for arm in arms:
            for k in folds:
                set_state(torch, mods[k], states[(arm, k)])
            for it in its:
                writers[arm].add(it, st.infer([models[k] for k in folds], it))
        for k in folds:
            set_state(torch, mods[k], src[k])
        drec["infer_s"] = round(time.time() - t2, 1)
        rep["arms_run"]["domains"][dom] = drec
        log(f"{st.name} domain {dom}: n={len(its)} folds {folds} calib {drec['calib_s']}s infer {drec['infer_s']}s "
            f"cache {[v.get('cache') for v in drec['cache'].values()]}")
        save_report(st.name, rep)
    reader = B0Reader(st)
    for it in by_dom.get(None, []):  # static images outside S660 / S800: B0 maps in every arm (unadapted)
        arr = reader.get(it, st.model_names)
        for arm in arms:
            writers[arm].add(it, arr)
    for w in writers.values():
        w.close()
    (REPORTS / f"bn_shift_{st.name}.json").write_text(json.dumps(shifts, indent=1))


def save_report(name: str, rep: dict) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    rep["elapsed_min"] = round((time.time() - T0) / 60, 2)
    (REPORTS / f"{name}.json").write_text(json.dumps(rep, indent=1, default=str))


def write_domains(st, items: list[dict]) -> None:
    n_dom = Counter(it["domain"] for it in items)
    with open(REPORTS / f"domains_{st.name}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["net", "dataset", "image_id", "domain", "adapted", "n_domain", "height", "width", "source_filename"])
        for it in items:
            w.writerow([st.name, it["ds"], it["key"], it["domain"] or "", int(it["domain"] is not None),
                        n_dom[it["domain"]] if it["domain"] else 0, it["H"], it["W"], it["src"]])


def env_info(torch, dev) -> dict:
    info = {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
            "device": str(dev), "cpus": os.cpu_count(), "mem_available_gb": round(mem_available() / 2 ** 30, 1)}
    if dev.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
    try:
        import segmentation_models_pytorch as smp
        import timm

        info["smp"], info["timm"] = smp.__version__, timm.__version__
    except Exception as e:  # noqa: BLE001
        info["smp"] = repr(e)
    return info


def run_net(st) -> None:
    rep: dict = {"net": st.name, "mode": MODE, "prereg_sha256": TABLES.get("prereg_sha256"),
                 "addendum_sha256": TABLES.get("addendum_sha256"), "phase_s": {},
                 "status": "running", "arms": ["B0"] + ARMS[st.name]}
    t = time.time()
    if isinstance(st, OrientStage):
        try:
            rep["oof_prep"] = st.prepare_oof()  # first: no torch / cv2 thread pool in this process yet
        except Exception:  # noqa: BLE001  (a failed OOF preparation must not cost the maps: OOF then reports n/a)
            st.oof_sel = {}
            rep["oof_prep"] = {"error": traceback.format_exc()}
            log(f"orient OOF preparation failed (OOF check will be empty = n/a):\n{rep['oof_prep']['error']}")
        rep["phase_s"]["oof_prep"] = round(time.time() - t, 1)
    torch, dev = torch_setup()
    st.torch, st.dev = torch, dev
    rep["env"] = env_info(torch, dev)
    log(f"{st.name}: env {rep['env']}")
    save_report(st.name, rep)
    t = time.time()
    items = st.build_items()
    items, rep["data"] = finish_items(st, items)
    rep["phase_s"]["data"] = round(time.time() - t, 1)
    t = time.time()
    models, rep["load"] = st.load_models(torch, dev)
    rep["phase_s"]["load"] = round(time.time() - t, 1)
    mods = {k: bn_modules(torch, mdl) for k, mdl in models.items()}
    counts = {str(k): len(v) for k, v in mods.items()}
    dtypes = sorted({str(m.running_var.dtype) for v in mods.values() for m in v.values()})
    k0 = min(models)
    first = items[0]
    order = exec_order(torch, models[k0], dev, lambda: models[k0](st.make_x(first["canvas"][None])))
    src = {k: get_state(torch, mods[k]) for k in models}
    bad = {k: bad_channels(torch, src[k]) for k in models}
    rep["bn"] = {"expected": EXPECTED_BN[st.name], "count_per_fold": counts, "buffer_dtypes": dtypes,
                 "exec_order": order, "bad_channels": {str(k): {n: len(ix) for n, ix in b.items()} for k, b in bad.items()}}
    if st.name == "orient":
        rep["bn"]["bad_expected"] = TABLES.get("orient_bad_expected")
    log(f"{st.name}: BN layers {counts}; non-finite source channels {rep['bn']['bad_channels']}")
    if any(c != EXPECTED_BN[st.name] for c in counts.values()) or dtypes != ["torch.float32"]:
        rep["status"] = "abort: BatchNorm inventory differs from the PREREG"
        save_report(st.name, rep)
        raise SystemExit(5)
    if st.name == "apo" and any(bad.values()):
        rep["status"] = "abort: apo B0 statistics not finite after the production repair"
        save_report(st.name, rep)
        raise SystemExit(5)
    t = time.time()
    rep["repro"] = repro_check(st, items, models)
    rep["phase_s"]["repro"] = round(time.time() - t, 1)
    ok = rep["repro"]["pass"] and rep["load"].get("fix_check_pass", True)
    save_report(st.name, rep)
    if not ok:
        if not SMOKE:
            rep["status"] = "abort: B0 does not reproduce the production maps"
            save_report(st.name, rep)
            raise SystemExit(4)
        log("cpusmoke: reproduction failed (expected for apo, which uses fold 0 only in the dry run); continuing")
    items = smoke_select(items)
    write_domains(st, items)
    t = time.time()
    for k in models:
        set_state(torch, mods[k], src[k])
    rep["B0_bank"] = write_bank("B0", st, items, models)
    rep["phase_s"]["B0_maps"] = round(time.time() - t, 1)
    save_report(st.name, rep)
    t = time.time()
    run_arms(st, items, models, mods, order, src, bad, rep)
    rep["phase_s"]["arms"] = round(time.time() - t, 1)
    save_report(st.name, rep)
    if hasattr(st, "run_oof"):
        t = time.time()
        if past_deadline():
            rep["oof"] = {"skipped": "deadline"}
        else:
            try:
                rep["oof"] = st.run_oof(models, mods, order, src, bad)
            except Exception:  # noqa: BLE001  (the maps are written; a failed check is reported, not fatal)
                rep["oof"] = {"error": traceback.format_exc()}
                log(f"OOF check failed:\n{rep['oof']['error']}")
        rep["phase_s"]["oof"] = round(time.time() - t, 1)
    rep["status"] = "done" if rep["arms_run"]["complete"] else "partial (deadline)"
    save_report(st.name, rep)
    log(f"{st.name}: {rep['status']} phases {rep['phase_s']}")


# ============================================================================================ finalize
def stage_finalize() -> None:
    reps = {}
    for n in ("apo", "ours", "orient"):
        p = REPORTS / f"{n}.json"
        if p.exists():
            reps[n] = json.loads(p.read_text())
    rows = []
    for n in ("apo", "ours", "orient"):
        p = REPORTS / f"domains_{n}.csv"
        if p.exists():
            rows += read_csv(p)
    if rows:
        with open(OUT / "domain_manifest.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    shifts = {n: json.loads((REPORTS / f"bn_shift_{n}.json").read_text())
              for n in ("apo", "ours", "orient") if (REPORTS / f"bn_shift_{n}.json").exists()}
    (OUT / "bn_shift.json").write_text(json.dumps(shifts, indent=1))
    (OUT / "repro.json").write_text(json.dumps(
        {n: {"repro": r.get("repro"), "apo_fix_check": r.get("load", {}).get("fix_check")} for n, r in reps.items()},
        indent=1, default=str))
    (OUT / "oof_check.json").write_text(json.dumps({n: r.get("oof") for n, r in reps.items()}, indent=1, default=str))
    files, total = [], 0
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            size = p.stat().st_size
            total += size
            files.append({"path": str(p.relative_to(OUT)), "bytes": size, "sha256": sha256_file(p)})
    codes = json.loads(os.environ.get("V25_CODES", "{}"))
    man = {"experiment": "v25 AdaBN (BatchNorm running statistics only, per domain, unlabeled images)",
           "prereg": "artifacts/candidates_v25/PREREG_v25_adabn.json", "prereg_sha256": TABLES.get("prereg_sha256"),
           "addendum": "artifacts/candidates_v25/PREREG_v25_addendum_A.json",
           "addendum_sha256": TABLES.get("addendum_sha256"),
           "source_sha256": TABLES.get("source_sha256"), "built": TABLES.get("built"), "mode": MODE,
           "exit_codes": codes, "minutes_since_launch": round((time.time() - LAUNCH_T0) / 60, 1),
           "networks": {n: {k: r.get(k) for k in ("status", "env", "data", "bn", "phase_s", "arms_run", "B0_bank",
                                                  "elapsed_min")}
                        | {"checkpoints": r.get("load", {}).get("checkpoints"),
                           "repro_pass": (r.get("repro") or {}).get("pass"),
                           "apo_fix_check_pass": r.get("load", {}).get("fix_check_pass")}
                        for n, r in reps.items()},
           "total_mb": round(total / 1e6, 1), "files": files}
    if total > 2e9:
        man["warning"] = "outputs exceed 2 GB"
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1, default=str))
    log(f"finalize: {len(files)} files, {total / 1e6:.1f} MB; statuses { {n: r.get('status') for n, r in reps.items()} }")


# ============================================================================================ local self-test
def selftest() -> None:
    """Torch-free checks of the helpers (runs on the laptop: numpy only)."""
    global TABLES
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (37, 53)).astype(np.uint8)
    bm = block_means(a, 4)
    ys, xs = np.linspace(0, 37, 5).round().astype(int), np.linspace(0, 53, 5).round().astype(int)
    ref = np.array([[a[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].mean() for j in range(4)] for i in range(4)])
    assert np.allclose(bm, ref), "block_means"
    assert tail3("vl_young/NeuAge/NAY_1/timepoint1/ANON_2.tiff") == "nay_1/timepoint1/anon_2.tiff"
    assert arr_sha1(a) == arr_sha1(a.copy()) and arr_sha1(a) != arr_sha1(a.T.copy())
    TABLES = {"test_group": {"IMG_00001": "SIE"}}
    assert domain_of("test", "IMG_00001", 1200, 800) == "SIE"
    assert domain_of("static", "im_01_arch", 660, 556) == "S660" and domain_of("static", "x", 512, 512) is None
    tmp = Path(os.environ.get("V25_SELFTEST_DIR", "."))
    p = tmp / "_v25_selftest.npz"
    w = NpzZip(p)
    w.add("k1", a)
    w.close()
    with np.load(p) as f:
        assert (f["k1"] == a).all()
    p.unlink()
    print("selftest ok")


def main() -> None:
    global STAGE, TABLES
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["apo", "ours", "orient", "finalize", "selftest"])
    a = ap.parse_args()
    STAGE = a.stage
    if a.stage == "selftest":
        selftest()
        return
    for d in (OUT, MAPS, BNDIR, REPORTS, LOGS):
        d.mkdir(parents=True, exist_ok=True)
    TABLES = json.loads(TABLES_PATH.read_text(encoding="utf-8"))
    if a.stage == "finalize":
        stage_finalize()
        return
    st = {"apo": ApoStage, "ours": OursStage, "orient": OrientStage}[a.stage]()
    run_net(st)


if __name__ == "__main__":
    main()
