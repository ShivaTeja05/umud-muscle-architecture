# UMUD Challenge solution code -- v30 S5 direct image -> (PA, FL) regressor (Kaggle kernel script)
# Copyright (C) 2026 Gaddam Shivateja
# SPDX-License-Identifier: GPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 (GPL-3.0-only). It is distributed in the hope that
# it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See <https://www.gnu.org/licenses/gpl-3.0.txt>. Third-party parts: see CREDITS below.
"""v30 S5: ONE pre-registered direct regressor (artifacts/candidates_v30/s5/PREREG_v30_s5.json).

Model   timm convnext_tiny.fb_in22k_ft_in1k, global average pool, two MLP regression heads:
        PA (deg, fascicle vs the DEEP aponeurosis) and log(FL_canvas / SIZE) (FL in canvas pixels; converted to mm
        exactly with the per-image letterbox scale and the per-source px/mm, so the network never has to read a scale bar).
Labels  geometric primitives in raw image pixels, transformed EXACTLY with the image geometry ("exact-affine label
        augmentation"): PA = acute angle between L f and L a (f = fascicle direction at the deep end, a = deep
        aponeurosis direction, L = linear part of the full raw-crop -> canvas matrix incl. augmentation); FL = r * sum |L d_i|
        over the expert fascicle chain (r makes it equal to the expert FL at L = s I). Dynamic FL (no geometry) is used
        only under similarity transforms (FL' = s FL, asserted).
Data    train: VL expert lines (OSF xbawc, 244 images with >= 2 tracks; subject fold 0 held out as an information-only
        validation fold), TimTrack paper datasets 3 + 4 (PA_deep = alpha - gamma per observer; FL masked), UMUD GM
        dynamic calf raise (FL only; PA masked). static35 (OSF benchmark) is NEVER trained on: it is the gate.
        No competition image is trained on. Test images (competition) are inference only.
Guards  training pool vs static35 by id, by pixel sha256 and by thumbnail NCC (> 0.95 -> dropped); vs the 309 test images
        by thumbnail NCC (> 0.95 -> dropped); the 25 static copies of competition training images are part of static35.
Output  /kaggle/working: static_pred.csv, test_pred.csv (image_id, pa, fl), vl_val_pred.csv, pred_by_seed.csv,
        metrics.json, logs/, CREDITS.txt. No image, map or checkpoint is written to /kaggle/working.
Modes   UMUD_S5_MODE=gpu (T4 x2: one seed per GPU, same fold) | cpusmoke (CPU: tiny subset, 3 steps, 1 seed, same files).
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")  # never initialise CUDA in the parent before fork
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import copy
import csv
import hashlib
import json
import math
import multiprocessing as mp
import shutil
import signal
import subprocess
import sys
import time
import traceback
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

# ----------------------------------------------------------------------------- declared configuration (PREREG)
MODE = os.environ.get("UMUD_S5_MODE", "gpu")  # "gpu" | "cpusmoke"
SMOKE = MODE == "cpusmoke"
T0 = time.time()
SIZE = 512
ENCODER = "convnext_tiny.fb_in22k_ft_in1k"
DROP_PATH = 0.1
SEEDS = [20261107] if SMOKE else [20261107, 20261108]
VAL_FOLD = 0                      # VL subject fold held out (information only; no checkpoint selection)
BATCH = 2 if SMOKE else 16
MAX_STEPS = 3 if SMOKE else None  # per epoch (smoke: 3 steps in total)
MAX_EPOCHS = 1 if SMOKE else 80
MIN_EPOCHS = 1 if SMOKE else 12
TRAIN_BUDGET_S = 0.85 * 3600      # per seed, from its training start (epochs fitted to it after epoch 1)
HARD_TRAIN_END_S = 1.25 * 3600    # from kernel start: the epoch loop stops here whatever the fit says
HARD_WALL_S = 2.0 * 3600          # from kernel start: the parent kills unfinished seed workers here
LR_BACKBONE, LR_HEAD, WEIGHT_DECAY = 1e-4, 1e-3, 0.05
WARMUP_EPOCHS, LR_FLOOR = 2.0, 0.02
EMA_DECAY = 0.995
GRAD_CLIP = 1.0
HUBER_BETA = 1.0
PA_MU, PA_SD = 20.0, 8.0          # target normalisation constants (fixed, declared)
LFL_MU, LFL_SD = -0.4, 0.35       # log(FL_canvas / SIZE)
MIX = {"vl": 2, "tt": 1, "dyn": 60}  # per epoch: each VL-train image x2, each TT image x1, 60 random dynamic frames
AUG = {"p_crop_w": 0.35, "crop_w": (0.4, 1.0), "p_crop_h": 0.15, "crop_h": (0.7, 1.0), "p_scale": 0.7,
       "scale": (0.8, 1.25), "p_aniso": 0.5, "aniso": 0.20, "p_rot": 0.6, "rot_deg": 10.0, "p_shift": 0.5,
       "shift": 0.06, "p_flip": 0.5}
ANISO_KINDS = ("vl", "tt")        # dynamic samples: similarity transforms only (FL' = s FL exactly)
DUP_NCC, THUMB = 0.90, 48         # flip-aware near-duplicate threshold (VL/TT non-copies <= 0.83 vs static35 on the laptop)
SMOKE_N = {"vl": 8, "tt": 4, "dyn": 4, "vlval": 3}
INFER_BATCH = 8
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
NATIVE_GROUPS = {"siemens_juniper", "philips_lumify"}
COMP_SLUG = "umud-challenge-muscle-architecture-in-ultrasound-data"
WORK = Path("/kaggle/working")
TMP = Path("/tmp/s5")
SRC = Path(os.environ.get("UMUD_S5_SRC", "/tmp/s5_src"))  # payload written by the runner (never in /kaggle/working)
LOG_DIR = WORK / "logs"
EXTRA_SOURCES = {
    "static": [("https://osf.io/download/hb5xj/", "static.zip")],
    "dynamic": [("https://osf.io/download/vqb7h/", "dynamic.zip")],
    "vl": [("https://osf.io/download/zcuwk/", "vl_young.zip"), ("https://osf.io/download/6ga94/", "vl_old.zip")],
}
_LOG_PREFIX = ""
_LOG_FILE: Path | None = None
DATA: dict = {}  # filled in main() before any fork; inherited by the GPU workers


def config_dict() -> dict:
    keys = ("MODE", "SIZE", "ENCODER", "DROP_PATH", "SEEDS", "VAL_FOLD", "BATCH", "MAX_STEPS", "MAX_EPOCHS",
            "MIN_EPOCHS", "TRAIN_BUDGET_S", "HARD_TRAIN_END_S", "HARD_WALL_S", "LR_BACKBONE", "LR_HEAD",
            "WEIGHT_DECAY", "WARMUP_EPOCHS", "LR_FLOOR", "EMA_DECAY", "GRAD_CLIP", "HUBER_BETA", "PA_MU", "PA_SD",
            "LFL_MU", "LFL_SD", "MIX", "AUG", "ANISO_KINDS", "DUP_NCC", "THUMB", "SMOKE_N", "INFER_BATCH")
    return {k: globals()[k] for k in keys}


# ----------------------------------------------------------------------------- logging
def log(msg: str) -> None:
    line = f"[{(time.time() - T0) / 60:7.2f}m]{_LOG_PREFIX} {msg}\n"
    try:
        os.write(1, line.encode())
    except OSError:
        print(line, end="", flush=True)
    if _LOG_FILE is not None:
        with open(_LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(line)


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout
    except Exception as exc:  # noqa: BLE001
        return f"<{exc!r}>"


# ----------------------------------------------------------------------------- viewport (= kaggle/fasc_orient_v1)
def _largest_run(values: np.ndarray) -> tuple[int, int]:
    padded = np.pad(values.astype(np.int8), 1)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    if not len(starts):
        return 0, len(values)
    k = int(np.argmax(ends - starts))
    return int(starts[k]), int(ends[k])


def _first_true(values: np.ndarray, start: int = 0) -> int | None:
    idx = np.flatnonzero(values[start:])
    return int(start + idx[0]) if len(idx) else None


def _last_true(values: np.ndarray) -> int | None:
    idx = np.flatnonzero(values)
    return int(idx[-1]) if len(idx) else None


def texture_viewport(gray: np.ndarray) -> tuple[int, int, int, int]:
    h, w = gray.shape
    g = gray.astype(np.float32)
    x1, x2 = _largest_run(np.abs(np.diff(g, axis=0)).mean(axis=0) > 1.0)
    if x2 - x1 < w * 0.35:
        bright = gray > max(8, int(np.percentile(gray, 35)))
        x1, x2 = _largest_run(bright.mean(axis=0) > 0.35)
        if x2 - x1 < w * 0.2:
            x1, x2 = 0, w
    y1, y2 = _largest_run(np.abs(np.diff(g[:, x1:x2], axis=1)).mean(axis=1) > 0.5)
    if y2 - y1 < h * 0.45:
        y1, y2 = 0, h
    return x1, y1, x2, y2


def layout_viewport(gray: np.ndarray, suffix: str) -> tuple[tuple[int, int, int, int], str] | None:
    """Scanner-layout rules (same as src/umud/geometry.ultrasound_metadata, no OCR).

    Layout rules adapted from AmbrosM, "UMUD Quick and Dirty"
    (https://www.kaggle.com/code/ambrosm/umud-quick-and-dirty), Apache-2.0; see THIRD_PARTY_NOTICES.md.
    """
    h, w = gray.shape
    if suffix == ".png" and (w, h) == (1200, 800):
        first = _first_true(gray[:, 6] > 50)
        second_minor = _first_true(gray[:, 6] > 50, 150)
        second_major = _first_true(gray[:, 9] > 50, 150)
        last = _last_true(gray[:, 6] > 50)
        if None not in (first, second_minor, second_major, last):
            half = w // 2
            span = int(np.argmin(gray[:, half:].sum(axis=0)))
            if span > 40 and last - first > 200:
                return (half - span, first, half + span, last), "philips_lumify"
    if (w, h) == (1200, 800):
        if np.all(np.abs(gray[87, 1147:1157].astype(int) - 175) <= 3):
            first = _first_true(gray[:, 1150] > 50)
            last = _last_true(gray[:, 1150] > 50)
            if first is not None and last is not None:
                second = _first_true(gray[:, 1150] > 50, first + 20)
                if second is not None and second > first:
                    n = round((last - first) / (second - first))
                    boxes = {7: 142, 8: 163, 9: 211, 10: 249, 11: 282, 12: 308, 13: 331, 14: 349, 15: 142}
                    if n in boxes and last > 300:
                        x1 = boxes[n]
                        return (x1, 91, 1200 - x1, last), "siemens_juniper"
        if np.all(gray[42, 67:74] > 115):
            return (171, 42, 1029, 798), "telemed_artus_1200"
    if (w, h) == (1088, 644):
        return (140, 0, 947, 643), "telemed_artus_1088"
    if h in (512, 513):
        return (0, 0, w, h - 10), "legacy_512"
    if h == 853:
        return (0, 0, w, h), "legacy_853"
    return None


def viewport(gray: np.ndarray, suffix: str) -> tuple[tuple[int, int, int, int], str]:
    h, w = gray.shape
    found = layout_viewport(gray, suffix)
    if found is None:
        box, kind = texture_viewport(gray), "texture"
    else:
        box, kind = found
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 32 or y2 - y1 < 32 or (x2 - x1) * (y2 - y1) < 0.08 * w * h:
        return (0, 0, w, h), kind + "_full"
    return (x1, y1, x2, y2), kind


# ----------------------------------------------------------------------------- image helpers
def read_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)


def pixel_sha256(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(arr.shape, np.int64).tobytes())
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def thumbs(gray: np.ndarray) -> np.ndarray:
    """(3, THUMB^2) unit-norm centred thumbnails: full frame, central 80 %, central 60 % (PIL BOX resize)."""
    h, w = gray.shape
    out = []
    for f in (0.0, 0.1, 0.2):
        c = gray[int(h * f):max(int(h * f) + 1, int(h * (1 - f))), int(w * f):max(int(w * f) + 1, int(w * (1 - f)))]
        t = np.asarray(Image.fromarray(np.ascontiguousarray(c)).resize((THUMB, THUMB), Image.BOX), np.float32).ravel()
        t -= t.mean()
        out.append(t / (np.linalg.norm(t) + 1e-6))
    return np.stack(out).astype(np.float32)


FLIPS = ("id", "hflip", "vflip", "hvflip")


def flips(gray: np.ndarray) -> list[np.ndarray]:
    return [gray, np.ascontiguousarray(gray[:, ::-1]), np.ascontiguousarray(gray[::-1]),
            np.ascontiguousarray(gray[::-1, ::-1])]


def thumbs_flips(gray: np.ndarray) -> np.ndarray:
    """(12, THUMB^2): thumbs() of the image and of its h / v / hv mirror (static im_10_arch is an exact mirror copy)."""
    return np.concatenate([thumbs(g) for g in flips(gray)])


def max_ncc(t: np.ndarray, bank: np.ndarray) -> tuple[float, int]:
    """max over bank images and all thumbnail pairs; t (k, D), bank (N, 3, D)."""
    if bank is None or not len(bank):
        return -1.0, -1
    c = np.einsum("kd,njd->nkj", t, bank).max(axis=(1, 2))
    j = int(np.argmax(c))
    return float(c[j]), j


def guard_pool(pool: list[dict], static_items: list[dict], test_bank: np.ndarray | None) -> tuple[list, dict]:
    """Drop a training image if its id equals a static35 id, if any of its 4 mirror variants has the pixel sha256 of a
    static35 image, or if its flip-aware thumbnail NCC vs any static35 image (or any test image) exceeds DUP_NCC.
    pool items need id / kind / sha_flips / th; static items need id / sha / th (3, D)."""
    static_ids = {it["id"] for it in static_items}
    static_sha = {it["sha"]: it["id"] for it in static_items}
    static_bank = np.stack([it["th"] for it in static_items])
    keep, dropped = [], []
    for it in pool:
        ncc_s, js = max_ncc(it["th"], static_bank)
        ncc_t, _ = max_ncc(it["th"], test_bank) if test_bank is not None else (-1.0, -1)
        reason = []
        if it["id"] in static_ids:
            reason.append("static_id")
        hit = [f"{FLIPS[k]}={static_sha[h]}" for k, h in enumerate(it["sha_flips"]) if h in static_sha]
        if hit:
            reason.append("static_sha256 " + ",".join(hit))
        if ncc_s > DUP_NCC:
            reason.append(f"static_ncc {ncc_s:.4f} {static_items[js]['id']}")
        if ncc_t > DUP_NCC:
            reason.append(f"test_ncc {ncc_t:.4f}")
        it["max_ncc_static"], it["max_ncc_static_id"], it["max_ncc_test"] = ncc_s, static_items[js]["id"], ncc_t
        (dropped if reason else keep).append(it if not reason else {"id": it["id"], "kind": it["kind"], "reason": reason})
    kept_sha = {h for it in keep for h in it["sha_flips"]}
    g = {"dropped": dropped, "dropped_ids": sorted(d["id"] for d in dropped), "pool_before": len(pool),
         "pool_after": len(keep), "max_ncc_static_kept": float(max(it["max_ncc_static"] for it in keep)),
         "max_ncc_test_kept": float(max(it["max_ncc_test"] for it in keep)),
         "static_id_overlap_after": len({it["id"] for it in keep} & static_ids),
         "static_sha_overlap_after_flip_aware": len(kept_sha & set(static_sha)),
         "max_ncc_kept_per_static": {}}
    for it in keep:
        k = it["max_ncc_static_id"]
        g["max_ncc_kept_per_static"][k] = max(g["max_ncc_kept_per_static"].get(k, -1.0), it["max_ncc_static"])
    return keep, g


# ----------------------------------------------------------------------------- geometry (3x3, src pixel -> canvas)
def letterbox(w: int, h: int, c: float = 1.0) -> np.ndarray:
    """Raw crop (w x h px; pixel-centre coords) -> SIZE canvas. c = label-space anisotropy: x_native = x / c, so the
    canvas shows the NATIVE square-pixel geometry; s0 is the canvas px per native px on both axes (= M[1, 1])."""
    s0 = SIZE / max(w / c, h)
    sx, sy = s0 / c, s0
    return np.array([[sx, 0.0, (SIZE - sx * w) / 2 + 0.5 * sx - 0.5],
                     [0.0, sy, (SIZE - sy * h) / 2 + 0.5 * sy - 0.5], [0, 0, 1]], np.float64)


def aug_matrix(rng: np.random.Generator, aniso: bool) -> np.ndarray:
    """Canvas-centred: hflip, then aspect stretch (image axes; VL/TT only), then iso scale, rotation, shift."""
    cc = (SIZE - 1) / 2.0
    sc = math.exp(rng.uniform(math.log(AUG["scale"][0]), math.log(AUG["scale"][1]))) if rng.random() < AUG["p_scale"] else 1.0
    ar = math.exp(rng.uniform(-AUG["aniso"], AUG["aniso"])) if rng.random() < AUG["p_aniso"] else 1.0
    if not aniso:
        ar = 1.0
    phi = math.radians(rng.uniform(-AUG["rot_deg"], AUG["rot_deg"])) if rng.random() < AUG["p_rot"] else 0.0
    if rng.random() < AUG["p_shift"]:
        tx, ty = rng.uniform(-AUG["shift"], AUG["shift"]) * SIZE, rng.uniform(-AUG["shift"], AUG["shift"]) * SIZE
    else:
        tx = ty = 0.0
    flip = -1.0 if rng.random() < AUG["p_flip"] else 1.0
    tm = np.array([[1, 0, -cc], [0, 1, -cc], [0, 0, 1]], np.float64)
    fm = np.diag([flip, 1.0, 1.0])
    am = np.diag([ar, 1.0 / ar, 1.0])
    sm = np.diag([sc, sc, 1.0])
    cs, sn = math.cos(phi), math.sin(phi)
    rm = np.array([[cs, -sn, 0], [sn, cs, 0], [0, 0, 1]], np.float64)
    tp = np.array([[1, 0, cc + tx], [0, 1, cc + ty], [0, 0, 1]], np.float64)
    return tp @ rm @ sm @ am @ fm @ tm


def warp_image(src: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Warp src with 3x3 m (src pixel coords -> SIZE canvas), anti-aliased when shrinking (= fasc_orient_v1)."""
    import cv2

    h, w = src.shape[:2]
    lin = math.sqrt(abs(np.linalg.det(m[:2, :2])))
    mw = m
    if lin < 0.75:
        f = lin / 0.9
        nw, nh = max(8, int(round(w * f))), max(8, int(round(h * f)))
        src = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_AREA)
        fx, fy = nw / w, nh / h
        pinv = np.array([[1 / fx, 0, 0.5 / fx - 0.5], [0, 1 / fy, 0.5 / fy - 0.5], [0, 0, 1]], np.float64)
        mw = m @ pinv
    return cv2.warpAffine(src, mw[:2].astype(np.float64), (SIZE, SIZE), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def poly_mask(m: np.ndarray, w: int, h: int) -> np.ndarray:
    import cv2

    corners = np.array([[-0.5, -0.5, 1], [w - 0.5, -0.5, 1], [w - 0.5, h - 0.5, 1], [-0.5, h - 0.5, 1]]).T
    p = (m @ corners)[:2].T
    out = np.zeros((SIZE, SIZE), np.uint8)
    cv2.fillConvexPoly(out, np.round(p * 16).astype(np.int32), 1, cv2.LINE_8, 4)
    return out


def intensity_aug(x8: np.ndarray, valid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """= kaggle/fasc_orient_v1 intensity_aug (gamma, gain, TGC-like profile, speckle, noise, blur, resample, JPEG)."""
    import cv2

    x = x8.astype(np.float32) * (1.0 / 255.0)
    if rng.random() < 0.8:
        x = np.power(x, rng.uniform(0.65, 1.5)).astype(np.float32)
    if rng.random() < 0.8:
        x = x * rng.uniform(0.8, 1.25) + rng.uniform(-0.08, 0.08)
    if rng.random() < 0.4:
        yy = np.linspace(-0.5, 0.5, SIZE, dtype=np.float32)
        g = 1.0 + rng.uniform(-0.5, 0.5) * yy + rng.uniform(-0.6, 0.6) * (yy * yy - 1.0 / 12)
        x = x * g[:, None]
    if rng.random() < 0.3:
        nz = rng.standard_normal((SIZE // 2, SIZE // 2), dtype=np.float32)
        nz = cv2.resize(nz, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        x = x * (1.0 + rng.uniform(0.05, 0.25) * nz)
    if rng.random() < 0.25:
        x = x + rng.uniform(0.01, 0.04) * rng.standard_normal((SIZE, SIZE), dtype=np.float32)
    if rng.random() < 0.25:
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.4, 1.4))
    if rng.random() < 0.15:
        f = rng.uniform(0.45, 0.8)
        small = cv2.resize(x, (int(SIZE * f), int(SIZE * f)), interpolation=cv2.INTER_AREA)
        x = cv2.resize(small, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    x8 = (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    if rng.random() < 0.3:
        ok, buf = cv2.imencode(".jpg", x8, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(30, 91))])
        if ok:
            x8 = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    x8[valid == 0] = 0
    return x8


# ----------------------------------------------------------------------------- exact-affine label geometry (numpy)
def wrap90(d: float) -> float:
    return (d + 90.0) % 180.0 - 90.0


def line_angle(v) -> float:
    """deg, image coords (x right, y down)."""
    return math.degrees(math.atan2(float(v[1]), float(v[0])))


def _dir(theta_deg: float) -> np.ndarray:
    t = math.radians(theta_deg)
    return np.array([math.cos(t), math.sin(t)])


def prims_vl(rec: dict) -> dict:
    """VL expert record (raw px) -> primitives. Per track: deep-aponeurosis line angle th_a, fascicle deep-end angle
    th_f = th_a +/- pa (sign from the traced deep end, magnitude = the expert's pa exactly), chain pieces d_i
    (superficial intersection -> traced points -> deep intersection) and r = expert FL px / sum |d_i|."""
    ppmm = float(rec["ppc"]) / 10.0
    th_a = line_angle(rec["deep"][2:])
    pairs, chains = [], []
    for tr in rec["tracks"]:
        P = np.asarray(tr["pts"], float)
        ps, pd = np.asarray(tr["ps"], float), np.asarray(tr["pd"], float)
        chain = np.vstack([ps[None], P, pd[None]])
        d = np.diff(chain, axis=0)
        d = d[np.hypot(d[:, 0], d[:, 1]) > 1e-9]
        n = len(P)
        k = max(2, int(math.ceil(0.12 * n)))
        dend = pd - P[max(0, n - k)]
        sgn = 1.0 if wrap90(line_angle(dend) - th_a) >= 0 else -1.0
        pairs.append((th_a + sgn * float(tr["pa"]), th_a))
        chains.append((d, float(tr["fl_mm"]) * ppmm / float(np.hypot(d[:, 0], d[:, 1]).sum())))
    return {"pairs": pairs, "chains": chains, "fl_px": None}


def prims_tt(rows: list[dict]) -> dict:
    """TimTrack observer angles (deg vs the image horizontal, + = rising to the right; image y points down)."""
    return {"pairs": [(-float(r["alpha"]), -float(r["gamma"])) for r in rows], "chains": [], "fl_px": None}


def prims_dyn(fl_px: float) -> dict:
    return {"pairs": [], "chains": [], "fl_px": float(fl_px)}


def _similarity_scale(L: np.ndarray) -> float:
    s2 = abs(float(np.linalg.det(L)))
    s = math.sqrt(s2)
    if np.abs(L.T @ L - s2 * np.eye(2)).max() > 1e-9 * max(1.0, s2):
        raise ValueError("scalar-FL primitive under a non-similarity transform")
    return s


def label_targets(pr: dict, L: np.ndarray) -> tuple[float, float]:
    """(PA deg, FL in transformed px) of the primitives under the 2x2 linear map L; NaN where absent."""
    pa = float("nan")
    if pr["pairs"]:
        vals = []
        for th_f, th_a in pr["pairs"]:
            vals.append(abs(wrap90(line_angle(L @ _dir(th_f)) - line_angle(L @ _dir(th_a)))))
        pa = float(np.mean(vals))
    fl = float("nan")
    if pr["chains"]:
        fl = float(np.mean([r * float(np.hypot(*(L @ d.T)).sum()) for d, r in pr["chains"]]))
    elif pr["fl_px"] is not None:
        fl = _similarity_scale(L) * pr["fl_px"]
    return pa, fl


def transform_prims(pr: dict, L: np.ndarray) -> dict:
    """Primitives re-expressed in the transformed frame (used by the identity / inverse checks)."""
    out = {"pairs": [(line_angle(L @ _dir(f)), line_angle(L @ _dir(a))) for f, a in pr["pairs"]],
           "chains": [((L @ d.T).T, r) for d, r in pr["chains"]], "fl_px": None}
    if pr["fl_px"] is not None:
        out["fl_px"] = _similarity_scale(L) * pr["fl_px"]
    return out


def label_aug_checks(items: list[dict], rng: np.random.Generator, n_mats: int = 24) -> dict:
    """identity: label_targets(I) == expert labels; inverse: T(L) then T(L^-1) == identity; similarity: PA invariant,
    FL scaled by s. All errors in deg (PA) and mm (FL)."""
    mats_any, mats_sim = [], []
    for _ in range(n_mats):
        th = rng.uniform(-math.pi, math.pi)
        R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        F = np.diag([rng.choice([-1.0, 1.0]), 1.0])
        s = math.exp(rng.uniform(-1.0, 1.0))
        ar = math.exp(rng.uniform(-0.4, 0.4))
        mats_sim.append((s * R @ F, s))
        mats_any.append(s * R @ np.diag([ar, 1 / ar]) @ F @ (np.eye(2) + rng.uniform(-0.2, 0.2, (2, 2))))
    e = {"identity_pa": 0.0, "identity_fl_mm": 0.0, "inverse_pa": 0.0, "inverse_fl_mm": 0.0, "similarity_pa": 0.0,
         "similarity_fl_mm": 0.0, "consistency_pa": 0.0, "consistency_fl_mm": 0.0}
    I2 = np.eye(2)
    for it in items:
        pr, ppmm = it["prims"], it["ppmm"]
        pa0, fl0 = label_targets(pr, I2)
        if it["has_pa"]:
            e["identity_pa"] = max(e["identity_pa"], abs(pa0 - it["pa_label"]))
        if it["has_fl"]:
            e["identity_fl_mm"] = max(e["identity_fl_mm"], abs(fl0 / ppmm - it["fl_label_mm"]))
        for L in mats_any if it["kind"] in ANISO_KINDS else []:
            back = transform_prims(transform_prims(pr, L), np.linalg.inv(L))
            pa1, fl1 = label_targets(back, I2)
            pa2, fl2 = label_targets(pr, L)
            pa3, fl3 = label_targets(transform_prims(pr, L), I2)
            if it["has_pa"]:
                e["inverse_pa"] = max(e["inverse_pa"], abs(pa1 - pa0))
                e["consistency_pa"] = max(e["consistency_pa"], abs(pa3 - pa2))
            if it["has_fl"]:
                e["inverse_fl_mm"] = max(e["inverse_fl_mm"], abs(fl1 - fl0) / ppmm)
                e["consistency_fl_mm"] = max(e["consistency_fl_mm"], abs(fl3 - fl2) / ppmm)
        for L, s in mats_sim:
            pa1, fl1 = label_targets(pr, L)
            if it["has_pa"]:
                e["similarity_pa"] = max(e["similarity_pa"], abs(pa1 - pa0))
            if it["has_fl"]:
                e["similarity_fl_mm"] = max(e["similarity_fl_mm"], abs(fl1 / s - fl0) / ppmm)
            back = transform_prims(transform_prims(pr, L), np.linalg.inv(L))
            pa2, fl2 = label_targets(back, I2)
            if it["has_pa"]:
                e["inverse_pa"] = max(e["inverse_pa"], abs(pa2 - pa0))
            if it["has_fl"]:
                e["inverse_fl_mm"] = max(e["inverse_fl_mm"], abs(fl2 - fl0) / ppmm)
    e["n_items"] = len(items)
    e["n_matrices"] = n_mats
    e["max_err"] = max(v for k, v in e.items() if k not in ("n_items", "n_matrices"))
    e["pass"] = bool(e["max_err"] < 1e-6)
    return e


# ----------------------------------------------------------------------------- samples
def make_sample(it: dict, rng: np.random.Generator | None) -> dict:
    """Training sample: canvas image + normalised targets (y) + masks (m). rng None -> no augmentation."""
    crop = it["crop"]
    h, w = crop.shape
    x0, y0, ww, hh = 0, 0, w, h
    if rng is not None:
        if rng.random() < AUG["p_crop_w"]:
            ww = max(32, int(round(rng.uniform(*AUG["crop_w"]) * w)))
            x0 = int(rng.integers(0, w - ww + 1))
        if rng.random() < AUG["p_crop_h"]:
            hh = max(32, int(round(rng.uniform(*AUG["crop_h"]) * h)))
            y0 = int(rng.integers(0, h - hh + 1))
    sub = crop[y0:y0 + hh, x0:x0 + ww]
    m = letterbox(ww, hh)
    if rng is not None:
        m = aug_matrix(rng, aniso=it["kind"] in ANISO_KINDS) @ m
    pa, flc = label_targets(it["prims"], m[:2, :2])
    img = warp_image(sub, m)
    if rng is not None:
        img = intensity_aug(img, poly_mask(m, ww, hh), rng)
    y = np.zeros(2, np.float32)
    msk = np.zeros(2, np.float32)
    if it["has_pa"] and np.isfinite(pa):
        y[0], msk[0] = (pa - PA_MU) / PA_SD, 1.0
    if it["has_fl"] and np.isfinite(flc) and flc > 0:
        y[1], msk[1] = (math.log(flc / SIZE) - LFL_MU) / LFL_SD, 1.0
    return {"img": img, "y": y, "m": msk}


def infer_input(it: dict) -> tuple[np.ndarray, float]:
    """Inference canvas (native geometry for Siemens / Lumify test rows) and the canvas px per native px."""
    h, w = it["crop"].shape
    m = letterbox(w, h, it["c"])
    return warp_image(it["crop"], m), float(m[1, 1])


# ----------------------------------------------------------------------------- data discovery / download
def find_dir(name: str) -> Path:
    for base in (Path("/kaggle/input/competitions") / COMP_SLUG, Path("/kaggle/input") / COMP_SLUG, Path("/kaggle/input")):
        if base.exists():
            hits = [p for p in base.rglob(name) if p.is_dir()]
            if hits:
                return sorted(hits, key=lambda p: len(str(p)))[0]
    raise FileNotFoundError(name)


def _fetch(url: str, dest: Path, timeout: int = 180) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (kaggle kernel)"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as fh:
        shutil.copyfileobj(r, fh, 1 << 20)


def download_zip(url: str, dest: Path) -> bool:
    for attempt in range(3):
        try:
            _fetch(url, dest)
            if zipfile.is_zipfile(dest):
                return True
            log(f"download {url}: not a zip ({dest.stat().st_size} bytes)")
        except Exception as exc:  # noqa: BLE001
            log(f"download {url} attempt {attempt + 1} failed: {exc!r}")
        time.sleep(10)
    return False


def gather_extra(name: str) -> list[tuple[str, Path, str]]:
    """[(image_id, path, source_rel)] of an OSF dataset (downloaded + unzipped into TMP; = fasc_orient_v1 rules)."""
    root = TMP / "ext" / name
    root.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    for url, fname in EXTRA_SOURCES[name]:
        zpath = TMP / "ext" / fname
        if not download_zip(url, zpath):
            raise RuntimeError(f"download failed: {url}")
        sub = root / Path(fname).stem
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(sub)
        zpath.unlink()
        files += [p for p in sub.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
                  and "__MACOSX" not in p.parts and not p.name.startswith("._")]
    if name == "static":
        files = [p for p in files if p.stem.startswith("im_") and p.stem.endswith("_arch")]
    elif name == "dynamic":
        files = [p for p in files if "frames" in p.parts or p.stem.startswith("image_")]
    files = sorted(files, key=lambda p: str(p.relative_to(root)))
    stem_counts: dict[str, int] = {}
    for p in files:
        stem_counts[p.stem] = stem_counts.get(p.stem, 0) + 1
    out, used = [], set()
    for p in files:
        key = p.stem
        if stem_counts[p.stem] > 1:
            key = "_".join([*p.parent.parts[-2:], p.stem])
        k2, n = key, 1
        while k2 in used:
            n += 1
            k2 = f"{key}_{n}"
        used.add(k2)
        out.append((k2, p, str(p.relative_to(root))))
    log(f"extra {name}: {len(out)} images")
    return out


def fetch_timtrack(rows: list[dict], base: str) -> dict[str, Path]:
    """Download the TimTrack paper images listed in the payload from the pinned commit; sha256-checked."""
    dest_root = TMP / "ext" / "timtrack"
    dest_root.mkdir(parents=True, exist_ok=True)
    want = {}
    for r in rows:
        want[r["file"]] = (r["file_sha256"], int(r["file_bytes"]))
    out = {}
    for i, (f, (sha, nb)) in enumerate(sorted(want.items())):
        dest = dest_root / f"tt_{i:04d}{Path(f).suffix.lower()}"
        url = base + urllib.parse.quote(f)
        ok = False
        for attempt in range(4):
            try:
                _fetch(url, dest, timeout=120)
                b = dest.read_bytes()
                if len(b) == nb and hashlib.sha256(b).hexdigest() == sha:
                    ok = True
                    break
                log(f"timtrack {f}: sha/bytes mismatch ({len(b)} B)")
            except Exception as exc:  # noqa: BLE001
                log(f"timtrack {f} attempt {attempt + 1} failed: {exc!r}")
            time.sleep(3 + 3 * attempt)
        if not ok:
            raise RuntimeError(f"TimTrack image not obtained with the pinned sha256: {f}")
        out[f] = dest
    log(f"timtrack: {len(out)} images, all sha256 OK")
    return out


def read_csv_dicts(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


# ----------------------------------------------------------------------------- data assembly (parent, before fork)
def build_data() -> dict:
    ids = json.loads((SRC / "ids.json").read_text())
    exp = ids["expected"]
    geom = {(r["dataset"], r["image_id"]): r for r in read_csv_dicts(SRC / "geom.csv")}
    counts, checks = {}, {}

    # ---- inference sets: static35 (OSF) and test309 (competition; inference only)
    static_files = gather_extra("static")
    static_items = []
    for key, path, src in static_files:
        g = geom[("static", key)]
        gray = read_gray(path)
        box, kind = viewport(gray, path.suffix.lower())
        x1, y1, x2, y2 = box
        static_items.append({"dataset": "static", "id": key, "crop": np.ascontiguousarray(gray[y1:y2, x1:x2]),
                             "c": float(g["c_label"]), "ppmm": float(g["px_per_cm"]) / 10.0, "layout": kind,
                             "sha": pixel_sha256(gray), "th": thumbs(gray)})
    counts["static"] = len(static_items)
    test_dir = find_dir("test_set_v2")
    test_files = sorted([p for p in test_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES], key=lambda p: p.name)
    test_items, lay_mismatch = [], 0
    lay_map = {"siemens_juniper": "siemens_juniper", "philips_lumify": "philips_lumify",
               "telemed_artus_1200": "telemed_artus_1200x800", "telemed_artus_1088": "telemed_artus_1088x644",
               "legacy_512": "legacy_512", "legacy_853": "legacy_853"}
    for p in test_files:
        g = geom[("test", p.name)]
        gray = read_gray(p)
        box, kind = viewport(gray, p.suffix.lower())
        lay_mismatch += int(lay_map.get(kind, kind) != g["group"])
        x1, y1, x2, y2 = box
        test_items.append({"dataset": "test", "id": p.name, "crop": np.ascontiguousarray(gray[y1:y2, x1:x2]),
                           "c": float(g["c_label"]), "ppmm": float(g["px_per_cm"]) / 10.0, "layout": kind,
                           "group": g["group"], "th": thumbs(gray)})
    counts["test"] = len(test_items)
    counts["test_layout_vs_group_mismatch"] = lay_mismatch
    counts["test_native_c_rows"] = int(sum(it["c"] != 1.0 for it in test_items))
    static_ids = {it["id"] for it in static_items}
    test_bank = np.stack([it["th"] for it in test_items])

    # ---- VL expert lines (OSF zips + payload)
    recs = json.loads((SRC / "vl_lines.json").read_text())
    counts["vl_payload"] = len(recs)
    vl_files = gather_extra("vl")
    by_rel = {}
    for key, path, src in vl_files:
        by_rel[src.split("/", 1)[1] if "/" in src else src] = path
    vl_items, n_match = [], 0
    for r in recs:
        path = by_rel.get(r["rel"])
        if path is None:
            continue
        gray = read_gray(path)
        if gray.shape != (r["H"], r["W"]):
            log(f"VL {r['key']}: shape {gray.shape} != payload {(r['H'], r['W'])}; skipped")
            continue
        n_match += 1
        if len(r["tracks"]) < 2:
            continue
        box, kind = viewport(gray, path.suffix.lower())
        x1, y1, x2, y2 = box
        vl_items.append({"kind": "vl", "id": r["key"], "subject": r["subject"], "fold": int(r["fold"]),
                         "crop": np.ascontiguousarray(gray[y1:y2, x1:x2]), "layout": kind,
                         "prims": prims_vl(r), "ppmm": float(r["ppc"]) / 10.0,
                         "pa_label": float(np.mean([t["pa"] for t in r["tracks"]])),
                         "fl_label_mm": float(np.mean([t["fl_mm"] for t in r["tracks"]])),
                         "has_pa": True, "has_fl": True, "sha_flips": [pixel_sha256(g) for g in flips(gray)],
                         "th": thumbs_flips(gray), "c": 1.0})
    counts["vl_matched"] = n_match
    counts["vl_used_ntracks_ge2"] = len(vl_items)

    # ---- TimTrack datasets 3 / 4 (PA only)
    tt_rows = read_csv_dicts(SRC / "tt_labels.csv")
    counts["tt_rows"] = len(tt_rows)
    tt_paths = fetch_timtrack(tt_rows, ids["sources"]["timtrack_raw_base"])
    by_file: dict[str, list[dict]] = {}
    for r in tt_rows:
        by_file.setdefault(r["file"], []).append(r)
    tt_items = []
    for f, rows in sorted(by_file.items()):
        gray = read_gray(tt_paths[f])
        if gray.shape != (int(rows[0]["H"]), int(rows[0]["W"])):
            raise RuntimeError(f"TimTrack {f}: shape {gray.shape}")
        box, kind = viewport(gray, Path(f).suffix.lower())
        x1, y1, x2, y2 = box
        tt_items.append({"kind": "tt", "id": "tt__" + rows[0]["key"], "subset": rows[0]["subset"],
                         "crop": np.ascontiguousarray(gray[y1:y2, x1:x2]), "layout": kind, "prims": prims_tt(rows),
                         "ppmm": 1.0, "pa_label": float(np.mean([float(r["alpha"]) - float(r["gamma"]) for r in rows])),
                         "fl_label_mm": float("nan"), "has_pa": True, "has_fl": False,
                         "sha_flips": [pixel_sha256(g) for g in flips(gray)], "th": thumbs_flips(gray), "c": 1.0})
    counts["tt_images"] = len(tt_items)
    counts["tt_by_subset"] = {s: int(sum(it["subset"] == s for it in tt_items)) for s in ("tt3", "tt4a", "tt4b")}

    # ---- dynamic calf raise (FL only)
    dyn_files = gather_extra("dynamic")
    counts["dyn_frames_zip"] = len(dyn_files)
    dlab = {r["image_id"]: r for r in read_csv_dicts(SRC / "dyn_labels.csv")}
    dyn_items = []
    for key, path, src in dyn_files:
        r = dlab.get(key)
        if r is None:
            continue
        gray = read_gray(path)
        box, kind = viewport(gray, path.suffix.lower())
        x1, y1, x2, y2 = box
        ppmm = float(r["px_per_cm"]) / 10.0
        dyn_items.append({"kind": "dyn", "id": "dyn__" + key, "crop": np.ascontiguousarray(gray[y1:y2, x1:x2]),
                          "layout": kind, "prims": prims_dyn(float(r["fl_mm"]) * ppmm), "ppmm": ppmm,
                          "pa_label": float("nan"), "fl_label_mm": float(r["fl_mm"]), "has_pa": False, "has_fl": True,
                          "sha_flips": [pixel_sha256(g) for g in flips(gray)], "th": thumbs_flips(gray), "c": 1.0})
    counts["dyn_fl_labels"] = len(dyn_items)

    # ---- guards: training pool vs static35 (id, flip-aware pixel sha256, flip-aware NCC) and vs test309 (NCC)
    pool = vl_items + tt_items + dyn_items
    keep, guard = guard_pool(pool, static_items, test_bank)
    tr_ids = {it["id"] for it in keep}
    tr_sha = {h for it in keep for h in it["sha_flips"]}
    copies25 = set(ids["copies25"])
    guard["copies25_in_training_by_id"] = len(tr_ids & copies25)
    guard["copies25_in_training_by_sha_flip_aware"] = len(tr_sha & {it["sha"] for it in static_items if it["id"] in copies25})
    guard["copies25_max_ncc_kept"] = float(max([v for k, v in guard["max_ncc_kept_per_static"].items() if k in copies25]
                                               or [-1.0]))
    guard["expected_exact_copies_dropped"] = sorted(set(ids["expected_drop_exact"]) & set(guard["dropped_ids"]))
    guard["expected_drop_all_local"] = ids["expected_drop_all_local"]
    guard["dropped_vs_local_symdiff"] = sorted(set(ids["expected_drop_all_local"]) ^ set(guard["dropped_ids"]))
    guard["competition_images_in_training"] = 0  # pools are VL / TimTrack / dynamic only (no competition dir is read for training)

    vl_keep = [it for it in keep if it["kind"] == "vl"]
    vl_train = [it for it in vl_keep if it["fold"] != VAL_FOLD]
    vl_val = [it for it in vl_keep if it["fold"] == VAL_FOLD]
    train = vl_train + [it for it in keep if it["kind"] in ("tt", "dyn")]
    counts.update(vl_train=len(vl_train), vl_val=len(vl_val), train_pool=len(train),
                  train_by_kind={k: int(sum(it["kind"] == k for it in train)) for k in ("vl", "tt", "dyn")})
    subj_overlap = len({it["subject"] for it in vl_train} & {it["subject"] for it in vl_val})
    ls = {}
    for name, grp in (("vl_train", vl_train), ("vl_val", vl_val), ("tt", [it for it in train if it["kind"] == "tt"]),
                      ("dyn", [it for it in train if it["kind"] == "dyn"])):
        pa = np.array([it["pa_label"] for it in grp if it["has_pa"]])
        fl = np.array([it["fl_label_mm"] for it in grp if it["has_fl"]])
        ls[name] = {"n": len(grp), "pa_mean": float(pa.mean()) if len(pa) else None,
                    "pa_sd": float(pa.std()) if len(pa) else None, "fl_mean_mm": float(fl.mean()) if len(fl) else None,
                    "fl_sd_mm": float(fl.std()) if len(fl) else None}
    counts["label_stats_identity"] = ls

    # ---- checks (all must hold; asserted before training)
    lab = label_aug_checks(train + vl_val, np.random.default_rng(20261106))
    checks.update({
        "count_static_35": counts["static"] == exp["static"] == 35,
        "count_test_309": counts["test"] == exp["test"] == 309,
        "test_ids_exact": sorted(it["id"] for it in test_items) == ids["test309"],
        "static_ids_exact": sorted(it["id"] for it in static_items) == ids["static35"],
        "count_vl_payload": counts["vl_payload"] == exp["vl_payload"],
        "count_vl_matched": counts["vl_matched"] == exp["vl_payload"],
        "count_vl_used": counts["vl_used_ntracks_ge2"] == exp["vl_used_ntracks_ge2"],
        "count_tt_images": counts["tt_images"] == exp["tt_images"],
        "count_tt_rows": counts["tt_rows"] == exp["tt_rows"],
        "count_tt_by_subset": counts["tt_by_subset"] == exp["tt_by_subset"],
        "count_dyn_frames": counts["dyn_frames_zip"] == exp["dyn_frames_zip"],
        "count_dyn_fl": counts["dyn_fl_labels"] == exp["dyn_fl_labels"],
        "static35_not_in_training_by_id": guard["static_id_overlap_after"] == 0,
        "static35_not_in_training_by_sha256_flip_aware": guard["static_sha_overlap_after_flip_aware"] == 0,
        "static35_not_in_training_by_ncc": guard["max_ncc_static_kept"] <= DUP_NCC,
        "copies25_excluded": (guard["copies25_in_training_by_id"] == 0
                              and guard["copies25_in_training_by_sha_flip_aware"] == 0
                              and guard["copies25_max_ncc_kept"] <= DUP_NCC
                              and copies25 <= static_ids and len(copies25) == 25),
        "expected_exact_copies_dropped": guard["expected_exact_copies_dropped"] == sorted(ids["expected_drop_exact"]),
        "test_not_in_training_by_ncc": guard["max_ncc_test_kept"] <= DUP_NCC,
        "vl_val_subjects_disjoint": subj_overlap == 0,
        "label_aug_identity_inverse_lt_1e-6": lab["pass"],
    })
    log("counts: " + json.dumps(counts))
    log("guard: " + json.dumps({k: v for k, v in guard.items() if k != "dropped"}) + f" dropped={guard['dropped_ids']}")
    log("label_aug: " + json.dumps(lab))
    log("checks: " + json.dumps(checks))
    if SMOKE:  # tiny subset (training only; inference stays on every static / test image)
        rng = np.random.default_rng(1)
        sub = []
        for k in ("vl", "tt", "dyn"):
            pool_k = [it for it in train if it["kind"] == k]
            sub += [pool_k[i] for i in sorted(rng.choice(len(pool_k), SMOKE_N[k], replace=False))]
        train = sub
        vl_val = vl_val[:SMOKE_N["vlval"]]
    for it in train + vl_val + static_items + test_items:
        it.pop("th", None)
    return {"train": train, "vl_val": vl_val, "static": static_items, "test": test_items, "counts": counts,
            "guard": guard, "label_aug": lab, "checks": checks, "ids": ids}


# ----------------------------------------------------------------------------- torch parts
def build_net(torch, timm, pretrained: bool):
    nn = torch.nn

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = timm.create_model(ENCODER, pretrained=pretrained, num_classes=0, drop_path_rate=DROP_PATH)
            nf = self.backbone.num_features
            self.head_pa = nn.Sequential(nn.Linear(nf, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 1))
            self.head_fl = nn.Sequential(nn.Linear(nf, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 1))

        def forward(self, x):
            f = self.backbone(x)
            return torch.cat([self.head_pa(f), self.head_fl(f)], 1)

    return Net()


def to_input(img_u8, torch):
    x = img_u8.float().unsqueeze(1) * (1.0 / 255.0)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return ((x.expand(-1, 3, -1, -1) - mean) / std).contiguous(memory_format=torch.channels_last)


class S5Set:
    """Epoch list of (item index, draw seed); every draw is a fresh exact-affine augmentation."""

    def __init__(self, items: list[dict], draws: list[tuple[int, int]]) -> None:
        self.items, self.draws = items, draws

    def __len__(self) -> int:
        return len(self.draws)

    def __getitem__(self, k: int):
        import torch

        i, seed = self.draws[k]
        s = make_sample(self.items[i], np.random.default_rng(seed))
        return torch.from_numpy(np.ascontiguousarray(s["img"])), torch.from_numpy(s["y"]), torch.from_numpy(s["m"])


def _worker_init(_):
    import cv2

    cv2.setNumThreads(1)
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:  # noqa: BLE001
        pass


def epoch_draws(items: list[dict], seed: int, epoch: int) -> list[tuple[int, int]]:
    rng = np.random.default_rng([seed, epoch])
    idx = {k: [i for i, it in enumerate(items) if it["kind"] == k] for k in ("vl", "tt", "dyn")}
    lst = idx["vl"] * MIX["vl"] + idx["tt"] * MIX["tt"]
    if idx["dyn"]:
        lst += [idx["dyn"][j] for j in rng.choice(len(idx["dyn"]), min(MIX["dyn"], len(idx["dyn"])), replace=False)]
    rng.shuffle(lst)
    return [(i, int(rng.integers(0, 2**63 - 1))) for i in lst]


def predict(model, items: list[dict], torch, dev, tta: bool = True) -> list[dict]:
    """-> per item: pa (deg), fl (mm). hflip TTA averaged in the normalised output space."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(items), INFER_BATCH):
            chunk = items[i:i + INFER_BATCH]
            imgs, scales = zip(*[infer_input(it) for it in chunk])
            x = to_input(torch.from_numpy(np.stack(imgs)).to(dev), torch)
            if tta:
                x = torch.cat([x, torch.flip(x, dims=[3])], 0)
            with amp_ctx(torch, dev):
                o = model(x).float()
            if tta:
                o = 0.5 * (o[:len(chunk)] + o[len(chunk):])
            o = o.cpu().numpy().astype(np.float64)
            for it, s, (a, b) in zip(chunk, scales, o):
                pa = PA_MU + PA_SD * a
                flc = SIZE * math.exp(LFL_MU + LFL_SD * b)
                out.append({"dataset": it.get("dataset", "vl_val"), "image_id": it["id"], "pa": pa,
                            "fl": flc / (s * it["ppmm"])})
    return out


def amp_ctx(torch, dev):
    import contextlib

    if dev.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _corr(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def val_metrics(rows: list[dict], items: list[dict]) -> dict:
    lab = {it["id"]: it for it in items}
    pa = np.array([r["pa"] for r in rows])
    fl = np.array([r["fl"] for r in rows])
    tp = np.array([lab[r["image_id"]]["pa_label"] for r in rows])
    tf = np.array([lab[r["image_id"]]["fl_label_mm"] for r in rows])
    return {"n": len(rows), "pa_mae": float(np.abs(pa - tp).mean()), "fl_mae": float(np.abs(fl - tf).mean()),
            "pa_bias": float((pa - tp).mean()), "fl_bias": float((fl - tf).mean()), "pa_corr": _corr(pa, tp),
            "fl_corr": _corr(fl, tf), "pa_sd_ratio": float(pa.std() / max(tp.std(), 1e-9)),
            "fl_sd_ratio": float(fl.std() / max(tf.std(), 1e-9))}


def train_seed(si: int, seed: int, torch, timm, dev) -> dict:
    from torch.utils.data import DataLoader

    t_start = time.time()
    torch.manual_seed(seed)
    train, vl_val = DATA["train"], DATA["vl_val"]
    model = build_net(torch, timm, pretrained=True).to(dev).to(memory_format=torch.channels_last)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    groups = {(b, d): [] for b in (True, False) for d in (True, False)}
    for name, p in model.named_parameters():
        groups[(name.startswith("backbone."), not (p.ndim <= 1 or name.endswith(".bias")))].append(p)
    opt = torch.optim.AdamW([{"params": groups[(b, d)], "lr": LR_BACKBONE if b else LR_HEAD,
                              "weight_decay": WEIGHT_DECAY if d else 0.0} for (b, d) in groups if groups[(b, d)]])
    n_workers = 0 if dev.type == "cpu" else 2
    steps_per_epoch = len(epoch_draws(train, seed, 0)) // BATCH
    if MAX_STEPS is not None:
        steps_per_epoch = min(steps_per_epoch, MAX_STEPS)
    sched_state = {"total": MAX_EPOCHS * steps_per_epoch, "warm": max(1, int(WARMUP_EPOCHS * steps_per_epoch))}

    def lr_lambda(step: int) -> float:
        warm = sched_state["warm"] if not SMOKE else 1
        if step < warm:
            return 0.05 + 0.95 * step / warm
        total = max(sched_state["total"], warm + 1)
        prog = min(1.0, (step - warm) / (total - warm))
        return LR_FLOOR + (1 - LR_FLOOR) * 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    hist, step_global, n_epochs, truncated = [], 0, MAX_EPOCHS, False
    log(f"seed {seed}: train {len(train)} ({ {k: sum(it['kind'] == k for it in train) for k in ('vl', 'tt', 'dyn')} }) "
        f"val {len(vl_val)} steps/epoch {steps_per_epoch} workers {n_workers}")
    epoch = 0
    while epoch < n_epochs:
        epoch += 1
        t_ep = time.time()
        model.train()
        ds = S5Set(train, epoch_draws(train, seed, epoch))
        kw = dict(batch_size=BATCH, shuffle=False, num_workers=n_workers, drop_last=True,
                  pin_memory=dev.type == "cuda")
        if n_workers:
            kw.update(worker_init_fn=_worker_init, prefetch_factor=4)
        sums, nb = np.zeros(3), 0
        for step, (img, y, m) in enumerate(DataLoader(ds, **kw), 1):
            x = to_input(img.to(dev, non_blocking=True), torch)
            y, m = y.to(dev).float(), m.to(dev).float()
            with amp_ctx(torch, dev):
                o = model(x)
            o = o.float()
            per = torch.nn.functional.smooth_l1_loss(o, y, reduction="none", beta=HUBER_BETA) * m
            l_pa = per[:, 0].sum() / m[:, 0].sum().clamp_min(1.0)
            l_fl = per[:, 1].sum() / m[:, 1].sum().clamp_min(1.0)
            loss = l_pa + l_fl
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step_global += 1
            d = min(EMA_DECAY, (1.0 + step_global) / (10.0 + step_global))
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(d).add_(pm.detach(), alpha=1.0 - d)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
            sums += [float(loss.detach()), float(l_pa.detach()), float(l_fl.detach())]
            nb += 1
            if step % 20 == 0 or SMOKE:
                log(f"seed {seed} ep {epoch} step {step}/{steps_per_epoch} loss={sums[0] / nb:.4f} "
                    f"pa={sums[1] / nb:.4f} fl={sums[2] / nb:.4f}")
            if MAX_STEPS is not None and step >= MAX_STEPS:
                break
        vm = val_metrics(predict(ema, vl_val, torch, dev, tta=False), vl_val) if vl_val else {}
        sec = time.time() - t_ep
        row = {"epoch": epoch, "steps": nb, "loss": sums[0] / max(1, nb), "loss_pa": sums[1] / max(1, nb),
               "loss_fl": sums[2] / max(1, nb), "lr": opt.param_groups[0]["lr"], "sec": sec,
               **{f"val_{k}": v for k, v in vm.items()}}
        hist.append(row)
        log(f"EPOCH seed={seed} ep={epoch}/{n_epochs} loss={row['loss']:.4f} | VLval pa_mae={vm.get('pa_mae', float('nan')):.2f} "
            f"fl_mae={vm.get('fl_mae', float('nan')):.2f} pa_r={vm.get('pa_corr', float('nan')):.2f} "
            f"fl_r={vm.get('fl_corr', float('nan')):.2f} | {sec:.0f}s")
        if epoch == 1 and not SMOKE:
            left = t_start + DATA.get("train_budget_s", TRAIN_BUDGET_S) - time.time()
            n_epochs = int(max(MIN_EPOCHS, min(MAX_EPOCHS, 1 + left // (sec * 1.05))))
            sched_state["total"] = n_epochs * steps_per_epoch
            log(f"seed {seed}: epoch time {sec:.0f}s -> n_epochs={n_epochs}")
        if time.time() > T0 + HARD_TRAIN_END_S and epoch < n_epochs:
            truncated = True
            log(f"seed {seed}: HARD training end reached after epoch {epoch}; stopping")
            break
    t_inf = time.time()
    rows = predict(ema, vl_val, torch, dev) + predict(ema, DATA["static"], torch, dev) + predict(ema, DATA["test"], torch, dev)
    res = {"seed": seed, "n_epochs_planned": n_epochs, "epochs_done": epoch, "truncated": truncated,
           "steps_total": step_global, "train_min": (t_inf - t_start) / 60, "infer_min": (time.time() - t_inf) / 60,
           "history": hist}
    with open(TMP / f"pred_seed{si}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "image_id", "pa", "fl"])
        w.writeheader()
        w.writerows(rows)
    (TMP / f"train_seed{si}.json").write_text(json.dumps(res, default=float))
    log(f"seed {seed}: done ({res['train_min']:.1f} min train, {res['infer_min']:.1f} min inference)")
    return res


def gpu_worker(gpu: int, si: int, seed: int) -> None:
    global _LOG_PREFIX, _LOG_FILE
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    _LOG_PREFIX = f"[gpu{gpu}]"
    _LOG_FILE = LOG_DIR / f"seed{si}.log"
    code = 0
    try:
        import cv2
        import timm
        import torch

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True
        torch.set_num_threads(1)
        cv2.setNumThreads(1)
        log(f"worker seed {seed} on {torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'}")
        train_seed(si, seed, torch, timm, dev)
    except BaseException:  # noqa: BLE001
        code = 1
        log(f"seed {seed} FAILED:\n{traceback.format_exc()}")
    os._exit(code)


# ----------------------------------------------------------------------------- outputs
CREDITS = """UMUD v30 S5 direct regressor -- credits and licences of external material (private kernel output)
Competition data: UMUD Challenge (Kaggle competition data, CC BY-NC-SA 4.0, competition rules apply): test images
  are read for INFERENCE ONLY; no competition image or label is trained on, written or redistributed.
OSF node xbawc (https://osf.io/xbawc/, CC BY 4.0): UMUD benchmark static (hb5xj; inference / gate only, never trained
  on), GM dynamic calf raise (vqb7h; FL labels only, embedded as dyn_labels.csv), 'Example Annotated Images' VL
  (zcuwk + 6ga94; expert lines parsed on the laptop with scripts/vl_labels.py rules, embedded as vl_lines.json).
TimTrack paper datasets 3 and 4 (repo https://github.com/timvanderzee/ultrasound-automated-algorithm @ d3ba5e2b, CC0-1.0
  repository; datasets 3-4 CC BY 4.0): dataset 3 J. Drazan, T. Hullfish, J. Baxter (https://zenodo.org/record/2598553,
  https://peerj.com/articles/7120/); dataset 4 O. Seynnes, N. Cronin (https://data.mendeley.com/datasets/dpmf9bz8pt/2,
  https://doi.org/10.1371/journal.pone.0229034). Observer angles from the repo's manual-estimation workbooks.
Backbone: convnext_tiny.fb_in22k_ft_in1k (ConvNeXt, Z. Liu et al. 2022, facebookresearch/ConvNeXt, MIT) via timm
  (Apache-2.0); weights downloaded at run time from the Hugging Face hub, not redistributed.
Viewport layout rules adapted from AmbrosM, 'UMUD Quick and Dirty' (Kaggle notebook), Apache-2.0.
Code: GPL-3.0-only (C) 2026 Gaddam Shivateja.
"""


def write_outputs(seed_res: list[dict], status: str, err: str | None) -> dict:
    import pandas as pd

    parts = []
    for si in range(len(SEEDS)):
        p = TMP / f"pred_seed{si}.csv"
        if p.exists():
            d = pd.read_csv(p)
            d["seed_index"] = si
            parts.append(d)
    metrics = {"status": status, "error": err, "mode": MODE, "elapsed_min": (time.time() - T0) / 60,
               "config": config_dict(), "counts": DATA.get("counts"), "guard": DATA.get("guard"),
               "label_aug": DATA.get("label_aug"), "checks": dict(DATA.get("checks", {})), "seeds": seed_res,
               "versions": DATA.get("versions"), "payload_sha256": DATA.get("payload_sha256")}
    if parts:
        P = pd.concat(parts, ignore_index=True)
        P.to_csv(WORK / "pred_by_seed.csv", index=False)
        M = P.groupby(["dataset", "image_id"], sort=False)[["pa", "fl"]].mean().reset_index()
        metrics["n_seeds_used"] = int(P.seed_index.nunique())
        for name, fn in (("static", "static_pred.csv"), ("test", "test_pred.csv"), ("vl_val", "vl_val_pred.csv")):
            S = M[M.dataset == name][["image_id", "pa", "fl"]].sort_values("image_id")
            S.to_csv(WORK / fn, index=False, float_format="%.6f")
        st = M[M.dataset == "static"]
        te = M[M.dataset == "test"]
        ids = DATA["ids"]
        ok_vals = bool(np.isfinite(M[["pa", "fl"]].to_numpy()).all())
        metrics["checks"].update({
            "outputs_finite": ok_vals,
            "static_pred_ids_exact": sorted(st.image_id) == ids["static35"],
            "test_pred_ids_exact": sorted(te.image_id) == ids["test309"] and len(te) == 309,
        })
        if SMOKE:
            metrics["checks"]["smoke_3_steps"] = all(r.get("steps_total") == 3 for r in seed_res) and len(seed_res) == 1
        vv = M[M.dataset == "vl_val"]
        if len(vv):
            metrics["vl_val_seedmean_tta"] = val_metrics(vv.to_dict("records"), DATA["vl_val"])
        grp = {it["id"]: it["group"] for it in DATA["test"]}
        te = te.assign(group=te.image_id.map(grp))
        metrics["pred_summary"] = {
            "static": {"pa_mean": float(st.pa.mean()), "pa_sd": float(st.pa.std()), "fl_mean": float(st.fl.mean()),
                       "fl_sd": float(st.fl.std())},
            "test_by_group": {g: {"n": int(len(G)), "pa_mean": float(G.pa.mean()), "pa_sd": float(G.pa.std()),
                                  "fl_mean": float(G.fl.mean()), "fl_sd": float(G.fl.std())}
                              for g, G in te.groupby("group")}}
    else:
        metrics["n_seeds_used"] = 0
        metrics["checks"]["outputs_finite"] = False
    metrics["all_checks_pass"] = bool(metrics["checks"]) and all(bool(v) for v in metrics["checks"].values())
    (WORK / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    (WORK / "CREDITS.txt").write_text(CREDITS)
    return metrics


# ----------------------------------------------------------------------------- main
def main() -> None:
    global _LOG_FILE, DATA
    for d in (LOG_DIR, TMP / "ext"):
        d.mkdir(parents=True, exist_ok=True)
    _LOG_FILE = LOG_DIR / "main.log"
    log(f"MODE={MODE} encoder={ENCODER} size={SIZE} seeds={SEEDS} python={sys.version.split()[0]} cpu={os.cpu_count()}")
    log("nvidia-smi: " + sh("nvidia-smi -L").strip().replace("\n", " | "))
    seed_res, status, err = [], "failed", None
    try:
        import cv2
        import timm
        import torch

        payload_sha = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SRC.iterdir()) if p.is_file()}
        versions = {"torch": torch.__version__, "timm": timm.__version__, "cv2": cv2.__version__,
                    "numpy": np.__version__, "python": sys.version.split()[0]}
        log(f"versions {versions}")
        DATA = build_data()
        DATA["versions"], DATA["payload_sha256"] = versions, payload_sha
        bad = [k for k, v in DATA["checks"].items() if not v]
        if bad:
            raise RuntimeError(f"pre-training checks FAILED: {bad}")
        t = time.time()
        torch.set_num_threads(1)  # = v16: keep the parent's OpenMP pool minimal before forking the seed workers
        _m = build_net(torch, timm, pretrained=True)  # caches the pretrained weights before any fork
        log(f"pretrained {ENCODER} cached ({sum(p.numel() for p in _m.parameters()) / 1e6:.1f}M params, {time.time() - t:.0f}s)")
        del _m
        n_gpu = sh("nvidia-smi -L").count("GPU ")
        if SMOKE:
            torch.set_num_threads(4)
            for si, seed in enumerate(SEEDS):
                seed_res.append(train_seed(si, seed, torch, timm, torch.device("cpu")))
        elif n_gpu == 0:
            raise RuntimeError("UMUD_S5_MODE=gpu but no GPU is attached (push with the T4 x2 accelerator)")
        else:
            # one seed per GPU in parallel; with fewer GPUs than seeds the seeds run one after the other and each
            # gets TRAIN_BUDGET_S / n_seeds (declared fallback; Kaggle T4 x2 is the planned machine)
            parallel = n_gpu >= len(SEEDS)
            DATA["train_budget_s"] = TRAIN_BUDGET_S if parallel else TRAIN_BUDGET_S / len(SEEDS)
            log(f"{n_gpu} GPU(s): seeds {'in parallel' if parallel else 'sequentially'}, "
                f"budget {DATA['train_budget_s'] / 3600:.2f} h per seed")
            ctx = mp.get_context("fork")
            waves = [list(enumerate(SEEDS))] if parallel else [[(si, s)] for si, s in enumerate(SEEDS)]
            for wave in waves:
                procs = []
                for si, seed in wave:
                    p = ctx.Process(target=gpu_worker, args=(si % n_gpu, si, seed), daemon=False)
                    p.start()
                    procs.append((si, p))
                    time.sleep(5)
                while any(p.is_alive() for _, p in procs):
                    if time.time() > T0 + HARD_WALL_S:
                        log("HARD wall-clock cap reached: terminating unfinished seed workers")
                        for _, p in procs:
                            if p.is_alive():
                                os.kill(p.pid, signal.SIGKILL)
                        break
                    time.sleep(20)
                for si, p in procs:
                    p.join(timeout=30)
                    log(f"seed worker {si} exit code {p.exitcode}")
                    f = TMP / f"train_seed{si}.json"
                    if f.exists():
                        seed_res.append(json.loads(f.read_text()))
                if time.time() > T0 + HARD_WALL_S:
                    break
        status = "complete" if len(seed_res) == len(SEEDS) else ("partial" if seed_res else "failed")
    except BaseException as exc:  # noqa: BLE001
        err = f"{exc!r}\n{traceback.format_exc()}"
        log(f"FAILED: {err}")
    metrics = write_outputs(seed_res, status, err)
    log(f"checks: {json.dumps(metrics['checks'])} all_pass={metrics['all_checks_pass']} n_seeds={metrics['n_seeds_used']}")
    shutil.rmtree(TMP, ignore_errors=True)
    log(f"DONE status={status} elapsed={(time.time() - T0) / 60:.1f} min")
    if status == "failed" or not metrics["all_checks_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
