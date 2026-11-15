"""UMUD LF-26 LFNet-v26 (pre-registered: artifacts/candidates_v26/PREREG_v26_longfasc.json + PREREG_v26_addendum_K.json).
Kaggle kernel template: a copy of kaggle/fasc_orient_v3/train.py (sha256 a7778983..., BACKBONE "convnext" = the
fasc_orient_v1 network) with these changes (everything else, incl. the v1 folds, evaluate(), the hflip TTA, the
FALLMUD fetch / parse / overlap guard and the MASKBANK encoding, is the template's code):
  1. fold k starts from the v1 fold-k checkpoint (attached kernel output shivateja005/umud-fasc-orient-v1); BatchNorm
     layers with non-finite buffers (v1 folds 2 and 4, fp16 overflow) are re-estimated on N_CAL training crops of
     the fold before fine-tuning;
  2. a 4th output channel "path" (sigmoid), initialised as a copy of the seg filter;
  3. FALLMUD Cunningham (Cronin dropped by the overlap guard): path-ridge targets (Gaussian ridge sigma PATH_SIGMA
     canvas px around the full-path skeleton, focal BCE gamma PATH_GAMMA + soft clDice, every non-path pixel a
     negative at weight PATH_NEG_W) and dense orientation targets by doubled-angle inverse-distance weighting of the
     path pieces within FM_IDW_R canvas px; the seg channel is masked on FALLMUD (as v3);
  4. competition fragments: the v1 seg / orientation targets + path positives (the seg lines) at weight
     COMP_PATH_POS_W with a sparsity prior COMP_PATH_SPARSITY * mean(path prob) (fragments are incomplete paths);
  5. VL is NOT used in training (held out for K2 / K0d);
  6. field-of-view crop augmentation for FALLMUD samples: FM_FOV_RANGE of the viewport width with p FM_FOV_P;
  7. LR 0.25 x v1 (LR_ENC 5e-5 / LR_DEC 1.5e-4), cosine, WARMUP_EPOCHS 0.5; MIN_EPOCHS..MAX_EPOCHS budget-fitted,
     training capped at TRAIN_CAP_S; 5 folds on 2 x T4; fp32 checkpoints;
  8. FALLMUD subject groups rebuilt from index-block breaks (else 8 clusters); FM_HOLDOUT_FRAC of the frames held out
     of training in every fold for K0c;
  9. after inference (5 folds x hflip on test / static / dynamic / VL): the LF readout (embedded lf_readout.py +
     geometry_v2.py, payload of production geometry exported on the laptop) in the kernel: L1prod (all payload
     images), L1expvl (VL, expert aponeuroses), L0port (the v1 maskbank, port check); K0 metrics; VL field samples at
     the payload K0d points. No FL / PA / MT label enters the kernel.
 10. revision r1 (PREREG_v26_addendum_K_r1.json, before any push): fail fast before training on a v1 maskbank that
     differs from the laptop copy or on payload images missing from the inference datasets; per-draw augmentation
     seeds from the loader worker's torch RNG (v1's seeding quirk fixed); Cunningham-only assert after the overlap
     guard; port runs L0port + L0portexp (VL expert geometry) on every payload image in every mode, compared with the
     embedded laptop L0 values at the r1 float-noise tolerance; K0c also reported at path thresholds 0.3 / 0.4 and
     input / inference coverage + epoch selection in K0.json (information only).
MODE (env UMUD_MODE): "full" | "smoke" (GPU, tiny) | "cpusmoke" (CPU-safe dry run: full data prep, FALLMUD fetch +
sha256, overlap guard, path-target viz, BN repair check, 3 training steps, L1 readouts on 20 images per set, port
readouts + port check on every payload image).
Outputs (/kaggle/working): readout/, metrics/ (history_fold*.csv, fold*.json, cv_summary.json, K0.json,
port_check.json), logs/, viz/ (<= MAX_VIZ jpg), lfnet_v26_config.json, MANIFEST.json, CREDITS.txt; maskbank/ and
checkpoints/ stay on Kaggle (download pattern in RUN.md).
GPL-3.0-only.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")  # never initialise CUDA in the parent before fork
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import csv
import hashlib
import html as htmlmod
import http.cookiejar
import json
import math
import multiprocessing as mp
import re
import shutil
import ssl
import subprocess
import sys
import time
import traceback
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# OpenCV's worker pool must never start in the parent: a forked child that touches it deadlocks (fasc_orient_v2).
cv2.setNumThreads(1)

# ----------------------------------------------------------------------------- config
MODE = os.environ.get("UMUD_MODE", "full")  # "full" | "smoke" | "cpusmoke" (CPU-safe dry run)
BACKBONE = "convnext"  # the fasc_orient_v1 network (smp U-Net, ConvNeXt-S); the v3 DINOv2 arm is removed
USE_FALLMUD = True
MODEL_NAME = "lfnet_v26"
V1_REF_EVAL = True  # evaluate the attached v1 fold checkpoints on the same validation caches (as-is + BN-repaired)
V1_INIT = True  # fold k starts from the v1 fold-k checkpoint (BN-repaired where non-finite)
SMOKE = MODE != "full"
CPU_SMOKE = MODE == "cpusmoke"
T0 = time.time()

SEED = 20261021  # same as v1
SIZE = 512
N_FOLDS = 5
TRAIN_FOLDS = [0] if CPU_SMOKE else ([0, 1] if SMOKE else [0, 1, 2, 3, 4])
SMOKE_TRAIN, SMOKE_VAL, SMOKE_REPEAT, SMOKE_FM = (6, 4, 1, 4) if CPU_SMOKE else (30, 10, 4, 12)
MAX_EPOCHS = 1 if CPU_SMOKE else (2 if SMOKE else int(os.environ.get("UMUD_MAX_EPOCHS", "12")))
MIN_EPOCHS = 1 if SMOKE else 6
MAX_STEPS = 3 if CPU_SMOKE else None  # per epoch (cpusmoke only)
TRAIN_CAP_S = 3.5 * 3600  # training proper (first epoch start -> last epoch end on a GPU) is capped at 3.5 h
TRAIN_BUDGET_S = 3.9 * 3600  # all training incl. preprocessing must end by T0 + this
HARD_TRAIN_END_S = 4.2 * 3600  # absolute stop for any epoch loop
BATCH = 2 if CPU_SMOKE else 8
LR_ENC, LR_DEC, WEIGHT_DECAY = 5e-5, 1.5e-4, 1e-2  # 0.25 x v1 (2e-4 / 6e-4)
WARMUP_EPOCHS, LR_FLOOR = 0.5, 0.02
POS_WEIGHT, W_BCE, W_DICE, W_SKEL, W_ORI = 3.0, 1.0, 1.0, 0.5, 1.0  # v1
SEG_WIDTH, ORI_RADIUS = 3, 8.0  # v1 competition targets
# LF-26 path channel
CHANNELS = ("seg", "cos2", "sin2", "path")
N_CLASSES = 4
PATH_SIGMA = 1.5  # canvas px: Gaussian ridge around the FALLMUD path skeleton
PATH_GAMMA = 2.0  # focal BCE
PATH_NEG_W = 0.3  # weight of every non-path pixel (target < PATH_POS_T) in the FALLMUD path loss
PATH_POS_T = 0.05
W_PATH, W_CLDICE = 1.0, 0.5  # FALLMUD path focal BCE / soft clDice
CLDICE_ITERS = 5
COMP_PATH_POS_W = 0.2  # competition fragments as path positives (positive pixels only)
COMP_PATH_SPARSITY = 0.02  # x mean path probability over valid pixels of competition samples
FM_IDW_R = 16.0  # canvas px: FALLMUD dense orientation = doubled-angle IDW (1/max(d,1)) of the pieces within this
FM_IDW_MIN_R = 0.3  # min resultant length of the IDW doubled-angle mean (else unsupervised)
FM_FOV_P, FM_FOV_RANGE = 0.3, (0.55, 1.0)  # FALLMUD field-of-view crop augmentation
FM_HOLDOUT_FRAC = 0.2  # FALLMUD frames held out of training (whole subject groups) for K0c
FM_BREAK_CORR = 0.80  # consecutive-frame thumbnail correlation below this = a new index block (subject / trial)
FM_N_CLUSTERS = 8
K0B_MAX, K0C_RECALL, K0C_PREC, K0C_TOL_PX = 1.60, 0.50, 0.30, 3.0
K0C_INFO_THR = (0.3, 0.4)  # LF-26 r1: K0c recall / precision also reported at these path thresholds (information only)
MAX_VIZ = 40
DUP_CORR, THUMB = 0.95, 48
VERIFY_TOL_ANG, VERIFY_TOL_DICE = 0.05, 0.01  # reloaded checkpoint must reproduce the saved best metrics
N_CAL = 8 if CPU_SMOKE else 160  # training crops for the BN re-estimate of non-finite v1 BatchNorm layers
ENCODER = "tu-convnext_small.fb_in22k_ft_in1k"
SMP_VERSION = "0.5.0"
# FALLMUD
FALLMUD_URL = "https://kalisteo.cea.fr/index.php/download/fallmud_dataset/?wpdmdl=3570"
FALLMUD_PAGE = "https://kalisteo.cea.fr/index.php/fallmud/"
FALLMUD_AIA = "http://crt.harica.gr/HARICA-GEANT-TLS-R1.cer"  # intermediate the CEA server does not send
FALLMUD_SHA256 = "3c1c0d4a2373e67be3d968d93b7630eca2437b01d1512b530355bc6b4f870cf9"  # v21 kernels (FACTS)
FM_PER_COMP = 1.0 / 3.0  # FALLMUD samples per competition sample in every epoch
FM_ORI_W = 1.0  # per-pixel weight of FALLMUD orientation pixels in the pooled orientation loss
FM_DUP_CORR = 0.90  # overlap guard: thumbnail correlation threshold
FM_PCA_R, FM_PIECE_HALF, FM_PIECE_STEP = 6.0, 4.0, 4.0  # canvas px (converted with the letterbox scale)
FM_MIN_ANISO, FM_MIN_BRANCH, FM_MIN_PIECES = 0.90, 8, 3
FM_MIN_FG, FM_MAX_FG = 0.0005, 0.35
FM_BLUR, FM_MIN_COMP = 1.0, 12  # mask smoothing sigma (mask px) / min component area before skeletonising
FM_MAX_JUNC_FRAC = 0.02  # skeleton junctions per skeleton px above this = noise mesh, not paths (image dropped)
TEST_LIMIT = 2 if CPU_SMOKE else (5 if SMOKE else None)  # inference only (the overlap guard always uses all images)
EXTRA_LIMIT = 1 if CPU_SMOKE else (2 if SMOKE else 800)
READOUT_LIMIT = 20 if SMOKE else None  # images per readout run in the smoke modes
READOUT_PROCS = 4
SEG_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# fasc_orient_v1 reference (attached via kernel_sources)
V1_MODEL = "fasc_orient_v1"
V1_FOLDS_SHA256 = "9cf568447913a47d6ae713dd443100a6250c9c921574a279a7e20dc16cfe93d3"
V1_TTA_ANG_MAE = (1.397, 1.428, 1.678, 1.466, 1.664)  # PREREG_v21 baseline (kernel OOF TTA), mean 1.5267

COMP_SLUG = "umud-challenge-muscle-architecture-in-ultrasound-data"
INPUT_ROOT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
SRC = Path(os.environ.get("UMUD_SRC", str(WORK / "lf26_src")))  # embedded lf_readout.py, geometry_v2.py, payload/
TMP = Path("/tmp/lf26_tmp")
MASKBANK = WORK / "maskbank"
CKPT_DIR = WORK / "checkpoints"
LOG_DIR = WORK / "logs"
METRIC_DIR = WORK / "metrics"
VIZ_DIR = WORK / "viz"
FM_DIR = WORK / "fallmud"
READOUT_DIR = WORK / "readout"
OOF_PART_DIR = TMP / "oof_parts"
EXTRA_SOURCES = {
    "static": [("https://osf.io/download/hb5xj/", "static.zip")],
    "dynamic": [("https://osf.io/download/vqb7h/", "dynamic.zip")],
    "vl": [
        ("https://osf.io/download/zcuwk/", "vl_young.zip"),
        ("https://osf.io/download/6ga94/", "vl_old.zip"),
    ],
}

ITEMS: list[dict] = []  # competition items; filled in main(), inherited by forked GPU workers
FMS: list[dict] = []  # FALLMUD items kept by the overlap guard (train-only)
FM_HOLD: list[dict] = []  # FALLMUD items held out of training in every fold (K0c)
V1_CKPTS: dict[int, Path] = {}
_VIZ_N = [0]  # images written to viz/ (capped at MAX_VIZ)
_LOG_PREFIX = ""
_LOG_FILE: Path | None = None


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


# ----------------------------------------------------------------------------- viewport (v1)
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


# ----------------------------------------------------------------------------- fragments (v1)
MIN_PTS = 6


def _fit(points: np.ndarray):
    c = points.mean(0)
    q = points - c
    evals, evecs = np.linalg.eigh(q.T @ q)
    d = evecs[:, 1]
    n = evecs[:, 0]
    return c, d, n, q @ d, q @ n


def decompose_component(points: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    """Split one connected component (N,2 float x,y) into straight segments [x1,y1,x2,y2]."""
    segs: list[np.ndarray] = []
    rem = points
    for _ in range(8):
        if len(rem) < MIN_PTS:
            break
        c, d, n, t, nn = _fit(rem)
        if np.abs(nn).max() <= 2.0:
            a, b = c + d * (t.min() - 0.5), c + d * (t.max() + 0.5)
            segs.append(np.array([a[0], a[1], b[0], b[1]], np.float32))
            break
        best_cnt, best = 0, None
        for _ in range(80):
            i, j = rng.integers(0, len(rem), 2)
            dv = rem[j] - rem[i]
            length = float(np.hypot(*dv))
            if length < 4:
                continue
            normal = np.array([-dv[1], dv[0]]) / length
            inl = np.abs((rem - rem[i]) @ normal) < 1.25
            cnt = int(inl.sum())
            if cnt > best_cnt:
                best_cnt, best = cnt, inl
        if best is None or best_cnt < MIN_PTS:
            break
        c, d, n, _, _ = _fit(rem[best])
        q = rem - c
        t = q @ d
        inl = np.abs(q @ n) < 1.5
        ts = np.sort(t[inl])
        if len(ts) < MIN_PTS:
            break
        cuts = np.flatnonzero(np.diff(ts) > 4.0)
        starts = np.r_[0, cuts + 1]
        ends = np.r_[cuts, len(ts) - 1]
        k = int(np.argmax(ends - starts))
        t0, t1 = ts[starts[k]], ts[ends[k]]
        take = inl & (t >= t0 - 1e-6) & (t <= t1 + 1e-6)
        if take.sum() >= MIN_PTS:
            a, b = c + d * (t0 - 0.5), c + d * (t1 + 0.5)
            segs.append(np.array([a[0], a[1], b[0], b[1]], np.float32))
        if take.sum() == 0:
            break
        rem = rem[~take]
    return segs


def mask_segments(mask: np.ndarray, img_w: int, img_h: int, rng: np.random.Generator):
    """Fragments of a binary mask as segments in IMAGE pixel coords (plain stretch)."""
    mh, mw = mask.shape
    sx, sy = img_w / mw, img_h / mh
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    segs, all_pts, n_split = [], [], 0
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 3:
            continue
        x0, y0 = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        ww, hh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        ys, xs = np.nonzero(lab[y0:y0 + hh, x0:x0 + ww] == i)
        pts = np.stack([(xs + x0 + 0.5) * sx - 0.5, (ys + y0 + 0.5) * sy - 0.5], 1)
        all_pts.append(pts)
        got = decompose_component(pts, rng)
        n_split += len(got) > 1
        segs.extend(got)
    segs_arr = np.stack(segs).astype(np.float32) if segs else np.zeros((0, 4), np.float32)
    pts_arr = np.concatenate(all_pts) if all_pts else np.zeros((0, 2))
    return segs_arr, pts_arr, n - 1, n_split


def read_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)


def sha1_array(arr: np.ndarray) -> str:
    h = hashlib.sha1()
    h.update(np.asarray(arr.shape, np.int64).tobytes())
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def thumbnail(crop: np.ndarray) -> np.ndarray:
    h, w = crop.shape
    c = crop[int(h * 0.1):max(int(h * 0.1) + 1, int(h * 0.9)), int(w * 0.1):max(int(w * 0.1) + 1, int(w * 0.9))]
    t = cv2.resize(c, (THUMB, THUMB), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    t -= t.mean()
    return (t / (np.linalg.norm(t) + 1e-6)).astype(np.float32)


def prep_train(pair: tuple[str, str]) -> dict:
    image_path, mask_path = Path(pair[0]), Path(pair[1])
    rng = np.random.default_rng(int(hashlib.md5(image_path.stem.encode()).hexdigest()[:8], 16))
    img = read_gray(image_path)
    mask = read_gray(mask_path) > 0
    out = {
        "stem": image_path.stem, "name": image_path.name, "mask_name": mask_path.name,
        "ih": sha1_array(img), "mh": sha1_array(np.packbits(mask)), "H": img.shape[0], "W": img.shape[1],
        "mask_shape": mask.shape, "empty": not mask.any(),
    }
    if out["empty"]:
        return out
    segs, pts, ncomp, nsplit = mask_segments(mask, img.shape[1], img.shape[0], rng)
    box, kind = viewport(img, image_path.suffix.lower())
    x1, y1, x2, y2 = box
    inside = ((pts[:, 0] >= x1 - 0.5) & (pts[:, 0] < x2 - 0.5) & (pts[:, 1] >= y1 - 0.5) & (pts[:, 1] < y2 - 0.5))
    coverage = float(inside.mean()) if len(pts) else 1.0
    if coverage < 0.97:
        box, kind = (0, 0, img.shape[1], img.shape[0]), kind + "_lowcov"
        x1, y1, x2, y2 = box
    crop = np.ascontiguousarray(img[y1:y2, x1:x2])
    segs = segs - np.array([x1, y1, x1, y1], np.float32)
    out.update(
        box=box, layout=kind, coverage=coverage, crop=crop, segs=segs.astype(np.float32),
        thumb=thumbnail(crop), ncomp=ncomp, nsplit=nsplit, empty=len(segs) == 0,
        thumb_full=thumbnail(img), ch=sha1_array(crop),  # v3: for the FALLMUD overlap guard
    )
    return out


# ----------------------------------------------------------------------------- geometry / sample (v1)
def letterbox_matrix(w: int, h: int) -> np.ndarray:
    s0 = SIZE / max(w, h)
    return np.array(
        [[s0, 0.0, (SIZE - s0 * w) / 2 + 0.5 * s0 - 0.5], [0.0, s0, (SIZE - s0 * h) / 2 + 0.5 * s0 - 0.5], [0, 0, 1]],
        np.float64,
    )


def aug_matrix(rng: np.random.Generator) -> np.ndarray:
    c = (SIZE - 1) / 2.0
    sc = math.exp(rng.uniform(math.log(0.8), math.log(1.25))) if rng.random() < 0.7 else 1.0
    ar = math.exp(rng.uniform(-0.12, 0.12)) if rng.random() < 0.3 else 1.0
    phi = math.radians(rng.uniform(-12.0, 12.0)) if rng.random() < 0.6 else 0.0
    if rng.random() < 0.5:
        tx, ty = rng.uniform(-0.06, 0.06) * SIZE, rng.uniform(-0.06, 0.06) * SIZE
    else:
        tx = ty = 0.0
    flip = -1.0 if rng.random() < 0.5 else 1.0
    tm = np.array([[1, 0, -c], [0, 1, -c], [0, 0, 1]], np.float64)
    fm = np.diag([flip, 1.0, 1.0])
    sm = np.diag([sc * ar, sc / ar, 1.0])
    cs, sn = math.cos(phi), math.sin(phi)
    rm = np.array([[cs, -sn, 0], [sn, cs, 0], [0, 0, 1]], np.float64)
    tp = np.array([[1, 0, c + tx], [0, 1, c + ty], [0, 0, 1]], np.float64)
    return tp @ rm @ sm @ fm @ tm


def warp_image(src: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Warp src with 3x3 m (src pixel coords -> SIZE canvas), anti-aliased when shrinking."""
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


def _poly_mask(m: np.ndarray, w: int, h: int) -> np.ndarray:
    corners = np.array([[-0.5, -0.5, 1], [w - 0.5, -0.5, 1], [w - 0.5, h - 0.5, 1], [-0.5, h - 0.5, 1]]).T
    p = (m @ corners)[:2].T
    out = np.zeros((SIZE, SIZE), np.uint8)
    cv2.fillConvexPoly(out, np.round(p * 16).astype(np.int32), 1, cv2.LINE_8, 4)
    return out


def intensity_aug(x8: np.ndarray, valid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = x8.astype(np.float32) * (1.0 / 255.0)
    if rng.random() < 0.8:
        x = np.power(x, rng.uniform(0.65, 1.5)).astype(np.float32)
    if rng.random() < 0.8:
        x = x * rng.uniform(0.8, 1.25) + rng.uniform(-0.08, 0.08)
    if rng.random() < 0.4:  # time-gain-compensation like vertical gain profile
        yy = np.linspace(-0.5, 0.5, SIZE, dtype=np.float32)
        g = 1.0 + rng.uniform(-0.5, 0.5) * yy + rng.uniform(-0.6, 0.6) * (yy * yy - 1.0 / 12)
        x = x * g[:, None]
    if rng.random() < 0.3:  # correlated multiplicative speckle
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


def render_targets(segs_out: np.ndarray, valid: np.ndarray) -> dict:
    """segs_out (K,4) in canvas coords -> seg/skel/orientation targets."""
    from scipy import ndimage

    seg = np.zeros((SIZE, SIZE), np.uint8)
    idm = np.zeros((SIZE, SIZE), np.uint16)
    k = len(segs_out)
    ctab = np.zeros(k + 1, np.float32)
    stab = np.zeros(k + 1, np.float32)
    okt = np.zeros(k + 1, bool)
    for i, (x1, y1, x2, y2) in enumerate(segs_out):
        if not (np.isfinite([x1, y1, x2, y2]).all()) or max(abs(x1), abs(y1), abs(x2), abs(y2)) > 50 * SIZE:
            continue
        p1 = (int(round(x1 * 16)), int(round(y1 * 16)))
        p2 = (int(round(x2 * 16)), int(round(y2 * 16)))
        cv2.line(seg, p1, p2, 1, SEG_WIDTH, cv2.LINE_8, 4)
        cv2.line(idm, p1, p2, i + 1, 1, cv2.LINE_8, 4)
        dx, dy = x2 - x1, y2 - y1
        l2 = dx * dx + dy * dy
        if l2 >= 9.0:
            ctab[i + 1] = (dx * dx - dy * dy) / l2
            stab[i + 1] = 2.0 * dx * dy / l2
            okt[i + 1] = True
    seg &= valid
    skel = ((idm > 0) & (valid > 0)).astype(np.uint8)
    if (idm > 0).any():
        dist, ind = ndimage.distance_transform_edt(idm == 0, return_indices=True)
        nid = idm[ind[0], ind[1]]
        om = (dist <= ORI_RADIUS) & okt[nid] & (valid > 0)
        cos2 = np.where(om, ctab[nid], 0.0).astype(np.float16)
        sin2 = np.where(om, stab[nid], 0.0).astype(np.float16)
        om = om.astype(np.uint8)
    else:
        om = np.zeros((SIZE, SIZE), np.uint8)
        cos2 = np.zeros((SIZE, SIZE), np.float16)
        sin2 = np.zeros((SIZE, SIZE), np.float16)
    return {"seg": seg, "skel": skel, "om": om, "cos2": cos2, "sin2": sin2}


_IDW_K = None


def idw_kernel() -> np.ndarray:
    global _IDW_K
    if _IDW_K is None:
        r = int(math.ceil(FM_IDW_R))
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
        d = np.hypot(xx, yy)
        _IDW_K = np.where(d <= FM_IDW_R, 1.0 / np.maximum(d, 1.0), 0.0).astype(np.float32)
    return _IDW_K


def render_fm_targets(segs_out: np.ndarray, skel_out: np.ndarray, valid: np.ndarray) -> dict:
    """LF-26 FALLMUD targets (canvas): path ridge exp(-d^2 / 2 sigma^2) around the full-path skeleton, dense
    orientation = doubled-angle IDW mean (weights 1/max(d,1), d <= FM_IDW_R) of the path pieces, unit-normalised where
    its resultant length >= FM_IDW_MIN_R. seg / skel are zero (the seg losses are masked on FALLMUD)."""
    from scipy import ndimage

    sk = np.zeros((SIZE, SIZE), np.uint8)
    if len(skel_out):
        q = np.round(skel_out).astype(np.int64)
        ok = (q[:, 0] >= 0) & (q[:, 0] < SIZE) & (q[:, 1] >= 0) & (q[:, 1] < SIZE)
        sk[q[ok, 1], q[ok, 0]] = 1
    sk &= valid
    if sk.any():
        d = ndimage.distance_transform_edt(sk == 0)
        path_t = (np.exp(-0.5 * (d / PATH_SIGMA) ** 2) * (valid > 0)).astype(np.float16)
    else:
        path_t = np.zeros((SIZE, SIZE), np.float16)
    cm = np.zeros((SIZE, SIZE), np.float32)
    smap = np.zeros((SIZE, SIZE), np.float32)
    lm = np.zeros((SIZE, SIZE), np.float32)
    for x1, y1, x2, y2 in segs_out:
        if not (np.isfinite([x1, y1, x2, y2]).all()) or max(abs(x1), abs(y1), abs(x2), abs(y2)) > 50 * SIZE:
            continue
        dx, dy = x2 - x1, y2 - y1
        l2 = dx * dx + dy * dy
        if l2 < 4.0:
            continue
        p1 = (int(round(x1 * 16)), int(round(y1 * 16)))
        p2 = (int(round(x2 * 16)), int(round(y2 * 16)))
        cv2.line(cm, p1, p2, float((dx * dx - dy * dy) / l2), 1, cv2.LINE_8, 4)
        cv2.line(smap, p1, p2, float(2.0 * dx * dy / l2), 1, cv2.LINE_8, 4)
        cv2.line(lm, p1, p2, 1.0, 1, cv2.LINE_8, 4)
    k = idw_kernel()
    den = cv2.filter2D(lm, -1, k, borderType=cv2.BORDER_CONSTANT)
    nc = cv2.filter2D(cm * lm, -1, k, borderType=cv2.BORDER_CONSTANT)
    ns = cv2.filter2D(smap * lm, -1, k, borderType=cv2.BORDER_CONSTANT)
    with np.errstate(invalid="ignore", divide="ignore"):
        c, s = nc / den, ns / den
        rr = np.hypot(c, s)
        om = (den > 1e-6) & (rr >= FM_IDW_MIN_R) & (valid > 0)
        cos2 = np.where(om, c / np.maximum(rr, 1e-6), 0.0).astype(np.float16)
        sin2 = np.where(om, s / np.maximum(rr, 1e-6), 0.0).astype(np.float16)
    z8 = np.zeros((SIZE, SIZE), np.uint8)
    return {"seg": z8, "skel": z8.copy(), "om": om.astype(np.uint8), "cos2": cos2, "sin2": sin2, "path_t": path_t}


def make_sample(crop: np.ndarray, segs: np.ndarray, rng: np.random.Generator | None,
                skel: np.ndarray | None = None) -> dict:
    """v1 sample (competition; RNG draws identical to v1) or, with `skel` (FALLMUD full-path skeleton, crop coords),
    a FALLMUD sample: LF-26 field-of-view crop instead of v1's narrow-FOV crop, IDW orientation + path-ridge targets."""
    fm = skel is not None
    h, w = crop.shape
    x0, y0, ww, hh = 0, 0, w, h
    if rng is not None:
        if fm:
            if rng.random() < FM_FOV_P:  # LF-26 field-of-view crop: paths cut at the FOV edge
                ww = max(32, int(round(rng.uniform(*FM_FOV_RANGE) * w)))
                x0 = int(rng.integers(0, w - ww + 1))
        elif rng.random() < 0.35:  # narrow field of view (e.g. Philips Lumify), v1
            ww = max(32, int(round(rng.uniform(0.3, 1.0) * w)))
            x0 = int(rng.integers(0, w - ww + 1))
        if rng.random() < 0.15:
            hh = max(32, int(round(rng.uniform(0.65, 1.0) * h)))
            y0 = int(rng.integers(0, h - hh + 1))
    sub = crop[y0:y0 + hh, x0:x0 + ww]
    m = letterbox_matrix(ww, hh)
    if rng is not None:
        m = aug_matrix(rng) @ m
    img = warp_image(sub, m)
    valid = _poly_mask(m, ww, hh)
    if len(segs):
        s = segs - np.array([x0, y0, x0, y0], np.float32)
        p1 = (m[:2, :2] @ s[:, :2].T + m[:2, 2:3]).T
        p2 = (m[:2, :2] @ s[:, 2:].T + m[:2, 2:3]).T
        segs_out = np.concatenate([p1, p2], 1)
    else:
        segs_out = np.zeros((0, 4))
    if fm:
        k = skel - np.array([x0, y0], np.float32)
        k = k[(k[:, 0] >= -0.5) & (k[:, 0] < ww - 0.5) & (k[:, 1] >= -0.5) & (k[:, 1] < hh - 0.5)]
        skel_out = (m[:2, :2] @ k.T.astype(np.float64) + m[:2, 2:3]).T if len(k) else np.zeros((0, 2))
        tg = render_fm_targets(segs_out, skel_out, valid)
    else:
        tg = render_targets(segs_out, valid)
        tg["path_t"] = np.zeros((SIZE, SIZE), np.float16)
    if rng is not None:
        img = intensity_aug(img, valid, rng)
    tg["img"] = img
    tg["valid"] = valid
    return tg


def make_input(crop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = crop.shape
    m = letterbox_matrix(w, h)
    return warp_image(crop, m), m


def unletterbox(maps: np.ndarray, m: np.ndarray, w: int, h: int) -> np.ndarray:
    """maps (SIZE,SIZE,C) float32 in canvas -> (h,w,C) in crop coords."""
    out = cv2.warpAffine(maps, m[:2].astype(np.float64), (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out.reshape(h, w, -1)


def encode_channels(seg: np.ndarray, cos2: np.ndarray, sin2: np.ndarray, path: np.ndarray | None = None) -> dict:
    out = {
        "seg": np.clip(np.round(seg * 255.0), 0, 255).astype(np.uint8),
        "cos2": np.clip(np.round((cos2 + 1.0) * 127.5), 0, 255).astype(np.uint8),
        "sin2": np.clip(np.round((sin2 + 1.0) * 127.5), 0, 255).astype(np.uint8),
    }
    if path is not None:  # LF-26: path probability, same encoding as seg
        out["path"] = np.clip(np.round(path * 255.0), 0, 255).astype(np.uint8)
    return out


# ----------------------------------------------------------------------------- npz writer (v1)
class NpzZip:
    """Incremental np.savez_compressed-compatible writer (np.load reads it lazily)."""

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


def merge_npz(parts: list[Path], dest: Path) -> int:
    w = NpzZip(dest)
    n = 0
    for part in parts:
        if not part.exists():
            continue
        with zipfile.ZipFile(part) as zin:
            for name in zin.namelist():
                key = name[:-4]
                if key in w.keys:
                    continue
                w.keys.add(key)
                with zin.open(name) as src, w.zf.open(name, "w", force_zip64=True) as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
                n += 1
    w.close()
    return n


def _merge_job(args):
    parts, dest = args
    return str(dest), merge_npz([Path(p) for p in parts], Path(dest))


# ----------------------------------------------------------------------------- FALLMUD: download + index
FM_RE_APO = re.compile(r"apo|aponeur")
FM_RE_FASC = re.compile(r"fasc|fibre|fiber|(?<![a-z])fas(?![a-z])")
FM_RE_MASK = re.compile(r"(?<![a-z])(masks?|labels?|seg|segmentation|gt|annotations?|annot)(?![a-z])")
FM_RE_STRIP = re.compile(r"aponeuros[a-z]*|apo|fascicules?|fascicles?|fasc|fas|fib(?:re|er)s?|masks?|labels?|"
                         r"segmentation|seg|gt|annotations?|annot|images?|img")


def _ssl_contexts():
    """(name, SSLContext) to try in order. kalisteo.cea.fr serves only its leaf certificate, so OpenSSL without AIA
    fetching cannot build the chain: add the HARICA GEANT TLS RSA 1 intermediate (fetched from the leaf's AIA URL);
    still verified against the system / certifi roots. Unverified only as a last resort (logged; sha256 recorded)."""
    yield "default", ssl.create_default_context()
    try:
        import certifi

        yield "certifi", ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001
        pass
    try:
        with urllib.request.urlopen(FALLMUD_AIA, timeout=60) as r:
            der = r.read()
        pem = der.decode() if der.lstrip().startswith(b"-----BEGIN") else ssl.DER_cert_to_PEM_cert(der)
        ctx = ssl.create_default_context()
        try:
            import certifi

            ctx.load_verify_locations(cafile=certifi.where())
        except Exception:  # noqa: BLE001
            pass
        ctx.load_verify_locations(cadata=pem)
        yield "aia_intermediate", ctx
    except Exception as exc:  # noqa: BLE001
        log(f"FALLMUD: AIA intermediate fetch failed: {exc!r}")
    yield "UNVERIFIED", ssl._create_unverified_context()  # noqa: S323


def _download_urls_from_html(text: str) -> list[str]:
    found = re.findall(r'data-downloadurl\s*=\s*"([^"]+)"', text) + re.findall(r"data-downloadurl\s*=\s*'([^']+)'", text)
    found += re.findall(r'href\s*=\s*"([^"]*wpdmdl=3570[^"]*)"', text)
    out = []
    for u in found:
        u = htmlmod.unescape(u)
        if u.startswith("http") and u not in out:
            out.append(u)
    return out


def fetch_fallmud(dest: Path) -> dict:
    info: dict = {"attempts": []}
    for ctx_name, ctx in _ssl_contexts():
        jar = http.cookiejar.CookieJar()
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx),
                                             urllib.request.HTTPCookieProcessor(jar))
        opener.addheaders = [("User-Agent", "Mozilla/5.0 (X11; Linux x86_64) kaggle-kernel umud-fasc-orient-v3"),
                             ("Accept", "*/*")]
        queue, tried, ssl_failed = [FALLMUD_URL], set(), False
        while queue and not ssl_failed:
            url = queue.pop(0)
            if url in tried:
                continue
            tried.add(url)
            for attempt in range(2):
                t = time.time()
                try:
                    with opener.open(url, timeout=180) as r, open(dest, "wb") as fh:
                        ctype = r.headers.get("Content-Type", "")
                        shutil.copyfileobj(r, fh, 1 << 20)
                    n = dest.stat().st_size
                    is_zip = zipfile.is_zipfile(dest)
                    info["attempts"].append({"url": url, "ssl": ctx_name, "bytes": n, "ctype": ctype, "zip": is_zip,
                                             "sec": round(time.time() - t, 1)})
                    log(f"FALLMUD GET {url} [{ctx_name}] -> {n} bytes {ctype!r} zip={is_zip} ({time.time() - t:.0f}s)")
                    if is_zip:
                        info.update(url=url, ssl=ctx_name)
                        return info
                    text = dest.read_bytes()[:3_000_000].decode("utf-8", "replace")
                    more = _download_urls_from_html(text)
                    log(f"FALLMUD: not a zip; download URLs found in the response: {more}")
                    queue.extend(more)
                    if FALLMUD_PAGE not in tried:
                        queue.append(FALLMUD_PAGE)
                    break
                except Exception as exc:  # noqa: BLE001
                    reason = getattr(exc, "reason", exc)
                    info["attempts"].append({"url": url, "ssl": ctx_name, "error": repr(exc)})
                    log(f"FALLMUD GET {url} [{ctx_name}] attempt {attempt + 1} failed: {exc!r}")
                    if isinstance(reason, ssl.SSLError) or isinstance(exc, ssl.SSLError):
                        ssl_failed = True  # certificate problem: next SSL context
                        break
                    time.sleep(10)
    raise RuntimeError(f"FALLMUD download failed: {json.dumps(info['attempts'])[:3000]}")


def safe_extract(zpath: Path, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    n = 0
    with zipfile.ZipFile(zpath) as zf:
        for m in zf.infolist():
            p = (dest / m.filename).resolve()
            if not str(p).startswith(str(root) + os.sep) or m.is_dir():
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m) as src, open(p, "wb") as dst:
                shutil.copyfileobj(src, dst)
            n += 1
    return n


def fm_author(parts: tuple[str, ...]) -> tuple[str, int]:
    for i, p in enumerate(parts[:-1]):
        pl = p.lower()
        if "cunningham" in pl:
            return "Cunningham", i
        if "cronin" in pl:
            return "Cronin", i
    return "unknown", -1


def fm_role(dirs: list[str], stem: str) -> str:
    s = stem.lower()
    if FM_RE_APO.search(s):
        return "apo"
    if FM_RE_FASC.search(s):
        return "fasc"
    for d in reversed(dirs):
        dl = d.lower()
        if FM_RE_APO.search(dl):
            return "apo"
        if FM_RE_FASC.search(dl):
            return "fasc"
    if FM_RE_MASK.search(s) or any(FM_RE_MASK.search(d.lower()) for d in dirs):
        return "mask_unknown"
    return "image"


def fm_key(stem: str) -> str:
    s = FM_RE_STRIP.sub("", stem.lower())
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def fallmud_index(xroot: Path) -> tuple[list[dict], dict, list[list]]:
    """Pair every FALLMUD image with its fascicle (and aponeurosis) mask, per author folder."""
    files = sorted(p for p in xroot.rglob("*") if p.is_file() and "__MACOSX" not in p.parts
                   and not p.name.startswith("."))
    top = {p.relative_to(xroot).parts[0] for p in files}
    rows, groups = [], {}
    ext_counts: dict[str, int] = {}
    for p in files:
        parts = p.relative_to(xroot).parts
        ext = p.suffix.lower()
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
        author, ai = fm_author(parts)
        if ai >= 0:
            dirs = list(parts[ai + 1:-1])
        else:
            dirs = list(parts[1:-1]) if len(top) == 1 else list(parts[:-1])
        role = fm_role(dirs, p.stem) if ext in IMAGE_SUFFIXES else "other"
        key = fm_key(p.stem)
        rows.append([str(p.relative_to(xroot)), p.stat().st_size, ext, author, role, key])
        if ext in IMAGE_SUFFIXES:
            groups.setdefault(author, {}).setdefault(role, []).append((key, p))
    recs, stats = [], {"ext_counts": ext_counts, "authors": {}}
    used_keys: set[str] = set()
    for author in sorted(groups):
        g = groups[author]
        imgs, fas, apo = g.get("image", []), g.get("fasc", []), g.get("apo", [])
        best = None
        for sname, kf in (("key", lambda k: k), ("digits", lambda k: "_".join(re.findall(r"\d+", k)))):
            ii, ff, aa = {}, {}, {}
            for lst, d in ((imgs, ii), (fas, ff), (apo, aa)):
                for k, p in lst:
                    kk = kf(k)
                    if kk:
                        d.setdefault(kk, []).append(p)
            n = sum(1 for k in ii if k in ff)
            if best is None or n > best[1]:
                best = (sname, n, ii, ff, aa)
        sname, n_match, ii, ff, aa = best
        for k in sorted(ii):
            if k not in ff:
                continue
            uk = f"{author}_{k}"
            j = 1
            while uk in used_keys:
                j += 1
                uk = f"{author}_{k}_{j}"
            used_keys.add(uk)
            recs.append({"author": author, "key": uk, "img": str(sorted(ii[k])[0]), "fasc": str(sorted(ff[k])[0]),
                         "apo": str(sorted(aa[k])[0]) if k in aa else "", "n_img_same_key": len(ii[k]),
                         "n_fasc_same_key": len(ff[k])})
        un_i = [str(ii[k][0].name) for k in sorted(ii) if k not in ff]
        un_f = [str(ff[k][0].name) for k in sorted(ff) if k not in ii]
        st = {"files": sum(len(v) for v in g.values()), "roles": {r: len(v) for r, v in g.items()},
              "strategy": sname, "matched_image_fasc": n_match, "matched_with_apo": sum(1 for k in ii if k in ff and k in aa),
              "unmatched_images": len(un_i), "unmatched_fasc_masks": len(un_f),
              "unmatched_image_examples": un_i[:8], "unmatched_fasc_examples": un_f[:8],
              "mask_unknown_examples": [p.name for _, p in g.get("mask_unknown", [])[:8]],
              "dup_image_keys": sum(1 for k in ii if len(ii[k]) > 1), "dup_fasc_keys": sum(1 for k in ff if len(ff[k]) > 1)}
        stats["authors"][author] = st
        log(f"FALLMUD index {author}: {json.dumps(st)}")
    stats["pairs"] = len(recs)
    return recs, stats, rows


# ----------------------------------------------------------------------------- FALLMUD: targets
def fm_binarize(m: np.ndarray) -> tuple[np.ndarray, int, float]:
    """FALLMUD masks are JPEGs (ringing / noise around the paths): light Gaussian smoothing + Otsu threshold, closing,
    small specks removed (cpusmoke r1: 'anything above background' turned JPEG noise into a skeleton mesh)."""
    u = np.unique(m)
    if len(u) <= 1:
        return np.zeros(m.shape, bool), len(u), float("nan")
    if len(u) == 2:
        thr, mf = float(u[0]), m
    else:
        mf = cv2.GaussianBlur(np.ascontiguousarray(m), (0, 0), FM_BLUR)
        thr, _ = cv2.threshold(mf, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = mf > thr
    if fg.mean() > 0.5:  # white background with dark paths
        fg = ~fg
    fg8 = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg8, connectivity=8)
    small = stats[:, cv2.CC_STAT_AREA] < FM_MIN_COMP
    small[0] = False
    fg = (fg8 > 0) & ~small[lab]
    return fg, len(u), float(thr)


def skel_junctions(sk8: np.ndarray) -> np.ndarray:
    """Branch points of a 1-px skeleton: crossing number (0->1 transitions around the 8-neighbourhood) >= 3, so that
    staircase pixels of oblique lines are not counted."""
    h, w = sk8.shape
    p = np.pad((sk8 > 0).astype(np.uint8), 1)
    nb = [p[0:h, 1:w + 1], p[0:h, 2:w + 2], p[1:h + 1, 2:w + 2], p[2:h + 2, 2:w + 2],
          p[2:h + 2, 1:w + 1], p[2:h + 2, 0:w], p[1:h + 1, 0:w], p[0:h, 0:w]]
    trans = sum(((nb[i] == 0) & (nb[(i + 1) % 8] == 1)).astype(np.uint8) for i in range(8))
    return (sk8 > 0) & (trans >= 3)


def skeletonize_mask(fg: np.ndarray) -> np.ndarray:
    try:
        from skimage.morphology import skeletonize

        return skeletonize(fg).astype(bool)
    except Exception:  # noqa: BLE001 - morphological skeleton fallback
        img = fg.astype(np.uint8)
        kern = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        sk = np.zeros_like(img)
        while img.any():
            er = cv2.erode(img, kern)
            sk |= img & ~cv2.dilate(er, kern)
            img = er
        return sk.astype(bool)


def fm_pieces(sk: np.ndarray, sx: float, sy: float, s0: float) -> tuple[np.ndarray, dict]:
    """Skeleton (mask px) -> short straight pieces [x1,y1,x2,y2] in IMAGE px: branch points (crossing number >= 3) are cut out,
    every branch is sampled every FM_PIECE_STEP canvas px and each piece follows the local PCA direction of the branch
    within FM_PCA_R canvas px (kept only if anisotropy >= FM_MIN_ANISO). s0 = letterbox scale (canvas px per image px)."""
    from scipy.spatial import cKDTree

    sk8 = sk.astype(np.uint8)
    junc = skel_junctions(sk8)
    r_img, h_img, step = FM_PCA_R / s0, FM_PIECE_HALF / s0, FM_PIECE_STEP / s0
    br = sk8.copy()
    if junc.any():  # cut a disk of the PCA radius around every branch point (skeletons bend near crossings)
        rj = int(math.ceil(r_img / max(sx, sy))) + 1
        br[cv2.dilate(junc.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rj + 1, 2 * rj + 1))) > 0] = 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(br, connectivity=8)
    segs, anis = [], []
    n_branch = n_rej = 0
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < FM_MIN_BRANCH:
            continue
        x0, y0 = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        ww, hh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        ys, xs = np.nonzero(lab[y0:y0 + hh, x0:x0 + ww] == i)
        pts = np.stack([(xs + x0 + 0.5) * sx - 0.5, (ys + y0 + 0.5) * sy - 0.5], 1)
        if float(np.hypot(*np.ptp(pts, 0))) < 2 * h_img:
            continue
        n_branch += 1
        tree = cKDTree(pts)
        cell = np.floor(pts / step).astype(np.int64)
        _, first = np.unique(cell, axis=0, return_index=True)
        cen = pts[np.sort(first)]
        for nbr in tree.query_ball_point(cen, r_img):
            if len(nbr) < 5:
                n_rej += 1
                continue
            q = pts[nbr]
            mu = q.mean(0)
            q = q - mu
            ev, evec = np.linalg.eigh(q.T @ q)
            a = 1.0 - float(ev[0]) / max(float(ev[1]), 1e-9)
            if a < FM_MIN_ANISO:
                n_rej += 1
                continue
            d = evec[:, 1]
            segs.append([mu[0] - h_img * d[0], mu[1] - h_img * d[1], mu[0] + h_img * d[0], mu[1] + h_img * d[1]])
            anis.append(a)
    arr = np.asarray(segs, np.float32).reshape(-1, 4)
    th = np.degrees(np.arctan2(arr[:, 3] - arr[:, 1], arr[:, 2] - arr[:, 0])) if len(arr) else np.zeros(0)
    th = (th + 90.0) % 180.0 - 90.0
    return arr, {"branches": n_branch, "junctions": int(junc.sum()), "rejected": n_rej,
                 "aniso_med": float(np.median(anis)) if anis else float("nan"),
                 "theta_med": float(np.median(th)) if len(th) else float("nan")}


def prep_fallmud(rec: dict) -> dict:
    out = {k: rec[k] for k in ("author", "key", "img", "fasc", "apo")}
    try:
        img_path = Path(rec["img"])
        img = read_gray(img_path)
        raw = read_gray(Path(rec["fasc"]))
        fg, levels, thr = fm_binarize(raw)
        frac = float(fg.mean())
        out.update(H=img.shape[0], W=img.shape[1], mask_shape=list(raw.shape), fg_frac=frac, mask_levels=levels,
                   mask_max=int(raw.max()), mask_p99=float(np.percentile(raw, 99)), mask_thr=thr,
                   ih=sha1_array(img), thumb_full=thumbnail(img))
        if not (FM_MIN_FG <= frac <= FM_MAX_FG):
            out["error"] = f"mask foreground fraction {frac:.5f} outside [{FM_MIN_FG}, {FM_MAX_FG}]"
            return out
        sk = skeletonize_mask(fg)
        sx, sy = img.shape[1] / raw.shape[1], img.shape[0] / raw.shape[0]
        ys, xs = np.nonzero(sk)
        pts = np.stack([(xs + 0.5) * sx - 0.5, (ys + 0.5) * sy - 0.5], 1)
        box, kind = viewport(img, img_path.suffix.lower())
        x1, y1, x2, y2 = box
        inside = ((pts[:, 0] >= x1 - 0.5) & (pts[:, 0] < x2 - 0.5) & (pts[:, 1] >= y1 - 0.5) & (pts[:, 1] < y2 - 0.5))
        coverage = float(inside.mean()) if len(pts) else 1.0
        if coverage < 0.97:
            box, kind = (0, 0, img.shape[1], img.shape[0]), kind + "_lowcov"
            x1, y1, x2, y2 = box
        crop = np.ascontiguousarray(img[y1:y2, x1:x2])
        s0 = SIZE / max(x2 - x1, y2 - y1)
        segs, st = fm_pieces(sk, sx, sy, s0)
        segs = segs - np.array([x1, y1, x1, y1], np.float32)
        # LF-26: the full-path skeleton (path-ridge targets and K0c ground truth), crop coords, junction disks kept
        skel = (pts - np.array([x1, y1], np.float64)).astype(np.float32)
        skel = skel[(skel[:, 0] >= -0.5) & (skel[:, 0] < (x2 - x1) - 0.5) & (skel[:, 1] >= -0.5) & (skel[:, 1] < (y2 - y1) - 0.5)]
        idx_ = re.findall(r"\d+", Path(rec["img"]).stem)
        out.update(box=box, layout=kind, coverage=coverage, crop=crop, segs=segs.astype(np.float32),
                   thumb=thumbnail(crop), ch=sha1_array(crop), n_pieces=int(len(segs)), skel_px=int(len(pts)),
                   skel=skel, fm_index=int(idx_[-1]) if idx_ else -1, **st)
        if len(segs) < FM_MIN_PIECES:
            out["error"] = f"only {len(segs)} orientation pieces"
        elif st["junctions"] > FM_MAX_JUNC_FRAC * max(1, len(pts)):
            out["error"] = f"mesh-like skeleton ({st['junctions']} junctions / {len(pts)} skeleton px)"
    except Exception as exc:  # noqa: BLE001
        out["error"] = repr(exc)
    return out


def prep_ref(entry: tuple[str, str, str]) -> dict:
    ds, key, path = entry
    out = {"ds": ds, "key": key}
    try:
        p = Path(path)
        gray = read_gray(p)
        box, _ = viewport(gray, p.suffix.lower())
        x1, y1, x2, y2 = box
        crop = gray[y1:y2, x1:x2]
        out.update(thumb=thumbnail(crop), thumb_full=thumbnail(gray), ih=sha1_array(gray), ch=sha1_array(crop))
    except Exception as exc:  # noqa: BLE001
        out["error"] = repr(exc)
    return out


def overlap_guard(fms: list[dict], refs: dict[str, dict]) -> tuple[list[dict], list[list], dict]:
    """Drop FALLMUD images that are near-duplicates (max thumbnail correlation over crop/full combinations >=
    FM_DUP_CORR) or pixel-identical (full-frame or crop hash) to any reference image."""
    a = np.stack([f["thumb"] for f in fms])
    af = np.stack([f["thumb_full"] for f in fms])
    names = list(refs)
    best = np.full((len(fms), len(names)), -1.0, np.float32)
    best_key = np.empty((len(fms), len(names)), object)
    hash_hit = np.zeros((len(fms), len(names)), bool)
    for j, nm in enumerate(names):
        r = refs[nm]
        if not len(r["keys"]):
            continue
        c = np.maximum.reduce([a @ r["thumb"].T, a @ r["thumb_full"].T, af @ r["thumb"].T, af @ r["thumb_full"].T])
        best[:, j] = c.max(1)
        arg = c.argmax(1)
        for i in range(len(fms)):
            best_key[i, j] = r["keys"][arg[i]]
            hash_hit[i, j] = bool({fms[i]["ih"], fms[i]["ch"]} & (r["ih"] | r["ch"]))  # full / crop, crosswise
    corr_hit = best >= FM_DUP_CORR
    drop = (corr_hit | hash_hit).any(1)
    rows, kept = [], []
    for i, f in enumerate(fms):
        reasons = [f"{names[j]}:{'hash' if hash_hit[i, j] else ''}{'corr' if corr_hit[i, j] else ''}"
                   for j in range(len(names)) if corr_hit[i, j] or hash_hit[i, j]]
        rows.append([f["author"], f["key"], Path(f["img"]).name, int(drop[i]), ";".join(reasons),
                     *[f"{best[i, j]:.4f}" for j in range(len(names))], *[best_key[i, j] for j in range(len(names))]])
        if not drop[i]:
            kept.append(f)
    mx = best.max(1)
    bins = [-1.0, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.01]
    hist = {f"{bins[k]:.2f}-{bins[k + 1]:.2f}": int(((mx >= bins[k]) & (mx < bins[k + 1])).sum()) for k in range(len(bins) - 1)}
    stats = {
        "n_in": len(fms), "n_kept": len(kept), "n_dropped": int(drop.sum()), "threshold": FM_DUP_CORR,
        "dropped_by_set_corr": {nm: int(corr_hit[:, j].sum()) for j, nm in enumerate(names)},
        "dropped_by_set_hash": {nm: int(hash_hit[:, j].sum()) for j, nm in enumerate(names)},
        "ref_set_sizes": {nm: len(refs[nm]["keys"]) for nm in names},
        "max_corr_hist": hist,
        "near_threshold_0.85_0.90": int(((mx >= 0.85) & (mx < 0.90)).sum()),
        "by_author": {au: {"in": sum(f["author"] == au for f in fms), "kept": sum(f["author"] == au for f in kept)}
                      for au in sorted({f["author"] for f in fms})},
    }
    return kept, rows, stats


def save_pieces_viz(path: Path, crop: np.ndarray, segs: np.ndarray) -> None:
    rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    for x1, y1, x2, y2 in segs:
        cv2.line(rgb, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), (0, 255, 0), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), rgb)


def save_mask_viz(path: Path, f: dict) -> None:
    """[full image with the orientation pieces (green, viewport box blue) | raw fascicle mask (as read)]."""
    if _VIZ_N[0] >= MAX_VIZ:
        return
    _VIZ_N[0] += 1
    img = read_gray(Path(f["img"]))
    raw = read_gray(Path(f["fasc"]))
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if "box" in f:
        x1, y1, x2, y2 = f["box"]
        cv2.rectangle(rgb, (x1, y1), (x2 - 1, y2 - 1), (255, 0, 0), 1)
        for a, b, c, d in f.get("segs", np.zeros((0, 4))):
            cv2.line(rgb, (int(round(a + x1)), int(round(b + y1))), (int(round(c + x1)), int(round(d + y1))),
                     (0, 255, 0), 1, cv2.LINE_AA)
    m = cv2.resize(raw, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    m = cv2.cvtColor(cv2.normalize(m, None, 0, 255, cv2.NORM_MINMAX), cv2.COLOR_GRAY2BGR)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.concatenate([rgb, m], 1))


def save_target_viz(path: Path, s: dict) -> None:
    """Augmented training sample: image + orientation target (green) where om, seg target (red)."""
    if _VIZ_N[0] >= MAX_VIZ:
        return
    _VIZ_N[0] += 1
    rgb = cv2.cvtColor(s["img"], cv2.COLOR_GRAY2BGR)
    rgb[..., 2] = np.where(s["seg"] > 0, 255, rgb[..., 2])
    if "path_t" in s:  # LF-26 path-ridge target (blue)
        pt = s["path_t"].astype(np.float32)
        rgb[..., 0] = np.maximum(rgb[..., 0], (pt * 255).astype(np.uint8))
    th = 0.5 * np.arctan2(s["sin2"].astype(np.float32), s["cos2"].astype(np.float32))
    for yy in range(8, SIZE, 16):
        for xx in range(8, SIZE, 16):
            if not s["om"][yy, xx]:
                continue
            dx, dy = math.cos(th[yy, xx]) * 7, math.sin(th[yy, xx]) * 7
            cv2.line(rgb, (int(xx - dx), int(yy - dy)), (int(xx + dx), int(yy + dy)), (0, 255, 0), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), rgb)


# ----------------------------------------------------------------------------- torch parts
def ensure_smp():
    try:
        import segmentation_models_pytorch as smp  # noqa: F401

        if smp.__version__ == SMP_VERSION:
            return smp
    except Exception:  # noqa: BLE001
        pass
    log(f"pip install segmentation-models-pytorch=={SMP_VERSION}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", f"segmentation-models-pytorch=={SMP_VERSION}"])
    import importlib

    for name in list(sys.modules):
        if name.startswith("segmentation_models_pytorch"):
            del sys.modules[name]
    smp = importlib.import_module("segmentation_models_pytorch")
    return smp


def build_v1_model(pretrained: bool = False, classes: int = 3):
    """The v1 network (smp U-Net, ConvNeXt-S encoder); classes=4 adds the LF-26 path channel."""
    import segmentation_models_pytorch as smp

    return smp.Unet(
        encoder_name=ENCODER,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=classes,
        decoder_channels=(256, 128, 64, 32, 16),
    )


def build_model(pretrained: bool):
    if BACKBONE != "convnext":
        raise ValueError(f"LF-26 uses the v1 network only (BACKBONE {BACKBONE!r})")
    return build_v1_model(pretrained, classes=N_CLASSES)


def init_from_v1(fold: int, items: list[dict], torch, dev) -> tuple:
    """LFNet-v26 fold model initialised from the v1 fold checkpoint: every v1 tensor copied; the 4th (path) output of
    the segmentation head initialised as a copy of the seg filter (channel 0); BatchNorm layers with non-finite
    buffers in the v1 checkpoint re-estimated on N_CAL un-augmented training crops of this fold (recalibrate_bn).
    Returns (model on dev, info)."""
    cp = V1_CKPTS.get(fold)
    if cp is None:
        raise RuntimeError(f"v1 fold {fold} checkpoint not attached")
    st = torch.load(cp, map_location="cpu", weights_only=False)
    sd = {k: v.float() if v.is_floating_point() else v for k, v in st["state_dict"].items()}
    bad = state_is_finite(sd, torch)
    model = build_model(pretrained=False)
    own = model.state_dict()
    copied, head = 0, []
    for k, v in sd.items():
        if k not in own:
            raise RuntimeError(f"v1 key {k} not in the LF-26 model")
        if own[k].shape == v.shape:
            own[k] = v.clone()
            copied += 1
        elif k.startswith("segmentation_head") and own[k].shape[0] == N_CLASSES and v.shape[0] == 3:
            w = torch.cat([v, v[:1]], 0)  # path channel = copy of the seg filter
            if tuple(w.shape) != tuple(own[k].shape):
                raise RuntimeError(f"head shape {k}: {tuple(w.shape)} vs {tuple(own[k].shape)}")
            own[k] = w.clone()
            head.append(k)
        else:
            raise RuntimeError(f"shape mismatch {k}: {tuple(v.shape)} vs {tuple(own[k].shape)}")
    missing = [k for k in own if k not in sd]
    if missing:
        raise RuntimeError(f"LF-26 keys not in the v1 checkpoint: {missing[:10]}")
    model.load_state_dict(own)
    model = model.to(dev).to(memory_format=torch.channels_last)
    info = dict(v1_ckpt=str(cp), v1_epoch=st.get("epoch"), copied=copied, head_expanded=head, nonfinite_v1=bad[:20])
    if bad:
        names = {k.rsplit(".", 1)[0] for k in bad}
        info["recalibrated_bn_layers"] = recalibrate_bn(model, items, torch, dev, names=names)
        info["recalibrated_names"] = sorted(names)
    still = state_is_finite(model.state_dict(), torch)
    info["nonfinite_after_repair"] = still[:20]
    if still:
        raise RuntimeError(f"fold {fold}: non-finite tensors after the BN repair: {still[:5]}")
    model.train()
    return model, info


def param_groups(model) -> tuple[list[dict], list[str]]:
    """v1: encoder LR_ENC / decoder (incl. the segmentation head with the path channel) LR_DEC, one AdamW weight decay."""
    enc_params = list(model.encoder.parameters())
    enc_ids = {id(p) for p in enc_params}
    dec_params = [p for p in model.parameters() if id(p) not in enc_ids]
    return ([{"params": enc_params, "lr": LR_ENC}, {"params": dec_params, "lr": LR_DEC}],
            [f"encoder lr={LR_ENC} n={len(enc_params)}", f"decoder lr={LR_DEC} n={len(dec_params)}"])


def amp(torch, dev):
    if dev.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return torch.autocast("cpu", enabled=False)


def to_input(img_u8, torch):
    x = img_u8.float().unsqueeze(1) * (1.0 / 255.0)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return ((x.expand(-1, 3, -1, -1) - mean) / std).contiguous(memory_format=torch.channels_last)


def forward_maps(model, x, torch, tta: bool):
    """Returns prob, cos2, sin2 (B,H,W) float32; hflip TTA flips the sin2 sign (v1)."""
    b = x.shape[0]
    xin = torch.cat([x, torch.flip(x, dims=[3])], 0) if tta else x
    with amp(torch, x.device):
        out = model(xin).float()
    p = torch.sigmoid(out[:b, 0])
    c = torch.tanh(out[:b, 1])
    s = torch.tanh(out[:b, 2])
    if tta:
        of = torch.flip(out[b:], dims=[3])  # (B,C,H,W): un-flip the W axis
        p = 0.5 * (p + torch.sigmoid(of[:, 0]))
        c = 0.5 * (c + torch.tanh(of[:, 1]))
        s = 0.5 * (s - torch.tanh(of[:, 2]))  # mirror: theta -> -theta, so sin(2 theta) changes sign
    return p, c, s


def forward_maps4(model, x, torch, tta: bool):
    """LF-26: prob, cos2, sin2, path (B,H,W) float32 with the v1 hflip TTA (path: plain un-flip and mean)."""
    b = x.shape[0]
    xin = torch.cat([x, torch.flip(x, dims=[3])], 0) if tta else x
    with amp(torch, x.device):
        out = model(xin).float()
    p = torch.sigmoid(out[:b, 0])
    c = torch.tanh(out[:b, 1])
    s = torch.tanh(out[:b, 2])
    q = torch.sigmoid(out[:b, 3])
    if tta:
        of = torch.flip(out[b:], dims=[3])
        p = 0.5 * (p + torch.sigmoid(of[:, 0]))
        c = 0.5 * (c + torch.tanh(of[:, 1]))
        s = 0.5 * (s - torch.tanh(of[:, 2]))
        q = 0.5 * (q + torch.sigmoid(of[:, 3]))
    return p, c, s, q


def soft_skel(x, torch, iters: int = CLDICE_ITERS):
    """Soft skeleton (Shit et al. 2021, clDice) of a (B,H,W) probability map."""
    F = torch.nn.functional
    x = x.unsqueeze(1)

    def erode(t):
        return -F.max_pool2d(-t, 3, 1, 1)

    def dilate(t):
        return F.max_pool2d(t, 3, 1, 1)
    sk = F.relu(x - dilate(erode(x)))
    for _ in range(iters):
        x = erode(x)
        d = F.relu(x - dilate(erode(x)))
        sk = sk + F.relu(d - sk * d)
    return sk.squeeze(1)


def state_is_finite(sd: dict, torch) -> list[str]:
    return [k for k, v in sd.items() if torch.is_tensor(v) and v.is_floating_point() and not bool(torch.isfinite(v).all())]


def recalibrate_bn(model, items: list[dict], torch, dev, names: set[str] | None = None, n_max: int = N_CAL) -> int:
    """Re-estimate the running stats (cumulative average over un-augmented training crops) of the BatchNorm layers
    in `names` (fasc_orient_v2)."""
    bns = [mm for n_, mm in model.named_modules() if isinstance(mm, torch.nn.modules.batchnorm._BatchNorm)
           and (names is None or n_ in names)]
    if not bns:
        return 0
    for b in bns:
        b.reset_running_stats()
        b.momentum = None
    model.eval()
    for b in bns:
        b.train()
    sel = np.random.default_rng(0).permutation(len(items))[:n_max]
    with torch.no_grad():
        for i in range(0, len(sel), BATCH):
            xs = [make_input(items[int(j)]["crop"])[0] for j in sel[i:i + BATCH]]
            x = to_input(torch.from_numpy(np.stack(xs)).to(dev), torch)
            with amp(torch, dev):
                model(x)
    for b in bns:
        b.momentum = 0.1
    model.eval()
    return len(bns)


class TrainSet:
    """Competition items (v1 sampling / augmentation) followed by FALLMUD items (orientation only)."""

    def __init__(self, items: list[dict], fm: list[dict], augment: bool, repeat: int = 1) -> None:
        self.items = items
        self.fm = fm
        self.augment = augment
        self.repeat = repeat
        self.n_comp = len(items) * repeat

    def __len__(self) -> int:
        return self.n_comp + len(self.fm)

    def __getitem__(self, idx: int):
        import torch

        is_fm = idx >= self.n_comp
        it = self.fm[idx - self.n_comp] if is_fm else self.items[idx % len(self.items)]
        # LF-26 r1 (addendum K_r1): v1's seed (torch.initial_seed() + idx * 7919) is constant inside a persistent
        # loader worker, so every sample got <= 3 distinct augmentations over the whole run (FACTS v21 seeding quirk).
        # The worker's torch RNG advances with every call: deterministic per run, a fresh augmentation per draw.
        seed = (int(torch.randint(0, 2**31 - 1, (1,)).item()) * 1000003 + idx * 7919) % (2**32)
        rng = np.random.default_rng(seed) if self.augment else None
        s = make_sample(it["crop"], it["segs"], rng, it["skel"] if is_fm else None)
        s["has_seg"] = np.array([0 if is_fm else 1], np.uint8)
        s["is_fm"] = np.array([1 if is_fm else 0], np.uint8)
        return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in s.items()}


class MixSampler:
    """Every epoch: all competition indices + a fresh random draw of n_fm FALLMUD indices, shuffled together."""

    def __init__(self, n_comp: int, n_fm_pool: int, n_fm: int, seed: int) -> None:
        self.n_comp, self.n_fm_pool, self.n_fm = n_comp, n_fm_pool, n_fm
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.n_comp + self.n_fm

    def __iter__(self):
        fm = self.n_comp + self.rng.choice(self.n_fm_pool, self.n_fm, replace=self.n_fm > self.n_fm_pool)
        idx = self.rng.permutation(np.concatenate([np.arange(self.n_comp), fm]))
        return iter(int(i) for i in idx)


def _worker_init(_):
    cv2.setNumThreads(1)
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:  # noqa: BLE001
        pass


def build_val_cache(items: list[dict], torch) -> list[dict]:
    samples = [make_sample(it["crop"], it["segs"], None) for it in items]
    for s in samples:  # LF-26: evaluate() does not use the path target
        s.pop("path_t", None)
    cache = []
    for i in range(0, len(samples), BATCH):
        chunk = samples[i:i + BATCH]
        cache.append({k: torch.from_numpy(np.stack([s[k] for s in chunk])) for k in chunk[0]})
    return cache


def evaluate(model, cache, torch, dev, tta: bool = False) -> dict:
    """v1 evaluate() (only .cuda() -> .to(dev) so that cpusmoke runs it)."""
    model.eval()
    dice_sum = {t: 0.0 for t in SEG_THRESHOLDS}
    n_img, ang_img_sum, ang_img_n, ang_pix_sum, ang_pix_n, med_sum, med_n = 0, 0.0, 0, 0.0, 0, 0.0, 0
    with torch.no_grad():
        for b in cache:
            x = to_input(b["img"].to(dev, non_blocking=True), torch)
            p, c, s = forward_maps(model, x, torch, tta)
            seg_t = b["seg"].to(dev) > 0
            skel = b["skel"].to(dev) > 0
            om = b["om"].to(dev) > 0
            valid = b["valid"].to(dev) > 0
            ct = b["cos2"].to(dev).float()
            st = b["sin2"].to(dev).float()
            for t in SEG_THRESHOLDS:
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
    dices = {t: dice_sum[t] / max(1, n_img) for t in SEG_THRESHOLDS}
    best_t = max(dices, key=dices.get)
    return {
        "dice50": dices[0.5], "dice_best": dices[best_t], "thr_best": best_t,
        "ang_mae": ang_img_sum / max(1, ang_img_n), "ang_mae_pix": ang_pix_sum / max(1, ang_pix_n),
        "med_err": med_sum / max(1, med_n), "n_val": n_img, "n_ang": ang_img_n,
    }


def _better(m: dict, best: dict | None) -> bool:
    """v1 epoch-selection rule."""
    if best is None:
        return True
    if m["ang_mae"] < best["ang_mae"] - 0.02:
        return True
    return abs(m["ang_mae"] - best["ang_mae"]) <= 0.02 and m["dice50"] > best["dice50"]


def predict_native(models, gray: np.ndarray, box, torch, dev) -> tuple[dict, np.ndarray]:
    """Fold-ensemble + hflip TTA maps at native image resolution (uint8 MASKBANK encoding), v1 + the path channel."""
    x1, y1, x2, y2 = box
    crop = gray[y1:y2, x1:x2]
    inp, m = make_input(crop)
    x = to_input(torch.from_numpy(inp[None]).to(dev), torch)
    acc = None
    with torch.no_grad():
        for model in models:
            p, c, s, q = forward_maps4(model, x, torch, True)
            cur = torch.stack([p[0], c[0], s[0], q[0]], -1)
            acc = cur if acc is None else acc + cur
    maps = (acc / len(models)).cpu().numpy().astype(np.float32)
    back = unletterbox(maps, m, x2 - x1, y2 - y1)
    full = np.zeros((gray.shape[0], gray.shape[1], 4), np.float32)
    full[y1:y2, x1:x2] = back
    return encode_channels(full[..., 0], full[..., 1], full[..., 2], full[..., 3]), maps


def save_viz(path: Path, gray: np.ndarray, enc: dict) -> None:
    if _VIZ_N[0] >= MAX_VIZ:
        return
    _VIZ_N[0] += 1
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if "path" in enc:  # LF-26 path probability (blue)
        q = enc["path"].astype(np.float32) / 255.0
        rgb[..., 0] = np.clip(rgb[..., 0] * (1 - q) + 255 * q, 0, 255).astype(np.uint8)
    seg = enc["seg"].astype(np.float32) / 255.0
    rgb[..., 2] = np.clip(rgb[..., 2] * (1 - seg) + 255 * seg, 0, 255).astype(np.uint8)
    c = enc["cos2"].astype(np.float32) / 127.5 - 1
    s = enc["sin2"].astype(np.float32) / 127.5 - 1
    th = 0.5 * np.arctan2(s, c)
    mag = np.hypot(c, s)
    step = max(16, gray.shape[1] // 40)
    for yy in range(step // 2, gray.shape[0], step):
        for xx in range(step // 2, gray.shape[1], step):
            if mag[yy, xx] < 0.3:
                continue
            dx, dy = math.cos(th[yy, xx]) * step * 0.45, math.sin(th[yy, xx]) * step * 0.45
            cv2.line(rgb, (int(xx - dx), int(yy - dy)), (int(xx + dx), int(yy + dy)), (0, 255, 0), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), rgb)


# ----------------------------------------------------------------------------- GPU worker
def v1_reference(fold: int, cache, tr: list[dict], torch, dev) -> dict:
    """The attached v1 fold checkpoint on this fold's validation cache: as v1 evaluated it, and BN-repaired."""
    cp = V1_CKPTS.get(fold)
    if cp is None:
        return {"available": False}
    t = time.time()
    st = torch.load(cp, map_location="cpu", weights_only=False)
    sd = st["state_dict"]
    bad = state_is_finite(sd, torch)
    m = build_v1_model(False)
    m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})  # exactly as v1 did
    m = m.to(dev).to(memory_format=torch.channels_last).eval()
    res = {"available": True, "ckpt_epoch": st.get("epoch"), "nonfinite": bad, "tta_asis": evaluate(m, cache, torch, dev, tta=True)}
    if bad:
        names = {k.rsplit(".", 1)[0] for k in bad}
        res["recalibrated_bn"] = recalibrate_bn(m, tr, torch, dev, names=names)
        res["tta_repaired"] = evaluate(m, cache, torch, dev, tta=True)
    else:
        res["tta_repaired"] = res["tta_asis"]
    res["sec"] = round(time.time() - t, 1)
    del m, st, sd
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return res


def train_fold(fold: int, gpu_state: dict, torch, dev) -> None:
    from torch.utils.data import DataLoader

    rng = np.random.default_rng(SEED + fold)
    tr = [it for it in ITEMS if it["fold"] != fold]
    va = [it for it in ITEMS if it["fold"] == fold]
    fm = list(FMS) if USE_FALLMUD else []
    if SMOKE:
        tr = [tr[i] for i in rng.choice(len(tr), SMOKE_TRAIN, replace=False)]
        va = [va[i] for i in rng.choice(len(va), SMOKE_VAL, replace=False)]
        if fm:
            fm = [fm[i] for i in rng.choice(len(fm), min(len(fm), SMOKE_FM), replace=False)]
    if {it["stem"] for it in tr} & {it["stem"] for it in va}:
        raise RuntimeError(f"fold {fold}: train/val overlap")
    torch.manual_seed(SEED + fold)
    n_cpu = os.cpu_count() or 4
    n_workers = 0 if CPU_SMOKE else max(2, min(4, n_cpu // gpu_state["n_parallel"] + 1))
    repeat = SMOKE_REPEAT if SMOKE else 1
    ds = TrainSet(tr, fm, augment=True, repeat=repeat)
    n_fm_epoch = int(round(len(tr) * repeat * FM_PER_COMP)) if fm else 0
    kw = dict(batch_size=BATCH, num_workers=n_workers, pin_memory=dev.type == "cuda", drop_last=True,
              worker_init_fn=_worker_init)
    if n_workers:
        kw.update(persistent_workers=True, prefetch_factor=4)
    if n_fm_epoch:
        dl = DataLoader(ds, sampler=MixSampler(ds.n_comp, len(fm), n_fm_epoch, SEED + 1000 + fold), **kw)
    else:
        dl = DataLoader(ds, shuffle=True, **kw)  # v1
    t_cache = time.time()
    cache = build_val_cache(va, torch)
    log(f"fold {fold}: train={len(tr)} val={len(va)} fallmud_pool={len(fm)} fallmud/epoch={n_fm_epoch} "
        f"(ratio {n_fm_epoch / max(1, len(tr) * repeat):.3f}) loader_workers={n_workers} steps/epoch={len(dl)} "
        f"val_cache={time.time() - t_cache:.1f}s")
    if fold == TRAIN_FOLDS[0]:  # what augmented training samples / targets look like
        try:
            for j in range(min(3, len(fm))):
                f_ = fm[(j * 7) % len(fm)]
                save_target_viz(VIZ_DIR / f"train_sample_fallmud_{f_['key']}.jpg",
                                make_sample(f_["crop"], f_["segs"], np.random.default_rng(j), f_["skel"]))
            save_target_viz(VIZ_DIR / f"train_sample_comp_{tr[0]['stem']}.jpg",
                            make_sample(tr[0]["crop"], tr[0]["segs"], np.random.default_rng(0)))
        except Exception:  # noqa: BLE001
            log(f"target viz failed:\n{traceback.format_exc()}")

    v1ref = {"available": False}
    if V1_REF_EVAL:
        try:
            v1ref = v1_reference(fold, cache, tr, torch, dev)
            log(f"fold {fold}: v1 reference {json.dumps(v1ref, default=str)}")
        except Exception:  # noqa: BLE001
            log(f"fold {fold}: v1 reference FAILED:\n{traceback.format_exc()}")
    torch.manual_seed(SEED + fold)  # the v1 reference build consumed RNG: decoder init / loader seeds as in v1

    if V1_INIT:
        model, init_info = init_from_v1(fold, tr, torch, dev)
        log(f"fold {fold}: initialised from v1: {json.dumps(init_info, default=str)}")
    else:
        model = build_model(pretrained=True).to(dev).to(memory_format=torch.channels_last)
        init_info = {"v1_init": False}
    torch.manual_seed(SEED + fold)
    groups, gdesc = param_groups(model)
    opt = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"fold {fold}: model {BACKBONE} {n_params / 1e6:.1f}M params; param groups: {'; '.join(gdesc)}")
    steps = len(dl) if MAX_STEPS is None else min(len(dl), MAX_STEPS)
    warm = max(1, int(steps * WARMUP_EPOCHS))
    gpu_state.setdefault("t_train_start", time.time())
    deadline = min(T0 + TRAIN_BUDGET_S, gpu_state["t_train_start"] + TRAIN_CAP_S)  # LF-26: <= 3.5 h of training
    n_epochs = gpu_state.get("n_epochs", MAX_EPOCHS)
    if "ep_time" in gpu_state:  # later folds on this GPU: re-check the remaining budget
        folds_left = max(1, gpu_state["plan_folds"] - gpu_state["done"])
        n_fit = int((deadline - time.time()) / folds_left / (gpu_state["ep_time"] * 1.03))
        n_epochs = int(max(MIN_EPOCHS, min(n_epochs, n_fit)))
        log(f"fold {fold}: n_epochs={n_epochs} (budget fit {n_fit})")
    sched_state = {"total": n_epochs * steps}

    def lr_lambda(step: int) -> float:
        if step < warm:
            return 0.05 + 0.95 * step / warm
        total = max(sched_state["total"], warm + 1)
        prog = min(1.0, (step - warm) / (total - warm))
        return LR_FLOOR + (1 - LR_FLOOR) * 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    pos_w = torch.tensor([POS_WEIGHT], device=dev)
    hist_path = METRIC_DIR / f"history_fold{fold}.csv"
    cols = ["epoch", "loss", "bce", "dice_l", "skel_l", "ori_l", "ori_comp", "ori_fm", "path_fm", "cldice_fm",
            "path_comp", "dice50", "dice_best", "thr_best", "ang_mae", "ang_mae_pix", "med_err", "lr", "sec",
            "n_nonfinite", "n_fm_seen"]
    with open(hist_path, "w", newline="") as fh:
        csv.writer(fh).writerow(cols)
    best, best_epoch = None, 0
    ckpt_path = CKPT_DIR / f"fold{fold}.pt"
    epoch = 0
    n_bad_total = 0
    while epoch < n_epochs:
        epoch += 1
        model.train()
        t_ep = time.time()
        sums = np.zeros(10)
        nb = 0
        nb_fm = 0
        n_bad = 0
        n_fm_seen = 0
        t_wait = 0.0
        t_prev = time.time()
        for step, b in enumerate(dl, 1):
            if MAX_STEPS is not None and step > MAX_STEPS:
                break
            t_wait += time.time() - t_prev
            x = to_input(b["img"].to(dev, non_blocking=True), torch)
            seg_t = b["seg"].to(dev, non_blocking=True).float()
            skel = b["skel"].to(dev, non_blocking=True).float()
            om = b["om"].to(dev, non_blocking=True).float()
            ct = b["cos2"].to(dev, non_blocking=True).float()
            st = b["sin2"].to(dev, non_blocking=True).float()
            hs = b["has_seg"].to(dev, non_blocking=True).float().view(-1)
            isf = b["is_fm"].to(dev, non_blocking=True).float().view(-1)
            path_t = b["path_t"].to(dev, non_blocking=True).float()
            valid_t = b["valid"].to(dev, non_blocking=True).float()
            n_fm_seen += int(b["is_fm"].sum())
            with amp(torch, dev):
                out = model(x)
            out = out.float()
            logit = out[:, 0]
            # segmentation losses (v1) on competition samples only; identical to v1 when every sample has has_seg=1
            n_hs = hs.sum().clamp_min(1)
            bce_s = torch.nn.functional.binary_cross_entropy_with_logits(logit, seg_t, pos_weight=pos_w,
                                                                         reduction="none").mean((1, 2))
            bce = (bce_s * hs).sum() / n_hs
            p = torch.sigmoid(logit)
            inter = (p * seg_t).sum((1, 2))
            dice_s = 1 - (2 * inter + 1) / (p.sum((1, 2)) + seg_t.sum((1, 2)) + 1)
            dice_l = (dice_s * hs).sum() / n_hs
            sk_n = skel.sum((1, 2))
            has = (sk_n > 0) & (hs > 0)
            if bool(has.any()):
                skel_l = (1 - (p * skel).sum((1, 2))[has] / sk_n[has]).mean()
            else:
                skel_l = logit.sum() * 0
            # orientation: pooled over all supervised pixels (v1), FALLMUD pixels weighted FM_ORI_W
            cp, sp = torch.tanh(out[:, 1]), torch.tanh(out[:, 2])
            err = (cp - ct).abs() + (sp - st).abs()
            om_c = om * (1 - isf).view(-1, 1, 1)
            om_f = om * isf.view(-1, 1, 1)
            e_c, e_f = (err * om_c).sum(), (err * om_f).sum()
            n_c, n_f = om_c.sum(), om_f.sum()
            ori_l = (e_c + FM_ORI_W * e_f) / (2 * (n_c + FM_ORI_W * n_f).clamp_min(1))
            # LF-26 path channel. FALLMUD: focal BCE on the ridge target (every non-path pixel a negative at weight
            # PATH_NEG_W) + soft clDice; competition: fragment pixels as positives (weight COMP_PATH_POS_W) + sparsity.
            plog = out[:, 3]
            pq = torch.sigmoid(plog)
            bce_p = torch.nn.functional.binary_cross_entropy_with_logits(plog, path_t, reduction="none")
            focal = (path_t - pq).abs().pow(PATH_GAMMA) * bce_p
            wpx = torch.where(path_t >= PATH_POS_T, torch.ones_like(path_t), torch.full_like(path_t, PATH_NEG_W)) * valid_t
            wpx_f = wpx * isf.view(-1, 1, 1)
            path_fm = (focal * wpx_f).sum() / wpx_f.sum().clamp_min(1)
            if bool((isf > 0).any()):
                fi = isf > 0
                sp_, st_ = soft_skel(pq[fi], torch), soft_skel(path_t[fi], torch)
                tprec = (sp_ * path_t[fi]).sum((1, 2)) / sp_.sum((1, 2)).clamp_min(1e-6)
                tsens = (st_ * pq[fi]).sum((1, 2)) / st_.sum((1, 2)).clamp_min(1e-6)
                cldice = (1 - 2 * tprec * tsens / (tprec + tsens).clamp_min(1e-6)).mean()
            else:
                cldice = plog.sum() * 0
            pos_c = seg_t * hs.view(-1, 1, 1)
            path_comp = (COMP_PATH_POS_W * (torch.nn.functional.softplus(-plog) * pos_c).sum() / pos_c.sum().clamp_min(1)
                         + COMP_PATH_SPARSITY * (pq * valid_t * hs.view(-1, 1, 1)).sum()
                         / (valid_t * hs.view(-1, 1, 1)).sum().clamp_min(1))
            loss = (W_BCE * bce + W_DICE * dice_l + W_SKEL * skel_l + W_ORI * ori_l
                    + W_PATH * path_fm + W_CLDICE * cldice + path_comp)
            opt.zero_grad(set_to_none=True)
            if not bool(torch.isfinite(loss)):
                n_bad += 1
                sched.step()
                t_prev = time.time()
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            e_c_, e_f_, n_c_, n_f_ = (float(v.detach()) for v in (e_c, e_f, n_c, n_f))
            ori_c = e_c_ / (2 * max(1.0, n_c_))
            ori_f = e_f_ / (2 * max(1.0, n_f_)) if n_f_ > 0 else float("nan")
            sums[:6] += [*(float(v.detach()) for v in (loss, bce, dice_l, skel_l, ori_l)), ori_c]
            if not math.isnan(ori_f):
                sums[6] += ori_f
                sums[7] += float(path_fm.detach())
                sums[8] += float(cldice.detach())
                nb_fm += 1
            sums[9] += float(path_comp.detach())
            nb += 1
            if step % 50 == 0 or (SMOKE and step % 5 == 0) or (CPU_SMOKE and step <= 3):
                el = time.time() - t_ep
                log(f"fold {fold} ep {epoch} step {step}/{steps} loss={sums[0] / max(1, nb):.4f} "
                    f"ori={sums[4] / max(1, nb):.4f} ms/step={1000 * el / step:.0f} wait_ms/step={1000 * t_wait / step:.0f} "
                    f"mem={(torch.cuda.max_memory_allocated() / 2**30) if dev.type == 'cuda' else 0:.1f}G")
            t_prev = time.time()
        t_train = time.time() - t_ep
        met = evaluate(model, cache, torch, dev)
        sec = time.time() - t_ep
        lr_now = opt.param_groups[0]["lr"]  # v1 logs the encoder LR (group 0)
        sums[:6] /= max(1, nb)
        sums[6:9] = sums[6:9] / nb_fm if nb_fm else float("nan")
        sums[9] /= max(1, nb)
        n_bad_total += n_bad
        log(f"EPOCH fold={fold} ep={epoch}/{n_epochs} loss={sums[0]:.4f} bce={sums[1]:.4f} dice_l={sums[2]:.4f} "
            f"skel_l={sums[3]:.4f} ori_l={sums[4]:.4f} ori_comp={sums[5]:.4f} ori_fm={sums[6]:.4f} "
            f"path_fm={sums[7]:.4f} cldice_fm={sums[8]:.4f} path_comp={sums[9]:.4f} | val "
            f"dice50={met['dice50']:.4f} dice_best={met['dice_best']:.4f}@{met['thr_best']} ang_mae={met['ang_mae']:.3f}deg "
            f"ang_pix={met['ang_mae_pix']:.2f} med_err={met['med_err']:.2f}deg | lr={lr_now:.2e} nonfinite={n_bad} "
            f"fm_seen={n_fm_seen} train={t_train:.0f}s total={sec:.0f}s")
        with open(hist_path, "a", newline="") as fh:
            csv.writer(fh).writerow([epoch, *[round(float(v), 5) for v in sums], met["dice50"], met["dice_best"],
                                     met["thr_best"], met["ang_mae"], met["ang_mae_pix"], met["med_err"], lr_now, sec,
                                     n_bad, n_fm_seen])
        if _better(met, best):
            best, best_epoch = met, epoch
            sd = {k: v.detach().float().cpu() if v.is_floating_point() else v.detach().cpu()
                  for k, v in model.state_dict().items()}  # float32 (v1 saved fp16: BN running_var overflow)
            torch.save({"state_dict": sd, "epoch": epoch, "metrics": met, "fold": fold, "backbone": BACKBONE,
                        "encoder": ENCODER, "size": SIZE, "classes": CHANNELS, "model": MODEL_NAME, "dtype": "float32",
                        "init": init_info}, ckpt_path)
            log(f"fold {fold}: saved best epoch {epoch} ang_mae={met['ang_mae']:.3f} dice50={met['dice50']:.4f}")
        if epoch == 1 and "n_epochs" not in gpu_state and not SMOKE:
            ep_time = sec
            left = deadline - time.time() + ep_time  # budget incl. this epoch (LF-26: <= TRAIN_CAP_S of training)
            per_fold = left / gpu_state["plan_folds"]
            n_epochs = int(max(MIN_EPOCHS, min(MAX_EPOCHS, per_fold // (ep_time * 1.03))))
            gpu_state["n_epochs"] = n_epochs
            gpu_state["ep_time"] = ep_time
            sched_state["total"] = n_epochs * steps
            log(f"fold {fold}: epoch time {ep_time:.0f}s -> n_epochs={n_epochs} for all folds on this GPU")
        if time.time() > T0 + HARD_TRAIN_END_S:
            log(f"fold {fold}: HARD time limit reached after epoch {epoch}; stopping this fold")
            break
    del dl
    log(f"fold {fold}: best epoch {best_epoch} {json.dumps(best)}")
    del model, opt
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # ---- reload the saved fp32 checkpoint from disk, check finiteness, re-validate (before any map is written)
    vmodel = build_model(pretrained=False)
    st_ = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    bad = state_is_finite(st_["state_dict"], torch)
    vmodel.load_state_dict(st_["state_dict"])
    vmodel = vmodel.to(dev).to(memory_format=torch.channels_last).eval()
    met_r = evaluate(vmodel, cache, torch, dev)
    ok = (not bad) and abs(met_r["ang_mae"] - best["ang_mae"]) <= VERIFY_TOL_ANG and \
        abs(met_r["dice50"] - best["dice50"]) <= VERIFY_TOL_DICE
    log(f"fold {fold}: reload check {'OK' if ok else 'FAILED'} nonfinite={bad[:5]} reloaded dice50={met_r['dice50']:.4f} "
        f"ang_mae={met_r['ang_mae']:.3f} vs best {best['dice50']:.4f}/{best['ang_mae']:.3f}")
    res = {"fold": fold, "best_epoch": best_epoch, "n_epochs": n_epochs, "epochs_run": epoch, "best": best,
           "n_train": len(tr), "n_val": len(va), "n_fm_pool": len(fm), "n_fm_epoch": n_fm_epoch,
           "nonfinite_steps": n_bad_total, "reload_ok": ok, "reload_metrics": met_r, "reload_nonfinite": bad[:20],
           "v1_ref": v1ref, "v1_kernel_tta_ang_mae": V1_TTA_ANG_MAE[fold], "init": init_info,
           "train_sec": round(time.time() - gpu_state["t_train_start"], 1)}
    if not ok:
        (METRIC_DIR / f"fold{fold}.json").write_text(json.dumps(res, indent=2, default=str))
        log(f"fold {fold}: checkpoint unusable -> excluded from the ensemble")
        return
    met_tta = evaluate(vmodel, cache, torch, dev, tta=True)
    res["tta"] = met_tta
    log(f"fold {fold}: TTA val dice50={met_tta['dice50']:.4f} ang_mae={met_tta['ang_mae']:.3f} "
        f"med_err={met_tta['med_err']:.3f} | v1 kernel {V1_TTA_ANG_MAE[fold]:.3f} | v1 re-evaluated as-is "
        f"{v1ref.get('tta_asis', {}).get('ang_mae', float('nan')):.3f} repaired "
        f"{v1ref.get('tta_repaired', {}).get('ang_mae', float('nan')):.3f}")
    # LF-26: no train_oof maps are written (K0b uses the TTA metric above; maps of the eval sets come from main())
    (METRIC_DIR / f"fold{fold}.json").write_text(json.dumps(res, indent=2, default=str))
    del vmodel, cache
    if dev.type == "cuda":
        torch.cuda.empty_cache()


def gpu_worker(gpu: int, folds: list[int], plan_folds: int, n_parallel: int) -> None:
    global _LOG_PREFIX, _LOG_FILE
    if not CPU_SMOKE:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    _LOG_PREFIX = f"[gpu{gpu}]"
    _LOG_FILE = LOG_DIR / f"gpu{gpu}.log"
    code = 0
    try:
        import torch

        torch.backends.cudnn.benchmark = True
        torch.set_num_threads(4 if CPU_SMOKE else 1)
        cv2.setNumThreads(1)
        dev = torch.device("cuda" if torch.cuda.is_available() and not CPU_SMOKE else "cpu")
        log(f"worker start pid={os.getpid()} folds={folds} device="
            f"{torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'}")
        state = {"plan_folds": plan_folds, "n_parallel": n_parallel, "done": 0}
        for k in folds:
            try:
                train_fold(k, state, torch, dev)
            except Exception:  # noqa: BLE001
                code = 1
                log(f"fold {k} FAILED:\n{traceback.format_exc()}")
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
            state["done"] += 1
    except BaseException:  # noqa: BLE001
        code = 2
        log(f"worker FAILED:\n{traceback.format_exc()}")
    os._exit(code)


# ----------------------------------------------------------------------------- isolated torch helper
PRETRAIN_CODE = r"""
import sys
backbone, name, res = sys.argv[1], sys.argv[2], int(sys.argv[3])
import timm
if backbone == "convnext":
    import segmentation_models_pytorch as smp
    m = smp.Unet(encoder_name=name, encoder_weights="imagenet", in_channels=3, classes=3,
                 decoder_channels=(256, 128, 64, 32, 16))
    print(f"pretrained {name}: {sum(p.numel() for p in m.parameters()) / 1e6:.1f}M params", flush=True)
else:
    m = timm.create_model(name, pretrained=True, img_size=res, dynamic_img_size=True, num_classes=0)
    print(f"pretrained {name}: timm {timm.__version__} {sum(p.numel() for p in m.parameters()) / 1e6:.1f}M params "
          f"grid={tuple(m.patch_embed.grid_size)} pos_embed={tuple(m.pos_embed.shape)} prefix={m.num_prefix_tokens} "
          f"forward_intermediates={hasattr(m, 'forward_intermediates')} hub={m.pretrained_cfg.get('hf_hub_id')} "
          f"license={m.pretrained_cfg.get('license')}", flush=True)
"""


def run_isolated(code: str, args: list[str]) -> tuple[int, list[str]]:
    """Run torch code in a fresh interpreter (never in the parent before fork); returns (code, stdout lines)."""
    r = subprocess.run([sys.executable, "-c", code, *args], capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        log(f"isolated helper failed (code {r.returncode}): {r.stderr[-3000:]}")
    return r.returncode, [ln for ln in r.stdout.splitlines() if ln.strip()]


# ----------------------------------------------------------------------------- data discovery (v1)
def find_dir(name: str) -> Path:
    for base in (INPUT_ROOT / "competitions" / COMP_SLUG, INPUT_ROOT / COMP_SLUG, INPUT_ROOT):
        if base.exists():
            hits = [p for p in base.rglob(name) if p.is_dir()]
            if hits:
                return sorted(hits, key=lambda p: len(str(p)))[0]
    raise FileNotFoundError(name)


def find_model_source(model: str) -> Path | None:
    """Root of an attached kernel output (kernel_sources) that holds <model>_config.json."""
    hits = sorted(INPUT_ROOT.rglob(f"{model}_config.json"), key=lambda p: len(str(p)))
    return hits[0].parent if hits else None


def find_pairs(image_dir: Path, mask_dir: Path) -> list[tuple[str, str]]:
    masks = {p.stem: p for p in mask_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES}
    pairs = [(str(p), str(masks[p.stem])) for p in image_dir.rglob("*")
             if p.suffix.lower() in IMAGE_SUFFIXES and p.stem in masks]
    return sorted(pairs, key=lambda t: Path(t[0]).stem)


def union_find_groups(n: int, edges: np.ndarray) -> np.ndarray:
    parent = np.arange(n)

    def root(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in edges:
        ra, rb = root(int(a)), root(int(b))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return np.array([root(i) for i in range(n)])


def assign_folds(groups: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    uniq, counts = np.unique(groups, return_counts=True)
    order = sorted(range(len(uniq)), key=lambda i: (-counts[i], rng.random()))
    load = np.zeros(N_FOLDS)
    gfold = {}
    for i in order:
        f = int(np.argmin(load + rng.random(N_FOLDS) * 1e-3))
        gfold[uniq[i]] = f
        load[f] += counts[i]
    return np.array([gfold[g] for g in groups])


def load_v1_folds(path: Path) -> tuple[dict[str, int], dict[str, tuple[str, int]]]:
    rep_fold, alias = {}, {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            alias[r["stem"]] = (r["representative"], int(r["fold"]))
            if r["stem"] == r["representative"]:
                rep_fold[r["stem"]] = int(r["fold"])
    return rep_fold, alias


def prepare_items(pairs: list[tuple[str, str]], v1_rep: dict[str, int], v1_alias: dict[str, tuple[str, int]]):
    """v1 preprocessing + de-duplication; folds are TAKEN FROM v1's folds.csv (every item must map, else fail)."""
    t = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(min(4, os.cpu_count() or 2)) as pool:
        results = list(pool.imap(prep_train, pairs, chunksize=8))
    log(f"preprocessed {len(results)} pairs in {time.time() - t:.0f}s")
    stats = {"pairs": len(results), "empty_masks": 0, "exact_duplicates": 0, "same_image_other_mask": 0}
    by_img: dict[str, list[dict]] = {}
    for r in results:
        by_img.setdefault(r["ih"], []).append(r)
    items, alias_rows = [], []
    for ih, rs in by_img.items():
        rs.sort(key=lambda r: r["stem"])
        seen_masks, keep = set(), []
        for r in rs:
            if r["mh"] in seen_masks:
                stats["exact_duplicates"] += 1
                continue
            seen_masks.add(r["mh"])
            if r["empty"]:
                stats["empty_masks"] += 1
                continue
            keep.append(r)
        if not keep:
            continue
        rep = keep[0]
        if len(keep) > 1:
            stats["same_image_other_mask"] += len(keep) - 1
            bx, by = rep["box"][0], rep["box"][1]
            allsegs = [rep["segs"]]
            for r in keep[1:]:
                off = np.array([r["box"][0] - bx, r["box"][1] - by] * 2, np.float32)
                allsegs.append(r["segs"] + off)
            segs = np.concatenate(allsegs)
            _, ui = np.unique(np.round(segs, 1), axis=0, return_index=True)
            rep["segs"] = segs[np.sort(ui)]
        rep["aliases"] = [r["stem"] for r in rs]
        items.append(rep)
    items.sort(key=lambda r: r["stem"])
    thumbs = np.stack([it["thumb"] for it in items])
    corr = thumbs @ thumbs.T
    ii, jj = np.nonzero(np.triu(corr > DUP_CORR, 1))
    groups = union_find_groups(len(items), np.stack([ii, jj], 1))
    folds = assign_folds(groups, SEED)
    missing, alias_bad, n_recomputed_match = [], [], 0
    for i, it in enumerate(items):
        it["group"] = int(groups[i])
        f = v1_rep.get(it["stem"])
        if f is None:
            missing.append(it["stem"])
            continue
        n_recomputed_match += int(folds[i]) == f
        it["fold"] = f
        for a in it["aliases"]:
            ra = v1_alias.get(a)
            if ra is None or ra[1] != f or ra[0] != it["stem"]:
                alias_bad.append((a, it["stem"], f, ra))
            alias_rows.append([a, it["stem"], it["group"], it["fold"]])
    if missing or alias_bad:
        raise RuntimeError(f"v1 fold mapping FAILED: {len(missing)} of {len(items)} items have no v1 fold "
                           f"(first {missing[:20]}); {len(alias_bad)} alias rows disagree (first {alias_bad[:10]})")
    fl = np.array([it["fold"] for it in items])
    _, gcounts = np.unique(groups, return_counts=True)
    stats.update(
        unique_items=len(items), n_groups=int(len(gcounts)), max_group=int(gcounts.max()),
        fold_sizes=[int((fl == f).sum()) for f in range(N_FOLDS)],
        v1_fold_mapped=len(items), v1_fold_recomputed_match=n_recomputed_match, v1_representatives=len(v1_rep),
        segments_median=float(np.median([len(it["segs"]) for it in items])),
        split_components=int(sum(it["nsplit"] for it in items)),
        components=int(sum(it["ncomp"] for it in items)),
        lowcov_fullframe=int(sum("_lowcov" in it["layout"] for it in items)),
        layouts={k: int(v) for k, v in zip(*np.unique([it["layout"] for it in items], return_counts=True))},
        crop_mb=float(sum(it["crop"].nbytes for it in items) / 2**20),
    )
    if len(items) != len(v1_rep):
        raise RuntimeError(f"v1 fold mapping FAILED: {len(items)} items here vs {len(v1_rep)} v1 representatives")
    return items, alias_rows, stats


# ----------------------------------------------------------------------------- extra datasets (v1)
def download(url: str, dest: Path) -> bool:
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (kaggle kernel)"})
            with urllib.request.urlopen(req, timeout=180) as r, open(dest, "wb") as fh:
                shutil.copyfileobj(r, fh, 1 << 20)
            if zipfile.is_zipfile(dest):
                return True
            log(f"download {url}: not a zip ({dest.stat().st_size} bytes)")
        except Exception as exc:  # noqa: BLE001
            log(f"download {url} attempt {attempt + 1} failed: {exc!r}")
            time.sleep(10)
    return False


def gather_extra(name: str) -> list[tuple[str, Path, str]]:
    """Returns [(image_id, path, source_filename)] for an OSF dataset (downloaded + unzipped); v1 keys."""
    root = TMP / "ext" / name
    root.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    for url, fname in EXTRA_SOURCES[name]:
        zpath = TMP / "ext" / fname
        if not download(url, zpath):
            continue
        sub = root / Path(fname).stem
        safe_extract(zpath, sub)
        zpath.unlink()
        files += [p for p in sub.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
                  and "__MACOSX" not in p.parts and not p.name.startswith("._")]
    if name == "static":
        sel = [p for p in files if p.stem.startswith("im_") and p.stem.endswith("_arch")]
        files = sel or files
    elif name == "dynamic":
        sel = [p for p in files if "frames" in p.parts or p.stem.startswith("image_")]
        files = sel or files
    files = sorted(files, key=lambda p: str(p.relative_to(root)))
    stem_counts: dict[str, int] = {}
    for p in files:
        stem_counts[p.stem] = stem_counts.get(p.stem, 0) + 1
    out, used = [], set()
    for p in files:
        key = p.stem  # non-unique stems (e.g. VL ANON_1 per subject) get their two parent folders prefixed
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


# ----------------------------------------------------------------------------- FALLMUD pipeline (parent)
def _kmeans_cos(X: np.ndarray, k: int, seed: int, iters: int = 30) -> np.ndarray:
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)].copy()
    lab = np.zeros(len(X), int)
    for _ in range(iters):
        lab = np.argmax(X @ C.T, 1)
        for j in range(k):
            if (lab == j).any():
                c = X[lab == j].mean(0)
                C[j] = c / (np.linalg.norm(c) + 1e-9)
    return lab


def fm_groups(fms: list[dict]) -> dict:
    """LF-26: FALLMUD subject groups (no IDs in the zip) from index-block breaks: frames sorted by (author, file
    index); a new block starts where the index jumps or the consecutive full-frame thumbnail correlation falls below
    FM_BREAK_CORR. If that gives fewer than 4 or more than 40 blocks: FM_N_CLUSTERS cosine k-means clusters of the
    thumbnails. Whole groups (seeded order) are held out until >= FM_HOLDOUT_FRAC of the frames (K0c); sets
    f['fm_group'], f['holdout']."""
    order = sorted(range(len(fms)), key=lambda i: (fms[i]["author"], fms[i]["fm_index"], fms[i]["key"]))
    T = np.stack([fms[i]["thumb_full"] for i in order]).astype(np.float64)
    idx = np.array([fms[i]["fm_index"] for i in order])
    corr = (T[:-1] * T[1:]).sum(1) if len(T) > 1 else np.zeros(0)
    brk = (corr < FM_BREAK_CORR) | (np.diff(idx) != 1)
    blocks = np.concatenate([[0], np.cumsum(brk)]).astype(int)
    nb = int(blocks.max()) + 1
    method = "index_blocks"
    if not (4 <= nb <= 40):
        method = f"kmeans{FM_N_CLUSTERS}"
        blocks = _kmeans_cos(T, FM_N_CLUSTERS, SEED)
        nb = FM_N_CLUSTERS
    for j, i in enumerate(order):
        fms[i]["fm_group"] = int(blocks[j])
    sizes = np.bincount(blocks, minlength=nb)
    hold, n = [], 0
    for gid in np.random.default_rng(SEED + 7).permutation(nb):
        if n >= FM_HOLDOUT_FRAC * len(fms):
            break
        if sizes[gid] == 0 or (hold and n + sizes[gid] > 0.35 * len(fms)):
            continue
        hold.append(int(gid))
        n += int(sizes[gid])
    for f in fms:
        f["holdout"] = f["fm_group"] in hold
    q = np.quantile(corr, [0.01, 0.05, 0.5]).round(4).tolist() if len(corr) else []
    return dict(method=method, n_groups=nb, sizes=sizes.tolist(), holdout_groups=hold, n_holdout=n,
                n_train=len(fms) - n, consecutive_corr_q01_q05_q50=q, n_breaks=int(brk.sum()))


def prepare_fallmud(refs: dict[str, dict]) -> tuple[list[dict], dict]:
    FM_DIR.mkdir(parents=True, exist_ok=True)
    root = TMP / "fallmud"
    root.mkdir(parents=True, exist_ok=True)
    zpath = root / "fallmud_dataset.zip"
    t = time.time()
    info = fetch_fallmud(zpath)
    h = hashlib.sha256()
    with open(zpath, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    size = zpath.stat().st_size
    info.update(sha256=h.hexdigest(), bytes=size, download_sec=round(time.time() - t, 1))
    log(f"FALLMUD zip: {size} bytes ({size / 2**20:.2f} MiB) sha256={info['sha256']} url={info['url']} ssl={info['ssl']}")
    info["sha256_expected"] = FALLMUD_SHA256
    info["sha256_ok"] = info["sha256"] == FALLMUD_SHA256
    if not info["sha256_ok"]:  # LF-26: the pre-registered zip only
        raise RuntimeError(f"FALLMUD sha256 {info['sha256']} != pre-registered {FALLMUD_SHA256}")
    with zipfile.ZipFile(zpath) as zf:
        infos = zf.infolist()
    with open(FM_DIR / "zip_listing.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "file_size", "compress_size", "date_time", "is_dir"])
        for zi in infos:
            w.writerow([zi.filename, zi.file_size, zi.compress_size, "%04d-%02d-%02d %02d:%02d:%02d" % zi.date_time,
                        int(zi.is_dir())])
    log(f"FALLMUD zip listing ({len(infos)} entries):")
    for zi in infos:
        log(f"FALLMUD| {zi.file_size:>10} {zi.filename}")
    xroot = root / "x"
    n_x = safe_extract(zpath, xroot)
    nested = sorted(p for p in xroot.rglob("*.zip") if "__MACOSX" not in p.parts)
    for p in nested:  # one level of nested archives
        n_n = safe_extract(p, p.with_suffix(""))
        log(f"FALLMUD nested zip {p.relative_to(xroot)}: {n_n} files")
        p.unlink()
    zpath.unlink()
    log(f"FALLMUD extracted {n_x} files")
    for p in sorted(xroot.rglob("*")):  # small text files (ReadMe): logged and copied to the output as data
        if p.is_file() and p.suffix.lower() in (".txt", ".md") and p.stat().st_size < 20000:
            txt = p.read_text(errors="replace")
            info.setdefault("text_files", {})[str(p.relative_to(xroot))] = txt
            shutil.copy(p, FM_DIR / ("zip_" + p.name))
            log(f"FALLMUD text file {p.relative_to(xroot)}:\n{txt}")
    recs, istats, rows = fallmud_index(xroot)
    with open(FM_DIR / "files.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["relpath", "bytes", "ext", "author", "role", "key"])
        w.writerows(rows)
    role_counts: dict[str, int] = {}
    for r in rows:
        role_counts[f"{r[3]}/{r[4]}"] = role_counts.get(f"{r[3]}/{r[4]}", 0) + 1
    log(f"FALLMUD files by author/role: {json.dumps(role_counts)}; matched image+fascicle pairs: {len(recs)}")
    t = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(min(4, os.cpu_count() or 2)) as pool:
        fms = list(pool.imap(prep_fallmud, recs, chunksize=4))
    good = [f for f in fms if "error" not in f]
    bad = [f for f in fms if "error" in f]
    log(f"FALLMUD prepared {len(fms)} pairs in {time.time() - t:.0f}s: ok={len(good)} failed={len(bad)}; first failures "
        f"{[(Path(f['img']).name, f['error']) for f in bad[:8]]}")
    pcols = ["author", "key", "img", "fasc", "apo", "H", "W", "mask_shape", "mask_levels", "mask_max", "mask_p99",
             "mask_thr", "fg_frac", "layout", "box", "coverage", "skel_px", "branches", "junctions", "rejected", "n_pieces",
             "aniso_med", "theta_med", "error"]
    with open(FM_DIR / "pairs.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(pcols)
        for f in fms:
            w.writerow([f.get(c, "") if c not in ("img", "fasc", "apo") else Path(f[c]).name if f[c] else ""
                        for c in pcols])
    pstats = {}
    for au in sorted({f["author"] for f in fms}):
        g = [f for f in good if f["author"] == au]
        allf = [f for f in fms if f["author"] == au]
        pstats[au] = {
            "pairs": len(allf), "ok": len(g),
            "fg_frac_median": float(np.median([f["fg_frac"] for f in allf if "fg_frac" in f])) if allf else None,
            "mask_levels_median": float(np.median([f["mask_levels"] for f in allf if "mask_levels" in f])) if allf else None,
            "pieces_median": float(np.median([f["n_pieces"] for f in g])) if g else None,
            "aniso_median": float(np.nanmedian([f["aniso_med"] for f in g])) if g else None,
            "theta_median_deg": float(np.nanmedian([f["theta_med"] for f in g])) if g else None,
            "image_shapes": {k: int(v) for k, v in zip(*np.unique([f"{f['W']}x{f['H']}" for f in allf if "W" in f],
                                                                   return_counts=True))} if allf else {},
            "layouts": {k: int(v) for k, v in zip(*np.unique([f["layout"] for f in g], return_counts=True))} if g else {},
        }
    log(f"FALLMUD per-author target stats: {json.dumps(pstats)}")
    for au in sorted({f["author"] for f in fms}):  # image + pieces | raw mask, for checking the parser
        g = [f for f in good if f["author"] == au]
        sel = g[:2] + sorted(g, key=lambda f: -f["fg_frac"])[:2] + [f for f in bad if f["author"] == au][:2]
        for f in sel:
            try:
                save_mask_viz(VIZ_DIR / f"fallmud_{'bad' if 'error' in f else 'ok'}_{f['key']}.jpg", f)
            except Exception as exc:  # noqa: BLE001
                log(f"FALLMUD viz {f['key']} failed: {exc!r}")
    kept, orows, gstats = overlap_guard(good, refs) if good else ([], [], {"n_in": 0, "n_kept": 0})
    names = list(refs)
    with open(FM_DIR / "overlap.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["author", "key", "image", "dropped", "reasons", *[f"maxcorr_{n}" for n in names],
                    *[f"nearest_{n}" for n in names]])
        w.writerows(orows)
    log(f"FALLMUD overlap guard: {json.dumps(gstats)}")
    non_cun = sorted({f["author"] for f in kept} - {"Cunningham"})
    if non_cun:  # LF-26 r1: only Cunningham et al. 2018 may survive the guard (v3cfm: Cronin 306/306 dropped)
        raise RuntimeError(f"FALLMUD images of {non_cun} survived the overlap guard "
                           f"({sum(f['author'] != 'Cunningham' for f in kept)} images); pre-registered: Cunningham only")
    grp_info = fm_groups(kept) if kept else {}
    log(f"FALLMUD subject groups: {json.dumps(grp_info)}")
    with open(FM_DIR / "groups.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["author", "key", "fm_index", "fm_group", "holdout"])
        for f in kept:
            w.writerow([f["author"], f["key"], f["fm_index"], f["fm_group"], int(f["holdout"])])
    for f in kept:
        for k in ("thumb", "thumb_full"):
            f.pop(k, None)
    stats = {"download": info, "index": istats, "role_counts": role_counts, "prepared_ok": len(good),
             "prepared_failed": len(bad), "per_author": pstats, "overlap_guard": gstats, "kept": len(kept),
             "kept_mb": float(sum(f["crop"].nbytes for f in kept) / 2**20), "groups": grp_info}
    (FM_DIR / "stats.json").write_text(json.dumps(stats, indent=2, default=str))
    shutil.rmtree(xroot, ignore_errors=True)
    if not kept:
        raise RuntimeError("USE_FALLMUD but no FALLMUD image survived parsing + the overlap guard")
    return kept, stats


def reference_thumbs(datasets: dict[str, list[tuple[str, Path, str]]], items: list[dict]) -> dict[str, dict]:
    refs = {"train": {"keys": [it["stem"] for it in items], "thumb": np.stack([it["thumb"] for it in items]),
                      "thumb_full": np.stack([it["thumb_full"] for it in items]),
                      "ih": {it["ih"] for it in items}, "ch": {it["ch"] for it in items}}}
    entries = [(ds, key, str(path)) for ds, ents in datasets.items() for key, path, _ in ents]
    t = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(min(4, os.cpu_count() or 2)) as pool:
        res = list(pool.imap(prep_ref, entries, chunksize=8))
    for ds in datasets:
        r = [x for x in res if x["ds"] == ds and "error" not in x]
        z = np.zeros((0, THUMB * THUMB), np.float32)
        refs[ds] = {"keys": [x["key"] for x in r], "thumb": np.stack([x["thumb"] for x in r]) if r else z,
                    "thumb_full": np.stack([x["thumb_full"] for x in r]) if r else z,
                    "ih": {x["ih"] for x in r}, "ch": {x["ch"] for x in r}}
    errs = [(x["ds"], x["key"], x["error"]) for x in res if "error" in x]
    log(f"reference thumbnails: {json.dumps({k: len(v['keys']) for k, v in refs.items()})} in {time.time() - t:.0f}s; "
        f"unreadable {len(errs)} {errs[:5]}")
    return refs


def write_credits(fm_stats: dict | None) -> None:
    lines = [
        f"{MODEL_NAME} (UMUD LF-26, from kaggle/fasc_orient_v3 / fasc_orient_v1) -- credits and licences of external material",
        "",
        "Competition data: UMUD Challenge (Kaggle competition data, competition rules apply).",
        "OSF benchmark / example images (static hb5xj, dynamic vqb7h, VL zcuwk + 6ga94): inference / overlap guard only;",
        "  VL is NOT used for training in LF-26.",
        "Viewport layout rules adapted from AmbrosM, 'UMUD Quick and Dirty' (Kaggle notebook), Apache-2.0.",
        f"Backbone: {ENCODER} (timm, Apache-2.0) via segmentation_models_pytorch {SMP_VERSION} (MIT); initialised from",
        "  our own fasc_orient_v1 fold checkpoints (attached kernel output shivateja005/umud-fasc-orient-v1).",
        "Soft clDice loss after S. Shit et al., 'clDice - a Novel Topology-Preserving Loss Function for Tubular Structure",
        "  Segmentation', CVPR 2021 (own implementation).",
    ]
    if USE_FALLMUD:
        d = (fm_stats or {}).get("download", {})
        lines += [
            "",
            "FALLMUD (FAscicle Lower Leg Muscle Ultrasound Dataset), CEA List:",
            f"  source page: {FALLMUD_PAGE}",
            f"  downloaded from: {d.get('url')} (TLS: {d.get('ssl')})",
            f"  zip sha256: {d.get('sha256')} (pre-registered {FALLMUD_SHA256})  bytes: {d.get('bytes')}",
            "  Used part: the Cunningham et al. images and their full fascicle-path masks only:",
            "  R. Cunningham, M. B. Sanchez, G. May, I. Loram, 'Estimating Full Regional Skeletal Muscle Fibre Orientation",
            "  from B-Mode Ultrasound Images Using Convolutional, Residual, and Deconvolutional Neural Networks', J. Imaging",
            "  4(2):29, 2018, doi 10.3390/jimaging4020029 (CC BY 4.0).",
            "  The Cronin et al. part (near-copies of competition training images) is dropped by the overlap guard; the",
            "  AW-Net aponeurosis masks (Michard et al., ACM BCB 2021) are never used.",
            "  Licence: CC BY 4.0 per the UMUD organisers' external-data catalogue and the J. Imaging article; the CEA page",
            "  states none. Used only as training targets (path ridge + orientation); no FALLMUD image or mask is",
            "  redistributed except a few small training-target visualisations in viz/ of this private kernel output.",
            f"  Kept after the overlap guard: {(fm_stats or {}).get('kept')} images; held out of training (K0c): "
            f"{(fm_stats or {}).get('groups', {}).get('n_holdout')} (fallmud/groups.csv).",
            "  Text files shipped inside the zip (copied verbatim to fallmud/zip_*):",
        ]
        for name, txt in (d.get("text_files") or {}).items():
            lines += [f"  --- {name} ---", *["    " + ln for ln in txt.splitlines()]]
    (WORK / "CREDITS.txt").write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- LF-26 embedded payload + readout
PAYLOAD: dict = {}
_RO: dict = {}  # per-process readout state (banks opened lazily after fork)
STAGE_SEC: dict = {}
INPUT_COVERAGE: dict = {}  # LF-26 r1: payload okeys found among the inference images, per set (before training)
# LF-26 r1 port tolerance (PREREG_v26_r1.json "G_impl_port"; float noise across cv2 / numpy / CPU platforms is allowed,
# a port bug is not): per compared column, median |d| <= PORT_MEDIAN_MAX, mean |d| <= PORT_MEAN_MAX, share of images
# with |d| > PORT_BIG_MM <= PORT_BIG_SHARE_MAX, NaN-pattern mismatches <= PORT_NAN_SHARE_MAX of the compared images.
PORT_MEDIAN_MAX, PORT_MEAN_MAX, PORT_BIG_MM, PORT_BIG_SHARE_MAX, PORT_NAN_SHARE_MAX = 1e-4, 0.02, 0.1, 0.01, 0.005


def load_embedded() -> dict:
    """Embedded modules (lf_readout.py + geometry_v2.py) and the payload (production geometry exported on the laptop)."""
    sys.path.insert(0, str(SRC))
    import geometry_v2  # noqa: F401  (the embedded copy of src/umud/geometry_v2.py)
    import lf_readout as lf
    pay = SRC / "payload"
    out = {"summary": json.loads((pay / "payload_summary.json").read_text()) if (pay / "payload_summary.json").exists() else {}}
    import gzip
    for p in sorted(pay.glob("*.json.gz")):
        with gzip.open(p, "rt") as fh:
            out[p.name[:-8]] = json.load(fh)
    ref = pay / "l0_ref.json"
    out["l0_ref"] = json.loads(ref.read_text()) if ref.exists() else {}
    log(f"embedded payload: {[k for k in out if k != 'summary']} complete={out['summary'].get('complete')} "
        f"readout={lf.VERSION} geometry_v2={hashlib.sha256((SRC / 'geometry_v2.py').read_bytes()).hexdigest()[:16]}")
    if not SMOKE and not out["summary"].get("complete"):
        raise RuntimeError("payload incomplete: re-run export_payload.py (4 chunks + --summary) and build_driver.py")
    return out


def _orient(bank, ds: str, model: str, key: str, with_path: bool):
    import geometry_v2 as g2
    seg, c, s = (bank.raw(ds, f"{model}_{ch}", key) for ch in ("seg", "cos2", "sin2"))
    if seg is None or c is None or s is None:
        return None, None
    path = bank.raw(ds, f"{model}_path", key) if with_path else None
    if with_path and path is None:
        return None, None
    return (g2.as_prob(seg), g2.decode_orient(c), g2.decode_orient(s)), (g2.as_prob(path) if with_path else None)


def _readout_job(task):
    """One image of one readout run (spawned worker: no fork after torch ops in the parent; banks opened once per
    process; the embedded modules are imported from SRC)."""
    run, ds, rec, v1_root = task
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
        cv2.setNumThreads(1)
    import geometry_v2 as g2
    import lf_readout as lf
    if "banks" not in _RO:
        _RO["banks"] = {"L1": g2.MaskBank(MASKBANK), "v1": g2.MaskBank(v1_root)}
    t = time.time()
    row = dict(run=run, set=ds, image_id=rec["image_id"], okey=rec["okey"], status=rec["status"])
    paths = []
    if rec["status"] != "ok":
        return row, paths
    try:
        geom = lf.geom_from_state(rec["geom"])
        if run in ("L0port", "L0portexp"):
            orient, extra = _orient(_RO["banks"]["v1"], ds, V1_MODEL, rec["okey"], False)
        else:
            orient, extra = _orient(_RO["banks"]["L1"], ds, MODEL_NAME, rec["okey"], True)
        if orient is None:
            row["status"] = "missing_maps"
            return row, paths
        if orient[1].shape != (geom["H"], geom["W"]):
            row["status"] = f"shape_mismatch:{orient[1].shape}"
            return row, paths
        r, paths = lf.readout_row(geom, orient, extra, keep_paths=(run == "L1prod"))
        row.update(r)
        row.update(mt_px=geom["mt_px"], mm=geom["mm"], vx0=geom["vx0"], vx1=geom["vx1"], W=geom["W"], H=geom["H"])
    except Exception as exc:  # noqa: BLE001
        row["status"] = f"readout_error:{type(exc).__name__}:{exc}"
    row["sec"] = round(time.time() - t, 3)
    return row, [dict(image_id=rec["image_id"], set=ds, **{k: v for k, v in p.items()}) for p in paths]


def run_readout(v1_bank_root: Path) -> dict:
    """L1prod / L1expvl / L0port / L0portexp readouts -> readout/images_<run>.csv (+ paths_L1prod.csv). The port runs
    (v1 maskbank, support seg) cover every payload image in every mode (LF-26 r1: the cross-platform float noise is
    measured on the full set by the cpusmoke, before any GPU time); READOUT_LIMIT applies to the L1 runs only."""
    t0 = time.time()
    runs = {"L1prod": [(ds, r) for ds in ("test", "static", "dynamic", "vl") for r in PAYLOAD.get(f"geom_{ds}", [])],
            "L1expvl": [("vl", r) for r in PAYLOAD.get("geom_vl_expert", [])],
            "L0port": [(ds, r) for ds in ("test", "static", "dynamic", "vl") for r in PAYLOAD.get(f"geom_{ds}", [])],
            "L0portexp": [("vl", r) for r in PAYLOAD.get("geom_vl_expert", [])]}
    info = {}
    READOUT_DIR.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    for run, recs in runs.items():
        if READOUT_LIMIT and run.startswith("L1"):  # smoke: the first READOUT_LIMIT payload records of EVERY set
            cnt: dict = {}
            sel = []
            for ds, r in recs:
                if cnt.get(ds, 0) < READOUT_LIMIT:
                    sel.append((ds, r))
                    cnt[ds] = cnt.get(ds, 0) + 1
            recs = sel
        t = time.time()
        tasks = [(run, ds, r, str(v1_bank_root)) for ds, r in recs]
        rows, prows = [], []
        with ctx.Pool(READOUT_PROCS) as pool:
            for row, pr in pool.imap(_readout_job, tasks, chunksize=4):
                rows.append(row)
                prows.extend(pr)
        cols = sorted({k for r in rows for k in r})
        with open(READOUT_DIR / f"images_{run}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        if prows:
            pc = sorted({k for r in prows for k in r})
            with open(READOUT_DIR / f"paths_{run}.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=pc)
                w.writeheader()
                w.writerows(prows)
        ok = [r for r in rows if r.get("status") == "ok"]
        fin = [r for r in ok if isinstance(r.get("fl_vwmed"), float) and math.isfinite(r["fl_vwmed"])]
        info[run] = dict(n=len(rows), n_ok=len(ok), n_finite=len(fin), n_paths=len(prows), sec=round(time.time() - t, 1),
                         status_counts={s: sum(r.get("status") == s for r in rows) for s in sorted({r.get("status") for r in rows})})
        log(f"readout {run}: {json.dumps(info[run])}")
    STAGE_SEC["readout"] = round(time.time() - t0, 1)
    return info


def _fnum(v) -> float:
    try:
        return float(v) if v not in (None, "", "nan", "None") else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def port_stats(pairs: list[tuple[float, float]]) -> dict:
    """LF-26 r1 port statistic of (kernel, laptop) value pairs (PREREG_v26_r1.json G_impl_port)."""
    n = len(pairs)
    nanm = sum(math.isnan(a) != math.isnan(b) for a, b in pairs)
    d = np.array([abs(a - b) for a, b in pairs if not (math.isnan(a) or math.isnan(b))], float)
    out = dict(n=n, n_finite_both=int(len(d)), nan_mismatch=int(nanm),
               median_abs=float(np.median(d)) if len(d) else None, mean_abs=float(d.mean()) if len(d) else None,
               max_abs=float(d.max()) if len(d) else None,
               share_gt_big=float((d > PORT_BIG_MM).mean()) if len(d) else None,
               share_le_1e6=float((d <= 1e-6).mean()) if len(d) else None,
               exact_1e6=bool(len(d) and nanm == 0 and float(d.max()) <= 1e-6))
    out["passed"] = bool(n > 0 and len(d) > 0 and out["median_abs"] <= PORT_MEDIAN_MAX
                         and out["mean_abs"] <= PORT_MEAN_MAX and out["share_gt_big"] <= PORT_BIG_SHARE_MAX
                         and nanm <= PORT_NAN_SHARE_MAX * n)
    return out


def port_check() -> dict:
    """In-kernel port check of L0port (production geometry) and L0portexp (VL expert geometry) against the embedded
    laptop L0 values (payload/l0_ref.json: {"prod": {"set|image_id": [fl_vwmed, fl_fixed]}, "expert": {...}}),
    LF-26 r1 tolerance (float noise allowed; the 1e-6 exactness statistic is reported as information)."""
    ref = PAYLOAD.get("l0_ref") or {}
    if not ref:
        return {"status": "no_reference_in_payload", "passed": False}
    out = {"tolerance": dict(median_max=PORT_MEDIAN_MAX, mean_max=PORT_MEAN_MAX, big_mm=PORT_BIG_MM,
                             big_share_max=PORT_BIG_SHARE_MAX, nan_share_max=PORT_NAN_SHARE_MAX)}
    ok = True
    for run, part in (("L0port", "prod"), ("L0portexp", "expert")):
        p = READOUT_DIR / f"images_{run}.csv"
        rf = ref.get(part) or {}
        if not p.exists() or not rf:
            out[run] = {"status": "no_readout" if not p.exists() else "no_reference"}
            ok = False
            continue
        with open(p) as fh:
            rows = list(csv.DictReader(fh))
        got = {f"{r['set']}|{r['image_id']}": r for r in rows}
        keys = sorted(rf)
        res = {"n_reference": len(keys), "n_readout": len(rows), "missing_in_readout": [k for k in keys if k not in got][:20],
               "status_counts": {s: sum(r.get("status") == s for r in rows) for s in sorted({r.get("status") for r in rows})}}
        for j, col in enumerate(("fl_vwmed", "fl_fixed")):
            pairs = [(_fnum(got[k].get(col)) if k in got else float("nan"), _fnum(rf[k][j])) for k in keys]
            res[col] = port_stats(pairs)
            ok &= res[col]["passed"]
        out[run] = res
    out["passed"] = bool(ok)
    (METRIC_DIR / "port_check.json").write_text(json.dumps(out, indent=2))
    log(f"port check (in-kernel, vs embedded laptop L0): passed={out['passed']} "
        f"{json.dumps({r: {c: out[r].get(c, {}).get('max_abs') for c in ('fl_vwmed', 'fl_fixed')} for r in ('L0port', 'L0portexp')})}")
    return out


def vl_field_samples(v1_bank_root: Path) -> dict:
    """K0d: L1 and v1 maps sampled (bilinear) at the payload points; the angles are computed on the laptop."""
    import geometry_v2 as g2
    pts = PAYLOAD.get("k0d_points_vl") or {}
    if not pts:
        return {"status": "no_points"}
    b1, bv = g2.MaskBank(MASKBANK), g2.MaskBank(v1_bank_root)
    rows, n_miss = [], 0

    def bil(a, x, y):
        H, W = a.shape
        gx = np.clip(x - 0.5, 0, W - 1.001); gy = np.clip(y - 0.5, 0, H - 1.001)
        j0 = gx.astype(int); i0 = gy.astype(int); fx = gx - j0; fy = gy - i0
        return (a[i0, j0] * (1 - fx) * (1 - fy) + a[i0, j0 + 1] * fx * (1 - fy) + a[i0 + 1, j0] * (1 - fx) * fy
                + a[i0 + 1, j0 + 1] * fx * fy)
    for k, (okey, lst) in enumerate(pts.items()):
        if READOUT_LIMIT and k >= READOUT_LIMIT:
            break
        o1, q1 = _orient(b1, "vl", MODEL_NAME, okey, True)
        ov, _ = _orient(bv, "vl", V1_MODEL, okey, False)
        if o1 is None or ov is None:
            n_miss += 1
            continue
        P = np.array([[p[1], p[2]] for p in lst], float)
        vals = {f"L1_{n}": bil(m, P[:, 0], P[:, 1]) for n, m in zip(("seg", "cos2", "sin2"), o1)}
        vals["L1_path"] = bil(q1, P[:, 0], P[:, 1])
        vals.update({f"v1_{n}": bil(m, P[:, 0], P[:, 1]) for n, m in zip(("seg", "cos2", "sin2"), ov)})
        for i, p in enumerate(lst):
            rows.append(dict(okey=okey, pid=p[0], x=p[1], y=p[2], **{c: float(v[i]) for c, v in vals.items()}))
    b1.close(); bv.close()
    if rows:
        with open(READOUT_DIR / "vl_field_samples.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    out = dict(n_points=len(rows), n_images=len(pts), n_missing=n_miss)
    log(f"VL field samples (K0d): {json.dumps(out)}")
    return out


def fm_holdout_eval(models, torch, dev) -> dict:
    """K0c: path channel (fold ensemble + hflip, prob >= 0.5, skeletonised) vs the held-out FALLMUD full-path
    skeleton at crop resolution; recall / precision at K0C_TOL_PX, pooled over the held-out frames."""
    from scipy import ndimage
    n_gt = n_gt_hit = n_pr = n_pr_hit = 0
    info_thr = {t: dict(g=0, pr=0, ph=0) for t in K0C_INFO_THR}  # LF-26 r1 information only (other thresholds)
    per = []
    for j, f in enumerate(FM_HOLD):
        crop = f["crop"]
        h, w = crop.shape
        enc, _ = predict_native(models, crop, (0, 0, w, h), torch, dev)
        prob = enc["path"].astype(np.float32) / 255.0
        pr = skeletonize_mask(prob >= 0.5)
        gt = np.zeros((h, w), bool)
        q = np.round(f["skel"]).astype(int)
        okq = (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)
        gt[q[okq, 1], q[okq, 0]] = True
        if not gt.any():
            continue
        d_pr = ndimage.distance_transform_edt(~pr) if pr.any() else np.full((h, w), 1e9)
        d_gt = ndimage.distance_transform_edt(~gt)
        g_hit = int((d_pr[gt] <= K0C_TOL_PX).sum())
        p_hit = int((d_gt[pr] <= K0C_TOL_PX).sum())
        n_gt += int(gt.sum()); n_gt_hit += g_hit; n_pr += int(pr.sum()); n_pr_hit += p_hit
        row = [f["key"], int(gt.sum()), g_hit, int(pr.sum()), p_hit]
        for t in K0C_INFO_THR:
            pr_t = skeletonize_mask(prob >= t)
            d_t = ndimage.distance_transform_edt(~pr_t) if pr_t.any() else np.full((h, w), 1e9)
            g_t, p_t = int((d_t[gt] <= K0C_TOL_PX).sum()), int((d_gt[pr_t] <= K0C_TOL_PX).sum())
            info_thr[t]["g"] += g_t
            info_thr[t]["pr"] += int(pr_t.sum())
            info_thr[t]["ph"] += p_t
            row += [g_t, int(pr_t.sum()), p_t]
        per.append(row)
        if j < 3:
            save_viz(VIZ_DIR / f"k0c_fallmud_{f['key']}.jpg", crop, enc)
    rec = n_gt_hit / max(1, n_gt)
    prec = n_pr_hit / max(1, n_pr)
    info = {}
    for t, v in info_thr.items():
        info[f"thr_{t}"] = dict(recall_3px=v["g"] / max(1, n_gt), precision_3px=v["ph"] / max(1, v["pr"]))
    with open(METRIC_DIR / "k0c_fallmud_holdout.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "gt_px", "gt_hit", "pred_px", "pred_hit",
                    *[f"{c}_t{t}" for t in K0C_INFO_THR for c in ("gt_hit", "pred_px", "pred_hit")]])
        w.writerows(per)
    return dict(n_images=len(per), recall_3px=rec, precision_3px=prec, recall_min=K0C_RECALL, precision_min=K0C_PREC,
                threshold=0.5, info_other_thresholds=info,
                passed=bool(len(per) > 0 and rec >= K0C_RECALL and prec >= K0C_PREC))


def write_manifest(extra: dict) -> None:
    files = []
    for sub in ("readout", "metrics", "logs", "viz", "fallmud"):
        d = WORK / sub
        if d.exists():
            for p in sorted(d.rglob("*")):
                if p.is_file():
                    files.append(p)
    files += [p for p in (WORK / f"{MODEL_NAME}_config.json", WORK / "CREDITS.txt") if p.exists()]
    man = dict(model=MODEL_NAME, mode=MODE, written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               elapsed_min=round((time.time() - T0) / 60, 1), stage_sec=STAGE_SEC,
               files={str(p.relative_to(WORK)): dict(bytes=p.stat().st_size,
                                                    sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files},
               kept_on_kaggle={sub: sum(1 for _ in (WORK / sub).glob("*")) for sub in ("maskbank", "checkpoints")
                               if (WORK / sub).exists()},
               **extra)
    hist = {}
    for p in sorted(METRIC_DIR.glob("history_fold*.csv")):
        with open(p) as fh:
            hist[p.stem] = list(csv.DictReader(fh))
    man["training_curves"] = hist
    (WORK / "MANIFEST.json").write_text(json.dumps(man, indent=1, default=str))


# ----------------------------------------------------------------------------- main
def main() -> None:
    global ITEMS, FMS, FM_HOLD, _LOG_FILE, PAYLOAD
    for d in (MASKBANK, CKPT_DIR, LOG_DIR, METRIC_DIR, VIZ_DIR, FM_DIR, READOUT_DIR, TMP / "ext"):
        d.mkdir(parents=True, exist_ok=True)
    _LOG_FILE = LOG_DIR / "main.log"
    log(f"MODE={MODE} MODEL_NAME={MODEL_NAME} V1_INIT={V1_INIT} MAX_EPOCHS={MAX_EPOCHS} python={sys.version.split()[0]} "
        f"cpu={os.cpu_count()}")
    log("nvidia-smi: " + sh("nvidia-smi -L").strip().replace("\n", " | "))
    log("mem: " + sh("free -g | head -2").strip().replace("\n", " | "))
    t_stage = time.time()
    PAYLOAD = load_embedded()
    import torch  # import only: no torch ops in the parent before the workers are forked

    smp = ensure_smp()
    import timm

    log(f"torch={torch.__version__} smp={smp.__version__} timm={timm.__version__} cv2={cv2.__version__} "
        f"numpy={np.__version__}")
    try:
        import skimage

        log(f"skimage={skimage.__version__}")
    except Exception as exc:  # noqa: BLE001
        log(f"skimage unavailable ({exc!r}): morphological skeleton fallback")

    # ---- v1 reference (attached kernel output): folds.csv (authoritative), fold checkpoints, maskbank (L0port)
    v1_root = find_model_source(V1_MODEL)
    log(f"v1 source: {v1_root}")
    if v1_root is None:
        raise RuntimeError("fasc_orient_v1 kernel output not attached (kernel_sources)")
    folds_csv = v1_root / "metrics" / "folds.csv"
    sha = hashlib.sha256(folds_csv.read_bytes()).hexdigest()
    log(f"v1 folds.csv sha256={sha} (expected {V1_FOLDS_SHA256})")
    if sha != V1_FOLDS_SHA256:
        raise RuntimeError(f"v1 folds.csv differs from the PREREG baseline run (sha256 {sha})")
    v1_rep, v1_alias = load_v1_folds(folds_csv)
    for k in range(N_FOLDS):
        cp = v1_root / "checkpoints" / f"fold{k}.pt"
        if cp.exists():
            V1_CKPTS[k] = cp
    log(f"v1 fold checkpoints: {sorted(V1_CKPTS)}")
    v1_bank = v1_root / "maskbank"
    want = (PAYLOAD.get("summary") or {}).get("v1_bank_sha256") or {}
    v1_sha = {}
    for name, h in want.items():
        p = v1_bank / name
        v1_sha[name] = dict(expected=h, got=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None)
    log(f"v1 maskbank vs laptop copy: {json.dumps({k: v['got'] == v['expected'] for k, v in v1_sha.items()})}")
    bad_v1 = sorted(k for k, v in v1_sha.items() if v["got"] != v["expected"])
    if bad_v1 or not want:  # LF-26 r1: fail fast (before any training) on a v1 bank that is not the laptop's copy
        raise RuntimeError(f"v1 maskbank differs from the laptop copy used for L0 / the payload: {bad_v1 or 'no shas'}")

    image_dir = find_dir("fasc_images_new_model_v1")
    mask_dir = find_dir("fasc_masks_new_model_v1")
    test_dir = find_dir("test_set_v2")
    pairs = find_pairs(image_dir, mask_dir)
    log(f"found {len(pairs)} fascicle pairs in {image_dir} / {mask_dir}; test dir {test_dir}")
    items, alias_rows, stats = prepare_items(pairs, v1_rep, v1_alias)
    log("data stats: " + json.dumps(stats))
    with open(METRIC_DIR / "folds.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stem", "representative", "group", "fold"])
        w.writerows(alias_rows)
    (METRIC_DIR / "data_stats.json").write_text(json.dumps(stats, indent=2))

    # ---- test + OSF images (inference datasets; reference set of the FALLMUD overlap guard)
    datasets: dict[str, list[tuple[str, Path, str]]] = {}
    test_files = sorted([p for p in test_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES], key=lambda p: p.name)
    datasets["test"] = [(p.stem, p, p.name) for p in test_files]
    log(f"test images: {len(test_files)}")
    for name in ("static", "dynamic", "vl"):
        try:
            datasets[name] = gather_extra(name)
        except Exception:  # noqa: BLE001
            log(f"extra {name} failed:\n{traceback.format_exc()}")
    # LF-26 r1: every payload image must be present BEFORE training (an OSF download failure used to be swallowed
    # and only showed up after ~2 h as missing_maps); full and cpusmoke modes raise, the GPU smoke mode only logs.
    want_keys = {ds: {r["okey"] for r in PAYLOAD.get(f"geom_{ds}", [])} for ds in ("test", "static", "dynamic", "vl")}
    have_keys = {ds: {e[0] for e in (datasets.get(ds) or [])} for ds in want_keys}
    missing_in = {ds: sorted(want_keys[ds] - have_keys[ds]) for ds in want_keys}
    INPUT_COVERAGE.update({ds: dict(payload=len(want_keys[ds]), found=len(want_keys[ds] & have_keys[ds]),
                                    missing=missing_in[ds][:20]) for ds in want_keys})
    log(f"input coverage (payload okeys found among the inference images): {json.dumps(INPUT_COVERAGE)}")
    if (not SMOKE or CPU_SMOKE) and (any(missing_in.values()) or not all(want_keys.values())):
        raise RuntimeError(f"payload images missing from the inference datasets: "
                           f"{ {ds: len(v) for ds, v in missing_in.items()} } (empty payload sets: "
                           f"{[ds for ds, v in want_keys.items() if not v]})")
    fm_stats = None
    if USE_FALLMUD:
        refs = reference_thumbs(datasets, items)
        kept, fm_stats = prepare_fallmud(refs)
        del refs
        FMS = [f for f in kept if not f["holdout"]]
        FM_HOLD = [f for f in kept if f["holdout"]]
        log(f"FALLMUD: {len(FMS)} training images, {len(FM_HOLD)} held out (K0c) "
            f"({sum(f['crop'].nbytes for f in kept) / 2**20:.0f} MB)")
    for it in items:
        for k in ("thumb", "thumb_full"):
            it.pop(k, None)
    ITEMS = items
    write_credits(fm_stats)
    STAGE_SEC["prep"] = round(time.time() - t_stage, 1)
    # ---- training: v1 fold weights (no pretrained download needed); workers forked before any torch op here
    t_stage = time.time()
    n_gpu = 0 if CPU_SMOKE else max(1, sh("nvidia-smi -L").count("GPU "))
    if n_gpu >= 2 and len(TRAIN_FOLDS) > 1:
        assign = [TRAIN_FOLDS[0::2], TRAIN_FOLDS[1::2]]
    else:
        assign = [TRAIN_FOLDS]
    plan = max(len(a) for a in assign)
    log(f"fold assignment {assign} plan_folds={plan} (n_gpu={n_gpu})")
    ctx = mp.get_context("fork")
    procs = []
    for g, folds in enumerate(assign):
        p = ctx.Process(target=gpu_worker, args=(g, folds, plan, len(assign)), daemon=False)
        p.start()
        log(f"started worker {g} pid={p.pid} folds={folds}")
        procs.append(p)
        time.sleep(5)
    for g, p in enumerate(procs):
        p.join()
        log(f"gpu worker {g} exit code {p.exitcode}")
    STAGE_SEC["train"] = round(time.time() - t_stage, 1)
    if CPU_SMOKE:  # the BN repair path on a fold whose v1 checkpoint has non-finite buffers (fold 2 or 4)
        try:
            torch.set_num_threads(4)
            k_bad = next((k for k in (2, 4) if k in V1_CKPTS), None)
            if k_bad is not None:
                m_, inf_ = init_from_v1(k_bad, [it for it in ITEMS if it["fold"] != k_bad][:N_CAL], torch,
                                        torch.device("cpu"))
                log(f"cpusmoke BN repair check fold {k_bad}: {json.dumps(inf_, default=str)}")
                (METRIC_DIR / "cpusmoke_bn_repair.json").write_text(json.dumps(inf_, indent=2, default=str))
                del m_
        except Exception:  # noqa: BLE001
            log(f"cpusmoke BN repair check FAILED:\n{traceback.format_exc()}")


    # ---- summaries + K0a / K0b
    fold_res = []
    for f in TRAIN_FOLDS:
        fp = METRIC_DIR / f"fold{f}.json"
        if fp.exists():
            fold_res.append(json.loads(fp.read_text()))
    ok_folds = [r["fold"] for r in fold_res if r.get("reload_ok")]
    summary: dict = {"folds": fold_res, "ok_folds": ok_folds, "model": MODEL_NAME, "backbone": BACKBONE,
                     "use_fallmud": USE_FALLMUD}
    ok_res = [r for r in fold_res if r.get("reload_ok") and "tta" in r]
    K0: dict = {}
    if ok_res:
        for key in ("dice50", "ang_mae", "ang_mae_pix", "med_err"):
            summary[f"mean_best_{key}"] = float(np.mean([r["best"][key] for r in ok_res]))
            summary[f"mean_tta_{key}"] = float(np.mean([r["tta"][key] for r in ok_res]))
    tta = {r["fold"]: r["tta"]["ang_mae"] for r in ok_res}
    K0["K0a"] = dict(folds=TRAIN_FOLDS, ok_folds=ok_folds,
                     nonfinite=[r["fold"] for r in fold_res if r.get("reload_nonfinite")],
                     passed=bool(len(ok_folds) == len(TRAIN_FOLDS) and all(not r.get("reload_nonfinite") for r in fold_res)))
    K0["K0b"] = dict(tta_ang_mae=tta, mean=float(np.mean(list(tta.values()))) if tta else None, max=K0B_MAX,
                     v1_kernel_tta_ang_mae=list(V1_TTA_ANG_MAE),
                     v1_reevaluated_repaired={r["fold"]: r.get("v1_ref", {}).get("tta_repaired", {}).get("ang_mae")
                                              for r in ok_res},
                     passed=bool(len(tta) == N_FOLDS and float(np.mean(list(tta.values()))) <= K0B_MAX))
    (METRIC_DIR / "cv_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log("CV summary: " + json.dumps({k: v for k, v in summary.items() if k != "folds"}, default=str))
    if not ok_folds:
        (METRIC_DIR / "K0.json").write_text(json.dumps(K0, indent=2, default=str))
        raise RuntimeError("no verified fold checkpoints: no maskbank written")

    # ---- fold-ensemble + hflip TTA inference (verified fp32 checkpoints only), 4 channels
    t_stage = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() and not CPU_SMOKE else "cpu")
    if dev.type == "cpu":
        torch.set_num_threads(4)
    models = []
    for f in ok_folds:
        cp = CKPT_DIR / f"fold{f}.pt"
        m = build_model(pretrained=False)
        st = torch.load(cp, map_location="cpu", weights_only=False)
        bad = state_is_finite(st["state_dict"], torch)
        if bad:
            log(f"skip {cp.name}: non-finite tensors {bad[:5]}")
            continue
        m.load_state_dict(st["state_dict"])
        models.append(m.to(dev).eval().to(memory_format=torch.channels_last))
    if not models:
        raise RuntimeError("no finite checkpoints for inference")
    log(f"inference with {len(models)} fold models {ok_folds} on {dev}")
    manifest_rows, vp_rows, n_ok_set = [], [], {}
    for name in ("test", "static", "dynamic", "vl"):
        entries = datasets.get(name) or []
        if want_keys.get(name) and not SMOKE:  # only the payload images (VL: 249 of the 498 files)
            entries = [e for e in entries if e[0] in want_keys[name]]
        elif SMOKE and want_keys.get(name):  # r1: exactly the images the smoke L1 readouts / K0d samples read
            first = {r["okey"] for r in PAYLOAD.get(f"geom_{name}", [])[:READOUT_LIMIT]}
            if name == "vl":
                first |= {r["okey"] for r in PAYLOAD.get("geom_vl_expert", [])[:READOUT_LIMIT]}
                first |= set(list((PAYLOAD.get("k0d_points_vl") or {}).keys())[:READOUT_LIMIT])
            entries = [e for e in entries if e[0] in first]
        else:
            entries = entries[:TEST_LIMIT] if name == "test" else entries[:EXTRA_LIMIT]
        if not entries:
            continue
        t = time.time()
        writers = {ch: NpzZip(MASKBANK / f"{name}__{MODEL_NAME}_{ch}.npz") for ch in CHANNELS}
        n_ok = 0
        for j, (key, path, src) in enumerate(entries):
            try:
                gray = read_gray(path)
                if max(gray.shape) > 4000 or min(gray.shape) < 64:
                    log(f"skip {name}/{src}: shape {gray.shape}")
                    continue
                box, kind = viewport(gray, path.suffix.lower())
                enc, _ = predict_native(models, gray, box, torch, dev)
                for ch in CHANNELS:
                    writers[ch].add(key, enc[ch])
                manifest_rows.append([name, key, src, gray.shape[0], gray.shape[1]])
                vp_rows.append([name, key, kind, *box])
                n_ok += 1
                if j < (3 if name == "test" else 1) or (name == "test" and j % 100 == 0):
                    save_viz(VIZ_DIR / f"{name}_{key}.jpg", gray, enc)
            except Exception:  # noqa: BLE001
                log(f"inference failed {name}/{src}:\n{traceback.format_exc()}")
        for w in writers.values():
            w.close()
        n_ok_set[name] = dict(n_ok=n_ok, n_entries=len(entries), n_payload=len(want_keys.get(name) or []))
        log(f"inference {name}: {n_ok}/{len(entries)} images in {time.time() - t:.0f}s")
    with open(MASKBANK / "manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "image_id", "source_filename", "height", "width"])
        w.writerows(manifest_rows)
    with open(MASKBANK / f"{MODEL_NAME}_viewports.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "image_id", "layout", "x1", "y1", "x2", "y2"])
        w.writerows(vp_rows)
    STAGE_SEC["inference"] = round(time.time() - t_stage, 1)
    t_stage = time.time()
    try:
        K0["K0c"] = fm_holdout_eval(models, torch, dev)
    except Exception:  # noqa: BLE001
        K0["K0c"] = dict(passed=False, error=traceback.format_exc()[-2000:])
    STAGE_SEC["k0c"] = round(time.time() - t_stage, 1)
    K0["K0d"] = {"computed_on": "laptop (gates.py k0d) from readout/vl_field_samples.csv"}
    # LF-26 r1 information (never gating): input / inference coverage per set and the epoch selection per fold
    # (epoch selection uses the competition ang_mae only, so a best_epoch of 1-2 means a barely trained path head).
    K0["info"] = dict(input_coverage=INPUT_COVERAGE, inference_coverage=n_ok_set,
                      fold_epochs={r["fold"]: dict(best_epoch=r.get("best_epoch"), epochs_run=r.get("epochs_run"),
                                                   n_epochs=r.get("n_epochs")) for r in fold_res})
    (METRIC_DIR / "K0.json").write_text(json.dumps(K0, indent=2, default=str))
    log(f"K0: {json.dumps({k: v.get('passed') for k, v in K0.items()})}")
    del models
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # ---- LF readout in the kernel (CPU, forked pool), K0d samples, port check
    ro = run_readout(v1_bank)
    fs = vl_field_samples(v1_bank)
    pc = port_check()
    config = {
        "model": MODEL_NAME, "mode": MODE, "backbone": BACKBONE, "encoder": ENCODER, "arch": "smp.Unet (v1) + path channel",
        "use_fallmud": USE_FALLMUD, "v1_init": V1_INIT, "channels": {"0": "seg logit", "1": "tanh cos2theta",
                                                                   "2": "tanh sin2theta", "3": "path logit"},
        "angle_convention": "theta = 0.5*atan2(sin2, cos2) in image pixel coords of the file (x right, y down)",
        "maskbank_encoding": "seg / path = prob*255; cos2/sin2 = (v+1)/2*255 of the fold x hflip mean",
        "targets": {"seg_width_px": SEG_WIDTH, "ori_radius_px": ORI_RADIUS, "pos_weight": POS_WEIGHT,
                    "loss_weights": [W_BCE, W_DICE, W_SKEL, W_ORI],
                    "path": {"sigma": PATH_SIGMA, "gamma": PATH_GAMMA, "neg_w": PATH_NEG_W, "pos_t": PATH_POS_T,
                             "w_path": W_PATH, "w_cldice": W_CLDICE, "cldice_iters": CLDICE_ITERS,
                             "comp_pos_w": COMP_PATH_POS_W, "comp_sparsity": COMP_PATH_SPARSITY},
                    "fallmud_orientation": {"idw_r": FM_IDW_R, "idw_min_r": FM_IDW_MIN_R},
                    "fallmud_fov": {"p": FM_FOV_P, "range": FM_FOV_RANGE}},
        "fallmud": {"per_comp": FM_PER_COMP, "ori_weight": FM_ORI_W, "dup_corr": FM_DUP_CORR, "n_train": len(FMS),
                    "n_holdout": len(FM_HOLD), "holdout_frac": FM_HOLDOUT_FRAC, "break_corr": FM_BREAK_CORR,
                    "stats": fm_stats} if USE_FALLMUD else None,
        "train": {"batch": BATCH, "lr_enc": LR_ENC, "lr_dec": LR_DEC, "wd": WEIGHT_DECAY, "warmup_epochs": WARMUP_EPOCHS,
                  "min_epochs": MIN_EPOCHS, "max_epochs": MAX_EPOCHS, "train_cap_h": TRAIN_CAP_S / 3600,
                  "folds": TRAIN_FOLDS, "ok_folds": ok_folds, "folds_source": "fasc_orient_v1 metrics/folds.csv",
                  "seed": SEED, "checkpoint_dtype": "float32"},
        "payload": PAYLOAD.get("summary", {}).get("files"), "v1_bank_check": v1_sha, "input_coverage": INPUT_COVERAGE,
        "readout": ro, "vl_field_samples": fs, "port_check": pc, "K0": {k: v.get("passed") for k, v in K0.items()},
        "data_stats": stats, "elapsed_min": (time.time() - T0) / 60, "stage_sec": STAGE_SEC,
    }
    (WORK / f"{MODEL_NAME}_config.json").write_text(json.dumps(config, indent=2, default=str))
    write_manifest(dict(K0={k: v.get("passed") for k, v in K0.items()}, readout=ro, port_check=pc))
    shutil.rmtree(TMP, ignore_errors=True)
    log(f"DONE elapsed={(time.time() - T0) / 60:.1f} min")


if __name__ == "__main__":
    main()
