"""UMUD fasc_orient_v3: the fasc_orient_v1 fascicle model with (a) an optional DINOv2 ViT-S/14 backbone and (b) optional
FALLMUD orientation supervision. Pre-registered experiment v21 track A (artifacts/candidates_v21/PREREG_v21.json).

Everything the v21 G1 gate compares is kept from fasc_orient_v1 (kaggle/fasc_orient_v1/train.py): the 3 output
channels and their activations, targets, input letterbox, augmentation, the 5 folds (read from v1's folds.csv),
the epoch-selection rule, evaluate(), the hflip TTA, the metric names and the MASKBANK format:
  ch0  fascicle fragment logit (targets: GT fragments rendered 3 px wide at 512)
  ch1  tanh -> cos(2 theta)   } supervised on pixels within 8 px of a GT fragment,
  ch2  tanh -> sin(2 theta)   } theta = that fragment's PCA angle (nearest fragment wins)
theta is measured in image pixel coordinates (x right, y down) of the image file.

Switches (top-level constants; kaggle/fasc_orient_v3/make_variant.py rewrites them):
  BACKBONE = "convnext"  smp U-Net with the ConvNeXt-S fb_in22k_ft_in1k encoder, exactly as v1.
           = "dinov2"    timm vit_small_patch14_dinov2.lvd142m (DINOv2, Apache-2.0). The 512 letterbox input is resized
                         to 518 (37x37 patches of 14 px) inside forward. DPT-lite decoder: blocks 2/5/8/11 (final-norm
                         applied) are reassembled to 148/74/37/19 px maps (x4 / x2 / x1 / x0.5 of the patch grid),
                         projected to 128 channels and fused coarse-to-fine with RefineNet residual units; the fused
                         148 px map is resized to 128 px (1/4 of 512) and merged with a shallow CNN stem on the 512
                         input (stride 1/2/4 features, 32/64/96 ch) in a U-Net style decoder back to 512, so that 3 px
                         fascicle lines survive. Layer-wise LR (blocks 3e-5 x 0.8^depth), decoder 6e-4.
  USE_FALLMUD            FALLMUD (812 lower-leg ultrasound images, CEA; Cunningham + Cronin folders) as extra
                         ORIENTATION supervision only: FALLMUD fascicle masks are full extrapolated fascicle paths (not
                         our visible fragments), so ch0 losses (bce/dice/skel) are masked out for FALLMUD samples and
                         the cos2/sin2 targets come from the local direction (skeleton + local PCA -> short straight
                         pieces) of the paths, within ORI_RADIUS. FALLMUD is train-only (never in a validation fold)
                         and every FALLMUD image that is a near-duplicate (48x48 thumbnail correlation >= 0.90, crop or
                         full frame, or identical pixels) of any competition train / test or OSF static / dynamic / VL
                         image is dropped. Mixed into every epoch at FM_PER_COMP FALLMUD samples per competition sample.
  MODE = "full" | "smoke" (GPU, tiny) | "cpusmoke" (CPU only: full data prep, fold check, FALLMUD parse, overlap guard,
                         tiny training, maskbank writing).

Also (V1_REF_EVAL): before training fold k, the attached v1 fold-k checkpoint is evaluated with the same evaluate()
and TTA on the same validation cache, (i) as v1 evaluated it (fp16 checkpoint, incl. the fp16-overflowed BatchNorm
running_var of v1 folds 2 and 4) and (ii) with only the non-finite BatchNorm layers re-estimated on training crops.
Checkpoints of this kernel are saved in float32 and every one is reloaded from disk, checked for inf/nan and
re-validated before any map is written.

Outputs (/kaggle/working): maskbank/{test,static,dynamic,vl,train_oof}__<MODEL_NAME>_{seg,cos2,sin2}.npz (v1 encoding),
manifest.csv, <MODEL_NAME>_viewports.csv, checkpoints/fold*.pt (fp32), metrics/ (history_fold*.csv, fold*.json,
cv_summary.json, folds.csv, data_stats.json), fallmud/ (zip listing, pairs, overlap guard), logs/, viz/,
<MODEL_NAME>_config.json, CREDITS.txt.
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
MODE = "full"  # "full" | "smoke" (GPU, tiny subsets) | "cpusmoke" (CPU only, tiny; kernel id *-cpusmoke)
BACKBONE = "dinov2"  # "convnext" (v1 encoder, smp U-Net exactly as v1) | "dinov2" (ViT-S/14 + DPT-lite + CNN stem)
USE_FALLMUD = True  # FALLMUD orientation-only supervision (train-only)
MODEL_NAME = "fasc_orient_v3dfm"
V1_REF_EVAL = True  # evaluate the attached v1 fold checkpoints on the same validation caches (as-is + BN-repaired)
SMOKE = MODE != "full"
CPU_SMOKE = MODE == "cpusmoke"
T0 = time.time()

SEED = 20261021  # same as v1
SIZE = 512
N_FOLDS = 5
TRAIN_FOLDS = [0] if CPU_SMOKE else ([0, 1] if SMOKE else [0, 1, 2, 3, 4])
SMOKE_TRAIN, SMOKE_VAL, SMOKE_REPEAT, SMOKE_FM = (6, 4, 1, 4) if CPU_SMOKE else (30, 10, 4, 12)
MAX_EPOCHS = 1 if CPU_SMOKE else (2 if SMOKE else 30)
MIN_EPOCHS = 1 if SMOKE else 8
MAX_STEPS = 3 if CPU_SMOKE else None  # per epoch (cpusmoke only)
TRAIN_BUDGET_S = 6.3 * 3600  # all training (incl. preprocessing) must end by T0 + this (kernel target <= 8 h)
HARD_TRAIN_END_S = 6.8 * 3600  # absolute stop for any epoch loop
BATCH = 2 if CPU_SMOKE else 8
LR_ENC, LR_DEC, WEIGHT_DECAY = 2e-4, 6e-4, 1e-2  # v1 (LR_ENC: ConvNeXt encoder only)
WARMUP_EPOCHS, LR_FLOOR = 1.0, 0.02
POS_WEIGHT, W_BCE, W_DICE, W_SKEL, W_ORI = 3.0, 1.0, 1.0, 0.5, 1.0
SEG_WIDTH, ORI_RADIUS = 3, 8.0
DUP_CORR, THUMB = 0.95, 48
VERIFY_TOL_ANG, VERIFY_TOL_DICE = 0.05, 0.01  # reloaded checkpoint must reproduce the saved best metrics
N_CAL = 8 if CPU_SMOKE else 160  # training crops for the BN re-estimate of the v1 reference checkpoints
ENCODER = "tu-convnext_small.fb_in22k_ft_in1k"
SMP_VERSION = "0.5.0"
# DINOv2 arm
DINO_NAME = "vit_small_patch14_dinov2.lvd142m"
DINO_RES = 518  # 37 x 14
DINO_LAYERS = (2, 5, 8, 11)
DINO_LR, DINO_LAYER_DECAY, DINO_DROP_PATH = 3e-5, 0.8, 0.1
DPT_CH, DPT_F = (96, 192, 384, 384), 128
STEM_CH, DEC_CH = (32, 64, 96), (128, 64, 32)
# FALLMUD
FALLMUD_URL = "https://kalisteo.cea.fr/index.php/download/fallmud_dataset/?wpdmdl=3570"
FALLMUD_PAGE = "https://kalisteo.cea.fr/index.php/fallmud/"
FALLMUD_AIA = "http://crt.harica.gr/HARICA-GEANT-TLS-R1.cer"  # intermediate the CEA server does not send
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
SEG_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# fasc_orient_v1 reference (attached via kernel_sources)
V1_MODEL = "fasc_orient_v1"
V1_FOLDS_SHA256 = "9cf568447913a47d6ae713dd443100a6250c9c921574a279a7e20dc16cfe93d3"
V1_TTA_ANG_MAE = (1.397, 1.428, 1.678, 1.466, 1.664)  # PREREG_v21 baseline (kernel OOF TTA), mean 1.5267
G1_MEAN_MAX, G1_MIN_FOLDS_BELOW = 1.427, 4

COMP_SLUG = "umud-challenge-muscle-architecture-in-ultrasound-data"
INPUT_ROOT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
TMP = Path("/tmp/fo3_tmp")
MASKBANK = WORK / "maskbank"
CKPT_DIR = WORK / "checkpoints"
LOG_DIR = WORK / "logs"
METRIC_DIR = WORK / "metrics"
VIZ_DIR = WORK / "viz"
FM_DIR = WORK / "fallmud"
OOF_PART_DIR = TMP / "oof_parts"
EXTRA_SOURCES = {
    "static": [("https://osf.io/download/hb5xj/", "static.zip")],
    "dynamic": [("https://osf.io/download/vqb7h/", "dynamic.zip")],
    "vl": [
        ("https://osf.io/download/zcuwk/", "vl_young.zip"),
        ("https://osf.io/download/6ga94/", "vl_old.zip"),
    ],
}
CHANNELS = ("seg", "cos2", "sin2")

ITEMS: list[dict] = []  # competition items; filled in main(), inherited by forked GPU workers
FMS: list[dict] = []  # FALLMUD items kept by the overlap guard (train-only)
V1_CKPTS: dict[int, Path] = {}
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


def make_sample(crop: np.ndarray, segs: np.ndarray, rng: np.random.Generator | None) -> dict:
    h, w = crop.shape
    x0, y0, ww, hh = 0, 0, w, h
    if rng is not None:
        if rng.random() < 0.35:  # narrow field of view (e.g. Philips Lumify)
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
    tg = render_targets(segs_out, valid)
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


def encode_channels(seg: np.ndarray, cos2: np.ndarray, sin2: np.ndarray) -> dict:
    return {
        "seg": np.clip(np.round(seg * 255.0), 0, 255).astype(np.uint8),
        "cos2": np.clip(np.round((cos2 + 1.0) * 127.5), 0, 255).astype(np.uint8),
        "sin2": np.clip(np.round((sin2 + 1.0) * 127.5), 0, 255).astype(np.uint8),
    }


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
        out.update(box=box, layout=kind, coverage=coverage, crop=crop, segs=segs.astype(np.float32),
                   thumb=thumbnail(crop), ch=sha1_array(crop), n_pieces=int(len(segs)), skel_px=int(len(pts)), **st)
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
    rgb = cv2.cvtColor(s["img"], cv2.COLOR_GRAY2BGR)
    rgb[..., 2] = np.where(s["seg"] > 0, 255, rgb[..., 2])
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


def build_v1_model(pretrained: bool = False):
    """The v1 network (smp U-Net, ConvNeXt-S encoder): BACKBONE='convnext' and the v1 reference checkpoints."""
    import segmentation_models_pytorch as smp

    return smp.Unet(
        encoder_name=ENCODER,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=3,
        decoder_channels=(256, 128, 64, 32, 16),
    )


_DINO_CLS = None


def dino_class():
    """DINOv2 ViT-S/14 encoder + DPT-lite decoder + full-resolution CNN stem (defined lazily: torch only in workers)."""
    global _DINO_CLS
    if _DINO_CLS is not None:
        return _DINO_CLS
    import timm
    import torch
    import torch.nn.functional as F
    from torch import nn

    def cbr(cin: int, cout: int, stride: int = 1):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def up_to(t, ref):
        return F.interpolate(t, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    class RCU(nn.Module):  # DPT / RefineNet residual conv unit
        def __init__(self, c: int) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(c, c, 3, 1, 1)
            self.conv2 = nn.Conv2d(c, c, 3, 1, 1)

        def forward(self, x):
            return x + self.conv2(F.relu(self.conv1(F.relu(x))))

    class Fuse(nn.Module):  # DPT feature fusion block (upsampling done on entry)
        def __init__(self, c: int, with_skip: bool) -> None:
            super().__init__()
            self.res1 = RCU(c) if with_skip else None
            self.res2 = RCU(c)
            self.proj = nn.Conv2d(c, c, 1)

        def forward(self, x, skip=None):
            if skip is not None:
                x = up_to(x, skip) + self.res1(skip)
            return self.proj(self.res2(x))

    class DinoDPT(nn.Module):
        def __init__(self, pretrained: bool) -> None:
            super().__init__()
            self.vit = timm.create_model(DINO_NAME, pretrained=pretrained, img_size=DINO_RES, dynamic_img_size=True,
                                         num_classes=0, drop_path_rate=DINO_DROP_PATH)
            d = self.vit.embed_dim
            c1, c2, c3, c4 = DPT_CH
            self.reassemble = nn.ModuleList([
                nn.Sequential(nn.Conv2d(d, c1, 1), nn.ConvTranspose2d(c1, c1, 4, 4)),  # 37 -> 148
                nn.Sequential(nn.Conv2d(d, c2, 1), nn.ConvTranspose2d(c2, c2, 2, 2)),  # 37 -> 74
                nn.Sequential(nn.Conv2d(d, c3, 1)),  # 37
                nn.Sequential(nn.Conv2d(d, c4, 1), nn.Conv2d(c4, c4, 3, 2, 1)),  # 37 -> 19
            ])
            self.layer_rn = nn.ModuleList([nn.Conv2d(c, DPT_F, 3, 1, 1, bias=False) for c in DPT_CH])
            self.fuse = nn.ModuleList([Fuse(DPT_F, True), Fuse(DPT_F, True), Fuse(DPT_F, True), Fuse(DPT_F, False)])
            s1, s2, s4 = STEM_CH
            self.stem1 = nn.Sequential(cbr(3, s1), cbr(s1, s1))  # 512
            self.stem2 = nn.Sequential(cbr(s1, s2, 2), cbr(s2, s2))  # 256
            self.stem4 = nn.Sequential(cbr(s2, s4, 2), cbr(s4, s4))  # 128
            d4, d2, d1 = DEC_CH
            self.dec4 = nn.Sequential(cbr(DPT_F + s4, d4), cbr(d4, d4))
            self.dec2 = nn.Sequential(cbr(d4 + s2, d2), cbr(d2, d2))
            self.dec1 = nn.Sequential(cbr(d2 + s1, d1), cbr(d1, d1))
            self.head = nn.Conv2d(d1, 3, 3, 1, 1)

        def vit_features(self, x):
            xv = F.interpolate(x, size=(DINO_RES, DINO_RES), mode="bilinear", align_corners=False)
            if hasattr(self.vit, "forward_intermediates"):
                return self.vit.forward_intermediates(xv, indices=list(DINO_LAYERS), norm=True, output_fmt="NCHW",
                                                      intermediates_only=True)
            return list(self.vit.get_intermediate_layers(xv, n=list(DINO_LAYERS), reshape=True, norm=True))

        def forward(self, x):
            feats = self.vit_features(x)
            r = [rn(rs(f)) for f, rs, rn in zip(feats, self.reassemble, self.layer_rn)]
            p = self.fuse[3](r[3])
            p = self.fuse[2](p, r[2])
            p = self.fuse[1](p, r[1])
            p = self.fuse[0](p, r[0])  # 148 px (1/3.5 of 518)
            s1 = self.stem1(x)
            s2 = self.stem2(s1)
            s4 = self.stem4(s2)
            d = self.dec4(torch.cat([up_to(p, s4), s4], 1))  # 128 px = 1/4 of 512 (same field of view)
            d = self.dec2(torch.cat([up_to(d, s2), s2], 1))
            d = self.dec1(torch.cat([up_to(d, s1), s1], 1))
            return self.head(d)

    _DINO_CLS = DinoDPT
    return DinoDPT


def build_model(pretrained: bool):
    if BACKBONE == "convnext":
        return build_v1_model(pretrained)
    if BACKBONE == "dinov2":
        return dino_class()(pretrained)
    raise ValueError(f"unknown BACKBONE {BACKBONE!r}")


def param_groups(model) -> tuple[list[dict], list[str]]:
    """v1: encoder LR_ENC / decoder LR_DEC (one AdamW weight decay). DINOv2: layer-wise LR decay inside the ViT
    (final norm DINO_LR, block i DINO_LR * decay^(12 - i), embeddings decay^13; no weight decay on 1-d params and
    tokens), decoder + stem LR_DEC."""
    if BACKBONE == "convnext":
        enc_params = list(model.encoder.parameters())
        enc_ids = {id(p) for p in enc_params}
        dec_params = [p for p in model.parameters() if id(p) not in enc_ids]
        return ([{"params": enc_params, "lr": LR_ENC}, {"params": dec_params, "lr": LR_DEC}],
                [f"encoder lr={LR_ENC} n={len(enc_params)}", f"decoder lr={LR_DEC} n={len(dec_params)}"])
    nb = len(model.vit.blocks)
    groups: dict[tuple, dict] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("vit."):
            sub = name[4:]
            if sub.startswith("blocks."):
                lid = int(sub.split(".")[1]) + 1
            elif sub.startswith(("cls_token", "pos_embed", "reg_token", "patch_embed", "mask_token")):
                lid = 0
            else:
                lid = nb + 1
            lr = DINO_LR * DINO_LAYER_DECAY ** (nb + 1 - lid)
            wd = 0.0 if (p.ndim <= 1 or sub in ("cls_token", "pos_embed", "reg_token", "mask_token")) else WEIGHT_DECAY
            key = ("vit", lid, wd)
        else:
            lr, wd, key = LR_DEC, WEIGHT_DECAY, ("dec", 0, WEIGHT_DECAY)
        g = groups.setdefault(key, {"params": [], "lr": lr, "weight_decay": wd})
        g["params"].append(p)
    desc = [f"{k[0]}{k[1]} lr={g['lr']:.2e} wd={g['weight_decay']} n={len(g['params'])}" for k, g in sorted(groups.items())]
    return list(groups.values()), desc


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
        seed = (torch.initial_seed() + idx * 7919) % (2**32)  # v1
        rng = np.random.default_rng(seed) if self.augment else None
        s = make_sample(it["crop"], it["segs"], rng)
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
    """Fold-ensemble + hflip TTA maps at native image resolution (uint8 MASKBANK encoding), v1."""
    x1, y1, x2, y2 = box
    crop = gray[y1:y2, x1:x2]
    inp, m = make_input(crop)
    x = to_input(torch.from_numpy(inp[None]).to(dev), torch)
    acc = None
    with torch.no_grad():
        for model in models:
            p, c, s = forward_maps(model, x, torch, True)
            cur = torch.stack([p[0], c[0], s[0]], -1)
            acc = cur if acc is None else acc + cur
    maps = (acc / len(models)).cpu().numpy().astype(np.float32)
    back = unletterbox(maps, m, x2 - x1, y2 - y1)
    full = np.zeros((gray.shape[0], gray.shape[1], 3), np.float32)
    full[y1:y2, x1:x2] = back
    return encode_channels(full[..., 0], full[..., 1], full[..., 2]), maps


def save_viz(path: Path, gray: np.ndarray, enc: dict) -> None:
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
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
                                make_sample(f_["crop"], f_["segs"], np.random.default_rng(j)))
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

    model = build_model(pretrained=True).to(dev).to(memory_format=torch.channels_last)
    groups, gdesc = param_groups(model)
    opt = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"fold {fold}: model {BACKBONE} {n_params / 1e6:.1f}M params; param groups: {'; '.join(gdesc)}")
    steps = len(dl) if MAX_STEPS is None else min(len(dl), MAX_STEPS)
    warm = max(1, int(steps * WARMUP_EPOCHS))
    n_epochs = gpu_state.get("n_epochs", MAX_EPOCHS)
    if "ep_time" in gpu_state:  # later folds on this GPU: re-check the remaining budget
        folds_left = max(1, gpu_state["plan_folds"] - gpu_state["done"])
        n_fit = int((T0 + TRAIN_BUDGET_S - time.time()) / folds_left / (gpu_state["ep_time"] * 1.03))
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
    cols = ["epoch", "loss", "bce", "dice_l", "skel_l", "ori_l", "ori_comp", "ori_fm", "dice50", "dice_best",
            "thr_best", "ang_mae", "ang_mae_pix", "med_err", "lr", "sec", "n_nonfinite", "n_fm_seen"]
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
        sums = np.zeros(7)
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
            loss = W_BCE * bce + W_DICE * dice_l + W_SKEL * skel_l + W_ORI * ori_l
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
                nb_fm += 1
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
        # v1 logs the encoder LR (group 0); DINOv2: the decoder LR (largest group)
        lr_now = opt.param_groups[0]["lr"] if BACKBONE == "convnext" else max(g["lr"] for g in opt.param_groups)
        sums[:6] /= max(1, nb)
        sums[6] = sums[6] / nb_fm if nb_fm else float("nan")
        n_bad_total += n_bad
        log(f"EPOCH fold={fold} ep={epoch}/{n_epochs} loss={sums[0]:.4f} bce={sums[1]:.4f} dice_l={sums[2]:.4f} "
            f"skel_l={sums[3]:.4f} ori_l={sums[4]:.4f} ori_comp={sums[5]:.4f} ori_fm={sums[6]:.4f} | val "
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
                        "encoder": ENCODER if BACKBONE == "convnext" else DINO_NAME, "size": SIZE,
                        "classes": CHANNELS, "model": MODEL_NAME, "dtype": "float32"}, ckpt_path)
            log(f"fold {fold}: saved best epoch {epoch} ang_mae={met['ang_mae']:.3f} dice50={met['dice50']:.4f}")
        if epoch == 1 and "n_epochs" not in gpu_state and not SMOKE:
            ep_time = sec
            left = T0 + TRAIN_BUDGET_S - time.time() + ep_time  # budget incl. this epoch
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
           "v1_ref": v1ref, "v1_kernel_tta_ang_mae": V1_TTA_ANG_MAE[fold]}
    if not ok:
        (METRIC_DIR / f"fold{fold}.json").write_text(json.dumps(res, indent=2, default=str))
        log(f"fold {fold}: checkpoint unusable -> no OOF maps, excluded from the ensemble")
        return
    met_tta = evaluate(vmodel, cache, torch, dev, tta=True)
    res["tta"] = met_tta
    log(f"fold {fold}: TTA val dice50={met_tta['dice50']:.4f} ang_mae={met_tta['ang_mae']:.3f} "
        f"med_err={met_tta['med_err']:.3f} | v1 kernel {V1_TTA_ANG_MAE[fold]:.3f} | v1 re-evaluated as-is "
        f"{v1ref.get('tta_asis', {}).get('ang_mae', float('nan')):.3f} repaired "
        f"{v1ref.get('tta_repaired', {}).get('ang_mae', float('nan')):.3f}")
    writers = {ch: NpzZip(OOF_PART_DIR / f"fold{fold}_{ch}.npz", level=6) for ch in CHANNELS}
    t_oof = time.time()
    rows = []
    for j, it in enumerate(va):
        gray = np.zeros((it["H"], it["W"]), np.uint8)
        x1, y1, x2, y2 = it["box"]
        gray[y1:y2, x1:x2] = it["crop"]
        enc, _ = predict_native([vmodel], gray, it["box"], torch, dev)
        for ch in CHANNELS:
            writers[ch].add(it["stem"], enc[ch])
        rows.append(["train_oof", it["stem"], it["name"], it["H"], it["W"]])
        if j < 3:
            save_viz(VIZ_DIR / f"oof_fold{fold}_{it['stem']}.jpg", gray, enc)
    for w in writers.values():
        w.close()
    with open(OOF_PART_DIR / f"fold{fold}_manifest.csv", "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    res["oof_seconds"] = time.time() - t_oof
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
    for f in kept:
        for k in ("thumb", "thumb_full"):
            f.pop(k, None)
    stats = {"download": info, "index": istats, "role_counts": role_counts, "prepared_ok": len(good),
             "prepared_failed": len(bad), "per_author": pstats, "overlap_guard": gstats, "kept": len(kept),
             "kept_mb": float(sum(f["crop"].nbytes for f in kept) / 2**20)}
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
        f"{MODEL_NAME} (UMUD fasc_orient_v3) -- credits and licences of external material used by this kernel",
        "",
        "Competition data: UMUD Challenge (Kaggle competition data, competition rules apply).",
        "OSF benchmark / example images (static hb5xj, dynamic vqb7h, VL zcuwk + 6ga94): inference / overlap guard only.",
        "Viewport layout rules adapted from AmbrosM, 'UMUD Quick and Dirty' (Kaggle notebook), Apache-2.0.",
    ]
    if BACKBONE == "dinov2":
        lines += [
            f"Backbone: DINOv2 ViT-S/14 ({DINO_NAME}) weights by Meta AI (Oquab et al. 2023, 'DINOv2: Learning Robust",
            "Visual Features without Supervision'), Apache-2.0; loaded through timm (Ross Wightman, Apache-2.0) from the",
            "Hugging Face hub (timm/vit_small_patch14_dinov2.lvd142m). Decoder follows DPT (Ranftl et al. 2021), own code.",
        ]
    else:
        lines += [f"Backbone: {ENCODER} (timm, Apache-2.0) via segmentation_models_pytorch {SMP_VERSION} (MIT)."]
    if USE_FALLMUD:
        d = (fm_stats or {}).get("download", {})
        lines += [
            "",
            "FALLMUD (FAscicle Lower Leg Muscle Ultrasound Dataset), 812 images, CEA List:",
            f"  source page: {FALLMUD_PAGE}",
            f"  downloaded from: {d.get('url')} (TLS: {d.get('ssl')})",
            f"  zip sha256: {d.get('sha256')}  bytes: {d.get('bytes')}",
            "  Paper: H. Michard, B. Luvison, Q.-C. Pham, A. J. Morales-Artacho, G. Guilhem, 'AW-Net: automatic muscle",
            "  structure analysis on B-mode ultrasound images for injury prevention', ACM BCB 2021.",
            "  Combines the datasets of R. Cunningham et al., 'Estimating Full Regional Skeletal Muscle Fibre Orientation",
            "  from B-Mode Ultrasound Images Using Convolutional, Residual, and Deconvolutional Neural Networks' (J. Imaging",
            "  2018) and N. J. Cronin et al., 'Automated Analysis of Musculoskeletal Ultrasound Images Using Deep Learning'",
            "  (2019/2020), with complementary annotations.",
            "  Licence: the UMUD organisers' external-data catalogue lists FALLMUD as CC BY 4.0; the CEA download page",
            "  states no licence. Used here only as training data (orientation targets); no FALLMUD image or mask is",
            "  redistributed except a few small training-target visualisations in viz/ of this private kernel output.",
            f"  Kept after the overlap guard: {(fm_stats or {}).get('kept')} images (see fallmud/stats.json).",
            "  Text files shipped inside the zip (copied verbatim to fallmud/zip_*):",
        ]
        for name, txt in (d.get("text_files") or {}).items():
            lines += [f"  --- {name} ---", *["    " + ln for ln in txt.splitlines()]]
    (WORK / "CREDITS.txt").write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- main
def main() -> None:
    global ITEMS, FMS, _LOG_FILE
    for d in (MASKBANK, CKPT_DIR, LOG_DIR, METRIC_DIR, VIZ_DIR, FM_DIR, OOF_PART_DIR, TMP / "ext"):
        d.mkdir(parents=True, exist_ok=True)
    _LOG_FILE = LOG_DIR / "main.log"
    log(f"MODE={MODE} BACKBONE={BACKBONE} USE_FALLMUD={USE_FALLMUD} MODEL_NAME={MODEL_NAME} V1_REF_EVAL={V1_REF_EVAL} "
        f"python={sys.version.split()[0]} cpu={os.cpu_count()}")
    log("nvidia-smi: " + sh("nvidia-smi -L").strip().replace("\n", " | "))
    log("mem: " + sh("free -g | head -2").strip().replace("\n", " | "))
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

    # ---- v1 reference (attached kernel output): folds.csv (authoritative), cv_summary, fold checkpoints
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
    v1_sum_path = v1_root / "metrics" / "cv_summary.json"
    if v1_sum_path.exists():
        v1_sum = json.loads(v1_sum_path.read_text())
        v1_f = [round(r["tta"]["ang_mae"], 3) for r in v1_sum.get("folds", [])]
        log(f"v1 cv_summary mean_tta_ang_mae={v1_sum.get('mean_tta_ang_mae')} folds={v1_f} (PREREG {V1_TTA_ANG_MAE})")
    for k in range(N_FOLDS):
        cp = v1_root / "checkpoints" / f"fold{k}.pt"
        if cp.exists():
            V1_CKPTS[k] = cp
    log(f"v1 fold checkpoints: {sorted(V1_CKPTS)}")

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
    fm_stats = None
    if USE_FALLMUD:
        refs = reference_thumbs(datasets, items)
        FMS, fm_stats = prepare_fallmud(refs)
        del refs
        log(f"FALLMUD kept for training: {len(FMS)} images ({sum(f['crop'].nbytes for f in FMS) / 2**20:.0f} MB)")
    for it in items:
        for k in ("thumb", "thumb_full"):
            it.pop(k, None)
    ITEMS = items
    write_credits(fm_stats)

    # ---- pretrained weights into the local cache (isolated interpreter: no torch ops in the parent before fork)
    t = time.time()
    rc, lines = run_isolated(PRETRAIN_CODE, [BACKBONE, ENCODER if BACKBONE == "convnext" else DINO_NAME, str(DINO_RES)])
    for line in lines:
        log(line)
    if rc != 0:
        raise RuntimeError("pretrained weights could not be loaded")
    log(f"pretrained weights cached ({time.time() - t:.0f}s)")

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

    # ---- summaries (v1 keys + v3 extras)
    fold_res = []
    for f in TRAIN_FOLDS:
        fp = METRIC_DIR / f"fold{f}.json"
        if fp.exists():
            fold_res.append(json.loads(fp.read_text()))
    ok_folds = [r["fold"] for r in fold_res if r.get("reload_ok")]
    summary: dict = {"folds": fold_res, "ok_folds": ok_folds, "model": MODEL_NAME, "backbone": BACKBONE,
                     "use_fallmud": USE_FALLMUD}
    ok_res = [r for r in fold_res if r.get("reload_ok") and "tta" in r]
    if ok_res:
        for key in ("dice50", "ang_mae", "ang_mae_pix", "med_err"):
            summary[f"mean_best_{key}"] = float(np.mean([r["best"][key] for r in ok_res]))
            summary[f"mean_tta_{key}"] = float(np.mean([r["tta"][key] for r in ok_res]))
        tta = {r["fold"]: r["tta"]["ang_mae"] for r in ok_res}
        v1k = {f: V1_TTA_ANG_MAE[f] for f in tta}
        g1 = {"folds_done": sorted(tta), "tta_ang_mae": tta, "v1_kernel_tta_ang_mae": v1k,
              "mean": float(np.mean(list(tta.values()))), "mean_max": G1_MEAN_MAX,
              "n_folds_below_v1": int(sum(tta[f] < v1k[f] for f in tta)), "min_folds_below": G1_MIN_FOLDS_BELOW}
        g1["pass"] = bool(len(tta) == N_FOLDS and g1["mean"] <= G1_MEAN_MAX and g1["n_folds_below_v1"] >= G1_MIN_FOLDS_BELOW)
        rep = {r["fold"]: r["v1_ref"]["tta_repaired"]["ang_mae"] for r in ok_res
               if r.get("v1_ref", {}).get("available") and "tta_repaired" in r["v1_ref"]}
        asis = {r["fold"]: r["v1_ref"]["tta_asis"]["ang_mae"] for r in ok_res
                if r.get("v1_ref", {}).get("available") and "tta_asis" in r["v1_ref"]}
        g1["info_v1_reevaluated_asis"] = asis
        g1["info_v1_bn_repaired"] = rep
        if rep:
            g1["info_mean_v1_bn_repaired"] = float(np.mean(list(rep.values())))
            g1["info_n_folds_below_v1_bn_repaired"] = int(sum(tta[f] < rep[f] for f in rep))
        summary["prereg_G1"] = g1
    (METRIC_DIR / "cv_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log("CV summary: " + json.dumps({k: v for k, v in summary.items() if k != "folds"}, default=str))
    if not ok_folds:
        raise RuntimeError("no verified fold checkpoints: no maskbank written")

    # ---- merge OOF parts
    manifest_rows: list[list] = []
    jobs = [([str(OOF_PART_DIR / f"fold{f}_{ch}.npz") for f in ok_folds], str(MASKBANK / f"train_oof__{MODEL_NAME}_{ch}.npz"))
            for ch in CHANNELS]
    t = time.time()
    with ctx.Pool(3) as pool:
        for dest, n in pool.imap_unordered(_merge_job, jobs):
            log(f"merged {n} OOF maps -> {Path(dest).name}")
    for f in ok_folds:
        mp_path = OOF_PART_DIR / f"fold{f}_manifest.csv"
        if mp_path.exists():
            with open(mp_path) as fh:
                manifest_rows += list(csv.reader(fh))
    log(f"OOF merge {time.time() - t:.0f}s")

    # ---- fold-ensemble + hflip TTA inference (verified fp32 checkpoints only)
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
    vp_rows = []
    for name in ("test", "static", "dynamic", "vl"):
        entries = datasets.get(name) or []
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
                if j < (4 if name == "test" else 2) or (name == "test" and j % 60 == 0):
                    save_viz(VIZ_DIR / f"{name}_{key}.jpg", gray, enc)
            except Exception:  # noqa: BLE001
                log(f"inference failed {name}/{src}:\n{traceback.format_exc()}")
        for w in writers.values():
            w.close()
        log(f"inference {name}: {n_ok}/{len(entries)} images in {time.time() - t:.0f}s")

    with open(MASKBANK / "manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "image_id", "source_filename", "height", "width"])
        w.writerows(manifest_rows)
    with open(MASKBANK / f"{MODEL_NAME}_viewports.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "image_id", "layout", "x1", "y1", "x2", "y2"])
        w.writerows(vp_rows)
    config = {
        "model": MODEL_NAME, "mode": MODE, "backbone": BACKBONE, "use_fallmud": USE_FALLMUD,
        "encoder": ENCODER if BACKBONE == "convnext" else DINO_NAME,
        "arch": "smp.Unet (v1)" if BACKBONE == "convnext" else
                f"DINOv2 ViT-S/14 @ {DINO_RES} (blocks {list(DINO_LAYERS)}, final norm) + DPT-lite (reassemble "
                f"{list(DPT_CH)} -> {DPT_F}, RefineNet fusion to 148 px) + CNN stem {list(STEM_CH)} (stride 1/2/4) + "
                f"U-Net decoder {list(DEC_CH)} to 512",
        "smp": smp.__version__, "timm": timm.__version__, "torch": torch.__version__, "size": SIZE,
        "channels": {"0": "seg logit", "1": "tanh cos2theta", "2": "tanh sin2theta"},
        "angle_convention": "theta = 0.5*atan2(sin2, cos2) in image pixel coords of the file (x right, y down); "
                            "rising-to-the-right fascicles have theta < 0",
        "maskbank_encoding": "seg = prob*255; cos2/sin2 = (v+1)/2*255 of the fold x hflip mean (vector NOT renormalised: "
                             "|v| < 1 means low agreement); outside the viewport seg=0, cos2=sin2=128",
        "input": "viewport crop -> aspect-preserving letterbox 512 (black pad), gray x3, ImageNet norm"
                 + (f"; resized to {DINO_RES} for the ViT inside forward" if BACKBONE == "dinov2" else ""),
        "targets": {"seg_width_px": SEG_WIDTH, "ori_radius_px": ORI_RADIUS, "pos_weight": POS_WEIGHT,
                    "loss_weights": [W_BCE, W_DICE, W_SKEL, W_ORI]},
        "fallmud": {"per_comp": FM_PER_COMP, "ori_weight": FM_ORI_W, "dup_corr": FM_DUP_CORR,
                    "pieces_canvas_px": {"pca_r": FM_PCA_R, "half": FM_PIECE_HALF, "step": FM_PIECE_STEP},
                    "min_aniso": FM_MIN_ANISO, "seg_loss": "masked (orientation only)", "n_kept": len(FMS),
                    "stats": fm_stats} if USE_FALLMUD else None,
        "train": {"batch": BATCH, "lr_enc": LR_ENC, "lr_dec": LR_DEC, "wd": WEIGHT_DECAY, "max_epochs": MAX_EPOCHS,
                  "dino_lr": DINO_LR, "dino_layer_decay": DINO_LAYER_DECAY, "dino_drop_path": DINO_DROP_PATH,
                  "folds": TRAIN_FOLDS, "ok_folds": ok_folds, "folds_source": "fasc_orient_v1 metrics/folds.csv",
                  "seed": SEED, "train_budget_h": TRAIN_BUDGET_S / 3600, "checkpoint_dtype": "float32"},
        "data_stats": stats, "elapsed_min": (time.time() - T0) / 60,
    }
    (WORK / f"{MODEL_NAME}_config.json").write_text(json.dumps(config, indent=2, default=str))
    shutil.rmtree(TMP, ignore_errors=True)
    log(f"DONE elapsed={(time.time() - T0) / 60:.1f} min")


if __name__ == "__main__":
    main()
