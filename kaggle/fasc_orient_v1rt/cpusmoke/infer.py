"""UMUD fasc_orient_v1rt: inference-only rich TTA for the fasc_orient_v1 fold checkpoints (Track D, PREREG v21).

Model: the 5 fasc_orient_v1 fold checkpoints (smp U-Net, ConvNeXt-S fb_in22k_ft_in1k; ch0 fragment logit, ch1 tanh
cos(2 theta), ch2 tanh sin(2 theta); theta in image pixel coords, x right / y down), attached via kernel_sources.
No training. What is new is the test-time augmentation:

  views = scales {0.8, 1.0, 1.25} x rotations {-6, 0, +6} deg about the letterbox canvas centre, each with v1's
  hflip TTA (forward_maps, unchanged: the flipped pass is un-flipped and its sin2 sign negated).

  A view is the letterboxed viewport crop warped once by M_v = A_v @ L (L = v1 letterbox matrix, A_v the view
  similarity), exactly how v1's training augmentation built its inputs. The view's output maps are warped back with
  the exact inverse affine (cv2 WARP_INVERSE_MAP), to the 512 base canvas (OOF metrics, v1 evaluate() frame) or to
  the native crop (maskbank), and the doubled angle is corrected for the view rotation phi (theta -> theta - phi,
  i.e. (cos2, sin2) rotated by -2 phi). seg probabilities and (cos2, sin2) vectors are averaged over the views,
  each pixel weighted only by the views whose valid (non-padded) region covers it (back-warped valid mask);
  pixels covered by no view keep the identity view's value, so the identity-only combo reproduces v1 exactly.

The angle correction is verified before any model runs: GT fragment targets (v1 render_targets) rendered on every
transformed view (and its hflip) are mapped back with the production code and compared with the targets rendered on
the untransformed canvas (rotation_check.json; must be < 0.2 deg everywhere, with a wrong-sign control).

Outputs (/kaggle/working):
  maskbank/{test,static,dynamic,vl,train_oof}__fasc_orient_v1rt_{seg,cos2,sin2}.npz  (uint8, native resolution, v1
  encoding, same keys / viewports as v1; train_oof = fold-k model on fold-k validation images), manifest.csv,
  fasc_orient_v1rt_viewports.csv; metrics/cv_summary.json (+ fold*.json, oof_per_image.csv): OOF metrics with v1's
  evaluate() math for the combos none / hflip (= v1 TTA, reproduction check) / scales / rots / rich (= tta);
  metrics/repro_v1_maskbank.csv (identity+hflip native maps vs v1's own maskbank), rotation_check.json, logs/, viz/.
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

# OpenCV's worker pool must never start in the parent: a forked child that touches it deadlocks (fasc_orient_v2 note).
cv2.setNumThreads(1)

# ----------------------------------------------------------------------------- config
MODE = "cpusmoke"  # "full" (GPU) | "cpusmoke" (CPU only, a few images, OOF folds 0-1; kernel id *-cpusmoke)
CPU_SMOKE = MODE == "cpusmoke"
T0 = time.time()

MODEL_NAME = "fasc_orient_v1rt"
BASE_MODEL = "fasc_orient_v1"
BASE_KERNEL_SLUG = "umud-fasc-orient-v1"
SEED = 20261021
SIZE = 512
N_FOLDS = 5
OOF_FOLDS = [0, 2] if CPU_SMOKE else [0, 1, 2, 3, 4]  # smoke: fold 2 has the non-finite BN stat
SMOKE_OOF_PER_FOLD = 3
TEST_LIMIT = 2 if CPU_SMOKE else None
EXTRA_LIMIT = 1 if CPU_SMOKE else 800  # v1 used 800 (no dataset is that large)
SCALES = (0.8, 1.0, 1.25)
ROTS_DEG = (-6.0, 0.0, 6.0)
IDENT_VIEW = (1.0, 0.0)
VIEWS = [(s, r) for s in SCALES for r in ROTS_DEG]
COMBOS = {  # every combo includes v1's hflip TTA inside each view
    "hflip": [IDENT_VIEW],
    "scales": [(s, 0.0) for s in SCALES],
    "rots": [(1.0, r) for r in ROTS_DEG],
    "rich": VIEWS,
}
MAIN_COMBO = "rich"
# Exploratory arm (not in PREREG_v21_addendum_D; found by the cpusmoke): v1's fold-2 checkpoint holds an fp16-overflowed
# (inf) BatchNorm running_var. The main model keeps the checkpoints exactly as v1 used them (inf -> that BN channel
# outputs beta; v1's own TTA metrics and maskbank were computed this way). The "fx" variant re-estimates only the
# non-finite BN channels from un-augmented canvases of that fold's TRAINING images (forward hooks, all layers in eval).
BN_FIX_ARM = True
FX_NAME = MODEL_NAME + "fx"  # rich TTA, repaired BN stats
V1FX_NAME = BASE_MODEL + "fx"  # hflip only (= v1 TTA), repaired BN stats
N_CAL = 8 if CPU_SMOKE else 256
CAL_BATCH = 4 if CPU_SMOKE else 8
VARIANTS = ("orig", "fx")
VIEW_CHUNK = 3 if CPU_SMOKE else 9  # views per forward call (x2 with hflip)
RC_N_ITEMS = 4 if CPU_SMOKE else 6
RC_TOL_DEG = 0.2
REPRO_TOL_ANG = 2e-3  # |hflip ang_mae - v1 TTA ang_mae| per fold (deg)
D1_MEAN_MAX, D1_MIN_FOLDS = 1.477, 4  # PREREG_v21_addendum_D (informational preview only)
TIME_LIMIT_S = 10.5 * 3600
SEG_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
ENCODER = "tu-convnext_small.fb_in22k_ft_in1k"
SMP_VERSION = "0.5.0"
DUP_CORR, THUMB = 0.95, 48
SEG_WIDTH, ORI_RADIUS = 3, 8.0
IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

COMP_SLUG = "umud-challenge-muscle-architecture-in-ultrasound-data"
INPUT_ROOT = Path(os.environ.get("UMUD_INPUT_ROOT", "/kaggle/input"))
WORK = Path(os.environ.get("UMUD_OUTPUT_DIR", "/kaggle/working"))
TMP = Path(os.environ.get("UMUD_TEMP_DIR", "/tmp/fo_rt_tmp"))
MASKBANK = WORK / "maskbank"
LOG_DIR = WORK / "logs"
METRIC_DIR = WORK / "metrics"
VIZ_DIR = WORK / "viz"
PART_DIR = TMP / "parts"
EXTRA_SOURCES = {
    "static": [("https://osf.io/download/hb5xj/", "static.zip")],
    "dynamic": [("https://osf.io/download/vqb7h/", "dynamic.zip")],
    "vl": [
        ("https://osf.io/download/zcuwk/", "vl_young.zip"),
        ("https://osf.io/download/6ga94/", "vl_old.zip"),
    ],
}
CHANNELS = ("seg", "cos2", "sin2")
COMBO_ORDER = ("none", "hflip", "scales", "rots", "rich")

ITEMS: list[dict] = []  # GT items (filled in main; inherited by forked workers)
UNL: list[dict] = []  # test / static / dynamic / vl images
V1_ROOT: Path | None = None
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


# ----------------------------------------------------------------------------- viewport (verbatim v1)
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


# ----------------------------------------------------------------------------- fragments (verbatim v1)
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
    )
    return out


# ----------------------------------------------------------------------------- geometry / targets (verbatim v1)
def letterbox_matrix(w: int, h: int) -> np.ndarray:
    s0 = SIZE / max(w, h)
    return np.array(
        [[s0, 0.0, (SIZE - s0 * w) / 2 + 0.5 * s0 - 0.5], [0.0, s0, (SIZE - s0 * h) / 2 + 0.5 * s0 - 0.5], [0, 0, 1]],
        np.float64,
    )


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


def transform_segs(segs: np.ndarray, m: np.ndarray) -> np.ndarray:
    """(K,4) segment end points through the 3x3 affine m (as in v1 make_sample)."""
    if not len(segs):
        return np.zeros((0, 4))
    p1 = (m[:2, :2] @ segs[:, :2].T + m[:2, 2:3]).T
    p2 = (m[:2, :2] @ segs[:, 2:].T + m[:2, 2:3]).T
    return np.concatenate([p1, p2], 1)


def make_sample(crop: np.ndarray, segs: np.ndarray) -> dict:
    """v1 make_sample(crop, segs, rng=None): the un-augmented validation sample (v1 build_val_cache)."""
    h, w = crop.shape
    m = letterbox_matrix(w, h)
    img = warp_image(crop, m)
    valid = _poly_mask(m, w, h)
    tg = render_targets(transform_segs(segs, m), valid)
    tg["img"] = img
    tg["valid"] = valid
    return tg


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


def to_full_enc(maps: np.ndarray, box, H: int, W: int) -> dict:
    """crop-frame maps -> full image, zeros outside the viewport (v1 predict_native) -> uint8 encoding."""
    x1, y1, x2, y2 = box
    full = np.zeros((H, W, 3), np.float32)
    full[y1:y2, x1:x2] = maps
    return encode_channels(full[..., 0], full[..., 1], full[..., 2])


# ----------------------------------------------------------------------------- rich TTA geometry (new)
FLIP = np.array([[-1.0, 0.0, SIZE - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # torch.flip(dims=[3]): x -> SIZE-1-x


def view_matrix(scale: float, rot_deg: float) -> np.ndarray:
    """Similarity about the canvas centre (base canvas px -> view canvas px); same composition as v1 aug_matrix."""
    c = (SIZE - 1) / 2.0
    phi = math.radians(rot_deg)
    cs, sn = math.cos(phi), math.sin(phi)
    tm = np.array([[1, 0, -c], [0, 1, -c], [0, 0, 1]], np.float64)
    rm = np.array([[cs, -sn, 0], [sn, cs, 0], [0, 0, 1]], np.float64)
    sm = np.diag([scale, scale, 1.0])
    tp = np.array([[1, 0, c], [0, 1, c], [0, 0, 1]], np.float64)
    return tp @ rm @ sm @ tm


def view_rotation(a: np.ndarray) -> float:
    """Rotation (rad) of a similarity in image coords: direction angle theta (atan2(dy, dx)) -> theta + phi."""
    return math.atan2(a[1, 0], a[0, 0])


def rotate_doubled(c: np.ndarray, s: np.ndarray, phi: float) -> tuple[np.ndarray, np.ndarray]:
    """(cos 2t, sin 2t) -> (cos 2(t - phi), sin 2(t - phi)): undo a view rotation of +phi."""
    c2, s2 = math.cos(2.0 * phi), math.sin(2.0 * phi)
    return c * c2 + s * s2, s * c2 - c * s2


def build_views(crop: np.ndarray, with_images: bool = True) -> tuple[list[dict], np.ndarray]:
    h, w = crop.shape
    L = letterbox_matrix(w, h)
    views = []
    for sc, rd in VIEWS:
        ident = (sc, rd) == IDENT_VIEW
        a = np.eye(3) if ident else view_matrix(sc, rd)
        m = L if ident else a @ L
        v = {"view": (sc, rd), "ident": ident, "A": a, "M": m, "phi": 0.0 if ident else view_rotation(a),
             "valid": _poly_mask(m, w, h)}
        if with_images:
            v["img"] = warp_image(crop, m)  # ident: identical to v1 make_input(crop)
        views.append(v)
    return views, L


def warp_weight(valid: np.ndarray, m: np.ndarray, w: int, h: int) -> np.ndarray:
    return cv2.warpAffine(valid.astype(np.float32), m[:2].astype(np.float64), (w, h),
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def back_map(maps: np.ndarray, v: dict, frame: str, out_w: int = SIZE, out_h: int = SIZE,
             phi: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """View-canvas maps (SIZE,SIZE,C; ch1/ch2 = cos2/sin2) -> output frame, + weight (back-warped valid mask).

    frame 'canvas': the base letterbox canvas (pixel q reads view pixel A q); 'native': crop pixels (x reads M x).
    The doubled angle is corrected by -2 phi (phi = view rotation; override only for the sign controls)."""
    phi = v["phi"] if phi is None else phi
    if frame == "canvas":
        if v["ident"]:
            back, wt = maps.astype(np.float32, copy=True), v["valid"].astype(np.float32)
        else:
            back, wt = unletterbox(maps, v["A"], SIZE, SIZE), warp_weight(v["valid"], v["A"], SIZE, SIZE)
    else:
        back, wt = unletterbox(maps, v["M"], out_w, out_h), warp_weight(v["valid"], v["M"], out_w, out_h)
    if phi != 0.0:
        c, s = rotate_doubled(back[..., 1].copy(), back[..., 2].copy(), phi)
        back[..., 1] = c
        back[..., 2] = s
    return back, wt


def combine(backs: dict, names: list) -> np.ndarray:
    """Valid-weighted mean of back-mapped views; pixels covered by no view keep the identity view's maps.
    A single-view combo returns that view unchanged (identity -> bit-exact v1 maps)."""
    ident = backs[IDENT_VIEW][0]
    if len(names) == 1:
        return backs[names[0]][0]
    acc = np.zeros_like(ident)
    wsum = np.zeros(ident.shape[:2], np.float32)
    for n_ in names:
        m_, wt = backs[n_]
        acc += m_ * wt[..., None]
        wsum += wt
    out = ident.copy()
    ok = wsum > 1e-6
    out[ok] = acc[ok] / wsum[ok][:, None]
    return out


def ang_diff_deg(c1, s1, c2, s2) -> np.ndarray:
    """Axial angle difference (deg) between two doubled-angle fields (same formula as v1 evaluate)."""
    t1 = 0.5 * np.arctan2(np.asarray(s1, np.float64), np.asarray(c1, np.float64))
    t2 = 0.5 * np.arctan2(np.asarray(s2, np.float64), np.asarray(c2, np.float64))
    d = np.remainder(t1 - t2, math.pi)
    return np.minimum(d, math.pi - d) * (180.0 / math.pi)


def _mean(d: np.ndarray, m: np.ndarray) -> float:
    return float(d[m].mean()) if m.any() else float("nan")


def rotation_check(items: list[dict], torch=None) -> dict:
    """GT-target round trip through every TTA view (see module docstring). numpy path per (view, flip); with torch,
    also the production path forward_maps(dummy model returning the view targets) -> back_map -> combine."""
    rows, trows = [], []
    for it in items:
        crop, segs = it["crop"], it["segs"]
        h, w = crop.shape
        views, L = build_views(crop, with_images=False)
        base = render_targets(transform_segs(segs, L), _poly_mask(L, w, h))
        bc, bs = base["cos2"].astype(np.float32), base["sin2"].astype(np.float32)
        bsup = (base["skel"] > 0) & (base["om"] > 0)  # v1 evaluate() angle pixels
        bmaps = np.stack([base["seg"].astype(np.float32), bc, bs, base["om"].astype(np.float32)], -1)
        bnat = unletterbox(bmaps, L, w, h)
        nat_ref = (bnat[..., 3] >= 0.999) & (np.hypot(bnat[..., 1], bnat[..., 2]) >= 0.99)
        fakes = {}
        for v in views:
            for flip in (False, True):
                m = FLIP @ v["M"] if flip else v["M"]
                tg = render_targets(transform_segs(segs, m), _poly_mask(m, w, h))
                maps = np.stack([tg["seg"], tg["cos2"], tg["sin2"], tg["om"]], -1).astype(np.float32)
                fakes[(v["view"], flip)] = maps  # what a perfect model would output for this (flipped) view
                un = maps
                if flip:  # numpy mirror of forward_maps' un-flip: flip W back, sin(2 theta) changes sign
                    un = maps[:, ::-1].copy()
                    un[..., 2] *= -1.0
                back, wt = back_map(un, v, "canvas")
                core = bsup & (back[..., 3] >= 0.999) & (wt >= 0.999)
                allm = bsup & (wt >= 0.999)
                d = ang_diff_deg(back[..., 1], back[..., 2], bc, bs)
                row = {"stem": it["stem"], "view": list(v["view"]), "flip": flip, "phi_deg": math.degrees(v["phi"]),
                       "n_sup": int(bsup.sum()), "n_core": int(core.sum()), "n_all": int(allm.sum()),
                       "core_deg": _mean(d, core), "all_deg": _mean(d, allm)}
                if not v["ident"] and v["phi"] != 0.0:
                    bw, _ = back_map(un, v, "canvas", phi=-v["phi"])
                    bn, _ = back_map(un, v, "canvas", phi=0.0)
                    row["wrong_sign_deg"] = _mean(ang_diff_deg(bw[..., 1], bw[..., 2], bc, bs), core)
                    row["no_corr_deg"] = _mean(ang_diff_deg(bn[..., 1], bn[..., 2], bc, bs), core)
                nb, nw = back_map(un, v, "native", w, h)
                ncore = nat_ref & (nb[..., 3] >= 0.999) & (nw >= 0.999) & (np.hypot(nb[..., 1], nb[..., 2]) >= 0.99)
                row["n_native"] = int(ncore.sum())
                row["native_deg"] = _mean(ang_diff_deg(nb[..., 1], nb[..., 2], bnat[..., 1], bnat[..., 2]), ncore)
                rows.append(row)
        if torch is None:
            continue

        class Dummy(torch.nn.Module):
            def __init__(self, out):
                super().__init__()
                self.out = out

            def forward(self, x):
                assert x.shape[0] == self.out.shape[0], (x.shape, self.out.shape)
                return self.out

        def logits(mp_: np.ndarray) -> np.ndarray:
            seg = np.clip(mp_[..., 0], 1e-4, 1 - 1e-4)
            return np.stack([np.log(seg / (1 - seg)), np.arctanh(np.clip(mp_[..., 1], -0.99999, 0.99999)),
                             np.arctanh(np.clip(mp_[..., 2], -0.99999, 0.99999))], 1).astype(np.float32)

        plain = np.stack([fakes[(v["view"], False)] for v in views])
        flipd = np.stack([fakes[(v["view"], True)] for v in views])
        out = torch.from_numpy(np.concatenate([logits(plain), logits(flipd)], 0))
        x = torch.zeros((len(views), 3, SIZE, SIZE))
        with torch.no_grad():
            p, c, s = forward_maps(Dummy(out), x, torch, True)
        vm = torch.stack([p, c, s], -1).numpy()
        backs, oms = {}, {}
        for i, v in enumerate(views):
            backs[v["view"]] = back_map(vm[i], v, "canvas")
            om_avg = 0.5 * (plain[i, ..., 3] + flipd[i, :, ::-1, 3])
            oms[v["view"]] = back_map(np.repeat(om_avg[..., None], 3, -1), v, "canvas", phi=0.0)[0][..., 0]
        for name in [*[("view", v["view"]) for v in views], *[("combo", k) for k in COMBOS]]:
            if name[0] == "view":
                mp_, wt = backs[name[1]]
                core = bsup & (oms[name[1]] >= 0.999) & (wt >= 0.999)
                allm = bsup & (wt >= 0.999)
            else:
                names = COMBOS[name[1]]
                mp_ = combine(backs, names)
                okc = np.ones_like(bsup)
                wsum = np.zeros(bsup.shape, np.float32)
                for n_ in names:
                    wt = backs[n_][1]
                    okc &= (wt < 1e-6) | ((wt >= 0.999) & (oms[n_] >= 0.999))
                    wsum += wt
                core = bsup & okc & (wsum >= 0.999)
                allm = bsup & (wsum > 1e-6)
            d = ang_diff_deg(mp_[..., 1], mp_[..., 2], bc, bs)
            seg_iou = float(((mp_[..., 0] > 0.5) & (base["seg"] > 0)).sum() / max(1, ((mp_[..., 0] > 0.5) | (base["seg"] > 0)).sum()))
            trows.append({"stem": it["stem"], "kind": name[0], "name": list(name[1]) if name[0] == "view" else name[1],
                          "n_core": int(core.sum()), "core_deg": _mean(d, core), "all_deg": _mean(d, allm),
                          "seg_iou": seg_iou})

    def wmean(rs, key, nkey):
        num = sum(r[key] * r[nkey] for r in rs if r[nkey] > 0 and math.isfinite(r[key]))
        den = sum(r[nkey] for r in rs if r[nkey] > 0 and math.isfinite(r[key]))
        return num / den if den else float("nan")

    per_view = []
    for v in VIEWS:
        for flip in (False, True):
            rs = [r for r in rows if tuple(r["view"]) == v and r["flip"] == flip]
            e = {"view": list(v), "flip": flip, "core_deg": wmean(rs, "core_deg", "n_core"),
                 "all_deg": wmean(rs, "all_deg", "n_all"), "native_deg": wmean(rs, "native_deg", "n_native"),
                 "core_frac": sum(r["n_core"] for r in rs) / max(1, sum(r["n_sup"] for r in rs))}
            if any("wrong_sign_deg" in r for r in rs):
                e["wrong_sign_deg"] = wmean(rs, "wrong_sign_deg", "n_core")
                e["no_corr_deg"] = wmean(rs, "no_corr_deg", "n_core")
            per_view.append(e)
    torch_rows = []
    for key in [("view", list(v)) for v in VIEWS] + [("combo", k) for k in COMBOS]:
        rs = [r for r in trows if r["kind"] == key[0] and r["name"] == key[1]]
        if rs:
            torch_rows.append({"kind": key[0], "name": key[1], "core_deg": wmean(rs, "core_deg", "n_core"),
                               "all_deg": float(np.nanmean([r["all_deg"] for r in rs])),
                               "seg_iou": float(np.mean([r["seg_iou"] for r in rs]))})
    worst_np = max(max(e["core_deg"], e["native_deg"]) for e in per_view)
    worst_t = max((e["core_deg"] for e in torch_rows), default=0.0)
    ctrl = [e["wrong_sign_deg"] for e in per_view if "wrong_sign_deg" in e]
    ok = (math.isfinite(worst_np) and worst_np < RC_TOL_DEG and (torch is None or worst_t < RC_TOL_DEG)
          and bool(ctrl) and min(ctrl) > 5.0)
    return {"pass": bool(ok), "tol_deg": RC_TOL_DEG, "worst_numpy_deg": worst_np,
            "worst_torch_deg": worst_t if torch is not None else None,
            "min_wrong_sign_control_deg": min(ctrl) if ctrl else None,
            "items": [it["stem"] for it in items], "per_view": per_view, "torch_path": torch_rows, "rows": rows}


# ----------------------------------------------------------------------------- npz writer (verbatim v1)
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


# ----------------------------------------------------------------------------- torch parts
def build_model(pretrained: bool):
    import segmentation_models_pytorch as smp

    return smp.Unet(
        encoder_name=ENCODER,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=3,
        decoder_channels=(256, 128, 64, 32, 16),
    )


def amp(torch, dev):
    if dev.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)  # v1
    return torch.autocast("cpu", enabled=False)


def to_input(img_u8, torch):
    x = img_u8.float().unsqueeze(1) * (1.0 / 255.0)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return ((x.expand(-1, 3, -1, -1) - mean) / std).contiguous(memory_format=torch.channels_last)


def forward_maps(model, x, torch, tta: bool):
    """Returns prob, cos2, sin2 (B,H,W) float32; hflip TTA flips the sin2 sign (verbatim v1, autocast per device)."""
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


def run_views(variants: dict, views: list[dict], torch, dev) -> dict:
    """For every variant (name -> list of models, v1 fold order): mean over its models of forward_maps(tta=True) for
    every view -> {name: (V,SIZE,SIZE,3) [p, cos2, sin2]}. A model shared by several variants runs once."""
    imgs = np.stack([v["img"] for v in views])
    out = {name: np.zeros((len(views), SIZE, SIZE, 3), np.float32) for name in variants}
    with torch.no_grad():
        for i in range(0, len(views), VIEW_CHUNK):
            x = to_input(torch.from_numpy(imgs[i:i + VIEW_CHUNK]).to(dev), torch)
            cache: dict = {}
            for name, models in variants.items():
                acc = None
                for model in models:
                    if id(model) not in cache:
                        p, c, s = forward_maps(model, x, torch, True)
                        cache[id(model)] = torch.stack([p, c, s], -1)
                    cur = cache[id(model)]
                    acc = cur if acc is None else acc + cur
                out[name][i:i + VIEW_CHUNK] = (acc / len(models)).cpu().numpy().astype(np.float32)
    return out


def metric_record(maps: np.ndarray, tg: dict, torch) -> dict:
    """Per-image terms of v1 evaluate() (same math, line by line); aggregated by aggregate_records()."""
    p = torch.from_numpy(np.ascontiguousarray(maps[..., 0]))
    c = torch.from_numpy(np.ascontiguousarray(maps[..., 1]))
    s = torch.from_numpy(np.ascontiguousarray(maps[..., 2]))
    seg_t = torch.from_numpy(tg["seg"]) > 0
    skel = torch.from_numpy(tg["skel"]) > 0
    om = torch.from_numpy(tg["om"]) > 0
    valid = torch.from_numpy(tg["valid"]) > 0
    ct = torch.from_numpy(tg["cos2"]).float()
    st = torch.from_numpy(tg["sin2"]).float()
    dice = []
    for t in SEG_THRESHOLDS:
        pred = p > t
        inter = (pred & seg_t).sum().float()
        den = pred.sum().float() + seg_t.sum().float()
        dice.append(float(torch.where(den > 0, 2 * inter / den.clamp_min(1), torch.ones_like(den))))
    rec: dict = {"dice": dice}
    th_p = 0.5 * torch.atan2(s, c)
    th_t = 0.5 * torch.atan2(st, ct)
    d = torch.remainder(th_p - th_t, math.pi)
    d = torch.minimum(d, math.pi - d) * (180.0 / math.pi)
    m = skel & om
    cnt = int(m.sum())
    if cnt < 10:
        return rec
    e = d[m]
    rec["ang"] = float(e.mean())
    rec["ang_sum"] = float(e.sum())
    rec["cnt"] = cnt
    gt_med = float(torch.median(th_t[m]))
    pr = (p > 0.5) & valid
    pm = float(torch.median(th_p[pr])) if int(pr.sum()) >= 50 else float(torch.median(th_p[m]))
    dm = abs(pm - gt_med) % math.pi
    rec["med"] = min(dm, math.pi - dm) * 180.0 / math.pi
    return rec


def aggregate_records(recs: list[dict]) -> dict:
    """v1 evaluate() return dict from per-image records."""
    n_img = len(recs)
    dices = {t: sum(r["dice"][i] for r in recs) / max(1, n_img) for i, t in enumerate(SEG_THRESHOLDS)}
    best_t = max(dices, key=dices.get)
    angs = [r for r in recs if "ang" in r]
    return {
        "dice50": dices[0.5], "dice_best": dices[best_t], "thr_best": best_t,
        "ang_mae": sum(r["ang"] for r in angs) / max(1, len(angs)),
        "ang_mae_pix": sum(r["ang_sum"] for r in angs) / max(1, sum(r["cnt"] for r in angs)),
        "med_err": sum(r["med"] for r in angs) / max(1, len(angs)), "n_val": n_img, "n_ang": len(angs),
    }


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


def compare_enc(mine: dict, ref: dict) -> dict:
    """uint8 maskbank maps vs v1's: per-channel max/mean |diff| and the angle difference on v1 fragment pixels."""
    out = {}
    for ch in CHANNELS:
        if mine[ch].shape != ref[ch].shape:
            return {"shape_mismatch": f"{mine[ch].shape} vs {ref[ch].shape}"}
        d = np.abs(mine[ch].astype(np.int16) - ref[ch].astype(np.int16))
        out[f"max_{ch}"] = int(d.max())
        out[f"mean_{ch}"] = float(d.mean())
        out[f"frac_gt1_{ch}"] = float((d > 1).mean())
    frag = ref["seg"] >= 128
    out["n_frag"] = int(frag.sum())
    if frag.any():
        a = ang_diff_deg(mine["cos2"][frag] / 127.5 - 1, mine["sin2"][frag] / 127.5 - 1,
                         ref["cos2"][frag] / 127.5 - 1, ref["sin2"][frag] / 127.5 - 1)
        out["ang_frag_mean"] = float(a.mean())
        out["ang_frag_max"] = float(a.max())
    return out


class V1Bank:
    """Lazy reader of the attached v1 maskbank (np.load reads one key at a time)."""

    def __init__(self, root: Path | None) -> None:
        self.root = None if root is None else root / "maskbank"
        self.files: dict = {}

    def get(self, ds: str, key: str) -> dict | None:
        if self.root is None:
            return None
        out = {}
        for ch in CHANNELS:
            f = self.files.get((ds, ch))
            if f is None:
                path = self.root / f"{ds}__{BASE_MODEL}_{ch}.npz"
                if not path.exists():
                    return None
                f = self.files[(ds, ch)] = np.load(path)
            if key not in f.files:
                return None
            out[ch] = f[key]
        return out


# ----------------------------------------------------------------------------- workers
def load_models(torch, dev) -> tuple[dict, dict]:
    """The 5 v1 fold checkpoints (fp16 on disk) loaded as fp32 exactly as v1 did. Non-finite BatchNorm running stats
    are kept (that is the model v1's TTA metrics and maskbank came from) and reported as {fold: {module: channels}};
    any other non-finite tensor aborts."""
    import torch.nn as nn

    models, bn_bad = {}, {}
    for k in range(N_FOLDS):
        path = V1_ROOT / "checkpoints" / f"fold{k}.pt"
        st = torch.load(path, map_location="cpu", weights_only=False)
        sd = st["state_dict"]
        m = build_model(pretrained=False)
        mods = dict(m.named_modules())
        bad, fatal = {}, []
        for n, v in sd.items():
            if not (torch.is_tensor(v) and v.is_floating_point()) or bool(torch.isfinite(v).all()):
                continue
            mod_name, attr = n.rsplit(".", 1)
            if isinstance(mods.get(mod_name), nn.modules.batchnorm._BatchNorm) and attr in ("running_mean", "running_var"):
                idx = torch.nonzero(~torch.isfinite(v)).flatten().tolist()
                bad.setdefault(mod_name, set()).update(idx)
                log(f"fold {k}: NON-FINITE {n} channels={idx[:20]}{'...' if len(idx) > 20 else ''} "
                    f"({len(idx)}/{v.numel()}) values={[float(v[j]) for j in idx[:5]]}")
            else:
                fatal.append(n)
        if fatal:
            raise RuntimeError(f"fold {k} checkpoint has non-finite non-BN-stat tensors: {fatal[:5]}")
        fin = max(float(v.float()[torch.isfinite(v)].abs().max()) for v in sd.values()
                  if torch.is_tensor(v) and v.is_floating_point() and v.numel() and bool(torch.isfinite(v).any()))
        dts = sorted({str(v.dtype) for v in sd.values() if torch.is_tensor(v)})
        m.load_state_dict({n: v.float() if v.is_floating_point() else v for n, v in sd.items()})
        models[k] = m.to(dev).eval().to(memory_format=torch.channels_last)
        bn_bad[k] = {name: sorted(ix) for name, ix in bad.items()}
        log(f"loaded fold{k}.pt epoch={st.get('epoch')} fold={st.get('fold')} encoder={st.get('encoder')} "
            f"dtypes={dts} tensors={len(sd)} nonfinite_bn={ {n_: len(ix) for n_, ix in bn_bad[k].items()} } "
            f"max|finite w|={fin:.1f}")
    return models, bn_bad


def repair_bn(model, bad: dict, fold: int, torch, dev) -> dict:
    """fx arm: replace the non-finite running stats (those channels only, mean and var) of the listed BatchNorm layers by
    the per-channel moments of the layer input over un-augmented canvases (v1 make_input) of N_CAL TRAINING images of
    this fold (other folds' items; never its validation images). All layers stay in eval mode (forward hooks only),
    fp32, so the moments are those of the activations the layer sees at inference."""
    mods = dict(model.named_modules())
    acc: dict = {name: [0, None, None] for name in bad}
    hooks = []
    for name in bad:
        def hook(mod, inp, out, name=name):
            xx = inp[0].detach().double()
            a = acc[name]
            a[0] += xx.shape[0] * xx.shape[2] * xx.shape[3]
            s1, s2 = xx.sum((0, 2, 3)), (xx * xx).sum((0, 2, 3))
            a[1] = s1 if a[1] is None else a[1] + s1
            a[2] = s2 if a[2] is None else a[2] + s2
        hooks.append(mods[name].register_forward_hook(hook))
    tr = [i for i, it in enumerate(ITEMS) if it["fold"] != fold]
    sel = np.random.default_rng(SEED + 100 + fold).choice(len(tr), min(N_CAL, len(tr)), replace=False)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(sel), CAL_BATCH):
            xs = np.stack([make_input_crop(ITEMS[tr[int(j)]]["crop"]) for j in sel[i:i + CAL_BATCH]])
            model(to_input(torch.from_numpy(xs).to(dev), torch))
    for h_ in hooks:
        h_.remove()
    info = {}
    for name, ix in bad.items():
        n, s1, s2 = acc[name]
        mean = s1 / n
        var = (s2 / n - mean * mean) * (n / max(1, n - 1))
        mod = mods[name]
        idx = torch.tensor(ix, dtype=torch.long, device=mod.running_var.device)
        old_mean = mod.running_mean[idx].tolist()
        mod.running_mean[idx] = mean[idx].to(mod.running_mean.dtype)
        mod.running_var[idx] = var[idx].to(mod.running_var.dtype)
        fin = torch.isfinite(mod.running_var)
        info[name] = {"channels": ix, "n_cal_images": int(len(sel)), "n_pixels": int(n),
                      "old_mean": old_mean[:10], "new_mean": mod.running_mean[idx].tolist()[:10],
                      "new_var": mod.running_var[idx].tolist()[:10],
                      "other_channels_var_median": float(mod.running_var[fin].median()),
                      "other_channels_var_max": float(mod.running_var[fin].max()),
                      "all_finite_after": bool(torch.isfinite(mod.running_var).all() and torch.isfinite(mod.running_mean).all())}
        log(f"fold {fold}: repaired {name} channels {ix[:10]} from {len(sel)} training canvases: "
            f"var -> {info[name]['new_var'][:5]} (other channels median {info[name]['other_channels_var_median']:.3g}, "
            f"max {info[name]['other_channels_var_max']:.3g}); mean {old_mean[:3]} -> {info[name]['new_mean'][:3]}")
    return info


def make_input_crop(crop: np.ndarray) -> np.ndarray:
    """v1 make_input(crop)[0]: the letterboxed 512 canvas."""
    h, w = crop.shape
    return warp_image(crop, letterbox_matrix(w, h))


def infer_worker(g: int, shard: list[tuple], forked: bool = True) -> int:
    global _LOG_PREFIX, _LOG_FILE
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if CPU_SMOKE else str(g)
    _LOG_PREFIX = f"[inf{g}]"
    _LOG_FILE = LOG_DIR / f"inf{g}.log"
    code = 0
    recs: list[dict] = []
    repro: list[dict] = []
    bn_info: dict = {}
    bn_bad: dict = {}
    timing: list[float] = []
    rec_path = PART_DIR / f"records{g}.json"

    def dump() -> None:
        rec_path.write_text(json.dumps({"oof": recs, "repro": repro, "bn_bad": bn_bad, "bn_fix": bn_info,
                                        "task_seconds": timing}))

    try:
        import torch

        torch.backends.cudnn.benchmark = True  # as in v1
        torch.set_num_threads(4 if CPU_SMOKE else 1)
        cv2.setNumThreads(1)
        dev = torch.device("cuda" if (torch.cuda.is_available() and not CPU_SMOKE) else "cpu")
        log(f"worker pid={os.getpid()} device={torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'} "
            f"tasks={len(shard)} torch={torch.__version__}")
        models, bn_bad = load_models(torch, dev)
        fx_models = {}
        if BN_FIX_ARM:
            for k in sorted(bn_bad):
                if bn_bad[k]:
                    fx_models[k] = copy.deepcopy(models[k])
                    bn_info[k] = repair_bn(fx_models[k], bn_bad[k], k, torch, dev)
        has_fx = bool(fx_models)
        mlist = [models[k] for k in sorted(models)]
        mlist_fx = [fx_models.get(k, models[k]) for k in sorted(models)]
        log(f"variants: orig = v1 checkpoints as-is; fx = repaired folds {sorted(fx_models)}" if has_fx
            else "no non-finite BN stats: fx variant not built")
        v1bank = V1Bank(V1_ROOT)
        writers: dict = {}
        viz_n: dict = {}
        t0 = time.time()
        for n_done, task in enumerate(shard):
            if time.time() - T0 > TIME_LIMIT_S:
                log(f"time limit: stopping after {n_done}/{len(shard)} tasks")
                break
            t_task = time.time()
            try:
                if task[0] == "oof":
                    _, ii, k = task
                    it = ITEMS[ii]
                    ds, key, crop, box, H, W = "train_oof", it["stem"], it["crop"], it["box"], it["H"], it["W"]
                    variants = {"orig": [models[k]]}
                    if k in fx_models:
                        variants["fx"] = [fx_models[k]]
                else:
                    u = UNL[task[1]]
                    ds, key, box, H, W = u["ds"], u["key"], u["box"], u["H"], u["W"]
                    gray = read_gray(u["path"])
                    if gray.shape != (H, W):
                        raise RuntimeError(f"shape changed {gray.shape} vs {(H, W)}")
                    x1, y1, x2, y2 = box
                    crop = np.ascontiguousarray(gray[y1:y2, x1:x2])
                    variants = {"orig": mlist, **({"fx": mlist_fx} if has_fx else {})}
                h, w = crop.shape
                views, _ = build_views(crop)
                vms = run_views(variants, views, torch, dev)
                for name, vm in vms.items():
                    if not np.isfinite(vm).all():
                        raise RuntimeError(f"non-finite maps in variant {name}")
                same_fx = has_fx and "fx" not in vms  # this fold's model is not repaired: fx == orig
                if task[0] == "oof":
                    tg = make_sample(crop, it["segs"])
                    if not np.array_equal(tg["img"], views[VIEWS.index(IDENT_VIEW)]["img"]):
                        raise RuntimeError("identity view input differs from v1 make_sample")
                    rr = {"fold": k, "stem": key}
                    for name, vm in vms.items():
                        backs = {v["view"]: back_map(vm[i], v, "canvas") for i, v in enumerate(views)}
                        mdl = fx_models[k] if name == "fx" else models[k]
                        with torch.no_grad():
                            x_id = to_input(torch.from_numpy(tg["img"][None]).to(dev), torch)
                            p, c, s = forward_maps(mdl, x_id, torch, False)
                            none_maps = torch.stack([p[0], c[0], s[0]], -1).cpu().numpy().astype(np.float32)
                        rr[name] = {"none": metric_record(none_maps, tg, torch)}
                        for cname, names in COMBOS.items():
                            rr[name][cname] = metric_record(combine(backs, names), tg, torch)
                    if same_fx:
                        rr["fx"] = rr["orig"]
                    recs.append(rr)
                encs: dict = {}
                for name, vm in vms.items():
                    nb = {v["view"]: back_map(vm[i], v, "native", w, h) for i, v in enumerate(views)}
                    rich = to_full_enc(combine(nb, COMBOS[MAIN_COMBO]), box, H, W)
                    hfl = to_full_enc(combine(nb, COMBOS["hflip"]), box, H, W)
                    if name == "orig":
                        encs[MODEL_NAME], enc_h = rich, hfl
                    else:
                        encs[FX_NAME], encs[V1FX_NAME] = rich, hfl
                if same_fx:
                    encs[FX_NAME], encs[V1FX_NAME] = encs[MODEL_NAME], enc_h
                for bank, enc_b in encs.items():
                    for ch in CHANNELS:
                        wk = (bank, ds, ch)
                        if wk not in writers:
                            writers[wk] = NpzZip(PART_DIR / f"inf{g}_{ds}__{bank}_{ch}.npz", level=6)
                        writers[wk].add(key, enc_b[ch])
                enc = encs[MODEL_NAME]
                ref = v1bank.get(ds, key)
                rep = {"ds": ds, "key": key, "found": ref is not None}
                if ref is not None:
                    rep.update(compare_enc(enc_h, ref))
                    rep["rich_vs_v1_ang_frag_mean"] = compare_enc(enc, ref).get("ang_frag_mean")
                    if has_fx:
                        rep["v1fx_vs_v1_ang_frag_mean"] = compare_enc(encs[V1FX_NAME], ref).get("ang_frag_mean")
                repro.append(rep)
                if viz_n.get(ds, 0) < (3 if ds != "test" else 4):
                    viz_n[ds] = viz_n.get(ds, 0) + 1
                    if task[0] == "oof":
                        gray = np.zeros((H, W), np.uint8)
                        x1, y1, x2, y2 = box
                        gray[y1:y2, x1:x2] = crop
                    save_viz(VIZ_DIR / f"{ds}_{key}.jpg", gray, enc)
            except Exception:  # noqa: BLE001
                code = 1
                log(f"task {task} FAILED:\n{traceback.format_exc()}")
            timing.append(round(time.time() - t_task, 3))
            if n_done % 100 == 0 or n_done == len(shard) - 1:
                el = time.time() - t0
                log(f"{n_done + 1}/{len(shard)} tasks ({task[0]}), {el:.0f}s ({el / (n_done + 1):.2f}s/task, "
                    f"last {timing[-1]:.2f}s)"
                    + (f" mem={torch.cuda.max_memory_allocated() / 2**30:.1f}G" if dev.type == "cuda" else ""))
                dump()
        for wr in writers.values():
            wr.close()
        dump()
        log(f"worker done in {time.time() - t0:.0f}s")
    except BaseException:  # noqa: BLE001
        code = 2
        log(f"worker FAILED:\n{traceback.format_exc()}")
        try:
            dump()
        except Exception:  # noqa: BLE001
            pass
    if forked:
        os._exit(code)
    return code


def rc_worker(idx: list[int]) -> None:
    """Rotation check incl. the torch production path, in a forked CPU-only child (no torch in the parent)."""
    global _LOG_PREFIX, _LOG_FILE
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    _LOG_PREFIX = "[rc]"
    _LOG_FILE = LOG_DIR / "main.log"
    code = 0
    try:
        import torch

        torch.set_num_threads(2)
        res = rotation_check([ITEMS[i] for i in idx], torch=torch)
        (METRIC_DIR / "rotation_check.json").write_text(json.dumps(res, indent=1))
        log_rotation_check(res)
        code = 0 if res["pass"] else 3
    except BaseException:  # noqa: BLE001
        code = 2
        log(f"rotation check FAILED:\n{traceback.format_exc()}")
    os._exit(code)


def log_rotation_check(res: dict) -> None:
    log(f"ROTATION CHECK pass={res['pass']} worst numpy={res['worst_numpy_deg']:.4f} deg "
        f"worst torch={res['worst_torch_deg']} min wrong-sign control={res['min_wrong_sign_control_deg']} "
        f"(tol {res['tol_deg']}) items={res['items']}")
    for e in res["per_view"]:
        log(f"  view scale={e['view'][0]:.2f} rot={e['view'][1]:+.0f} flip={int(e['flip'])}: core={e['core_deg']:.4f} "
            f"all={e['all_deg']:.4f} native={e['native_deg']:.4f} deg (core frac {e['core_frac']:.3f})"
            + (f" | wrong-sign={e['wrong_sign_deg']:.2f} no-corr={e['no_corr_deg']:.2f}" if "wrong_sign_deg" in e else ""))
    for e in res["torch_path"]:
        log(f"  torch {e['kind']} {e['name']}: core={e['core_deg']:.4f} all={e['all_deg']:.4f} deg seg_iou={e['seg_iou']:.3f}")


# ----------------------------------------------------------------------------- data discovery
def find_dir(name: str) -> Path:
    for base in (INPUT_ROOT / "competitions" / COMP_SLUG, INPUT_ROOT / COMP_SLUG, INPUT_ROOT):
        if base.exists():
            hits = [p for p in base.rglob(name) if p.is_dir()]
            if hits:
                return sorted(hits, key=lambda p: len(str(p)))[0]
    raise FileNotFoundError(name)


def find_model_source(model: str) -> Path | None:
    """Root of the attached kernel output (kernel_sources) that holds <model>_config.json."""
    for cand in (INPUT_ROOT / BASE_KERNEL_SLUG, INPUT_ROOT / "notebooks" / "shivateja005" / BASE_KERNEL_SLUG):
        if (cand / f"{model}_config.json").exists():
            return cand
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


def prepare_items(pairs: list[tuple[str, str]], v1_folds: dict[str, int]) -> tuple[list[dict], dict]:
    """v1 prepare_items; the fold of every item is taken from v1's folds.csv (authoritative, honest OOF)."""
    t = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(min(4, os.cpu_count() or 2)) as pool:
        results = list(pool.imap(prep_train, pairs, chunksize=8))
    log(f"preprocessed {len(results)} pairs in {time.time() - t:.0f}s")
    stats = {"pairs": len(results), "empty_masks": 0, "exact_duplicates": 0, "same_image_other_mask": 0}
    by_img: dict[str, list[dict]] = {}
    for r in results:
        by_img.setdefault(r["ih"], []).append(r)
    items = []
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
    n_match = n_diff = n_missing = 0
    for i, it in enumerate(items):
        it["group"] = int(groups[i])
        f = int(folds[i])
        wf = v1_folds.get(it["stem"])
        if wf is None:
            n_missing += 1
        elif wf == f:
            n_match += 1
        else:
            n_diff += 1
            f = wf
        it["fold"] = f
        del it["thumb"]
    _, gcounts = np.unique(groups, return_counts=True)
    fl = np.array([it["fold"] for it in items])
    stats.update(
        unique_items=len(items), n_groups=int(len(gcounts)), max_group=int(gcounts.max()),
        fold_sizes=[int((fl == f).sum()) for f in range(N_FOLDS)],
        v1_fold_match=n_match, v1_fold_diff=n_diff, v1_fold_missing=n_missing,
        segments_median=float(np.median([len(it["segs"]) for it in items])),
        split_components=int(sum(it["nsplit"] for it in items)),
        components=int(sum(it["ncomp"] for it in items)),
        lowcov_fullframe=int(sum("_lowcov" in it["layout"] for it in items)),
        layouts={k: int(v) for k, v in zip(*np.unique([it["layout"] for it in items], return_counts=True))},
        crop_mb=float(sum(it["crop"].nbytes for it in items) / 2**20),
    )
    return items, stats


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
    """Returns [(image_id, path, source_filename)] for an OSF dataset (downloaded + unzipped); verbatim v1."""
    root = TMP / "ext" / name
    root.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    for url, fname in EXTRA_SOURCES[name]:
        zpath = TMP / "ext" / fname
        if not download(url, zpath):
            continue
        sub = root / Path(fname).stem
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(sub)
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
    return out[:EXTRA_LIMIT]


def ensure_smp() -> str:
    """smp 0.5.0 as in v1, checked in a subprocess (the parent never imports torch)."""
    probe = [sys.executable, "-c", "import segmentation_models_pytorch as s, timm, torch; "
             "print(s.__version__, timm.__version__, torch.__version__)"]
    r = subprocess.run(probe, capture_output=True, text=True)
    ver = r.stdout.split()[0] if r.returncode == 0 and r.stdout.split() else None
    if ver != SMP_VERSION:
        log(f"pip install segmentation-models-pytorch=={SMP_VERSION} (found {ver})")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", f"segmentation-models-pytorch=={SMP_VERSION}"])
        r = subprocess.run(probe, capture_output=True, text=True)
    return r.stdout.strip() or r.stderr[-500:]


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


# ----------------------------------------------------------------------------- main
def main() -> None:
    global ITEMS, UNL, V1_ROOT, _LOG_FILE
    for d in (MASKBANK, LOG_DIR, METRIC_DIR, VIZ_DIR, PART_DIR, TMP / "ext"):
        d.mkdir(parents=True, exist_ok=True)
    _LOG_FILE = LOG_DIR / "main.log"
    log(f"MODE={MODE} model={MODEL_NAME} base={BASE_MODEL} views={VIEWS} combos={list(COMBOS)} "
        f"python={sys.version.split()[0]} cpu={os.cpu_count()}")
    log("nvidia-smi: " + sh("nvidia-smi -L").strip().replace("\n", " | "))
    log("mem: " + sh("free -g | head -2").strip().replace("\n", " | "))

    # ---- attached v1 kernel output: checkpoints are required
    V1_ROOT = find_model_source(BASE_MODEL)
    log(f"v1 source: {V1_ROOT}")
    if V1_ROOT is None:
        raise RuntimeError(f"{BASE_MODEL}_config.json not found under {INPUT_ROOT}: attach kernel_sources")
    for sub in ("checkpoints", "metrics", "maskbank"):
        p = V1_ROOT / sub
        log(f"v1 {sub}/: {sorted(x.name for x in p.iterdir()) if p.exists() else 'MISSING'}")
    missing = [f"fold{k}.pt" for k in range(N_FOLDS) if not (V1_ROOT / "checkpoints" / f"fold{k}.pt").exists()]
    if missing:
        raise RuntimeError(f"v1 checkpoints missing: {missing} -> stop")
    v1_folds = {r["stem"]: int(r["fold"]) for r in read_csv(V1_ROOT / "metrics" / "folds.csv")
                if r["stem"] == r["representative"]}
    v1_fold_json = {k: json.loads((V1_ROOT / "metrics" / f"fold{k}.json").read_text()) for k in range(N_FOLDS)}
    v1_manifest: dict = {}
    for r in read_csv(V1_ROOT / "maskbank" / "manifest.csv"):
        v1_manifest.setdefault(r["dataset"], {})[r["image_id"]] = r
    v1_vp = {(r["dataset"], r["image_id"]): r for r in read_csv(V1_ROOT / "maskbank" / f"{BASE_MODEL}_viewports.csv")}
    shutil.copy(V1_ROOT / "metrics" / "folds.csv", METRIC_DIR / "folds.csv")
    log(f"v1 folds.csv: {len(v1_folds)} representatives; v1 manifest: "
        f"{ {k: len(v) for k, v in v1_manifest.items()} }; v1 viewports: {len(v1_vp)}")
    versions = ensure_smp()
    log("versions smp/timm/torch: " + versions)

    # ---- GT items with v1's folds
    image_dir = find_dir("fasc_images_new_model_v1")
    mask_dir = find_dir("fasc_masks_new_model_v1")
    test_dir = find_dir("test_set_v2")
    pairs = find_pairs(image_dir, mask_dir)
    log(f"found {len(pairs)} fascicle pairs; test dir {test_dir}")
    items, stats = prepare_items(pairs, v1_folds)
    ITEMS = items
    v1_stats = json.loads((V1_ROOT / "metrics" / "data_stats.json").read_text()) \
        if (V1_ROOT / "metrics" / "data_stats.json").exists() else {}
    diffs = {k: (stats.get(k), v1_stats.get(k)) for k in v1_stats if k in stats and stats.get(k) != v1_stats.get(k)
             and k != "crop_mb"}
    log("data stats: " + json.dumps(stats))
    log(f"data stats differing from v1: {diffs if diffs else 'none'}")
    (METRIC_DIR / "data_stats.json").write_text(json.dumps(stats, indent=2))
    if stats["v1_fold_missing"] or stats["v1_fold_diff"]:
        log(f"WARNING fold mapping: {stats['v1_fold_missing']} items missing from v1 folds.csv, "
            f"{stats['v1_fold_diff']} recomputed folds differ (v1's fold used)")

    # ---- rotation check (GT targets through every view; torch production path in a CPU child)
    cand = [i for i, it in enumerate(ITEMS) if len(it["segs"]) >= 15]
    rc_idx = [cand[int(j)] for j in np.linspace(0, len(cand) - 1, RC_N_ITEMS).round().astype(int)]
    rc_idx += [i for i, it in enumerate(ITEMS) if it["layout"] == "legacy_512" and len(it["segs"]) >= 15][:1]
    rc_idx = sorted(set(rc_idx))
    ctx = mp.get_context("fork")
    t = time.time()
    p = ctx.Process(target=rc_worker, args=(rc_idx,), daemon=False)
    p.start()
    p.join()
    log(f"rotation check exit code {p.exitcode} ({time.time() - t:.0f}s)")
    if p.exitcode != 0:
        raise RuntimeError("rotation check failed: no maskbank written (see metrics/rotation_check.json)")

    # ---- test / OSF datasets: same keys and viewports as v1
    datasets: dict[str, list[tuple[str, Path, str]]] = {}
    test_files = sorted([p for p in test_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES], key=lambda p: p.name)
    datasets["test"] = [(p.stem, p, p.name) for p in test_files][:TEST_LIMIT]
    for name in ("static", "dynamic", "vl"):
        try:
            datasets[name] = gather_extra(name)
        except Exception:  # noqa: BLE001
            log(f"extra {name} failed:\n{traceback.format_exc()}")
    unl, vp_stats = [], {}
    for name, entries in datasets.items():
        st = {"n": len(entries), "v1_n": len(v1_manifest.get(name, {})), "not_in_v1": 0, "vp_mismatch": 0,
              "skipped": 0, "shape_mismatch": 0}
        keys = set()
        for key, path, src in entries:
            try:
                gray = read_gray(path)
            except Exception:  # noqa: BLE001
                log(f"read failed {name}/{src}:\n{traceback.format_exc()}")
                continue
            if max(gray.shape) > 4000 or min(gray.shape) < 64:
                log(f"skip {name}/{src}: shape {gray.shape}")
                st["skipped"] += 1
                continue
            box, kind = viewport(gray, path.suffix.lower())
            ref = v1_vp.get((name, key))
            if ref is None:
                st["not_in_v1"] += 1
            else:
                rbox = tuple(int(ref[c]) for c in ("x1", "y1", "x2", "y2"))
                if rbox != tuple(box) or ref["layout"] != kind:
                    st["vp_mismatch"] += 1
                    log(f"viewport mismatch {name}/{key}: mine {kind} {box} v1 {ref['layout']} {rbox} -> v1's used")
                    box, kind = rbox, ref["layout"]
                mref = v1_manifest.get(name, {}).get(key)
                if mref is not None and (int(mref["height"]), int(mref["width"])) != gray.shape:
                    st["shape_mismatch"] += 1
            keys.add(key)
            unl.append({"ds": name, "key": key, "path": path, "src": src, "H": gray.shape[0], "W": gray.shape[1],
                        "box": tuple(int(b) for b in box), "layout": kind})
        st["v1_not_here"] = len(set(v1_manifest.get(name, {})) - keys)
        vp_stats[name] = st
    UNL = unl
    log(f"images for inference: {len(UNL)}; key/viewport check vs v1: {json.dumps(vp_stats)}")

    # ---- tasks -> workers
    oof_tasks = []
    for k in OOF_FOLDS:
        va = [i for i, it in enumerate(ITEMS) if it["fold"] == k]
        if CPU_SMOKE:
            sel = np.random.default_rng(SEED + k).choice(len(va), SMOKE_OOF_PER_FOLD, replace=False)
            va = [va[int(j)] for j in sorted(sel)]
        oof_tasks += [("oof", i, k) for i in va]
    ens_tasks = [("ens", i) for i in range(len(UNL))]
    n_gpu = 0 if CPU_SMOKE else max(1, sh("nvidia-smi -L").count("GPU "))
    n_w = 1 if CPU_SMOKE else max(1, min(2, n_gpu))
    shards = [oof_tasks[g::n_w] + ens_tasks[g::n_w] for g in range(n_w)]
    log(f"tasks: oof={len(oof_tasks)} ens={len(ens_tasks)} workers={n_w} shard sizes={[len(s) for s in shards]}")
    t = time.time()
    procs = []
    for g in range(n_w):
        pr = ctx.Process(target=infer_worker, args=(g, shards[g]), daemon=False)
        pr.start()
        procs.append(pr)
        log(f"started worker {g} pid={pr.pid}")
    codes = []
    for g, pr in enumerate(procs):
        pr.join()
        codes.append(pr.exitcode)
        log(f"worker {g} exit code {pr.exitcode}")
    log(f"inference {time.time() - t:.0f}s")

    # ---- OOF metrics (v1 evaluate() aggregation), per fold, variant and combo
    recs, repro, bn_bad, bn_fix, tsec = [], [], {}, {}, []
    for g in range(n_w):
        rp = PART_DIR / f"records{g}.json"
        if rp.exists():
            d = json.loads(rp.read_text())
            recs += d["oof"]
            repro += d["repro"]
            bn_bad = bn_bad or d.get("bn_bad", {})
            bn_fix[f"worker{g}"] = d.get("bn_fix", {})
            tsec += d.get("task_seconds", [])
    variants = [vn for vn in VARIANTS if recs and all(vn in r for r in recs)]
    metric_keys = ("dice50", "ang_mae", "ang_mae_pix", "med_err")
    with open(METRIC_DIR / "oof_per_image.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fold", "stem", "variant", "combo", "dice50", "ang_mae", "ang_cnt", "med_err"])
        for r in sorted(recs, key=lambda r: (r["fold"], r["stem"])):
            for vn in variants:
                for cname in COMBO_ORDER:
                    x = r[vn][cname]
                    w.writerow([r["fold"], r["stem"], vn, cname, x["dice"][2], x.get("ang", ""), x.get("cnt", ""),
                                x.get("med", "")])
    fold_res = []
    for k in OOF_FOLDS:
        rk = sorted([r for r in recs if r["fold"] == k], key=lambda r: r["stem"])
        if not rk:
            continue
        res = {"fold": k, "n_val": len(rk), "n_val_v1": v1_fold_json[k]["tta"]["n_val"],
               "nonfinite_bn_in_checkpoint": bn_bad.get(str(k), {})}
        for cname in COMBO_ORDER:
            res[cname] = aggregate_records([r["orig"][cname] for r in rk])
        res["tta"] = res[MAIN_COMBO]  # same key as v1's fold json: the model's TTA metrics
        res["v1_tta"] = v1_fold_json[k]["tta"]
        res["v1_best"] = v1_fold_json[k]["best"]
        res["hflip_minus_v1_tta"] = {m_: res["hflip"][m_] - res["v1_tta"][m_] for m_ in metric_keys}
        res["none_minus_v1_best"] = {m_: res["none"][m_] - res["v1_best"][m_] for m_ in metric_keys}
        if "fx" in variants:
            res["fx"] = {cname: aggregate_records([r["fx"][cname] for r in rk]) for cname in COMBO_ORDER}
            res["fx"]["tta"] = res["fx"][MAIN_COMBO]
            res["fx"]["repaired"] = bool(bn_bad.get(str(k)))
        (METRIC_DIR / f"fold{k}.json").write_text(json.dumps(res, indent=2))
        fold_res.append(res)
        log(f"fold {k} (n={len(rk)}/{res['n_val_v1']}): " + " | ".join(
            f"{c} ang={res[c]['ang_mae']:.4f} pix={res[c]['ang_mae_pix']:.4f} med={res[c]['med_err']:.4f} "
            f"dice50={res[c]['dice50']:.4f}" for c in COMBO_ORDER) + f" || v1 TTA ang={res['v1_tta']['ang_mae']:.4f} "
            f"hflip-v1={res['hflip_minus_v1_tta']['ang_mae']:+.5f} | v1 best(no TTA) ang={res['v1_best']['ang_mae']:.4f} "
            f"none-v1best={res['none_minus_v1_best']['ang_mae']:+.5f}")
        if "fx" in res and res["fx"]["repaired"]:
            log(f"fold {k} fx (repaired BN): " + " | ".join(
                f"{c} ang={res['fx'][c]['ang_mae']:.4f} med={res['fx'][c]['med_err']:.4f} dice50={res['fx'][c]['dice50']:.4f}"
                for c in COMBO_ORDER))
    summary: dict = {"model": MODEL_NAME, "base_model": BASE_MODEL, "mode": MODE, "views": VIEWS,
                     "combos": {k: v for k, v in COMBOS.items()}, "main_combo": MAIN_COMBO,
                     "nonfinite_bn_in_v1_checkpoints": bn_bad, "folds": fold_res}

    def summarise(get, prefix: str) -> dict:
        out = {}
        for key in metric_keys:
            out[f"mean_tta_{key}"] = float(np.mean([get(r)["tta"][key] for r in fold_res]))
            for cname in COMBO_ORDER:
                out[f"mean_{cname}_{key}"] = float(np.mean([get(r)[cname][key] for r in fold_res]))
        for cname in ("tta", *COMBO_ORDER):
            out[f"{cname}_ang_mae_folds"] = [get(r)[cname]["ang_mae"] for r in fold_res]
        if len(fold_res) == N_FOLDS:
            below = sum(get(r)["tta"]["ang_mae"] < r["v1_tta"]["ang_mae"] for r in fold_res)
            out["d1_preview_informational"] = {
                "arm": prefix, "mean_tta_ang_mae": out["mean_tta_ang_mae"], "mean_max": D1_MEAN_MAX,
                "folds_below_v1": int(below), "min_folds": D1_MIN_FOLDS,
                "pass": bool(out["mean_tta_ang_mae"] <= D1_MEAN_MAX and below >= D1_MIN_FOLDS)}
        return out

    if fold_res:
        summary.update(summarise(lambda r: r, "orig (preregistered fasc_orient_v1rt)"))
        for key in metric_keys:
            summary[f"mean_v1_tta_{key}"] = float(np.mean([r["v1_tta"][key] for r in fold_res]))
            summary[f"mean_v1_best_{key}"] = float(np.mean([r["v1_best"][key] for r in fold_res]))
        summary["v1_tta_ang_mae_folds"] = [r["v1_tta"]["ang_mae"] for r in fold_res]
        full_folds = all(r["n_val"] == r["n_val_v1"] for r in fold_res)
        dev_ = [abs(r["hflip_minus_v1_tta"]["ang_mae"]) for r in fold_res]
        summary["hflip_repro"] = {"full_folds": full_folds, "max_abs_ang_mae_diff": max(dev_),
                                  "max_abs_dice50_diff": max(abs(r["hflip_minus_v1_tta"]["dice50"]) for r in fold_res),
                                  "per_fold_ang_mae_diff": [r["hflip_minus_v1_tta"]["ang_mae"] for r in fold_res],
                                  "tol": REPRO_TOL_ANG, "ok": bool(full_folds and max(dev_) <= REPRO_TOL_ANG)}
        if "fx" in variants:
            summary["fx_exploratory"] = {
                "note": "NOT preregistered: v1 checkpoints with the non-finite BN running stats re-estimated "
                        "(banks " + FX_NAME + " = rich TTA, " + V1FX_NAME + " = hflip only)",
                "bn_fix": bn_fix, **summarise(lambda r: r["fx"], "fx (exploratory)")}
    if tsec:
        summary["task_seconds"] = {"n": len(tsec), "mean": float(np.mean(tsec)), "sum": float(np.sum(tsec))}
    # ---- reproduction of v1's own maskbank (identity view + hflip, native, uint8)
    if repro:
        with open(METRIC_DIR / "repro_v1_maskbank.csv", "w", newline="") as fh:
            cols = sorted({k for r in repro for k in r})
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(repro)
        rsum = {}
        for ds in sorted({r["ds"] for r in repro}):
            rs = [r for r in repro if r["ds"] == ds and r.get("found") and "max_seg" in r]
            if not rs:
                rsum[ds] = {"n": 0}
                continue
            rsum[ds] = {"n": len(rs), "n_not_found": sum(1 for r in repro if r["ds"] == ds and not r.get("found")),
                        **{f"max_{ch}": max(r[f"max_{ch}"] for r in rs) for ch in CHANNELS},
                        **{f"mean_{ch}": float(np.mean([r[f"mean_{ch}"] for r in rs])) for ch in CHANNELS},
                        "frac_gt1_max": max(max(r[f"frac_gt1_{ch}"] for ch in CHANNELS) for r in rs),
                        "ang_frag_mean": float(np.nanmean([r.get("ang_frag_mean", np.nan) for r in rs])),
                        "rich_vs_v1_ang_frag_mean": float(np.nanmean([r.get("rich_vs_v1_ang_frag_mean") or np.nan
                                                                      for r in rs])),
                        "v1fx_vs_v1_ang_frag_mean": float(np.nanmean([r.get("v1fx_vs_v1_ang_frag_mean") or np.nan
                                                                      for r in rs]))}
        summary["repro_v1_maskbank"] = rsum
        log("repro vs v1 maskbank (identity+hflip, uint8): " + json.dumps(rsum))
    rcj = METRIC_DIR / "rotation_check.json"
    if rcj.exists():
        rc = json.loads(rcj.read_text())
        summary["rotation_check"] = {k: rc[k] for k in ("pass", "tol_deg", "worst_numpy_deg", "worst_torch_deg",
                                                        "min_wrong_sign_control_deg", "items")}
    summary["worker_exit_codes"] = codes
    (METRIC_DIR / "cv_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log("CV summary: " + json.dumps({k: v for k, v in summary.items() if k not in ("folds", "repro_v1_maskbank")},
                                    default=str))

    # ---- merge maskbank parts (main bank + exploratory fx banks)
    t = time.time()
    banks = [b_ for b_ in (MODEL_NAME, FX_NAME, V1FX_NAME) if list(PART_DIR.glob(f"inf*__{b_}_seg.npz"))]
    jobs = [([str(p_) for p_ in sorted(PART_DIR.glob(f"inf*_{ds}__{bank}_{ch}.npz"))],
             str(MASKBANK / f"{ds}__{bank}_{ch}.npz"))
            for bank in banks for ds in ("train_oof", *datasets) for ch in CHANNELS]
    jobs = [j for j in jobs if j[0]]
    with ctx.Pool(3) as pool:
        for dest, n in pool.imap_unordered(_merge_job, jobs):
            log(f"merged {n} maps -> {Path(dest).name}")
    log(f"merge {time.time() - t:.0f}s; banks {banks}")
    written: dict = {}
    for ds in ("train_oof", *datasets):
        f = MASKBANK / f"{ds}__{MODEL_NAME}_seg.npz"
        if f.exists():
            with zipfile.ZipFile(f) as z:
                written[ds] = {n[:-4] for n in z.namelist()}
    for bank in banks[1:]:
        for ds in written:
            f = MASKBANK / f"{ds}__{bank}_seg.npz"
            ks = set()
            if f.exists():
                with zipfile.ZipFile(f) as z:
                    ks = {n[:-4] for n in z.namelist()}
            if ks != written[ds]:
                log(f"WARNING bank {bank} {ds}: {len(ks)} keys vs {len(written[ds])} in {MODEL_NAME}")
    with open(MASKBANK / "manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "image_id", "source_filename", "height", "width"])
        for it in sorted(ITEMS, key=lambda it: (it["fold"], it["stem"])):
            if it["stem"] in written.get("train_oof", set()):
                w.writerow(["train_oof", it["stem"], it["name"], it["H"], it["W"]])
        for u in UNL:
            if u["key"] in written.get(u["ds"], set()):
                w.writerow([u["ds"], u["key"], u["src"], u["H"], u["W"]])
    for bank in banks:
        with open(MASKBANK / f"{bank}_viewports.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["dataset", "image_id", "layout", "x1", "y1", "x2", "y2"])
            for u in UNL:
                if u["key"] in written.get(u["ds"], set()):
                    w.writerow([u["ds"], u["key"], u["layout"], *u["box"]])
    log(f"maskbank written: { {k: len(v) for k, v in written.items()} }")
    config = {
        "model": MODEL_NAME, "base_model": BASE_MODEL, "base_kernel": f"shivateja005/{BASE_KERNEL_SLUG}",
        "banks": banks, "mode": MODE, "encoder": ENCODER, "arch": "smp.Unet", "size": SIZE,
        "channels": {"0": "seg logit", "1": "tanh cos2theta", "2": "tanh sin2theta"},
        "angle_convention": "theta = 0.5*atan2(sin2, cos2) in image pixel coords of the file (x right, y down); "
                            "rising-to-the-right fascicles have theta < 0",
        "maskbank_encoding": "seg = prob*255; cos2/sin2 = (v+1)/2*255 of the valid-weighted mean over 9 views x hflip "
                             "(x 5 folds for test/static/dynamic/vl; the fold-k model for train_oof); vector NOT "
                             "renormalised (|v| < 1 means low agreement); outside the viewport seg=0, cos2=sin2=128",
        "tta": {"scales": SCALES, "rotations_deg": ROTS_DEG, "hflip": True, "centre": "letterbox canvas centre",
                "angle_correction": "theta -> theta - phi ((cos2, sin2) rotated by -2 phi)",
                "weighting": "per-pixel back-warped valid mask of each view; uncovered pixels keep the identity view"},
        "fx_banks": {FX_NAME: "rich TTA, non-finite BN running stats re-estimated (exploratory)",
                     V1FX_NAME: "v1 hflip TTA only, non-finite BN running stats re-estimated (exploratory)"},
        "nonfinite_bn_in_v1_checkpoints": bn_bad, "bn_fix": bn_fix,
        "input": "viewport crop -> aspect-preserving letterbox 512 (black pad) -> view similarity, gray x3, ImageNet norm",
        "folds": "v1 folds.csv", "data_stats": stats, "viewport_check": vp_stats,
        "versions": versions, "elapsed_min": (time.time() - T0) / 60,
    }
    (WORK / f"{MODEL_NAME}_config.json").write_text(json.dumps(config, indent=2, default=str))
    shutil.rmtree(TMP, ignore_errors=True)
    log(f"DONE elapsed={(time.time() - T0) / 60:.1f} min worker codes={codes}")
    if any(c not in (0,) for c in codes):
        log("WARNING some worker reported failures (see logs/inf*.log)")


if __name__ == "__main__":
    main()
