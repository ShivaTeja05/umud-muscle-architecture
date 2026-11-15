"""UMUD manual-labelling kit: build the WORK set from the public UltraTimTrack GM videos.

Code: GPL-3.0-only.  The frames it writes are CC BY 4.0 (CREDITS.txt is written next to them).

Source: "Gastrocnemius medialis muscle ultrasonography dataset", van der Zee, Raiteri, Tecchio, Hahn
(2024), figshare, https://doi.org/10.6084/m9.figshare.26502868.v1 (CC BY 4.0).  Telemed ArtUs EXT-1H +
LV8-5N60-A2, 60 x 50 mm (width x depth) field of view, 8 participants, left GM.  Paper: van der Zee,
Tecchio, Hahn, Raiteri (2025) PeerJ Comput Sci 11:e2636.

Kaggle kernel: internet ON, GPU OFF, no competition data attached.  Nothing here touches the
competition test images.

  1. Checks the figshare metadata API against the pinned table VIDEOS / ESTIMATES below (id, name,
     size, computed_md5, folder).  The participant of a file is its folder in the API's
     "folder_structure" (ultrasonography/pN/ultrasound/original); file names repeat across people.
     Any difference -> stop before downloading.
  2. Downloads ONLY the 8 passive-rotation videos at 5 deg/s (pas_005, one per participant,
     769,930,362 bytes) and, if WITH_ESTIMATES, fascicle_tracking_estimates.zip (393,949,300 bytes):
     its UltraTimTrack result files hold TrackingData.res = the image depth in mm that the authors
     used (UltraTimTrack: mm per pixel = depth / image height, both axes).  Every file is md5-checked.
     One video is on disk at a time; it is deleted after use.
  3. Pass 1 (frame by frame): a 48x36 thumbnail, sharpness and brightness of every frame.
     Selection: N_PER_VIDEO frames per video, one per equal time window, by greedy farthest-point on
     thumbnail NCC (spreads over ankle angles, avoids near-duplicates), weighted by sharpness.
  4. Pass 2: WRK_XXX.png (8-bit grey, native pixels) + view-only neighbours WRK_XXX_n1 / _n2 at
     +0.1 s / +0.2 s (label on frame 1, like the old CAL_XX_nK files).  If the output would pass
     the size budget, all _n2 and then all _n1 files are dropped.
  5. Writes work/manifest.json (annotator format, tools/annotator/README.md), work/CREDITS.txt,
     private/work_selection.json (selection + scale evidence + algorithm estimates; never serve it)
     and /kaggle/working/labelling_work_v1.zip (< 80 MB).  Unzip it inside data/manual_labelling/.

Local fallback without network: tools/annotator/prepare_work.py (imports this file).
Set MODE = "api-check" for a first run that only checks the API and downloads nothing.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import shutil
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------- settings
MODE = os.environ.get("UMUD_WORK_MODE", "full")            # "full" | "api-check" (no download)
WITH_ESTIMATES = os.environ.get("UMUD_WORK_ESTIMATES", "1") != "0"   # + 393.95 MB, for the scale
N_PER_VIDEO = 12              # 8 videos x 12 = 96 tasks (WRK_001 .. WRK_096)
NB_DT_S = (0.1, 0.2)          # view-only neighbours after the labelled frame (n1, n2)
EDGE_S = 2.0                  # skip the first / last 2 s of a video
MIN_SEP_S = 2.0               # two chosen frames of one video are >= 2 s apart
DESC_SIZE = (48, 36)          # thumbnail (width, height) for the similarity
DUP_NCC = 0.98                # thumbnail NCC above this = near-duplicate (avoided, then flagged)
QUALITY_PCT = 25              # frames below this sharpness percentile are only a last resort
BUDGET_BYTES = 78_000_000     # zip budget (the kit asks for < 80 MB)
NOMINAL_DEPTH_MM = 50.0       # paper: 60 x 50 mm (width x depth) field of view
NOMINAL_WIDTH_MM = 60.0
UTT_DEFAULT_DEPTH_MM = 50.7   # UltraTimTrack.m default ImageDepth (used when no TVD .mat is found)
GROUP = "telemed_artus_utt"
OUT_ZIP_NAME = "labelling_work_v1.zip"

# ----------------------------------------------------------------------------- pinned files
ARTICLE_ID = 26502868
DOI = "10.6084/m9.figshare.26502868.v1"
API_ARTICLE = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}"
DOWNLOAD_URL = "https://ndownloader.figshare.com/files/{id}"
# From the figshare API on 2026-11-01 (version 1, published 2024-08-14T08:23:45Z).  folder = the
# API folder_structure entry of the file id; the participant comes from it.
VIDEOS = [
    {"participant": "p1", "name": "pas_005_01.mp4", "id": 48438310, "size": 91680151,
     "md5": "4a9b7af9b9b1e3ed60a794ca93d23fc7"},
    {"participant": "p2", "name": "pas_005_01.mp4", "id": 48438325, "size": 98876905,
     "md5": "8c03016793720713a814a08979ad3bfa"},
    {"participant": "p3", "name": "pas_005_01.mp4", "id": 48438388, "size": 105373227,
     "md5": "7d502514f23f9558ffa25b06690df2a3"},
    {"participant": "p4", "name": "pas_005_01.mp4", "id": 48438358, "size": 103348047,
     "md5": "88768abb9edbcf3e03af4e2f0f52d661"},
    {"participant": "p5", "name": "pas_005_01.mp4", "id": 48438328, "size": 102554650,
     "md5": "6f22fc7951e651494b23cd68016152c8"},
    {"participant": "p6", "name": "pas_005_02.mp4", "id": 48438037, "size": 38283919,
     "md5": "a8a34d3bebcc061d4bb3833403555bd8"},
    {"participant": "p7", "name": "pas_005_01.mp4", "id": 48438355, "size": 114807267,
     "md5": "dfb42bc661db88ca77412c6489ab45c9"},
    {"participant": "p8", "name": "pas_005_01.mp4", "id": 48438322, "size": 115006196,
     "md5": "fc6666f4fc316bde386125ddb863062e"},
]
for _v in VIDEOS:
    _v["folder"] = f"ultrasonography/{_v['participant']}/ultrasound/original"
ESTIMATES = {"participant": None, "name": "fascicle_tracking_estimates.zip", "id": 48368683,
             "size": 393949300, "md5": "e31a29a9382a091293b6556732943163", "folder": ""}

CREDITS = """\
These images are single frames taken from videos of this dataset:

  Gastrocnemius medialis muscle ultrasonography dataset.
  Tim van der Zee, Brent J. Raiteri, Paolo Tecchio, Daniel Hahn (2024). figshare. Dataset.
  https://doi.org/10.6084/m9.figshare.26502868.v1
  Licence: CC BY 4.0  https://creativecommons.org/licenses/by/4.0/

Paper: van der Zee TJ, Tecchio P, Hahn D, Raiteri BJ (2025). UltraTimTrack: a Kalman-filter-based
algorithm to track muscle fascicles in ultrasound image sequences. PeerJ Computer Science 11:e2636.

Changes made by us: we took single frames from the MP4 videos, saved them as 8-bit grey PNG at the
original pixel size, and gave them new names (WRK_XXX, WRK_XXX_n1, WRK_XXX_n2). Nothing else was
changed. The authors do not endorse this use.

Videos used (participant, figshare file id, file name):
{videos}
"""


def log(*parts) -> None:
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


# ----------------------------------------------------------------------------- files / network
def md5_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _this_file() -> Path:
    try:
        return Path(__file__)
    except NameError:  # some notebook runners have no __file__
        return Path("kernel.py")


def http_json(url: str, timeout: int = 60) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "umud-labelling-frames/1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def check_api() -> dict:
    """Compare the live figshare metadata with the pinned table; return {'problems': [...], ...}."""

    art = http_json(API_ARTICLE)
    files = {f["id"]: f for f in art.get("files", [])}
    folders = art.get("folder_structure") or {}
    problems = []
    for entry in VIDEOS + [ESTIMATES]:
        live = files.get(entry["id"])
        if live is None:
            problems.append(f"file {entry['id']} ({entry['name']}) is no longer in the article")
            continue
        for api_key, key in (("name", "name"), ("size", "size"), ("computed_md5", "md5")):
            if live.get(api_key) != entry[key]:
                problems.append(f"file {entry['id']}: {api_key} {live.get(api_key)!r} != pinned {entry[key]!r}")
        if folders.get(str(entry["id"]), "") != entry["folder"]:
            problems.append(f"file {entry['id']}: folder {folders.get(str(entry['id']))!r} != pinned {entry['folder']!r}")
    lic = (art.get("license") or {}).get("name")
    if lic != "CC BY 4.0":
        problems.append(f"licence is {lic!r}, not CC BY 4.0")
    return {"article": ARTICLE_ID, "version": art.get("version"), "modified_date": art.get("modified_date"),
            "licence": lic, "n_files": len(files), "problems": problems}


def download(entry: dict, dst_dir: Path, tries: int = 3) -> Path:
    """Stream one figshare file to dst_dir and check its size and md5 (computed_md5)."""

    dst = Path(dst_dir) / f"{entry['id']}_{entry['name']}"
    url = DOWNLOAD_URL.format(id=entry["id"])
    for attempt in range(1, tries + 1):
        try:
            h, n, t0 = hashlib.md5(), 0, time.time()
            req = urllib.request.Request(url, headers={"User-Agent": "umud-labelling-frames/1"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(dst, "wb") as fh:
                while True:
                    block = resp.read(1 << 20)
                    if not block:
                        break
                    fh.write(block)
                    h.update(block)
                    n += len(block)
            if n == entry["size"] and h.hexdigest() == entry["md5"]:
                log(f"  downloaded {entry['participant'] or ''} {entry['name']} ({n / 1e6:.2f} MB, "
                    f"{time.time() - t0:.0f} s, md5 ok)")
                return dst
            log(f"  bad download {entry['id']}: {n} bytes, md5 {h.hexdigest()} (try {attempt})")
        except Exception as exc:  # network errors: retry
            log(f"  download error {entry['id']}: {exc!r} (try {attempt})")
        time.sleep(5 * attempt)
    if dst.exists():
        dst.unlink()
    raise RuntimeError(f"could not download file {entry['id']} ({entry['name']})")


# ----------------------------------------------------------------------------- video passes
def _cv2():
    import cv2

    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    return cv2


def open_capture(path: Path):
    cv2 = _cv2()
    cap = cv2.VideoCapture(str(path))
    if hasattr(cv2, "CAP_PROP_N_THREADS"):
        try:
            cap.set(cv2.CAP_PROP_N_THREADS, 1)
        except Exception:
            pass
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {path}")
    return cap


def to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return _cv2().cvtColor(frame, _cv2().COLOR_BGR2GRAY)


def thumb(gray: np.ndarray) -> np.ndarray:
    """z-scored thumbnail; NCC of two frames = dot(a, b) / len(a)."""

    cv2 = _cv2()
    small = cv2.resize(gray, DESC_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    small -= small.mean()
    sd = float(small.std())
    return small / sd if sd > 1e-6 else np.zeros_like(small)


def sharpness(gray: np.ndarray) -> float:
    """Mean squared Sobel gradient of the half-size frame (Tenengrad)."""

    cv2 = _cv2()
    h, w = gray.shape
    small = cv2.resize(gray, (max(8, w // 2), max(8, h // 2)), interpolation=cv2.INTER_AREA).astype(np.float32)
    gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(gx * gx + gy * gy))


def scan_video(path: Path) -> dict:
    """Pass 1: read every frame once (sequentially) and keep only small per-frame features."""

    cv2 = _cv2()
    cap = open_capture(path)
    fps_raw = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n_header = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    descs, sharp, bright = [], [], []
    width = height = None
    col_sum = row_sum = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = to_gray(frame)
        if width is None:
            height, width = gray.shape
            col_sum = np.zeros(width, np.float64)
            row_sum = np.zeros(height, np.float64)
        elif gray.shape != (height, width):
            cap.release()
            raise RuntimeError(f"frame size changes inside {path}")
        descs.append(thumb(gray))
        sharp.append(sharpness(gray))
        bright.append(float(gray.mean()))
        col_sum += gray.mean(axis=0)
        row_sum += gray.mean(axis=1)
    cap.release()
    n = len(descs)
    if n == 0:
        raise RuntimeError(f"no frames decoded from {path}")
    fps_ok = 5.0 <= fps_raw <= 400.0
    return {
        "n_frames": n, "n_frames_header": n_header, "fps": fps_raw if fps_ok else 33.333,
        "fps_raw": fps_raw, "fps_ok": fps_ok, "width": int(width), "height": int(height),
        "desc": np.stack(descs).astype(np.float32), "sharp": np.asarray(sharp), "bright": np.asarray(bright),
        "col_mean": col_sum / n, "row_mean": row_sum / n,
    }


def viewport(scan: dict) -> list[int]:
    """Inclusive-exclusive box of the image area (the TVD2ALL export is already cropped to it)."""

    w, h = scan["width"], scan["height"]
    col, row = scan["col_mean"], scan["row_mean"]
    xs = np.flatnonzero(col > max(3.0, 0.1 * float(np.percentile(col, 90))))
    x1, x2 = (int(xs[0]), int(xs[-1]) + 1) if xs.size else (0, w)
    if x2 - x1 < 0.5 * w:
        x1, x2 = 0, w
    ys = np.flatnonzero(row > 1.0)
    y1, y2 = (int(ys[0]), int(ys[-1]) + 1) if ys.size else (0, h)
    if y2 - y1 < 0.5 * h:
        y1, y2 = 0, h
    return [x1, y1, x2, y2]


def neighbour_offsets(fps: float) -> list[int]:
    offs = []
    for dt in NB_DT_S:
        k = max(1, int(round(dt * fps)))
        while k in offs:
            k += 1
        offs.append(k)
    return offs


def select_frames(scan: dict, n_sel: int = N_PER_VIDEO) -> list[dict]:
    """One frame per equal time window, greedy farthest-point on thumbnail NCC x sharpness."""

    desc, sharp, bright, fps = scan["desc"], scan["sharp"], scan["bright"], scan["fps"]
    n, dim = desc.shape
    pad = max(neighbour_offsets(fps)) if NB_DT_S else 0
    lo, hi = max(pad, int(math.ceil(EDGE_S * fps))), n - max(pad, int(math.ceil(EDGE_S * fps))) - 1
    if hi - lo < 2 * n_sel:  # short video: only keep room for the neighbours
        lo, hi = 0, n - pad - 1
    if hi - lo < n_sel:
        raise RuntimeError(f"video too short ({n} frames) for {n_sel} frames")
    idx = np.arange(n)
    usable = (idx >= lo) & (idx < hi)
    usable &= bright > 0.5 * float(np.median(bright[usable]))  # dropouts / blank frames
    q_thr = float(np.percentile(sharp[usable], QUALITY_PCT))
    good = usable & (sharp >= q_thr)
    srank = np.zeros(n)
    ranks = np.argsort(np.argsort(sharp[usable]))
    srank[usable] = ranks / max(1, int(usable.sum()) - 1)
    edges = np.linspace(lo, hi, n_sel + 1).round().astype(int)
    window = np.full(n, -1)
    for k in range(n_sel):
        window[edges[k]:edges[k + 1]] = k
    min_sep = int(round(MIN_SEP_S * fps))
    mind = np.full(n, np.inf)  # 1 - NCC to the nearest chosen frame
    chosen: list[int] = []
    out: list[dict] = []
    for _ in range(n_sel):
        free = (window >= 0) & ~np.isin(window, [int(window[c]) for c in chosen])
        sep_ok = np.ones(n, bool)
        for c in chosen:
            sep_ok[max(0, c - min_sep): c + min_sep + 1] = False
        level, cand = None, None
        for lvl, mask in enumerate((good & free & sep_ok, usable & free & sep_ok, usable & free)):
            if mask.any():
                level, cand = lvl, mask
                break
        if cand is None:
            break
        if not chosen:
            pool = cand & (window == n_sel // 2)
            pool = pool if pool.any() else cand
            j = int(np.flatnonzero(pool)[np.argmax(sharp[pool])])
        else:
            score = np.minimum(mind, 2.0) * (0.5 + 0.5 * srank)
            nodup = cand & (mind > 1.0 - DUP_NCC)
            mask = nodup if nodup.any() else cand
            j = int(np.argmax(np.where(mask, score, -np.inf)))
        chosen.append(j)
        out.append({"rank": len(out), "frame": j, "time_s": round(j / fps, 3), "window": int(window[j]),
                    "sharpness": round(float(sharp[j]), 2), "sharp_rank": round(float(srank[j]), 3),
                    "relaxed": level, "dist_to_chosen": None if not np.isfinite(mind[j]) else round(float(mind[j]), 4)})
        mind = np.minimum(mind, 1.0 - desc @ desc[j] / dim)
    if len(chosen) > 1:
        sub = desc[chosen] @ desc[chosen].T / dim
        np.fill_diagonal(sub, -1.0)
        for rec, row in zip(out, sub):
            rec["max_ncc_to_others"] = round(float(row.max()), 4)
            rec["near_duplicate"] = bool(row.max() > DUP_NCC)
    return out


def extract_frames(path: Path, wanted: dict[int, list[Path]], check: dict[int, np.ndarray]) -> dict[str, int]:
    """Pass 2: decode sequentially again and write the wanted frames as 8-bit grey PNG."""

    cv2 = _cv2()
    cap = open_capture(path)
    last, i, written = max(wanted), 0, {}
    while i <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            gray = np.ascontiguousarray(to_gray(frame))
            if i in check:  # the same frame as in pass 1 (sequential decoding is deterministic)
                ncc = float(thumb(gray) @ check[i] / check[i].size)
                if ncc < 0.999:
                    cap.release()
                    raise RuntimeError(f"frame {i} of {path} differs between passes (NCC {ncc:.4f})")
            for dst in wanted[i]:
                if not cv2.imwrite(str(dst), gray, [cv2.IMWRITE_PNG_COMPRESSION, 9]):
                    cap.release()
                    raise RuntimeError(f"cannot write {dst}")
                written[dst.name] = dst.stat().st_size
        i += 1
    cap.release()
    missing = [d.name for ds in wanted.values() for d in ds if d.name not in written]
    if missing:
        raise RuntimeError(f"{path}: frames not decoded on pass 2: {missing[:5]}")
    return written


# ----------------------------------------------------------------------------- estimates / scale
def _first(x):
    if isinstance(x, np.ndarray) and x.dtype == object and x.size:
        return x.flat[0]
    return x


def _attr(obj, name):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _vec(x) -> np.ndarray | None:
    if x is None:
        return None
    try:
        a = np.asarray(x, dtype=float)
    except (TypeError, ValueError):
        return None
    if a.ndim == 0:
        a = a.reshape(1)
    if a.ndim == 2:
        a = a[0] if a.shape[0] <= a.shape[1] else a[:, 0]
    return a.ravel()


def _scalar(x) -> float | None:
    v = _vec(x)
    return float(v[0]) if v is not None and v.size and np.isfinite(v[0]) else None


def load_mat(data: bytes) -> dict:
    """TrackingData.res, TVDdata scale fields and the Fdat.Region FL / PEN / ANG / Time series."""

    try:
        from scipy.io import loadmat

        mat = loadmat(io.BytesIO(data), squeeze_me=True, struct_as_record=False,
                      variable_names=["TrackingData", "Fdat", "TVDdata"])
        fmt = "v5"
        td, fd, tvd = (_first(mat.get(k)) for k in ("TrackingData", "Fdat", "TVDdata"))
        region = _first(_attr(fd, "Region"))
        out = {"format": fmt, "res_mm": _scalar(_attr(td, "res")),
               "series": {k: _vec(_attr(region, k)) for k in ("FL", "PEN", "ANG", "Time")}}
        if tvd is not None:
            out["tvd"] = {k: _scalar(_attr(tvd, k)) for k in ("cmPerPixX", "cmPerPixY", "Height", "Width", "fps")}
        return out
    except (NotImplementedError, ValueError):
        import h5py  # MATLAB v7.3 files are HDF5

        out = {"format": "v7.3", "res_mm": None, "series": {}}
        with h5py.File(io.BytesIO(data), "r") as f:
            if "TrackingData" in f and "res" in f["TrackingData"]:
                out["res_mm"] = _scalar(np.array(f["TrackingData"]["res"]))
            if "TVDdata" in f:
                out["tvd"] = {k: _scalar(np.array(f["TVDdata"][k])) if k in f["TVDdata"] else None
                              for k in ("cmPerPixX", "cmPerPixY", "Height", "Width", "fps")}
            try:
                reg = f["Fdat"]["Region"]
                for k in ("FL", "PEN", "ANG", "Time"):
                    if k in reg:
                        ds = reg[k]
                        arr = np.array(ds)
                        if arr.dtype == object or ds.dtype.kind == "O":  # struct array: first region
                            arr = np.array(f[arr.flat[0]])
                        out["series"][k] = _vec(arr)
            except Exception as exc:  # best effort: the scale is what matters
                out["series_error"] = repr(exc)
        return out


def read_estimates(zip_path: Path, stems: dict[str, str]) -> dict:
    """{participant: {...}} for the pas_005 result files of every participant in the tracking zip."""

    out: dict = {"zip": Path(zip_path).name, "participants": {}}
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".mat")]
        out["n_mat_members"] = len(names)
        for part, stem in stems.items():
            num = int(part[1:])
            in_part = [n for n in names if re.search(rf"(?:^|/)p0*{num}(?:/|_)", n, flags=re.I)
                       and "pas_005" in n.lower()]
            exact = [n for n in in_part if stem.lower() in n.rsplit("/", 1)[-1].lower()]
            rec = {"stem": stem, "stem_match": bool(exact), "members": exact or in_part, "res_mm": {},
                   "tvd": None, "series": {}, "errors": []}
            for name in rec["members"]:
                try:
                    mat = load_mat(zf.read(name))
                except Exception as exc:
                    rec["errors"].append(f"{name}: {exc!r}")
                    continue
                base = name.rsplit("/", 1)[-1]
                rec["res_mm"][base] = mat.get("res_mm")
                if mat.get("tvd") and rec["tvd"] is None:
                    rec["tvd"] = mat["tvd"]
                rec["series"][base] = {k: v for k, v in (mat.get("series") or {}).items() if v is not None}
            out["participants"][part] = rec
            log(f"  estimates {part}: {len(rec['members'])} files, res_mm {rec['res_mm']}"
                + ("" if rec["stem_match"] else "  (no exact name match)") + (f"  errors {rec['errors']}" if rec["errors"] else ""))
    return out


def decide_scale(height: int, width: int, est: dict | None) -> dict:
    """px_per_cm (vertical, used for both axes) with its evidence."""

    ev: dict = {"height_px": height, "width_px": width,
                "alt_px_per_cm": {"depth_50.0mm": round(height / 5.0, 4), "depth_50.7mm": round(height / 5.07, 4)}}
    ppc, source, trusted = None, None, False
    tvd = (est or {}).get("tvd") or {}
    if tvd.get("cmPerPixY") and (not tvd.get("Height") or int(tvd["Height"]) == height):
        ppc, source, trusted = 1.0 / tvd["cmPerPixY"], "TVDdata.cmPerPixY (Telemed SDK pixel size)", True
        if tvd.get("cmPerPixX"):
            ev["pixel_aspect_x_over_y"] = round(tvd["cmPerPixX"] / tvd["cmPerPixY"], 5)
    res = [v for v in ((est or {}).get("res_mm") or {}).values() if v is not None and 20.0 < v < 120.0]
    if res:
        ev["res_mm_values"] = sorted({round(v, 4) for v in res})
        med = float(np.median(res))
        ev["res_consistent"] = (max(res) - min(res)) / med <= 0.005
        ev["res_equals_utt_default"] = all(abs(v - UTT_DEFAULT_DEPTH_MM) < 1e-6 for v in res)
        if ppc is None and ev["res_consistent"]:
            ppc, source, trusted = 10.0 * height / med, "TrackingData.res (image depth used by the authors)", True
            ev["depth_mm"] = round(med, 4)
    if ppc is None:
        ppc, source, trusted = 10.0 * height / NOMINAL_DEPTH_MM, "nominal 50 mm depth (paper: 60 x 50 mm field of view)", False
        ev["depth_mm"] = NOMINAL_DEPTH_MM
    ev["depth_mm_implied"] = round(10.0 * height / ppc, 3)
    ev["width_mm_implied"] = round(10.0 * width / ppc, 3)
    ev["width_check_ok"] = abs(ev["width_mm_implied"] - NOMINAL_WIDTH_MM) / NOMINAL_WIDTH_MM <= 0.04
    if trusted and not ev["width_check_ok"]:
        trusted = False
    return {"px_per_cm": round(float(ppc), 4), "source": source, "trusted": bool(trusted), "evidence": ev}


def estimates_at(frames: list[int], fps: float, n_frames: int, series: dict) -> dict:
    """Algorithm FL / PEN at the chosen frames (NOT ground truth; for later comparison only)."""

    out = {}
    for member, s in (series or {}).items():
        fl, pen = s.get("FL"), s.get("PEN")
        if fl is None or pen is None or fl.size != pen.size:
            continue
        rec = {"n": int(fl.size)}
        if abs(fl.size - n_frames) <= 1:
            pick, rec["align"] = [min(f, fl.size - 1) for f in frames], "index"
        elif s.get("Time") is not None and s["Time"].size == fl.size:
            t = s["Time"] - s["Time"][0]
            pick = [int(np.argmin(np.abs(t - f / fps))) for f in frames]
            rec["align"] = "time (frame / fps vs Fdat.Region.Time)"
            rec["max_dt_s"] = round(float(max(abs(t[p] - f / fps) for p, f in zip(pick, frames))), 3)
        else:
            rec["align"] = "none"
            rec["summary"] = {"FL_min_med_max": [round(float(np.nanpercentile(fl, q)), 2) for q in (0, 50, 100)],
                              "PEN_min_med_max": [round(float(np.nanpercentile(pen, q)), 2) for q in (0, 50, 100)]}
            out[member] = rec
            continue
        rec["FL"] = [round(float(fl[p]), 3) for p in pick]
        rec["PEN"] = [round(float(pen[p]), 3) for p in pick]
        out[member] = rec
    return out


# ----------------------------------------------------------------------------- the work set
def build_work_set(videos: list[dict], out_root: Path, estimates: dict | None = None, fetch=None,
                   delete_after: bool = False, n_per_video: int = N_PER_VIDEO,
                   budget_bytes: int | None = BUDGET_BYTES, api: dict | None = None, clean: bool = False) -> dict:
    """Write out_root/work/{img/,manifest.json,CREDITS.txt} and out_root/private/work_selection.json.

    videos: VIDEOS entries; each needs "path" unless fetch(entry) -> Path is given.  One video at a
    time; with delete_after the video file is removed after its frames are written.
    """

    out_root = Path(out_root)
    work, priv = out_root / "work", out_root / "private"
    img = work / "img"
    img.mkdir(parents=True, exist_ok=True)
    priv.mkdir(parents=True, exist_ok=True)
    if clean:
        for old in list(img.glob("WRK_*.png")) + list(img.glob("_tmp_*.png")):
            old.unlink()
    elif any(img.glob("WRK_*.png")):
        raise RuntimeError(f"{img} already has WRK_*.png; remove them or use clean=True")
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    per_video, items = [], []
    for vi, entry in enumerate(videos):
        part = entry["participant"]
        rec = {k: entry[k] for k in ("participant", "name", "id", "size", "md5", "folder") if k in entry}
        path = None
        try:
            path = Path(fetch(entry)) if fetch else Path(entry["path"])
            t0 = time.time()
            scan = scan_video(path)
            sel = select_frames(scan, n_per_video)
            est = ((estimates or {}).get("participants") or {}).get(part)
            scale = decide_scale(scan["height"], scan["width"], est)
            vp = viewport(scan)
            offs = neighbour_offsets(scan["fps"])
            wanted: dict[int, list[Path]] = {}
            for s in sel:
                tmp = f"_tmp_{part}_r{s['rank']:02d}"
                s["tmp"] = tmp
                s["neighbour_frames"] = [s["frame"] + o for o in offs]
                wanted.setdefault(s["frame"], []).append(img / f"{tmp}.png")
                for k, fr in enumerate(s["neighbour_frames"], 1):
                    wanted.setdefault(fr, []).append(img / f"{tmp}_n{k}.png")
            sizes = extract_frames(path, wanted, {s["frame"]: scan["desc"][s["frame"]] for s in sel})
            alg = estimates_at([s["frame"] for s in sel], scan["fps"], scan["n_frames"],
                               (est or {}).get("series")) if est else {}
            rec.update({"ok": True, "n_frames": scan["n_frames"], "n_frames_header": scan["n_frames_header"],
                        "fps": round(scan["fps"], 4), "fps_raw": scan["fps_raw"], "fps_ok": scan["fps_ok"],
                        "width": scan["width"], "height": scan["height"], "viewport": vp, "scale": scale,
                        "neighbour_offsets_frames": offs, "selected": sel, "algorithm_estimates_at_selected": alg,
                        "estimate_files": (est or {}).get("members"), "estimate_errors": (est or {}).get("errors"),
                        "png_bytes": int(sum(sizes.values())), "seconds": round(time.time() - t0, 1)})
            for s in sel:
                items.append({"vi": vi, "participant": part, "video": entry["name"], "file_id": entry["id"],
                              "rank": s["rank"], "frame": s["frame"], "time_s": s["time_s"], "tmp": s["tmp"],
                              "neighbour_frames": s["neighbour_frames"], "width": scan["width"],
                              "height": scan["height"], "viewport": vp, "px_per_cm": scale["px_per_cm"],
                              "scale_trusted": scale["trusted"]})
            log(f"{part} {entry['name']}: {scan['n_frames']} frames {scan['width']}x{scan['height']} "
                f"@ {scan['fps']:.2f} fps, {len(sel)} chosen, {scale['px_per_cm']} px/cm ({scale['source']}; "
                f"width {scale['evidence']['width_mm_implied']} mm), dup flags "
                f"{sum(bool(s.get('near_duplicate')) for s in sel)}, {rec['png_bytes'] / 1e6:.1f} MB PNG")
            del scan
        except Exception as exc:
            rec.update({"ok": False, "error": repr(exc)})
            log(f"ERROR {part} {entry.get('name')}: {exc!r}")
        finally:
            if delete_after and path is not None and path.exists():
                path.unlink()
        per_video.append(rec)
    if not items:
        raise RuntimeError("no frames were extracted")

    # Final ids: round-robin over participants by selection rank (the first 8 tasks cover all people).
    items.sort(key=lambda it: (it["rank"], it["vi"]))
    for k, it in enumerate(items, 1):
        it["task_id"] = f"WRK_{k:03d}"
        (img / f"{it['tmp']}.png").rename(img / f"{it['task_id']}.png")
        it["neighbours"] = []
        for j, fr in enumerate(it["neighbour_frames"], 1):
            src = img / f"{it['tmp']}_n{j}.png"
            if src.exists():
                src.rename(img / f"{it['task_id']}_n{j}.png")
                it["neighbours"].append({"image_id": f"{it['task_id']}_n{j}", "frame": fr})

    # Size budget: drop every _n2, then every _n1, until the PNGs fit.
    def total_png() -> int:
        return sum(p.stat().st_size for p in img.glob("WRK_*.png"))

    dropped = []
    max_nb = max((len(it["neighbours"]) for it in items), default=0)
    for level in range(max_nb, 0, -1):
        if budget_bytes is None or total_png() + 2_000_000 <= budget_bytes:
            break
        for it in items:
            for nb in [n for n in it["neighbours"] if n["image_id"].endswith(f"_n{level}")]:
                (img / f"{nb['image_id']}.png").unlink()
                it["neighbours"].remove(nb)
        dropped.append(f"_n{level}")
    leftovers = sorted(p.name for p in img.glob("_tmp_*.png"))
    for p in img.glob("_tmp_*.png"):
        p.unlink()

    images, tasks = [], []
    for k, it in enumerate(items, 1):
        common = {"clip": None, "width": it["width"], "height": it["height"], "px_per_cm": it["px_per_cm"],
                  "group": GROUP, "viewport": it["viewport"], "scale_trusted": it["scale_trusted"],
                  "kind": "work", "task_id": it["task_id"], "participant": it["participant"],
                  "source": {"dataset": DOI, "file_id": it["file_id"], "video": it["video"]}}
        images.append({"image_id": it["task_id"], "stem": it["task_id"], "file": f"img/{it['task_id']}.png",
                       "number": k, **common, "source": {**common["source"], "frame": it["frame"],
                                                          "time_s": it["time_s"]}})
        for nb in it["neighbours"]:
            images.append({"image_id": nb["image_id"], "stem": nb["image_id"], "file": f"img/{nb['image_id']}.png",
                           "number": None, **common, "view_only": True,
                           "source": {**common["source"], "frame": nb["frame"]}})
        task = {"task_id": it["task_id"], "image_id": it["task_id"], "clip": None, "members": [it["task_id"]]}
        if it["neighbours"]:
            task["frames"] = [it["task_id"]] + [nb["image_id"] for nb in it["neighbours"]]
        tasks.append(task)
    untrusted = sorted({it["participant"] for it in items if not it["scale_trusted"]})
    n_left = max(len(it["neighbours"]) for it in items)
    help_order = [f"Work set: {len(tasks)} Telemed ArtUs GM frames, WRK_001 to WRK_{len(tasks):03d}. "
                  "There are no answers: your labels are the answers."]
    if n_left:
        later = ", ".join(f"{d:g}" for d in NB_DT_S[:n_left])
        help_order.append(f"Measure on frame 1. The other frames ({later} s later) are only for looking "
                          "(keys , and .).")
    help_order.append("Measure as in practice: 3 MT, 3 FL, 3 PA. Superficial muscle only. "
                      "Flag (X) a frame you cannot measure.")
    manifest = {
        "version": 1, "kind": "work", "created": started,
        # optional annotator keys (tools/annotator/app.js): own browser storage, phase name, help text
        "storage_key": "umud_manual_work_v1", "phase_name": "Work set", "progress_label": "work",
        "help_order": help_order,
        "source_dir": f"figshare {DOI} (UltraTimTrack GM dataset, pas_005 videos)",
        "scale_note": "px_per_cm = 10 * image height (px) / image depth (mm), the VERTICAL scale, applied to "
                      "both axes (square pixels), as UltraTimTrack does (mm/px = depth / height). See "
                      "private/work_selection.json for the evidence per video.",
        "licence": "Images: CC BY 4.0, see CREDITS.txt", "hints_source": None,
        "n_images": len(images), "n_tasks": len(tasks), "groups": {GROUP: len(images)},
        "untrusted_scale": [e["image_id"] for e in images if not e["scale_trusted"]],
        "images": images, "tasks": tasks,
    }
    _write_json(work / "manifest.json", manifest)
    vids = "\n".join(f"  {v['participant']}  {v['id']}  {v['name']}" for v in videos)
    (work / "CREDITS.txt").write_text(CREDITS.format(videos=vids), encoding="utf-8")

    selection = {
        "version": 1, "created": started, "note": "PRIVATE: never serve or send. Algorithm estimates are "
        "UltraTrack / TimTrack / UltraTimTrack outputs from the authors, not ground truth.",
        "code_sha256": sha256_file(_this_file()), "api_check": api,
        "settings": {"N_PER_VIDEO": n_per_video, "NB_DT_S": NB_DT_S, "EDGE_S": EDGE_S, "MIN_SEP_S": MIN_SEP_S,
                     "DESC_SIZE": DESC_SIZE, "DUP_NCC": DUP_NCC, "QUALITY_PCT": QUALITY_PCT,
                     "BUDGET_BYTES": budget_bytes, "NOMINAL_DEPTH_MM": NOMINAL_DEPTH_MM},
        "estimates_zip": None if not estimates else {k: v for k, v in estimates.items() if k != "participants"},
        "dropped_neighbour_levels": dropped, "leftover_tmp_files": leftovers,
        "untrusted_scale_participants": untrusted, "png_bytes_total": total_png(),
        "tasks": [{k: it[k] for k in ("task_id", "participant", "video", "file_id", "rank", "frame", "time_s",
                                      "px_per_cm", "scale_trusted")} | {"neighbours": it["neighbours"]}
                  for it in items],
        "videos": per_video,
    }
    _write_json(priv / "work_selection.json", selection)
    problems = validate(work)
    summary = {"n_tasks": len(tasks), "n_images": len(images), "png_mb": round(total_png() / 1e6, 2),
               "videos_ok": sum(v.get("ok", False) for v in per_video), "videos_failed":
               [v["participant"] for v in per_video if not v.get("ok")], "dropped_neighbours": dropped,
               "untrusted_scale_participants": untrusted, "problems": problems}
    log("summary", json.dumps(summary))
    return summary


def _write_json(path: Path, obj) -> None:
    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        raise TypeError(type(o))

    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=default), encoding="utf-8")
    tmp.replace(path)


def validate(work: Path) -> list[str]:
    """Light checks that the annotator can load the folder."""

    problems = []
    man = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
    ids = [e["image_id"] for e in man["images"]]
    if len(ids) != len(set(ids)):
        problems.append("duplicate image ids")
    by_id = {e["image_id"]: e for e in man["images"]}
    for e in man["images"]:
        if not (work / e["file"]).exists():
            problems.append(f"missing {e['file']}")
        if not e["px_per_cm"] > 0:
            problems.append(f"bad px_per_cm {e['image_id']}")
    for t in man["tasks"]:
        for m in [t["image_id"], *t["members"], *t.get("frames", [])]:
            if m not in by_id:
                problems.append(f"task {t['task_id']} refers to unknown image {m}")
    for p in work.rglob("*"):
        if p.is_file() and re.search(r"IMG_\d{5}", p.name):
            problems.append(f"competition-style name in work set: {p.name}")
    return problems


def make_zip(out_root: Path, zip_path: Path) -> int:
    out_root = Path(out_root)
    files = [p for p in sorted((out_root / "work").rglob("*")) if p.is_file()]
    files.append(out_root / "private" / "work_selection.json")
    with zipfile.ZipFile(zip_path, "w") as zf:
        for p in files:
            comp = zipfile.ZIP_STORED if p.suffix.lower() == ".png" else zipfile.ZIP_DEFLATED
            zf.write(p, p.relative_to(out_root).as_posix(), compress_type=comp)
    return Path(zip_path).stat().st_size


# ----------------------------------------------------------------------------- Kaggle entry point
def main() -> None:
    t0 = time.time()
    working = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd() / "labelling_frames_out"
    working.mkdir(parents=True, exist_ok=True)
    api = check_api()
    log("figshare API:", json.dumps(api))
    plan = {"videos": VIDEOS, "estimates": ESTIMATES if WITH_ESTIMATES else None,
            "bytes": sum(v["size"] for v in VIDEOS) + (ESTIMATES["size"] if WITH_ESTIMATES else 0), "api": api}
    _write_json(working / "labelling_work_plan.json", plan)
    if api["problems"]:
        raise SystemExit("figshare metadata changed; not downloading: " + "; ".join(api["problems"]))
    if MODE == "api-check":
        log(f"api-check only: would download {plan['bytes']:,} bytes; nothing downloaded")
        return
    tmp_root = Path("/tmp") if Path("/tmp").is_dir() else None
    dl_dir = Path(tempfile.mkdtemp(prefix="utt_dl_", dir=tmp_root))
    estimates = None
    try:
        if WITH_ESTIMATES:  # optional: without it the scale falls back to the nominal 50 mm depth
            zpath = None
            try:
                zpath = download(ESTIMATES, dl_dir)
                estimates = read_estimates(zpath, {v["participant"]: Path(v["name"]).stem for v in VIDEOS})
            except Exception as exc:
                log(f"WARNING: tracking estimates not used ({exc!r}); scale = nominal 50 mm depth")
            finally:
                if zpath is not None and zpath.exists():
                    zpath.unlink()
        out_root = working / "labelling_build"
        if out_root.exists():
            shutil.rmtree(out_root)
        summary = build_work_set(VIDEOS, out_root, estimates=estimates, fetch=lambda v: download(v, dl_dir),
                                 delete_after=True, api=api)
        zip_path = working / OUT_ZIP_NAME
        size = make_zip(out_root, zip_path)
        summary["zip_bytes"] = size
        summary["zip_sha256"] = sha256_file(zip_path)
        summary["minutes"] = round((time.time() - t0) / 60, 1)
        _write_json(working / "labelling_work_summary.json", summary)
        shutil.rmtree(out_root)  # the zip is the only large output
        log(f"wrote {zip_path} ({size / 1e6:.1f} MB)" + ("  WARNING: over 80 MB" if size > 80_000_000 else ""))
        if summary["problems"]:
            raise SystemExit("validation problems: " + "; ".join(summary["problems"][:10]))
    finally:
        shutil.rmtree(dl_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
