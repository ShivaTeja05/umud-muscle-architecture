"""GASFAS-BENCH (v31, lane gasfas): the production map + geometry pipeline on the GasFasTrack validation frames.

Dataset: Drazan, Hullfish & Baxter 2019 (PeerJ 7:e7120), Zenodo record 2598553, CC BY 4.0. Medial gastrocnemius,
Telemed (Echo Wave II, 6 cm probe, 5 cm depth), 5 subjects, maximal isometric / isokinetic plantarflexion. 3 examiners
x 3 rating days re-drew the fascicle line on ~6 validation frames per push; fascicle length (mm) and pennation (deg)
are stored in validation_data/Day d/Examiner e/Subject s/TrackedData.mat.

Stages (each runs in its own subprocess, launched by kernel.py; CPU only):
  data     Zenodo zips are NEVER downloaded whole: the zip central directories and single members are read through
           HTTP Range requests (about 1.3 GB in total). (1) every .mat of validation_data.zip -> labels.csv,
           params.csv, schema_dump.json (generic recursive parser: the field names were inferred from the authors'
           MATLAB code and are verified here). (2) the *_automatic.mp4 clips of the validation zip carry burned-in
           overlays (aponeurosis / fascicle lines, text) and cannot be fed to a network; they are used ONLY to find
           the raw frames: one overlay clip per (subject, trial, push) is aligned as a sequence to the raw full-trial
           MP4 of the subject zip (masked NCC on thumbnails, one frame offset per clip), then the raw frame of every
           labelled clip frame is cropped with the authors' crop box (found by template matching, cross-checked with
           Params.imCropRect) and written as PNG. Map frame = that crop (label pixel coordinates apply directly).
  apo      apo_v2fix band / upper / lower  \
  ours     ours_apo384 / ours_fasc384       > kaggle/tt_bench/tt_stages.py, unchanged code, re-pointed at the crops
  orient   fasc_orient_v1 seg / cos2 / sin2 /
  measure  the production geometry (scripts/build_candidates_v3.measure_one body + telemed_fl.extra_one, i.e. the
           cs_measure.measure function of artifacts/candidates_v17/crop-stress) from the embedded production sources
           -> gf_out/cache/gasfas__v2.csv in the production v3 cache schema (+ scale / viewport info).
Outputs (small): gf_out/{labels.csv, params.csv, schema_dump.json, matches.csv, manifest.csv, data_info.json,
cache/gasfas__v2.csv, viewports/*.csv, qa/*.jpg, logs/}. The mask bank is deleted after a complete measure stage
unless GF_KEEP_BANK=1 (or smoke mode).
Nothing here uses competition test images, test labels or leaderboard information.
GPL-3.0-only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import struct
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import zlib
from pathlib import Path

import numpy as np

WORK = Path("/kaggle/working")
TMP = Path(os.environ.get("GF_TMP", "/tmp/gf"))
OUT = Path(os.environ.get("GF_OUT", str(WORK / "gf_out")))
BANK = OUT / "maskbank"
CROPS = TMP / "crops"
DL = TMP / "dl"
MANIFEST = OUT / "manifest.csv"
SMOKE = int(os.environ.get("GF_SMOKE", "0"))            # >0: frames per subject sent through the networks
KEEP_BANK = int(os.environ.get("GF_KEEP_BANK", "0"))
MAX_FRAMES = int(os.environ.get("GF_MAX_FRAMES", "1100"))
RECORD = "2598553"
API = f"https://zenodo.org/api/records/{RECORD}/files"
UA = {"User-Agent": "UMUD-GasFasTrack-bench/1.0 (research use of CC BY 4.0 record 2598553)"}
PAUSE = 1.0                                              # seconds between requests (Zenodo guest rate limits)
VAL_ZIP = "validation_data.zip"
RAW_ZIP = {"1": "ultrasound_data_AGR001.zip", "3": "ultrasound_data_AGR003.zip", "4": "ultrasound_data_AGR004.zip",
           "5": "ultrasound_data_AGR005.zip", "6": "ultrasound_data_GAP020.zip"}   # assumed; verified by matching
SMOKE_TRIALS = ("p6_isometric_max_01", "p6_isokinetic500_01")
ANCHOR_TRIAL = "p6_isometric_max_01"
THUMB_W = 416
TOP_FRAC = 0.85                                          # rows used for matching (the bottom carries overlay text)
WHITE = 240                                              # overlay pixels (white lines / markers / text)
ACCEPT_SEQ = 0.70                                        # median sequence NCC to accept a raw video for a trial
DEFAULT_PX_PER_MM = 11.30                                # params_6cm.m default, for a 728 px wide crop
DEFAULT_CROP_W = 728
T0 = time.time()


def log(msg: str) -> None:
    print(f"[gf {time.time() - T0:7.1f}s] {msg}", flush=True)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------------------------- remote zip
def _backoff(e: Exception, attempt: int, what: str) -> None:
    wait = min(600, 20 * 2 ** attempt)
    if isinstance(e, urllib.error.HTTPError) and e.code == 429:
        ra = e.headers.get("Retry-After") if e.headers else None
        wait = max(wait, int(ra) + 5 if ra and str(ra).isdigit() else 90)
    log(f"  {what}: {e!r}; retry in {wait}s")
    time.sleep(wait)


def http_size(url: str) -> int:
    for a in range(6):
        try:
            req = urllib.request.Request(url, headers=UA, method="HEAD")
            with urllib.request.urlopen(req, timeout=180) as r:
                n = int(r.headers["Content-Length"])
            time.sleep(PAUSE)
            return n
        except Exception as e:  # noqa: BLE001
            _backoff(e, a, f"HEAD {url}")
    raise RuntimeError(f"no size for {url}")


def http_range(url: str, start: int, end: int) -> bytes:
    want = end - start + 1
    for a in range(7):
        try:
            req = urllib.request.Request(url, headers=dict(UA, Range=f"bytes={start}-{end}"))
            with urllib.request.urlopen(req, timeout=900) as r:
                if r.status != 206:
                    raise RuntimeError(f"server ignored Range (status {r.status})")
                data = r.read()
            if len(data) != want:
                raise RuntimeError(f"short read {len(data)} of {want}")
            time.sleep(PAUSE)
            return data
        except Exception as e:  # noqa: BLE001
            _backoff(e, a, f"GET {url.rsplit('/', 2)[-2]} [{start}-{end}]")
    raise RuntimeError(f"giving up on {url} [{start}-{end}]")


class RemoteZip:
    """Random access to single members of a zip through a byte-range reader (the archive is never read whole)."""

    def __init__(self, name: str, reader=None, total: int | None = None):
        self.name = name
        if reader is None:
            url = f"{API}/{urllib.parse.quote(name)}/content"
            reader = lambda s, e: http_range(url, s, e)  # noqa: E731
            total = http_size(url)
        self.reader, self.total = reader, int(total)
        self.bytes_read = 0
        self.members = self._central_directory()

    def _get(self, start: int, end: int) -> bytes:
        b = self.reader(start, end)
        self.bytes_read += len(b)
        return b

    def _central_directory(self) -> dict:
        tail_len = min(self.total, 256 * 1024)
        tail = self._get(self.total - tail_len, self.total - 1)
        i = tail.rfind(b"PK\x05\x06")
        if i < 0:
            raise RuntimeError(f"{self.name}: no end-of-central-directory record")
        n_total, cd_size, cd_off = struct.unpack("<HII", tail[i + 10:i + 20])
        if cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF or n_total == 0xFFFF:
            j = tail.rfind(b"PK\x06\x07", 0, i)
            if j < 0:
                raise RuntimeError(f"{self.name}: zip64 archive without locator")
            z64_off = struct.unpack("<Q", tail[j + 8:j + 16])[0]
            z = self._get(z64_off, z64_off + 55)
            if z[:4] != b"PK\x06\x06":
                raise RuntimeError(f"{self.name}: bad zip64 end record")
            n_total, cd_size, cd_off = struct.unpack("<QQQ", z[32:56])
        if cd_size > 64 * 1024 * 1024:
            raise RuntimeError(f"{self.name}: central directory unexpectedly large ({cd_size} B)")
        cd = self._get(cd_off, cd_off + cd_size - 1)
        out, p = {}, 0
        while p + 46 <= len(cd) and cd[p:p + 4] == b"PK\x01\x02":
            method, _t, _d, crc, csize, usize, nlen, elen, clen = struct.unpack("<HHHIIIHHH", cd[p + 10:p + 34])
            lho = struct.unpack("<I", cd[p + 42:p + 46])[0]
            name = cd[p + 46:p + 46 + nlen].decode("utf-8", errors="replace")
            extra = cd[p + 46 + nlen:p + 46 + nlen + elen]
            q = 0
            while q + 4 <= len(extra):                       # zip64 extended information
                hid, hsz = struct.unpack("<HH", extra[q:q + 4])
                if hid == 1:
                    body, b = extra[q + 4:q + 4 + hsz], 0
                    if usize == 0xFFFFFFFF:
                        usize = struct.unpack("<Q", body[b:b + 8])[0]
                        b += 8
                    if csize == 0xFFFFFFFF:
                        csize = struct.unpack("<Q", body[b:b + 8])[0]
                        b += 8
                    if lho == 0xFFFFFFFF:
                        lho = struct.unpack("<Q", body[b:b + 8])[0]
                q += 4 + hsz
            out[name] = dict(method=method, crc=crc, csize=csize, usize=usize, offset=lho, nlen=nlen)
            p += 46 + nlen + elen + clen
        return out

    def read(self, member: str, dest: Path | None = None, chunk: int = 16 << 20):
        """Fetch one member (local header + data by Range), inflate, verify size and CRC-32."""
        m = self.members[member]
        if m["method"] not in (0, 8):
            raise RuntimeError(f"{member}: unsupported compression method {m['method']}")
        head = 30 + m["nlen"] + 1024
        first_end = min(self.total - 1, m["offset"] + head + min(m["csize"], chunk) - 1)
        buf = self._get(m["offset"], first_end)
        if buf[:4] != b"PK\x03\x04":
            raise RuntimeError(f"{member}: bad local header")
        n, e = struct.unpack("<HH", buf[26:30])
        if 30 + n + e > len(buf):
            raise RuntimeError(f"{member}: local header longer than expected")
        data_start = m["offset"] + 30 + n + e
        data_end = data_start + m["csize"]                   # exclusive
        dec = zlib.decompressobj(-15) if m["method"] == 8 else None
        sink = io.BytesIO() if dest is None else open(dest, "wb")
        crc, size = 0, 0
        try:
            part = buf[30 + n + e:30 + n + e + m["csize"]]
            pos = data_start + len(part)
            while True:
                raw = dec.decompress(part) if dec else part
                crc = zlib.crc32(raw, crc)
                size += len(raw)
                sink.write(raw)
                if pos >= data_end:
                    break
                part = self._get(pos, min(data_end, pos + chunk) - 1)
                pos += len(part)
            if dec:
                raw = dec.flush()
                crc = zlib.crc32(raw, crc)
                size += len(raw)
                sink.write(raw)
            if size != m["usize"] or (crc & 0xFFFFFFFF) != m["crc"]:
                raise RuntimeError(f"{member}: size/CRC mismatch ({size} vs {m['usize']})")
            return sink.getvalue() if dest is None else dest
        finally:
            if dest is not None:
                sink.close()


# ----------------------------------------------------------------------------------------------- .mat parsing
def plain(o):
    """scipy.io.loadmat(struct_as_record=False, squeeze_me=True) object -> dict / list / ndarray / scalar."""
    if hasattr(o, "_fieldnames"):
        return {f: plain(getattr(o, f)) for f in o._fieldnames}
    if isinstance(o, np.ndarray) and o.dtype == object:
        return [plain(v) for v in o.ravel()]
    return o


def plain_h5(f, o, depth: int = 0):
    """Best-effort MATLAB v7.3 (HDF5) reader with the same output convention as plain()."""
    import h5py

    if depth > 12:
        return None
    if isinstance(o, h5py.Group):
        return {k: plain_h5(f, o[k], depth + 1) for k in o.keys()}
    a = o[()]
    if isinstance(a, np.ndarray) and a.dtype == object:
        return [plain_h5(f, f[r], depth + 1) if r else None for r in a.ravel()]
    if o.attrs.get("MATLAB_class", b"") == b"char":
        return "".join(chr(int(c)) for c in np.asarray(a).ravel())
    a = np.asarray(a)
    return np.squeeze(a.T) if a.ndim else a


def load_mat(path: Path) -> tuple[dict, str]:
    import scipy.io

    try:
        d = scipy.io.loadmat(str(path), struct_as_record=False, squeeze_me=True)
        return {k: plain(v) for k, v in d.items() if not k.startswith("__")}, "v5"
    except NotImplementedError:
        import h5py

        with h5py.File(str(path), "r") as f:
            return {k: plain_h5(f, f[k]) for k in f.keys() if k != "#refs#"}, "v73"


def describe(node, path: str, out: dict, limit: int = 2500) -> None:
    """Flat schema description (path -> type / shape / small values) of a parsed .mat tree."""
    if len(out) >= limit:
        return
    if isinstance(node, dict):
        out[path or "."] = f"struct[{len(node)}]: {','.join(list(node)[:40])}"
        for k, v in node.items():
            describe(v, f"{path}.{k}" if path else k, out, limit)
    elif isinstance(node, list):
        out[path] = f"cell/structarray[{len(node)}]"
        if node:
            describe(node[0], path + "[0]", out, limit)
    elif isinstance(node, np.ndarray):
        s = f"ndarray{tuple(node.shape)} {node.dtype}"
        if node.size <= 8 and node.dtype.kind in "fiub":
            s += f" = {np.asarray(node, float).ravel().round(5).tolist()}"
        out[path] = s
    else:
        out[path] = f"{type(node).__name__} = {str(node)[:80]}"


def find_key(node, name: str, depth: int = 0):
    """First value stored under the key `name` (depth-first), or None."""
    if depth > 8:
        return None
    if isinstance(node, dict):
        if name in node:
            return node[name]
        for v in node.values():
            r = find_key(v, name, depth + 1)
            if r is not None:
                return r
    elif isinstance(node, list):
        for v in node[:4]:
            r = find_key(v, name, depth + 1)
            if r is not None:
                return r
    return None


def find_fascicles(node, path: tuple = (), parent=None):
    """Yield (path, node, parent) for every struct that has a 'validate' or an 'automatic' field."""
    if isinstance(node, dict):
        if "validate" in node or "automatic" in node:
            yield path, node, parent
            return
        for k, v in node.items():
            yield from find_fascicles(v, path + (str(k),), node)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from find_fascicles(v, path + (f"[{i}]",), node)


def trial_push(path: tuple) -> tuple[str | None, int | None]:
    trial, push = None, None
    for el in path:
        m = re.match(r"^(.*?)_?push_?0*(\d+)$", el)
        if m:
            push = int(m.group(2))
            if m.group(1):
                trial = m.group(1)
        elif re.match(r"^p\d+_", el) and trial is None:
            trial = el
    if push is None:
        idx = [int(el[1:-1]) for el in path if re.match(r"^\[\d+\]$", el)]
        if idx:
            push = idx[-1] + 1
    return trial, push


def vec(x) -> np.ndarray:
    if x is None or isinstance(x, (dict, list, str)):
        return np.zeros(0)
    try:
        return np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    except Exception:  # noqa: BLE001
        return np.zeros(0)


def mat2(x, n: int) -> np.ndarray:
    """(n, 2) point array or NaNs."""
    try:
        a = np.asarray(x, dtype=float)
        if a.size == 2 * n:
            return a.reshape(n, 2) if a.ndim < 2 or a.shape[-1] == 2 else a.T.reshape(n, 2)
    except Exception:  # noqa: BLE001
        pass
    return np.full((n, 2), np.nan)


def lines12(pi) -> list[float]:
    """plotInsertions entry -> [x1, y1, x2, y2] of its three 2-point lines (order as stored), NaN if unreadable."""
    out = [float("nan")] * 12
    try:
        if isinstance(pi, dict) and "x" in pi and "y" in pi:       # struct of cell fields
            xs, ys = pi["x"], pi["y"]
            ent = [dict(x=xs[i], y=ys[i]) for i in range(min(3, len(xs)))] if isinstance(xs, list) else [pi]
        else:
            ent = list(pi) if isinstance(pi, (list, tuple)) else []
        for i, e in enumerate(ent[:3]):
            x, y = vec(e.get("x")), vec(e.get("y"))
            if x.size >= 2 and y.size >= 2:
                out[4 * i:4 * i + 4] = [float(x[0]), float(y[0]), float(x[-1]), float(y[-1])]
    except Exception:  # noqa: BLE001
        pass
    return out


def cell_list(x, n: int) -> list:
    """A per-frame cell array as a list of n entries (handles a flattened n x 3 struct array)."""
    if isinstance(x, list):
        if len(x) == n:
            return x
        if n and len(x) == 3 * n and all(isinstance(v, dict) for v in x):
            return [x[3 * i:3 * i + 3] for i in range(n)]
    return [None] * n


def pick(a: np.ndarray, k: int, f: int, n: int) -> float:
    """Value of a per-frame array for the k-th validated frame f (arrays are either per validated frame or per clip
    frame)."""
    if a.size == n:
        return float(a[k])
    if a.size >= f >= 1:
        return float(a[f - 1])
    return float("nan")


LINE_COLS = [f"{p}_l{i}{c}" for p in ("v", "a") for i in (1, 2, 3) for c in ("x1", "y1", "x2", "y2")]
LABEL_COLS = (["mat", "source", "day", "examiner", "subject", "session", "trial", "push", "path", "k", "frame", "n_val",
               "t", "fl_mm", "pa_deg", "ins_deep_x", "ins_deep_y", "ins_sup_x", "ins_sup_y", "a_fl_mm", "a_pa_deg",
               "n_auto", "sample_rate", "fi_first", "fi_last", "fi_n"] + LINE_COLS +
              ["px2mmX", "px2mmY", "crop_x", "crop_y", "crop_w", "crop_h", "mat_sha1", "fmt"])


def params_of(tree: dict) -> dict:
    out = {}
    for k in ("px2mmX", "px2mmY"):
        v = vec(find_key(tree, k))
        out[k] = float(v[0]) if v.size else float("nan")
    rect = vec(find_key(tree, "imCropRect"))
    for i, k in enumerate(("crop_x", "crop_y", "crop_w", "crop_h")):
        out[k] = float(rect[i]) if rect.size >= 4 else float("nan")
    return out


def label_rows(tree: dict, meta: dict) -> list[dict]:
    """All examiner-validated frames of one TrackedData.mat (one row per clip and validated frame)."""
    rows = []
    par = params_of(tree)
    for path, node, parent in find_fascicles(tree):
        val, aut = node.get("validate"), node.get("automatic")
        if not isinstance(val, dict):
            continue
        trial, push = trial_push(path)
        L, Pn, tm = vec(val.get("length")), vec(val.get("pennation")), vec(val.get("time"))
        fr = vec(val.get("trackFrames"))
        if fr.size == 0 and L.size:
            fr = np.flatnonzero(np.isfinite(L)) + 1.0
        fr = np.rint(fr[np.isfinite(fr)]).astype(int)
        n = int(fr.size)
        if n == 0:
            continue
        iD, iS = mat2(val.get("insertDeep"), n), mat2(val.get("insertSup"), n)
        PI = cell_list(val.get("plotInsertions"), n)
        a = aut if isinstance(aut, dict) else {}
        La, Pa = vec(a.get("length")), vec(a.get("pennation"))
        fa = np.rint(vec(a.get("trackFrames"))).astype(int)
        PIa = a.get("plotInsertions") if isinstance(a.get("plotInsertions"), list) else []
        sr = vec(val.get("sampleRate") if val.get("sampleRate") is not None else a.get("sampleRate"))
        fi = vec(find_key(parent, "frameInds")) if isinstance(parent, dict) else np.zeros(0)
        for k, f in enumerate(fr.tolist()):
            ja = int(np.flatnonzero(fa == f)[0]) if fa.size and (fa == f).any() else (f - 1 if not fa.size else -1)
            al = lines12(PIa[ja]) if 0 <= ja < len(PIa) else [float("nan")] * 12
            row = dict(meta, trial=trial, push=push, path=".".join(path), k=k, frame=f, n_val=n,
                       t=pick(tm, k, f, n), fl_mm=pick(L, k, f, n), pa_deg=pick(Pn, k, f, n),
                       ins_deep_x=float(iD[k, 0]), ins_deep_y=float(iD[k, 1]), ins_sup_x=float(iS[k, 0]),
                       ins_sup_y=float(iS[k, 1]),
                       a_fl_mm=float(La[ja]) if 0 <= ja < La.size else float("nan"),
                       a_pa_deg=float(Pa[ja]) if 0 <= ja < Pa.size else float("nan"),
                       n_auto=int(max(fa.size, La.size)), sample_rate=float(sr[0]) if sr.size else float("nan"),
                       fi_first=float(fi[0]) if fi.size else float("nan"),
                       fi_last=float(fi[-1]) if fi.size else float("nan"), fi_n=int(fi.size), **par)
            row.update(dict(zip(LINE_COLS, lines12(PI[k]) + al)))
            rows.append(row)
    return rows


def member_meta(name: str) -> dict:
    m = re.search(r"Day (\d+)/Examiner (\d+)/Subject (\d+)/(?:([^/]+)/)?[^/]+$", name)
    if not m:
        return dict(mat=name, source="other", day="", examiner="", subject="", session="")
    return dict(mat=name, source="session" if m.group(4) else "root", day=m.group(1), examiner=m.group(2),
                subject=m.group(3), session=m.group(4) or "")


def parse_all_labels(val: RemoteZip) -> tuple[list[dict], list[dict], dict]:
    labels, params, schema = [], [], {}
    mats = sorted(n for n in val.members if n.lower().endswith(".mat"))
    DL.mkdir(parents=True, exist_ok=True)
    seen_schema = set()
    for name in mats:
        meta = member_meta(name)
        try:
            raw = val.read(name)
            p = DL / "label.mat"
            p.write_bytes(raw)
            tree, fmt = load_mat(p)
            sha = hashlib.sha1(raw).hexdigest()
            base = name.rsplit("/", 1)[-1]
            key = (base, meta["examiner"])
            if key not in seen_schema:                      # one schema dump per file kind and examiner
                seen_schema.add(key)
                d = {}
                describe(tree, "", d)
                schema[name] = d
            if base.lower().startswith("tracked"):
                rows = label_rows(tree, dict(meta, mat_sha1=sha, fmt=fmt))
                labels.extend(rows)
                log(f"  {name}: {len(rows)} validated frames ({fmt})")
            pr = dict(meta, mat_sha1=sha, fmt=fmt, kind=base, **params_of(tree))
            sr = vec(find_key(tree, "sampleRate"))
            pr["sample_rate"] = float(sr[0]) if sr.size else float("nan")
            params.append(pr)
        except Exception:  # noqa: BLE001
            log(f"  FAILED {name}:\n{traceback.format_exc()}")
            schema[name] = {"error": traceback.format_exc()[-600:]}
    return labels, params, schema


def write_csv(path: Path, rows: list[dict], cols: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = cols or sorted({k for r in rows for k in r}, key=lambda c: (c not in rows[0], list(rows[0]).index(c)
                                                                    if c in rows[0] else 0, c)) if rows else (cols or [])
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ----------------------------------------------------------------------------------------------- video alignment
def gray_frames(path: Path):
    """Sequential decode: yields (index, gray uint8)."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {path}")
    i = 0
    try:
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            yield i, (cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if fr.ndim == 3 else fr)
            i += 1
    finally:
        cap.release()


def video_fps(path: Path) -> float:
    import cv2

    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else float("nan")
    cap.release()
    return fps


def load_video(path: Path) -> dict:
    """Thumbnails of every frame of a raw full-trial video."""
    import cv2

    thumbs, H, W, tw, th = [], None, None, None, None
    for _, g in gray_frames(path):
        if H is None:
            H, W = g.shape
            tw = min(THUMB_W, W)
            th = max(8, int(round(H * tw / W)))
        thumbs.append(cv2.resize(g, (tw, th), interpolation=cv2.INTER_AREA))
    if not thumbs:
        raise RuntimeError(f"no frames in {path}")
    return dict(thumbs=np.stack(thumbs), H=H, W=W, sx=tw / W, sy=th / H, n=len(thumbs), fps=video_fps(path))


def load_clip(path: Path, sx: float, sy: float, keep: set, size: tuple | None = None) -> dict:
    """Overlay clip: thumbnails at the scale of the raw-video thumbnails, overlay masks, full frames for `keep`.
    size = (w, h): resize every clip frame to this size first (fallback when the clip is not pixel-identical)."""
    import cv2

    ths, oks, full, h, w = [], [], {}, 0, 0
    for i, g in gray_frames(path):
        if size is not None and (g.shape[1], g.shape[0]) != tuple(size):
            g = cv2.resize(g, tuple(size), interpolation=cv2.INTER_AREA)
        h, w = g.shape
        tw, th = max(8, int(round(w * sx))), max(8, int(round(h * sy)))
        ths.append(cv2.resize(g, (tw, th), interpolation=cv2.INTER_AREA))
        white = cv2.resize((g >= WHITE).astype(np.float32), (tw, th), interpolation=cv2.INTER_AREA)
        oks.append(cv2.dilate((white > 0.04).astype(np.uint8), np.ones((3, 3), np.uint8)) == 0)
        if i in keep:
            full[i] = g
    if not ths:
        raise RuntimeError(f"no frames in {path}")
    return dict(th=np.stack(ths), ok=np.stack(oks), full=full, h=h, w=w, n=len(ths))


def locate_crop(V: dict, C: dict, j: int) -> tuple[float, int, int, int] | None:
    """(ncc, x, y, frame) of clip frame j inside the raw frames at thumbnail scale (coarse-to-fine in time)."""
    import cv2

    th = C["th"].shape[1]
    tpl = C["th"][j][: max(8, int(TOP_FRAC * th))]
    if tpl.shape[0] > V["thumbs"].shape[1] or tpl.shape[1] > V["thumbs"].shape[2]:
        return None
    best = (-2.0, 0, 0, -1)
    step = max(1, V["n"] // 240)

    def scan(idx):
        nonlocal best
        for i in idx:
            r = cv2.matchTemplate(V["thumbs"][i], tpl, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(r)
            if mx > best[0]:
                best = (float(mx), int(loc[0]), int(loc[1]), int(i))

    scan(range(0, V["n"], step))
    if step > 1 and best[3] >= 0:
        scan(range(max(0, best[3] - step), min(V["n"], best[3] + step + 1)))
    return best


def ncc_rows(X: np.ndarray, q: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """NCC of every row of X with q over the pixels ok (overlay pixels excluded)."""
    if ok.mean() < 0.3:
        ok = np.ones_like(ok)
    a = X[:, ok]
    a = a - a.mean(1, keepdims=True)
    a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-6
    b = q[ok].astype(np.float32)
    b = b - b.mean()
    b /= np.linalg.norm(b) + 1e-6
    return a @ b


def match_clip(V: dict, C: dict, x: int, y: int, frames: list[int]) -> dict | None:
    """Sequence alignment of an overlay clip to the raw video: one frame offset for the whole clip (the clip is a
    contiguous cut of the trial), chosen by the summed masked NCC of ~9 spread frames + the labelled frames."""
    th, tw = C["th"].shape[1:]
    hh = max(8, int(TOP_FRAC * th))
    if y + hh > V["thumbs"].shape[1] or x + tw > V["thumbs"].shape[2]:
        return None
    X = V["thumbs"][:, y:y + hh:2, x:x + tw:2].reshape(V["n"], -1).astype(np.float32)
    n, N = C["n"], V["n"]
    js = sorted(set(int(v) for v in np.rint(np.linspace(0, n - 1, 9))) | {f - 1 for f in frames if 1 <= f <= n})
    S = np.stack([ncc_rows(X, C["th"][j][:hh:2, ::2].ravel(), C["ok"][j][:hh:2, ::2].ravel()) for j in js])
    del X
    lo, hi = -js[0], N - 1 - js[-1]
    res = dict(n_clip=n, n_samples=len(js), off=None, seq=float("nan"), margin=float("nan"), cons=float("nan"),
               frames={})
    if hi >= lo:
        offs = np.arange(lo, hi + 1)
        tot = sum(S[k, offs + j] for k, j in enumerate(js)) / len(js)
        b = int(np.argmax(tot))
        off = int(offs[b])
        far = np.abs(offs - off) > 2
        res.update(off=off, seq=float(tot[b]), margin=float(tot[b] - tot[far].max()) if far.any() else float("nan"),
                   cons=float(np.mean([S[k, off + j] >= S[k].max() - 0.02 for k, j in enumerate(js)])))
    for f in frames:
        j = f - 1
        if not 0 <= j < n:
            res["frames"][f] = dict(in_clip=False)
            continue
        k = js.index(j)
        d = int(np.argmax(S[k]))
        fr = dict(in_clip=True, idx_direct=d, ncc_direct=float(S[k, d]))
        if res["off"] is not None:
            fr.update(idx_model=res["off"] + j, ncc_model=float(S[k, res["off"] + j]))
        res["frames"][f] = fr
    return res


def grab_frames(path: Path, idxs) -> dict:
    need = set(int(i) for i in idxs)
    out = {}
    if not need:
        return out
    last = max(need)
    for i, g in gray_frames(path):
        if i in need:
            out[i] = g
        if i >= last:
            break
    return out


def refine_box(full: np.ndarray, clipf: np.ndarray, x0: int, y0: int, rad: int = 14):
    """Exact integer position of the clip frame inside the raw frame (template = overlay-free part of the clip)."""
    import cv2

    h, w = clipf.shape
    mx_ = int(0.08 * w)
    tpl = clipf[: int(TOP_FRAC * h), mx_: w - mx_]
    ax0 = x0 + mx_
    sx0, sy0 = max(0, ax0 - rad), max(0, y0 - rad)
    sx1, sy1 = min(full.shape[1], ax0 + tpl.shape[1] + rad), min(full.shape[0], y0 + tpl.shape[0] + rad)
    win = full[sy0:sy1, sx0:sx1]
    if win.shape[0] < tpl.shape[0] or win.shape[1] < tpl.shape[1]:
        return None
    r = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
    _, mx, _, loc = cv2.minMaxLoc(r)
    return int(sx0 + loc[0] - mx_), int(sy0 + loc[1]), float(mx)


def masked_ncc(a: np.ndarray, b: np.ndarray) -> float:
    """NCC of two equal-size images over the top rows, excluding near-white pixels of b (the overlay clip frame)."""
    h = int(TOP_FRAC * min(a.shape[0], b.shape[0]))
    w = min(a.shape[1], b.shape[1])
    a, b = a[:h, :w].astype(np.float32), b[:h, :w].astype(np.float32)
    ok = b < WHITE
    if ok.mean() < 0.3:
        ok = np.ones_like(ok)
    a, b = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))


def line_whiteness(g: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> float:
    """Mean gray value along a line (MATLAB 1-based pixel coordinates) - high where a white overlay line was drawn."""
    if not np.all(np.isfinite([x1, y1, x2, y2])):
        return float("nan")
    t = np.linspace(0.1, 0.9, 60)
    xs = np.clip(np.rint(x1 - 1 + t * (x2 - x1)).astype(int), 0, g.shape[1] - 1)
    ys = np.clip(np.rint(y1 - 1 + t * (y2 - y1)).astype(int), 0, g.shape[0] - 1)
    return float(g[ys, xs].mean())


def raw_candidates(zf: RemoteZip, trial: str, used: set) -> list[str]:
    """Raw full-trial MP4 members that may hold `trial`: the exact name first, then unused p6_* trials (the
    500 deg/s trials were renamed in some subject folders)."""
    vids = sorted(n for n in zf.members if n.lower().endswith(".mp4") and "/mp4/" in ("/" + n.lower()))
    stem = lambda n: n.rsplit("/", 1)[-1][:-4]  # noqa: E731
    exact = [n for n in vids if stem(n) == trial]
    if exact:
        return exact[:1]

    def rank(n):
        s = stem(n)
        return (0 if s.startswith("p6_isokinetic") else 1 if s.startswith("p6_eccentric") else
                2 if s.startswith("p6_") and "passive" not in s and "practice" not in s else 9, s)

    alt = [n for n in vids if stem(n) not in used and rank(n)[0] < 9]
    return sorted(alt, key=rank)[:5]


def clip_member(val: RemoteZip, subject: str, trial: str, push: int) -> str | None:
    pat = re.compile(rf"/Subject {re.escape(subject)}/[^/]+/{re.escape(trial)}_push_0*{push}_automatic\.mp4$")
    c = sorted(n for n in val.members if pat.search(n))
    pref = [n for n in c if "/Day 1/Examiner 1/" in n]
    return (pref or c or [None])[0]


def fetch(zf: RemoteZip, member: str, tag: str) -> Path:
    p = DL / tag / (hashlib.sha1(member.encode()).hexdigest()[:12] + "_" + member.rsplit("/", 1)[-1])
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".part")
        zf.read(member, tmp)
        tmp.rename(p)
    return p


def try_trial(val: RemoteZip, zf: RemoteZip, member: str, subject: str, trial: str, pushes: dict,
              crop_hint: tuple | None) -> dict | None:
    """Align all overlay clips of one trial to one raw video. pushes = {push: [labelled frames]}."""
    vpath = fetch(zf, member, "raw")
    V = load_video(vpath)
    log(f"    raw {member}: {V['n']} frames {V['W']}x{V['H']} fps {V['fps']:.2f}")
    clips, locs = {}, []
    for push, frames in sorted(pushes.items()):
        cm = clip_member(val, subject, trial, push)
        if cm is None:
            log(f"    no overlay clip for {trial} push {push}")
            continue
        keep = {f - 1 + d for f in frames for d in (-1, 0, 1)}
        C = load_clip(fetch(val, cm, "clip"), V["sx"], V["sy"], keep)
        j = min(max(0, frames[len(frames) // 2] - 1), C["n"] - 1)
        loc = locate_crop(V, C, j)
        resized = False
        if (loc is None or loc[0] < 0.5) and crop_hint is not None:    # clip not pixel-identical: Params crop size
            C = load_clip(fetch(val, cm, "clip"), V["sx"], V["sy"], keep, size=(crop_hint[2], crop_hint[3]))
            loc = locate_crop(V, C, j)
            resized = True
        if loc is None:
            continue
        clips[push] = dict(C=C, member=cm, loc=loc, resized=resized, frames=frames)
        locs.append(loc)
    if not clips:
        return None
    good = [l for l in locs if l[0] >= 0.5] or locs
    xy = max(set((l[1], l[2]) for l in good), key=lambda p: sum((l[1], l[2]) == p for l in good))
    seqs = []
    for push, c in clips.items():
        c["match"] = match_clip(V, c["C"], xy[0], xy[1], c["frames"])
        if c["match"] is not None and np.isfinite(c["match"]["seq"]):
            seqs.append(c["match"]["seq"])
    quality = float(np.median(seqs)) if seqs else -1.0
    return dict(V={k: v for k, v in V.items() if k != "thumbs"}, vpath=vpath, member=member, clips=clips, xy=xy,
                quality=quality, loc_ncc=float(np.median([l[0] for l in locs])))


def finish_trial(res: dict, subject: str, trial: str, raw_zip: str, crop_hint: tuple | None, fo) -> list[dict]:
    """Extract the raw frame of every labelled clip frame, fix the crop box, write the crops; -> match rows."""
    import cv2
    from PIL import Image

    V = res["V"]
    todo = []
    for push, c in sorted(res["clips"].items()):
        m = c["match"]
        if m is None:
            continue
        for f, fr in sorted(m["frames"].items()):
            if not fr.get("in_clip"):
                todo.append(dict(push=push, frame=f, reason="frame_outside_clip", n_clip=m["n_clip"]))
                continue
            use_model = "idx_model" in fr and m["seq"] >= 0.5
            todo.append(dict(push=push, frame=f, reason="", n_clip=m["n_clip"], off=m["off"], seq=m["seq"],
                             margin=m["margin"], cons=m["cons"], n_samples=m["n_samples"],
                             idx_direct=fr["idx_direct"], ncc_direct=fr["ncc_direct"],
                             idx_model=fr.get("idx_model", -1), ncc_model=fr.get("ncc_model", float("nan")),
                             idx_use=fr["idx_model"] if use_model else fr["idx_direct"],
                             idx_src="sequence" if use_model else "direct"))
    frames = grab_frames(res["vpath"], [t["idx_use"] for t in todo if not t["reason"]])
    x0p, y0p = int(round(res["xy"][0] / V["sx"])), int(round(res["xy"][1] / V["sy"]))
    boxes = []
    for t in todo:
        if t["reason"]:
            continue
        C = res["clips"][t["push"]]["C"]
        clipf, full = C["full"].get(t["frame"] - 1), frames.get(t["idx_use"])
        if clipf is None or full is None:
            t["reason"] = "frame_not_decoded"
            continue
        rb = refine_box(full, clipf, x0p, y0p)
        t["box_x"], t["box_y"], t["box_ncc"] = rb if rb else (x0p, y0p, float("nan"))
        if rb and rb[2] >= 0.6:
            boxes.append((rb[0], rb[1]))
    if boxes:
        bx, by = max(set(boxes), key=boxes.count)
        box_src = "match"
    elif crop_hint is not None:
        bx, by, box_src = int(crop_hint[0]), int(crop_hint[1]), "params"
    else:
        bx, by, box_src = x0p, y0p, "thumb"
    any_c = next(iter(res["clips"].values()))["C"]
    h, w = any_c["h"], any_c["w"]
    if crop_hint is not None and 0 <= w - crop_hint[2] <= 1 and 0 <= h - crop_hint[3] <= 1:
        w, h = int(crop_hint[2]), int(crop_hint[3])          # drop the even-size padding of the MPEG-4 writer
    bx, by = max(0, min(bx, V["W"] - w)), max(0, min(by, V["H"] - h))
    vp = ("", -1, -1, -1, -1)
    rows = []
    for t in todo:
        row = dict(subject=subject, trial=trial, raw_zip=raw_zip, raw_member=res["member"],
                   clip_member=res["clips"][t["push"]]["member"], clip_resized=res["clips"][t["push"]]["resized"],
                   n_full=V["n"], fps=V["fps"], H0=V["H"], W0=V["W"], loc_ncc=res["loc_ncc"],
                   trial_quality=res["quality"], box_src=box_src, x1=bx, y1=by, x2=bx + w, y2=by + h, **t)
        if not t["reason"]:
            C = res["clips"][t["push"]]["C"]
            full, clipf = frames[t["idx_use"]], C["full"][t["frame"] - 1]
            crop = np.ascontiguousarray(full[by:by + h, bx:bx + w])
            key = f"{trial}__p{int(t['push']):02d}__f{int(t['frame']):04d}"
            p = CROPS / f"gf{subject}" / f"{key}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(crop).save(p)
            if vp[0] == "" and fo is not None:
                try:
                    b, kind = fo.viewport(full, ".png")
                    vp = (str(kind), *[int(v) for v in b])
                except Exception as e:  # noqa: BLE001
                    vp = (f"error:{type(e).__name__}", -1, -1, -1, -1)
            lab = res["clips"][t["push"]].get("auto_fas", {}).get(t["frame"])
            wh = [line_whiteness(C["full"][t["frame"] - 1 + d], *lab) if lab and (t["frame"] - 1 + d) in C["full"]
                  else float("nan") for d in (-1, 0, 1)]
            row.update(subset=f"gf{subject}", key=key, src=f"{raw_zip}::{res['member']}#{t['idx_use']}",
                       kind="authors_crop", Hc=crop.shape[0], Wc=crop.shape[1],
                       crop_sha1=hashlib.sha1(crop.tobytes()).hexdigest(), ncc_full=masked_ncc(crop, clipf),
                       crop_mean=float(crop.mean()), ov_white_m1=wh[0], ov_white_0=wh[1], ov_white_p1=wh[2],
                       vp_kind=vp[0], vp_x1=vp[1], vp_y1=vp[2], vp_x2=vp[3], vp_y2=vp[4])
        rows.append(row)
    # QA montage: raw crop | overlay clip frame, first labelled frame of each push
    try:
        qa = OUT / "qa"
        qa.mkdir(parents=True, exist_ok=True)
        done = set()
        for r in rows:
            if r.get("key") and r["push"] not in done and len(done) < 2:
                done.add(r["push"])
                a = np.asarray(Image.open(CROPS / r["subset"] / f"{r['key']}.png"))
                b = res["clips"][r["push"]]["C"]["full"][r["frame"] - 1]
                hh = min(a.shape[0], b.shape[0])
                pair = np.concatenate([a[:hh], b[:hh]], axis=1)
                s = 760.0 / pair.shape[1]
                pair = cv2.resize(pair, (760, max(8, int(pair.shape[0] * s))), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(qa / f"gf{subject}__{r['key']}.jpg"), pair, [cv2.IMWRITE_JPEG_QUALITY, 70])
    except Exception:  # noqa: BLE001
        log(f"    qa montage failed:\n{traceback.format_exc()}")
    return rows


MATCH_COLS = ["subset", "key", "src", "H0", "W0", "x1", "y1", "x2", "y2", "kind", "Hc", "Wc", "crop_sha1",
              "subject", "trial", "push", "frame", "reason", "raw_zip", "raw_member", "clip_member", "clip_resized",
              "n_full", "fps", "n_clip", "n_samples", "off", "seq", "margin", "cons", "idx_use", "idx_src",
              "idx_model", "ncc_model", "idx_direct", "ncc_direct", "loc_ncc", "trial_quality", "box_src", "box_x",
              "box_y", "box_ncc", "ncc_full", "crop_mean", "ov_white_m1", "ov_white_0", "ov_white_p1", "vp_kind",
              "vp_x1", "vp_y1", "vp_x2", "vp_y2", "px2mmX", "px2mmY", "px2mm_spread", "px_per_cm", "ppc_src",
              "fi_first", "fi_n"]


def nanmedian(v) -> float:
    a = np.asarray([x for x in v if x is not None and np.isfinite(x)], dtype=float)
    return float(np.median(a)) if a.size else float("nan")


def stage_data() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    DL.mkdir(parents=True, exist_ok=True)
    info = dict(record=RECORD, smoke=SMOKE, started=time.strftime("%Y-%m-%d %H:%M:%S"), subjects={}, errors=[])
    val = RemoteZip(VAL_ZIP)
    log(f"{VAL_ZIP}: {len(val.members)} members")
    labels, params, schema = parse_all_labels(val)
    write_csv(OUT / "labels.csv", labels, LABEL_COLS)
    write_csv(OUT / "params.csv", params)
    (OUT / "schema_dump.json").write_text(json.dumps(schema, indent=1, default=str))
    info["n_label_rows"], info["n_mats"] = len(labels), len(params)
    log(f"labels: {len(labels)} rows from {len(params)} .mat files")
    if not labels:
        info["errors"].append("no validated frames parsed - read schema_dump.json")
        (OUT / "data_info.json").write_text(json.dumps(info, indent=1, default=str))
        raise SystemExit(2)

    try:
        fo = load_module("fo_src", TMP / "fo_src.py")
    except Exception as e:  # noqa: BLE001
        log(f"fo_src not loaded ({e!r}); full-frame viewport info skipped")
        fo = None
    units, par, auto_fas, fi = {}, {}, {}, {}
    for r in labels:
        if not r["trial"] or r["push"] is None or not r["subject"]:
            continue
        s = str(r["subject"])
        units.setdefault(s, {}).setdefault(r["trial"], {}).setdefault(int(r["push"]), set()).add(int(r["frame"]))
        par.setdefault(s, []).append(r)
        a = [r[f"a_l3{c}"] for c in ("x1", "y1", "x2", "y2")]
        if np.all(np.isfinite(a)):
            auto_fas.setdefault((s, r["trial"], int(r["push"])), {}).setdefault(int(r["frame"]), a)
        if np.isfinite(r["fi_first"]):
            fi[(s, r["trial"], int(r["push"]))] = (r["fi_first"], r["fi_n"])
    zips: dict[str, RemoteZip] = {}

    def rz(name: str) -> RemoteZip:
        if name not in zips:
            zips[name] = RemoteZip(name)
            log(f"{name}: {len(zips[name].members)} members")
        return zips[name]

    matches = []
    for s in sorted(units):
        rows_s = par[s]
        sx_, sy_ = nanmedian(r["px2mmX"] for r in rows_s), nanmedian(r["px2mmY"] for r in rows_s)
        spread = float(np.nanmax([abs(r["px2mmX"] / sx_ - 1) for r in rows_s] + [0.0])) if np.isfinite(sx_) else np.nan
        rect = [nanmedian(r[k] for r in rows_s) for k in ("crop_x", "crop_y", "crop_w", "crop_h")]
        hint = (int(rect[0]) - 1, int(rect[1]) - 1, int(rect[2]) + 1, int(rect[3]) + 1) if np.all(np.isfinite(rect)) \
            else None                                         # MATLAB imcrop [x y w h] -> 0-based box, (w+1) x (h+1)
        trials = sorted(units[s])
        if SMOKE:
            trials = [t for t in trials if t in SMOKE_TRIALS] or trials[:1]
        trials = sorted(trials, key=lambda t: (t != ANCHOR_TRIAL, t))
        zname, used, sinfo = RAW_ZIP.get(s), set(), dict(trials={}, crop_hint=hint, px2mmX=sx_, px2mmY=sy_)
        order = ([zname] if zname else []) + [z for z in RAW_ZIP.values() if z != zname]
        for ti, trial in enumerate(trials):
            pushes = {p: sorted(f) for p, f in units[s][trial].items()}
            best = None
            for zi, zn in enumerate(order if ti == 0 else [zname]):   # the first trial also settles the subject zip
                try:
                    zf = rz(zn)
                    for member in raw_candidates(zf, trial, used):
                        res = try_trial(val, zf, member, s, trial, pushes, hint)
                        q = res["quality"] if res else -1.0
                        log(f"  subject {s} {trial} <- {zn}::{member}: quality {q:.3f}")
                        if res and (best is None or q > best[0]):
                            best = (q, zn, res)
                        if q >= ACCEPT_SEQ:
                            break
                except Exception:  # noqa: BLE001
                    log(f"  subject {s} {trial} via {zn} failed:\n{traceback.format_exc()}")
                    info["errors"].append(f"{s}/{trial}/{zn}: {traceback.format_exc()[-300:]}")
                if best and best[0] >= ACCEPT_SEQ:
                    break
            if best is None or best[0] < ACCEPT_SEQ:
                sinfo["trials"][trial] = dict(status="no_raw_video", best_quality=best[0] if best else None)
                for p, fr in pushes.items():
                    matches.extend(dict(subject=s, trial=trial, push=p, frame=f, reason="no_raw_video") for f in fr)
                continue
            q, zn, res = best
            if ti == 0:
                zname = zn
            used.add(res["member"].rsplit("/", 1)[-1][:-4])
            for p, c in res["clips"].items():
                c["auto_fas"] = auto_fas.get((s, trial, p), {})
            try:
                rows = finish_trial(res, s, trial, zn, hint, fo)
            except Exception:  # noqa: BLE001
                log(f"  finish {s}/{trial} failed:\n{traceback.format_exc()}")
                info["errors"].append(f"finish {s}/{trial}: {traceback.format_exc()[-300:]}")
                rows = []
            for r in rows:
                w_px = r["x2"] - r["x1"]
                if np.isfinite(sx_) and np.isfinite(sy_) and sx_ > 0 and sy_ > 0:
                    ppc, src = 10.0 / float(np.sqrt(sx_ * sy_)), "params"
                else:
                    ppc, src = 10.0 * DEFAULT_PX_PER_MM * w_px / DEFAULT_CROP_W, "default_scaled"
                f1 = fi.get((s, trial, int(r["push"])), (float("nan"), 0))
                r.update(px2mmX=sx_, px2mmY=sy_, px2mm_spread=spread, px_per_cm=ppc, ppc_src=src, fi_first=f1[0],
                         fi_n=f1[1])
            matches.extend(rows)
            sinfo["trials"][trial] = dict(status="ok", raw=f"{zn}::{res['member']}", quality=q, n_frames=len(rows),
                                          n_crops=sum(bool(r.get("key")) for r in rows))
            del res
        sinfo["raw_zip"] = zname
        info["subjects"][s] = sinfo
        shutil.rmtree(DL / "raw", ignore_errors=True)        # free the disk of this subject's raw videos
    write_csv(OUT / "matches.csv", matches, MATCH_COLS)
    man = [r for r in matches if r.get("key")]
    if SMOKE:
        keep = []
        for s in sorted({r["subject"] for r in man}):
            rs = [r for r in man if r["subject"] == s]
            idx = sorted(set(int(v) for v in np.rint(np.linspace(0, len(rs) - 1, min(SMOKE, len(rs))))))
            keep.extend(rs[i] for i in idx)
        man = keep
    elif len(man) > MAX_FRAMES:
        idx = sorted(set(int(v) for v in np.rint(np.linspace(0, len(man) - 1, MAX_FRAMES))))
        man = [man[i] for i in idx]
        info["capped"] = MAX_FRAMES
    write_csv(MANIFEST, man, MATCH_COLS)
    info.update(n_matches=len(matches), n_crops=sum(bool(r.get("key")) for r in matches), n_manifest=len(man),
                bytes_read={k: z.bytes_read for k, z in dict(zips, **{VAL_ZIP: val}).items()},
                elapsed_s=time.time() - T0)
    (OUT / "data_info.json").write_text(json.dumps(info, indent=1, default=str))
    log(f"data: {len(matches)} labelled frames, {info['n_crops']} crops, manifest {len(man)}; "
        f"read {sum(info['bytes_read'].values()) / 1e6:.0f} MB")
    if not man:
        raise SystemExit(3)


# ----------------------------------------------------------------------------------------------- map stages
def tt_module():
    """kaggle/tt_bench/tt_stages.py (unchanged), re-pointed at the GasFas crops and manifest."""
    tt = load_module("tt_stages", TMP / "tt_stages.py")
    tt.TMP, tt.OUT, tt.BANK, tt.CROPS, tt.MANIFEST, tt.SMOKE = TMP, OUT, BANK, CROPS, MANIFEST, SMOKE
    with MANIFEST.open(newline="", encoding="utf-8") as fh:
        tt.ORDER = sorted({r["subset"] for r in csv.DictReader(fh)})
    return tt


# ----------------------------------------------------------------------------------------------- measure
def measure(mods, A, up, lo, fbs, thr, edges, F, fthr, img, ppc, orient) -> dict:
    """build_candidates_v3.measure_one body + telemed_fl.extra_one on the same GeoResult (= cs_measure.measure of
    artifacts/candidates_v17/crop-stress with horizontal=False)."""
    cv2, g2, b3, ea, eo, tf = mods
    cfg = ea.make_cfg("v2e", thr, edges).with_(fasc_thr=fthr)
    if img is not None and img.shape != A.shape:
        img = cv2.resize(img, (A.shape[1], A.shape[0]), interpolation=cv2.INTER_AREA)
    res = g2.measure_apo(A, F, img, ppc, cfg, apo_upper=up, apo_lower=lo, fallbacks=fbs, keep_details=True,
                         orient=orient)
    if orient is not None and res.curves:
        out = g2.field_architecture(res, orient, b3.FCFG, cfg.pa_reference)
        res.variants.update(out["variants"])
        res.diag.update(out["diag"])
    row = eo.row_of(res)
    row.update(b3.constructed_grid(res, cfg.pa_reference))
    row["d_frag_sign_c"] = b3.frag_sign(res) if res.curves else np.nan
    if res.curves and "sup_curve" in res.curves:
        try:
            row.update({f"x5_{k}": v for k, v in tf.extra_one(res, A.shape).items()})
        except Exception as e:  # noqa: BLE001
            row["x5_error"] = f"{type(e).__name__}:{e}"
    return row


def prod_modules(prod: Path):
    import warnings

    warnings.filterwarnings("ignore")
    for p in (prod / "scripts", prod / "src"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import cv2
    from umud import geometry_v2 as g2
    import build_candidates_v3 as b3
    import eval_apo_v2 as ea
    import eval_orient as eo
    import telemed_fl as tf

    return cv2, g2, b3, ea, eo, tf


def stage_measure() -> None:
    import pandas as pd
    from PIL import Image

    mods = prod_modules(TMP / "prod")
    g2 = mods[1]
    man = pd.read_csv(MANIFEST)
    bank = g2.MaskBank(BANK)
    out = OUT / "cache" / "gasfas__v2.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    for r in man.itertuples():
        with Image.open(CROPS / r.subset / f"{r.key}.png") as im:
            img = np.asarray(im.convert("L"), dtype=np.uint8)
        sha_ok = hashlib.sha1(img.tobytes()).hexdigest() == r.crop_sha1
        A = bank.prob(r.subset, "apo_v2fix_band", r.key)
        up = bank.prob(r.subset, "apo_v2fix_upper", r.key)
        lo = bank.prob(r.subset, "apo_v2fix_lower", r.key)
        fb = bank.prob(r.subset, "ours_apo384_noflip", r.key)
        F = bank.prob(r.subset, "ours_fasc384_noflip", r.key)
        seg = bank.raw(r.subset, "fasc_orient_v1_seg", r.key)
        c = bank.raw(r.subset, "fasc_orient_v1_cos2", r.key)
        s = bank.raw(r.subset, "fasc_orient_v1_sin2", r.key)
        orient = None if seg is None or c is None or s is None else (g2.as_prob(seg), g2.decode_orient(c),
                                                                    g2.decode_orient(s))
        ppc = float(r.px_per_cm)
        if A is None or F is None or orient is None or orient[1].shape != A.shape:
            row = g2._nan_result("missing_maps").as_row()
        else:
            fbs = [("ours", fb, 0.8)] if fb is not None else []
            try:
                row = measure(mods, A, up, lo, fbs, 0.5, True, F, 0.95, img, ppc, orient)
            except Exception as e:  # noqa: BLE001
                row = g2._nan_result(f"error:{type(e).__name__}:{e}").as_row()
        meta = dict(image_id=f"{r.subset}__{r.key}", okey=r.key, subset=r.subset, px_per_cm=ppc, subject=r.subject,
                    trial=r.trial, push=int(r.push), frame=int(r.frame), crop_sha_ok=sha_ok, crop_w=img.shape[1],
                    crop_h=img.shape[0])
        rows.append(dict(meta, **{k: v for k, v in row.items() if not str(k).startswith("x_")}))
        if len(rows) % 25 == 0:
            pd.DataFrame(rows).to_csv(out, index=False)
            log(f"  measure {len(rows)}/{len(man)} {time.time() - t0:.0f}s")
    D = pd.DataFrame(rows)
    D.to_csv(out, index=False)
    bank.close()
    n_mt = int(pd.to_numeric(D.get("mt_mm"), errors="coerce").notna().sum()) if "mt_mm" in D else 0
    vp = OUT / "viewports"
    vp.mkdir(parents=True, exist_ok=True)
    for p in sorted(BANK.glob("*.csv")) + sorted(BANK.glob("*.json")):
        shutil.copy2(p, vp / p.name)
    n_missing = int((D["d_fail"].astype(str) == "missing_maps").sum()) if "d_fail" in D else 0
    n_error = int(D["d_fail"].astype(str).str.startswith("error:").sum()) if "d_fail" in D else 0
    complete = len(D) == len(man) and len(D) > 0 and n_missing == 0
    removed = False
    if complete and not KEEP_BANK and not SMOKE:
        shutil.rmtree(BANK, ignore_errors=True)
        removed = True
    (OUT / "measure_info.json").write_text(json.dumps(dict(
        n=len(D), n_manifest=len(man), n_mt_finite=n_mt, n_missing_maps=n_missing, n_error=n_error,
        crop_sha_ok=bool(D.crop_sha_ok.all()) if len(D) else False, bank_removed=removed,
        elapsed_s=time.time() - t0)))
    log(f"measure: {len(D)} rows, finite mt {n_mt}, missing maps {n_missing}, errors {n_error}, "
        f"bank removed {removed} -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["data", "apo", "ours", "orient", "measure"])
    a = ap.parse_args()
    if a.stage == "data":
        stage_data()
    elif a.stage == "measure":
        stage_measure()
    else:
        tt = tt_module()
        {"apo": tt.stage_apo, "ours": tt.stage_ours, "orient": tt.stage_orient}[a.stage]()


if __name__ == "__main__":
    main()
