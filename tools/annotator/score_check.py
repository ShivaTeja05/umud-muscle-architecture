"""Grade a 10-image skills check (CHK_01..CHK_10) against the hidden expert answers.

    # the browser annotator (serve.py --data data/manual_labelling/check), its autosave or calib CSV export
    PYTHONPATH=src .venv/bin/python tools/annotator/score_check.py --state data/manual_labelling/check/autosave.json
    PYTHONPATH=src .venv/bin/python tools/annotator/score_check.py --csv ~/Downloads/umud_calib_labels_XXXX.csv
    # an outside expert's filled expert_package/answer_template.csv (FIJI values, 3 per parameter), plus the
    # returned CHK_XX_rois.zip files (recomputed and compared with the CSV)
    PYTHONPATH=src .venv/bin/python tools/annotator/score_check.py --csv answers_from_expert.csv --rois <folder> --name dr_x
    # practice progress: the FIRST complete attempt of the last 5 finished practice images vs the experts
    PYTHONPATH=src .venv/bin/python tools/annotator/score_check.py --practice

Inputs
    --key  data/manual_labelling/private/check_key.json   per image: every rater's values (R1..R7), their mean
    --bar  data/manual_labelling/private/pass_bar.json    the experts' leave-one-out scores and the verdict rule

What it prints
    * per image and component: the candidate value, the expert mean, the error and error / tau (the expert value is
      printed only for cells the candidate filled in);
    * the overall score S = mean over PA / FL / MT of MAE / tau (tau 6 deg / 12 mm / 3 mm, the competition core
      without its tie-breakers), per-component MAE / tau and bias (mean signed error);
    * the same score for every expert rater, each against the mean of the OTHER raters (leave-one-out), on the
      exact same image x component cells the candidate filled in, the candidate's rank among them, and a
      bootstrap over images (the check has only 10 images);
    * a copy check: images where 2 or more parameter means are within 0.05 of one published value (a rater, the
      rater mean, or the SMA / DL_Track value in the public workbook); 2 or more such images = SUSPECT;
    * the verdict: GOOD = S <= median expert, PASS = S <= worst expert, FAIL otherwise; INCOMPLETE when fewer
      than 8 images have all three parameters.

The check can be scored only once.  While it is INCOMPLETE, nothing that depends on the answers is printed (no
expert value, no error, no score), unless --reveal is given.  A missing or empty input is an error.

The candidate is graded against the mean of ALL raters.  A rater graded the same way would include their own
value in the reference: their error would shrink by (n - 1) / n.  That is why the bar uses leave-one-out
scores.  ``--reference-without R3`` grades against the mean without R3 instead (used by the self-test: R3's own
values then score exactly R3's leave-one-out score).

Writes a JSON report to <key dir>/check_results/ (never inside a served data dir).  Stdlib + labels.py only.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import re
import statistics
import struct
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import labels as labels_mod  # noqa: E402  (tools/annotator/labels.py: the annotator's maths)

ROOT = Path(__file__).resolve().parents[2]
KIT = ROOT / "data" / "manual_labelling"
COMPS = ("pa_deg", "fl_mm", "mt_mm")
SHORT = {"pa_deg": "PA", "fl_mm": "FL", "mt_mm": "MT"}
UNIT = {"pa_deg": "deg", "fl_mm": "mm", "mt_mm": "mm"}
TOL = {"pa_deg": 6.0, "fl_mm": 12.0, "mt_mm": 3.0}  # src/umud/metric.py TOL
MIN_COMPLETE_IMAGES = 8
N_BOOT = 2000
TEMPLATE_COLS = {"mt_mm": ("MT1_mm", "MT2_mm", "MT3_mm"), "fl_mm": ("FL1_mm", "FL2_mm", "FL3_mm"),
                 "pa_deg": ("PA1_deg", "PA2_deg", "PA3_deg")}
# plausible ranges for a single measurement, and the unit slips we can recognise
PLAUSIBLE = {"mt_mm": (5.0, 50.0), "fl_mm": (20.0, 250.0), "pa_deg": (3.0, 60.0)}
# copy check: a parameter mean this close to a published value counts as a match.  On the workbook, an honest
# rater has 2 or more parameters matching one other published value on 0-2 of 35 images; any single value
# matches on most images, so single values are not used.
COPY_TOL = 0.05
COPY_MIN_PARAMS = 2
COPY_MIN_IMAGES = 2
# ROI files vs the CSV: differences above these are reported
ROI_TOL = {"mt_mm": 0.1, "fl_mm": 0.5, "pa_deg": 0.5}
CSV_ENCODINGS = ("utf-8-sig", "cp1252", "mac_roman")


# ----------------------------------------------------------------------------- small helpers
def _mean(xs) -> float:
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return sum(xs) / len(xs) if xs else math.nan


def _finite(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def _clean(obj):
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else round(obj, 5)
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def norm_id(text: str) -> str | None:
    """'CHK_01', 'chk_1.png', 'CHK-01' or '1' -> 'CHK_01'."""

    m = re.search(r"(\d+)", str(text or ""))
    return f"CHK_{int(m.group(1)):02d}" if m else None


def read_text_any(path: Path) -> tuple[str, str]:
    """(text, encoding).  Excel's plain 'CSV' is cp1252 (Windows) or MacRoman (Mac), not UTF-8."""

    data = Path(path).read_bytes()
    for enc in CSV_ENCODINGS:
        try:
            return data.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1"), "latin-1"


# ----------------------------------------------------------------------------- key / references
def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rater_names(key: dict) -> list[str]:
    names = key.get("raters")
    if names:
        return list(names)
    seen: list[str] = []
    for item in key["items"].values():
        for r in item["raters"]:
            if r not in seen:
                seen.append(r)
    return seen


def reference(item: dict, comp: str, without: str | None = None) -> float:
    """Mean of the raters' values for one image and component (optionally leaving one rater out)."""

    return _mean([v.get(comp) for r, v in item["raters"].items() if r != without])


def rater_values(key: dict, rater: str) -> dict[str, dict[str, float]]:
    out = {}
    for cid, item in key["items"].items():
        v = item["raters"].get(rater) or {}
        out[cid] = {c: (float(v[c]) if _finite(v.get(c)) else math.nan) for c in COMPS}
    return out


def errors(values: dict, key: dict, without: str | None = None, cells: set | None = None) -> dict:
    """{(cid, comp): signed error} for every cell where the value and the reference are finite."""

    out = {}
    for cid, item in key["items"].items():
        for comp in COMPS:
            if cells is not None and (cid, comp) not in cells:
                continue
            v = (values.get(cid) or {}).get(comp, math.nan)
            ref = reference(item, comp, without)
            if _finite(v) and math.isfinite(ref):
                out[(cid, comp)] = v - ref
    return out


def summarize(errs: dict, tol: dict = TOL) -> dict:
    """Competition core on a set of cells: S = mean over components (with data) of MAE / tau."""

    comps = {}
    for comp in COMPS:
        e = [x for (cid, c), x in errs.items() if c == comp]
        n = len(e)
        if not n:
            comps[comp] = {"n": 0}
            continue
        mae = sum(abs(x) for x in e) / n
        bias = sum(e) / n
        sd = statistics.stdev(e) if n > 1 else math.nan
        se = sd / math.sqrt(n) if n > 1 else math.nan
        comps[comp] = {"n": n, "mae": mae, "mae_tau": mae / tol[comp], "bias": bias, "sd": sd,
                       "t_bias": bias / se if math.isfinite(se) and se > 1e-9 else math.nan,
                       "rmse": math.sqrt(sum(x * x for x in e) / n)}
    vals = [v["mae_tau"] for v in comps.values() if v.get("n")]
    return {"S": sum(vals) / len(vals) if vals else math.nan, "components": comps,
            "n_images": len({cid for cid, _ in errs}), "n_cells": len(errs)}


def loo_errors(key: dict, rater: str, cells: set | None = None) -> dict:
    """Rater vs the mean of the OTHER raters (leave-one-out), cell by cell."""

    return errors(rater_values(key, rater), key, without=rater, cells=cells)


def default_bar_raters(key: dict, min_share: float = 0.8) -> list[str]:
    """Raters with values on at least min_share of the key's cells (R5 rated only images 1-17)."""

    n_cells = len(key["items"]) * len(COMPS)
    out = []
    for r in rater_names(key):
        have = sum(_finite(v) for vals in rater_values(key, r).values() for v in vals.values())
        if n_cells and have / n_cells >= min_share:
            out.append(r)
    return out


def expert_table(key: dict, raters: list[str], cells: set | None = None, tol: dict = TOL) -> dict:
    return {r: summarize(loo_errors(key, r, cells), tol) for r in raters}


def bar_stats(table: dict) -> dict:
    s = sorted((v["S"], r) for r, v in table.items() if math.isfinite(v["S"]))
    if not s:
        return {}
    vals = [x for x, _ in s]
    return {"best": vals[0], "best_rater": s[0][1], "median": statistics.median(vals),
            "worst": vals[-1], "worst_rater": s[-1][1], "n_raters": len(vals),
            "per_component": {c: {"best": min(table[r]["components"][c]["mae_tau"] for r in table if table[r]["components"][c].get("n")),
                                  "median": statistics.median([table[r]["components"][c]["mae_tau"] for r in table if table[r]["components"][c].get("n")]),
                                  "worst": max(table[r]["components"][c]["mae_tau"] for r in table if table[r]["components"][c].get("n"))}
                              for c in COMPS if any(table[r]["components"][c].get("n") for r in table)}}


# ----------------------------------------------------------------------------- bootstrap over images
def _norm_matrix(errs: dict, ids: list[str], tol: dict) -> list[list[float]]:
    return [[abs(errs[(cid, c)]) / tol[c] if (cid, c) in errs else math.nan for c in COMPS] for cid in ids]


def _score_rows(rows: list[list[float]]) -> float:
    per = []
    for j in range(len(COMPS)):
        col = [r[j] for r in rows if math.isfinite(r[j])]
        if col:
            per.append(sum(col) / len(col))
    return sum(per) / len(per) if per else math.nan


def bootstrap(series: dict[str, dict], ids: list[str], tol: dict = TOL, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Resample the images (with replacement).  series: name -> errs.  Returns per-name SD and 5/95 %, plus
    the same for the worst / median of the names starting with 'R' (the experts), and the raw draws."""

    rng = random.Random(seed)
    mats = {k: _norm_matrix(v, ids, tol) for k, v in series.items()}
    draws = {k: [] for k in series}
    draws["_worst"], draws["_median"] = [], []
    n = len(ids)
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        exp = []
        for k, m in mats.items():
            s = _score_rows([m[i] for i in idx])
            draws[k].append(s)
            if k.startswith("R") and math.isfinite(s):
                exp.append(s)
        draws["_worst"].append(max(exp) if exp else math.nan)
        draws["_median"].append(statistics.median(exp) if exp else math.nan)

    def _q(xs, p):
        xs = sorted(x for x in xs if math.isfinite(x))
        return xs[min(len(xs) - 1, max(0, int(round(p * (len(xs) - 1)))))] if xs else math.nan

    out = {k: {"sd": statistics.pstdev([x for x in v if math.isfinite(x)]) if any(math.isfinite(x) for x in v) else math.nan,
               "p05": _q(v, 0.05), "p95": _q(v, 0.95)} for k, v in draws.items()}
    return {"n_boot": n_boot, "seed": seed, "stats": out, "draws": draws}


# ----------------------------------------------------------------------------- candidate values
def values_from_state(state: dict, key: dict, use_first: bool = False) -> tuple[dict, list[str]]:
    """Annotator labels (autosave.json state) -> per image the mean of the measurements present.

    use_first: use the copy of the clicks the annotator froze when the image was first complete (``first``), so
    that points moved after the answer was shown (practice) do not count."""

    values, notes = {}, []
    labels = state.get("labels") or {}
    for cid, item in key["items"].items():
        lab = labels.get(cid)
        if not lab:
            continue
        if use_first:
            if isinstance(lab.get("first"), dict):
                lab = {**lab, **{t: lab["first"].get(t) or [] for t in ("mt", "fl", "pa")}}
            else:
                notes.append(f"{cid}: no first-attempt copy (labelled before the tool kept one); current points used")
        s = labels_mod.summarize(lab, float(item["px_per_cm"]))
        rec = {}
        for comp, tool in (("pa_deg", "pa"), ("fl_mm", "fl"), ("mt_mm", "mt")):
            n = s[f"n_{tool}"]
            rec[comp] = s[tool] if n else math.nan
            if 0 < n < 3:
                notes.append(f"{cid} {SHORT[comp]}: only {n} of 3 measurements; using their mean")
        if lab.get("flag"):
            notes.append(f"{cid}: flagged in the annotator (values still scored if present)")
        values[cid] = rec
    return values, notes


def _sniff_delimiter(text: str) -> str:
    head = text.splitlines()[0] if text.strip() else ""
    counts = {d: head.count(d) for d in (",", ";", "\t")}
    return max(counts, key=counts.get) if any(counts.values()) else ","


def _num(cell: str | None) -> float:
    s = (cell or "").strip().replace("−", "-")
    if not s:
        return math.nan
    s = s.replace(" ", "")
    if "," in s and "." not in s:  # decimal comma (Excel in many locales)
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return math.nan


def _length_slips(v: float, px_per_mm: float, lo: float, hi: float) -> dict[str, float]:
    """For a length that is too large: the plausible readings.  PIXELS = Set Scale not done; X10 = Set Scale with
    px_per_mm as 'Distance in pixels' and Known distance 10 (every length 10x too large)."""

    out = {}
    if px_per_mm > 0 and lo <= v / px_per_mm <= hi:
        out["PIXELS"] = v / px_per_mm
    if lo <= v / 10.0 <= hi:
        out["X10"] = v / 10.0
    return out


def check_units(comp: str, v: float, px_per_mm: float) -> tuple[str | None, float | None]:
    """(warning, corrected value) for one measurement that looks like a unit slip; (None, None) if fine.  The
    corrected value is None when the slip is not clear (then the value is never changed)."""

    lo, hi = PLAUSIBLE[comp]
    if not math.isfinite(v) or lo <= v <= hi:
        return None, None
    if comp == "pa_deg":
        if 0.05 <= v <= 1.1:
            return f"{v:g} looks like RADIANS (= {math.degrees(v):.1f} deg)", math.degrees(v)
        if 60 < v <= 90:
            return f"{v:g} deg is very steep: angle against the vertical? (90 - x = {90 - v:.1f}); not changed", None
        if 90 < v < 180 and lo <= 180 - v <= hi:
            return f"{v:g} deg is OBTUSE: FIJI showed 180 - x (= {180 - v:.1f} deg)", 180 - v
        return f"{v:g} deg is outside {lo:g}-{hi:g}", None
    if lo / 10 <= v < lo:
        return f"{v:g} looks like CENTIMETRES (= {v * 10:g} mm)", v * 10
    if v > hi:
        ok = _length_slips(v, px_per_mm, lo, hi)
        if "PIXELS" in ok and "X10" in ok and abs(ok["PIXELS"] - ok["X10"]) > 0.03 * ok["X10"]:
            return (f"{v:g} is too large: PIXELS (= {ok['PIXELS']:.1f} mm at {px_per_mm:g} px/mm, no Set Scale) or 10x "
                    f"(= {ok['X10']:.1f} mm, Set Scale with px_per_mm and Known distance 10)? AMBIGUOUS, not changed "
                    f"(the ROI files settle it: --rois)"), None
        if "PIXELS" in ok:
            return f"{v:g} looks like PIXELS (= {ok['PIXELS']:.1f} mm at {px_per_mm:g} px/mm): was Set Scale done?", ok["PIXELS"]
        if "X10" in ok:
            return f"{v:g} looks 10x too large (= {ok['X10']:.1f} mm): Set Scale with px_per_mm and Known distance 10?", ok["X10"]
    return f"{v:g} {UNIT[comp]} is outside {lo:g}-{hi:g}", None


def _row_length_factor(raw: list[tuple[str, str, float]], px_per_mm: float) -> tuple[str | None, str]:
    """One scale slip per image row: the reading (PIXELS / X10) that makes EVERY too-large MT and FL value of the
    row plausible.  (None, why) when there is no such reading or both remain (ambiguous)."""

    big = [(comp, v) for _, comp, v in raw if comp != "pa_deg" and math.isfinite(v) and v > PLAUSIBLE[comp][1]]
    if not big:
        return None, ""
    sets = [set(_length_slips(v, px_per_mm, *PLAUSIBLE[comp])) for comp, v in big]
    common = set.intersection(*sets) if sets else set()
    if px_per_mm > 0 and abs(px_per_mm - 10.0) <= 0.3 and common:  # both readings are the same number here
        return "PIXELS", ""
    if len(common) == 1:
        return common.pop(), ""
    return None, "ambiguous" if common else "no single reading fits"


def values_from_template(path: Path, key: dict, fix_units: bool = False) -> tuple[dict, list[str], dict]:
    """Filled answer_template.csv -> per image the mean of its (up to) 3 values per parameter."""

    text, enc = read_text_any(path)
    notes: list[str] = []
    if enc != "utf-8-sig":
        notes.append(f"the CSV is not UTF-8; read it as {enc} (only the notes column can be affected)")
    delim = _sniff_delimiter(text)
    reader = csv.DictReader(io.StringIO(text), delimiter=delim)
    fields = {(f or "").strip().lower(): f for f in reader.fieldnames or []}
    values, meta, raw_values = {}, {}, {}
    for row in reader:
        cid = norm_id(row.get(fields.get("image_id", "image_id"), ""))
        if cid is None:
            continue
        if cid not in key["items"]:
            notes.append(f"row {row.get(fields.get('image_id', 'image_id'))!r}: not a check image, ignored")
            continue
        if cid in values:
            notes.append(f"{cid}: duplicate row, the first one is used")
            continue
        px_per_mm = float(key["items"][cid]["px_per_cm"]) / 10.0
        raw = [(col, comp, _num(row.get(fields.get(col.lower(), col)))) for comp, cols in TEMPLATE_COLS.items() for col in cols]
        factor, why = _row_length_factor(raw, px_per_mm)
        rec, got = {}, {c: [] for c in COMPS}
        for col, comp, v in raw:
            if not math.isfinite(v):
                continue
            warn, fixed = check_units(comp, v, px_per_mm)
            if warn and comp != "pa_deg" and v > PLAUSIBLE[comp][1]:
                # too-large lengths: one reading per row (all of the row's lengths share the scale)
                fixed = _length_slips(v, px_per_mm, *PLAUSIBLE[comp]).get(factor) if factor else None
                if fixed is not None:
                    warn = f"{v:g} looks like {factor} (row-wide reading; = {fixed:.1f} mm)"
                elif why == "ambiguous" and "AMBIGUOUS" not in warn:
                    warn += "; the row as a whole is AMBIGUOUS (pixels or 10x), not changed"
                elif why == "no single reading fits":
                    warn += "; the other lengths of this row need a different reading, not changed"
            if warn:
                act = f"; using {fixed:.3f}" if (fix_units and fixed is not None) else ""
                notes.append(f"UNITS? {cid} {col}: {warn}{act}")
                if fix_units and fixed is not None:
                    v = fixed
            got[comp].append(v)
        for comp in COMPS:
            rec[comp] = _mean(got[comp])
            if 0 < len(got[comp]) < 3:
                notes.append(f"{cid} {SHORT[comp]}: {len(got[comp])} of 3 values; using their mean")
        values[cid] = rec
        raw_values[cid] = got
        meta[cid] = {k: (row.get(fields.get(k, k)) or "").strip() for k in ("minutes", "confidence_1to5", "flag", "notes")}
    meta["_raw_values"] = raw_values
    return values, notes, meta


def _has_values(values: dict) -> bool:
    return any(_finite(v) for rec in values.values() for v in rec.values())


def load_candidate(args, key: dict) -> tuple[dict, list[str], dict, str]:
    if args.csv:
        if not Path(args.csv).is_file():
            raise SystemExit(f"no such file: {args.csv}")
        head = read_text_any(Path(args.csv))[0].splitlines()[:1]
        if head and "clicks_json" in head[0]:
            state = labels_mod.state_from_csv(Path(args.csv))
            v, n = values_from_state(state, key, use_first=args.practice)
            out = (v, n, {}, "annotator CSV export (clicks_json)")
        else:
            v, n, meta = values_from_template(Path(args.csv), key, args.fix_units)
            out = (v, n, meta, "answer_template CSV (FIJI)")
    else:
        if not Path(args.state).is_file():
            raise SystemExit(f"no annotator save file at {args.state}. Nothing to score yet: measure the images first "
                             "(the tool saves by itself), or give --csv.")
        state = load_json(args.state)
        v, n = values_from_state(state, key, use_first=args.practice)
        out = (v, n, {}, "annotator autosave.json" + (" (first attempts)" if args.practice else ""))
    if not _has_values(out[0]):
        raise SystemExit(f"no measurements of these images in {args.csv or args.state}. Nothing to score.")
    return out


# ----------------------------------------------------------------------------- copy check
def copy_check(values: dict, key: dict, tol: float = COPY_TOL) -> dict:
    """Images where >= 2 parameter means are within tol of ONE published value: a rater, the rater mean, or a
    workbook tool value (SMA, DL_Track).  Copying the public workbook shows up here; honest work almost never."""

    hits = {}
    for cid, item in key["items"].items():
        v = values.get(cid) or {}
        refs = dict(item.get("raters") or {})
        refs["rater mean"] = {c: reference(item, c) for c in COMPS}
        refs.update({f"{t} (workbook)": x for t, x in (item.get("tools") or {}).items()})
        for name, ref in refs.items():
            match = [c for c in COMPS if _finite(v.get(c)) and _finite((ref or {}).get(c)) and abs(v[c] - ref[c]) <= tol]
            if len(match) >= COPY_MIN_PARAMS:
                hits.setdefault(cid, []).append({"reference": name, "params": [SHORT[c] for c in match]})
    return {"tolerance": tol, "min_params": COPY_MIN_PARAMS, "min_images": COPY_MIN_IMAGES,
            "images": hits, "n_images": len(hits), "suspect": len(hits) >= COPY_MIN_IMAGES}


# ----------------------------------------------------------------------------- ImageJ / FIJI ROI files
ROI_TYPES = {0: "polygon", 1: "rect", 2: "oval", 3: "line", 4: "freeline", 5: "polyline", 6: "noroi", 7: "freehand",
             8: "traced", 9: "angle", 10: "point"}


def parse_roi(data: bytes) -> dict:
    """One ImageJ .roi file -> {"type", "points" [(x, y) image pixels], "spline"}.  Lines, segmented lines and
    angles only (the 3 tools in the instructions).  Big-endian; float coordinates when sub-pixel is on."""

    if len(data) < 64 or data[:4] != b"Iout":
        raise ValueError("not an ImageJ ROI")
    version = struct.unpack(">h", data[4:6])[0]
    typ = ROI_TYPES.get(data[6], str(data[6]))
    top, left = struct.unpack(">hh", data[8:12])
    n = struct.unpack(">H", data[16:18])[0]
    options = struct.unpack(">h", data[50:52])[0]
    if typ == "line":
        x1, y1, x2, y2 = struct.unpack(">ffff", data[18:34])
        return {"type": typ, "points": [(x1, y1), (x2, y2)], "spline": False}
    if typ not in ("polyline", "freeline", "angle", "polygon", "point"):
        return {"type": typ, "points": [], "spline": False}
    xs = struct.unpack(f">{n}h", data[64:64 + 2 * n])
    ys = struct.unpack(f">{n}h", data[64 + 2 * n:64 + 4 * n])
    pts = [(float(left + x), float(top + y)) for x, y in zip(xs, ys)]
    if options & 128 and version >= 222:  # SUB_PIXEL_RESOLUTION: absolute float coordinates follow
        b1 = 64 + 4 * n
        xf = struct.unpack(f">{n}f", data[b1:b1 + 4 * n])
        yf = struct.unpack(f">{n}f", data[b1 + 4 * n:b1 + 8 * n])
        pts = list(zip(xf, yf))
    return {"type": typ, "points": pts, "spline": bool(options & 1)}


def read_rois(path: Path) -> list[dict]:
    path = Path(path)
    if path.suffix.lower() == ".roi":
        return [dict(parse_roi(path.read_bytes()), name=path.name)]
    out = []
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if name.lower().endswith(".roi"):
                try:
                    out.append(dict(parse_roi(zf.read(name)), name=name))
                except (ValueError, struct.error) as err:
                    out.append({"type": "unreadable", "points": [], "spline": False, "name": name, "error": str(err)})
    return out


def roi_values(rois: list[dict], px_per_mm: float) -> tuple[dict[str, list[float]], list[str]]:
    """MT = near-vertical straight lines, FL = segmented lines (or other straight lines), PA = angles."""

    vals, notes = {c: [] for c in COMPS}, []
    for r in rois:
        p = r["points"]
        if r["type"] == "angle" and len(p) >= 3:
            a = math.degrees(abs(math.atan2(p[0][1] - p[1][1], p[0][0] - p[1][0]) - math.atan2(p[2][1] - p[1][1], p[2][0] - p[1][0])))
            a = a % 360.0
            a = 360.0 - a if a > 180.0 else a
            vals["pa_deg"].append(min(a, 180.0 - a))
        elif r["type"] == "line" and len(p) == 2:
            dx, dy = p[1][0] - p[0][0], p[1][1] - p[0][1]
            comp = "mt_mm" if abs(dx) <= math.tan(math.radians(10)) * abs(dy) else "fl_mm"
            vals[comp].append(math.hypot(dx, dy) / px_per_mm)
        elif r["type"] in ("polyline", "freeline") and len(p) >= 2:
            vals["fl_mm"].append(sum(math.dist(p[i - 1], p[i]) for i in range(1, len(p))) / px_per_mm)
            if r["spline"]:
                notes.append(f"{r['name']}: spline fit was on; recomputed as the straight segments")
        else:
            notes.append(f"{r['name']}: {r['type']} selection ignored")
    return vals, notes


def compare_rois(roi_dir: Path, key: dict, raw_values: dict) -> dict:
    """Recompute every image's values from its returned CHK_XX_rois.zip and compare them with the CSV."""

    if not Path(roi_dir).is_dir():
        raise SystemExit(f"--rois: not a folder: {roi_dir}")
    found = {}
    for p in sorted(Path(roi_dir).rglob("*")):
        if p.is_file() and p.suffix.lower() in {".zip", ".roi"} and "chk" in p.name.lower():
            cid = norm_id(p.name)
            if cid in key["items"]:
                found.setdefault(cid, []).append(p)
    per, notes, values = {}, [], {}
    for cid in key["items"]:
        if cid not in found:
            notes.append(f"{cid}: no ROI file")
            continue
        rois = [r for p in found[cid] for r in read_rois(p)]
        px_per_mm = float(key["items"][cid]["px_per_cm"]) / 10.0
        vals, n2 = roi_values(rois, px_per_mm)
        notes += [f"{cid} {x}" for x in n2]
        csv_vals = raw_values.get(cid) or {}
        row = {}
        for comp in COMPS:
            a, b = sorted(vals[comp]), sorted(x for x in csv_vals.get(comp, []) if math.isfinite(x))
            diff = max((abs(x - y) for x, y in zip(a, b)), default=math.nan) if len(a) == len(b) else math.nan
            row[comp] = {"roi": a, "csv": b, "max_abs_diff": diff}
            if len(a) != len(b):
                notes.append(f"MISMATCH {cid} {SHORT[comp]}: {len(a)} ROI(s) but {len(b)} CSV value(s)")
            elif math.isfinite(diff) and diff > ROI_TOL[comp]:
                notes.append(f"MISMATCH {cid} {SHORT[comp]}: ROI {', '.join(f'{x:.2f}' for x in a)} vs CSV "
                             f"{', '.join(f'{x:.2f}' for x in b)} {UNIT[comp]}")
        per[cid] = row
        values[cid] = {c: _mean(vals[c]) for c in COMPS}
    return {"per_image": per, "notes": notes, "values": values,
            "n_images": len(found), "n_mismatch": sum(n.startswith("MISMATCH") for n in notes)}


# ----------------------------------------------------------------------------- grading
def grade(values: dict, key: dict, bar: dict | None = None, without: str | None = None,
          tol: dict = TOL, n_boot: int = N_BOOT, min_complete: int = MIN_COMPLETE_IMAGES) -> dict:
    ids = list(key["items"])
    errs = errors(values, key, without=without)
    cells = set(errs)
    you = summarize(errs, tol)
    raters = (bar or {}).get("bar_raters") or default_bar_raters(key)
    if without:
        raters = [r for r in raters if r != without]
    table = expert_table(key, raters, cells, tol)
    bs = bar_stats(table)
    complete = [cid for cid in ids if all((cid, c) in cells for c in COMPS)]
    others = [r for r in rater_names(key) if r not in raters and r != without]
    extra = {r: summarize(loo_errors(key, r, cells), tol) for r in others}
    boot = bootstrap({"you": errs, **{r: loo_errors(key, r, cells) for r in raters}}, ids, tol, n_boot)
    d = boot["draws"]
    pairs = [(y, w, m) for y, w, m in zip(d["you"], d["_worst"], d["_median"]) if math.isfinite(y) and math.isfinite(w)]
    p_pass = sum(y <= w for y, w, _ in pairs) / len(pairs) if pairs else math.nan
    p_good = sum(y <= m for y, _, m in pairs) / len(pairs) if pairs else math.nan
    s = you["S"]
    if not math.isfinite(s) or not bs:
        verdict = "NO DATA"
    else:
        verdict = "GOOD" if s <= bs["median"] else "PASS" if s <= bs["worst"] else "FAIL"
        if len(complete) < min_complete:
            verdict = f"INCOMPLETE (provisional {verdict})"
    rank = 1 + sum(v["S"] < s for v in table.values() if math.isfinite(v["S"])) if math.isfinite(s) else None
    per_image = []
    for cid in ids:
        item = key["items"][cid]
        row = {"image_id": cid}
        norm = []
        for comp in COMPS:
            v = (values.get(cid) or {}).get(comp, math.nan)
            ref = reference(item, comp, without)
            e = errs.get((cid, comp), math.nan)
            row[comp] = {"you": v, "expert": ref, "err": e, "err_tau": abs(e) / tol[comp] if math.isfinite(e) else math.nan,
                         "expert_min": _mean([min([x[comp] for x in item["raters"].values() if _finite(x.get(comp))] or [math.nan])]),
                         "expert_max": _mean([max([x[comp] for x in item["raters"].values() if _finite(x.get(comp))] or [math.nan])])}
            if math.isfinite(e):
                norm.append(abs(e) / tol[comp])
        row["err_tau_mean"] = _mean(norm)
        per_image.append(row)
    return {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "tolerances": tol,
        "reference": f"mean of all raters except {without}" if without else "mean of all raters",
        "n_images_with_values": you["n_images"], "n_images_complete": len(complete), "n_cells": you["n_cells"],
        "complete_images": complete, "min_complete": min_complete,
        "per_image": per_image, "you": you, "experts_loo_same_cells": table, "experts_not_in_bar": extra,
        "bar_raters": raters, "bar": bs, "rank": rank, "n_ranked": len(table) + 1,
        "bootstrap": {"n_boot": boot["n_boot"], "you": boot["stats"]["you"], "worst": boot["stats"]["_worst"],
                      "median": boot["stats"]["_median"], "p_you_le_worst": p_pass, "p_you_le_median": p_good},
        "verdict": verdict,
        "rule": "GOOD: S <= median expert LOO score; PASS: S <= worst expert LOO score; FAIL otherwise. "
                "Expert scores are leave-one-out (each rater vs the mean of the other raters) on the same "
                f"image x component cells. INCOMPLETE if fewer than {min_complete} images have PA, FL and MT.",
    }


def hidden(res: dict) -> bool:
    """True while the answers must stay hidden: too few complete images."""

    return res["n_images_complete"] < res.get("min_complete", MIN_COMPLETE_IMAGES)


def redact(res: dict) -> dict:
    """The report without anything that depends on the answers for cells the candidate did not fill in; while the
    check is INCOMPLETE, without any answer-dependent number at all."""

    out = dict(res)
    if hidden(res):
        for k in ("per_image", "you", "experts_loo_same_cells", "experts_not_in_bar", "bar", "rank", "bootstrap"):
            out.pop(k, None)
        out["verdict"] = "INCOMPLETE"
        out["redacted"] = "INCOMPLETE: no answer-dependent values kept (use --reveal to override)"
        return out
    rows = []
    for r in res["per_image"]:
        r = dict(r)
        for c in COMPS:
            if not _finite(r[c]["you"]):
                r[c] = {"you": r[c]["you"], "expert": None, "err": None, "err_tau": None, "expert_min": None, "expert_max": None}
        rows.append(r)
    out["per_image"] = rows
    return out


# ----------------------------------------------------------------------------- printing
def _f(v, d=2, sign=False) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "-"
    return f"{v:+.{d}f}" if sign else f"{v:.{d}f}"


def report(res: dict, bar_file: dict | None, source: str, notes: list[str], reveal: bool = False,
           practice: bool = False) -> None:
    tol = res["tolerances"]
    print(f"\nUMUD {'practice progress' if practice else 'skills check'}: {source}. Reference = {res['reference']}.")
    print(f"{res['n_images_with_values']} images with values, {res['n_images_complete']} with PA, FL and MT "
          f"({res['n_cells']} image x parameter cells).")
    if notes:
        print("\nNotes:")
        for n in notes:
            print("  " + n)
    if hidden(res) and not reveal:
        missing = [r["image_id"] for r in res["per_image"] if r["image_id"] not in res["complete_images"]]
        print(f"\nNot scored yet: {res['n_images_complete']} of {len(res['per_image'])} images have all 9 measurements; "
              f"at least {res['min_complete']} are needed.")
        print("Still to finish: " + ", ".join(missing))
        print("No expert value, error or score is shown before that, so the check stays blind (it can be scored "
              "only once). --reveal shows everything anyway, and then the check is used up.")
        print("\nVERDICT: INCOMPLETE")
        return
    print("\nPer image (you / expert mean / error / |error| / tau; '-' = not measured, the expert value stays hidden):")
    print(f"{'image':7s} | {'PA you':>6s} {'exp':>6s} {'err':>6s} {'/tau':>5s} | {'FL you':>6s} {'exp':>6s} {'err':>6s} {'/tau':>5s}"
          f" | {'MT you':>6s} {'exp':>6s} {'err':>6s} {'/tau':>5s} | {'mean':>5s}")
    for r in res["per_image"]:
        cells = []
        for c in COMPS:
            x = r[c]
            show = reveal or _finite(x["you"])
            cells.append(f"{_f(x['you'], 1):>6s} {_f(x['expert'] if show else None, 1):>6s} {_f(x['err'], 1, True):>6s} "
                         f"{_f(x['err_tau'], 2):>5s}")
        print(f"{r['image_id']:7s} | " + " | ".join(cells) + f" | {_f(r['err_tau_mean'], 2):>5s}")
    you = res["you"]
    print("\nPer parameter:")
    print(f"{'':4s} {'n':>3s} {'MAE':>6s} {'MAE/tau':>8s} {'bias':>7s} {'t':>6s}   experts (LOO) MAE/tau best / median / worst")
    pc = (res["bar"] or {}).get("per_component", {})
    for c in COMPS:
        s = you["components"][c]
        if not s.get("n"):
            print(f"{SHORT[c]:4s}   0")
            continue
        b = pc.get(c, {})
        print(f"{SHORT[c]:4s} {s['n']:3d} {_f(s['mae']):>6s} {_f(s['mae_tau'], 3):>8s} {_f(s['bias'], 2, True):>7s} {_f(s['t_bias'], 1):>6s}"
              f"   {_f(b.get('best'), 3)} / {_f(b.get('median'), 3)} / {_f(b.get('worst'), 3)}  ({UNIT[c]}, tau {tol[c]:g})")
    print(f"\nScore S (mean over PA/FL/MT of MAE/tau) = {_f(you['S'], 4)}   (0 = the expert mean; lower is better)")
    bs = res["bar"]
    if bs:
        print("\nExperts on the same cells, each vs the mean of the OTHER raters (leave-one-out):")
        rows = sorted([(v["S"], r) for r, v in res["experts_loo_same_cells"].items()] + [(you["S"], "YOU")])
        print("  " + "  ".join(f"{r} {_f(s, 3)}" for s, r in rows))
        for r, v in res["experts_not_in_bar"].items():
            if v["n_images"]:
                print(f"  ({r} not in the bar: values on only {v['n_images']} of these images, S {_f(v['S'], 3)})")
        print(f"  best {_f(bs['best'], 3)} ({bs['best_rater']}), median {_f(bs['median'], 3)}, worst {_f(bs['worst'], 3)} ({bs['worst_rater']})")
        print(f"  your rank: {res['rank']} of {res['n_ranked']} (1 = closest to the expert consensus)")
        b = res["bootstrap"]
        print(f"  bootstrap over images ({b['n_boot']}x): your S {_f(you['S'], 3)} +/- {_f(b['you']['sd'], 3)} "
              f"[5-95%: {_f(b['you']['p05'], 3)}-{_f(b['you']['p95'], 3)}]; worst expert +/- {_f(b['worst']['sd'], 3)}, "
              f"median expert +/- {_f(b['median']['sd'], 3)}")
        print(f"  share of resamples where you <= worst expert: {_f(b['p_you_le_worst'], 2)}, <= median expert: {_f(b['p_you_le_median'], 2)}")
    if bar_file:
        full = bar_file.get("check_10", {}).get("bar", {})
        if full:
            print(f"\nFull-check bar (pass_bar.json, all 30 cells): median {_f(full.get('median'), 3)}, worst {_f(full.get('worst'), 3)}.")
        models = {k: m for k, m in (bar_file.get("models_on_check") or {}).items() if m.get("S") is not None}
        if models:
            print("For context, models and tools on the check images they cover (vs the mean of all raters; see pass_bar.json):")
        for name, m in models.items():
            hon = (f"; on the {m['n_honest_images']} images not in its training data {_f(m.get('S_honest_images'), 3)}"
                   if m.get("n_honest_images") != m.get("n_images") else "")
            print(f"  {name}: S {_f(m['S'], 3)} on {m['n_images']} images{hon}")
    cc = res.get("copy_check")
    if cc:
        print(f"\nCopy check (2+ parameter means within {cc['tolerance']:g} of one published value): {cc['n_images']} image(s)"
              + (" -> SUSPECT: the values may be copied from the public workbook. Review before trusting the verdict."
                 if cc["suspect"] else "; no sign of copying."))
        for cid, hits in cc["images"].items():
            print(f"  {cid}: " + "; ".join(f"{h['reference']} ({'/'.join(h['params'])})" for h in hits))
    rc = res.get("roi_check")
    if rc:
        print(f"\nROI files: {rc['n_images']} image(s) found, {rc['n_mismatch']} mismatch(es) with the CSV.")
        for n in rc["notes"]:
            print("  " + n)
        if rc.get("S_from_rois") is not None:
            print(f"  score S recomputed from the ROI files alone: {_f(rc['S_from_rois'], 4)}")
    if practice:
        v = res["verdict"]
        meaning = {"GOOD": "as good as a typical expert on these images: ready for the check",
                   "PASS": "within the expert range on these images: ready for the check",
                   "FAIL": "not yet at the expert level on these images: keep practising"}
        print(f"\nPRACTICE LEVEL: {v}" + (f" ({meaning[v]})" if v in meaning else ""))
        print("Based on the FIRST complete attempt of each image (points moved after the answer was shown do not count).")
        return
    warn = " (copy check SUSPECT: review before trusting it)" if (cc or {}).get("suspect") else ""
    print(f"\nVERDICT: {res['verdict']}{warn}")
    print("Rule: " + res["rule"])
    print("Only 10 images: treat a result close to the bar as uncertain (see the bootstrap line).")


def practice_subset(state: dict, key: dict, last: int) -> tuple[dict, list[str]]:
    """The key restricted to the `last` practice images whose first attempt was completed most recently."""

    labels = state.get("labels") or {}
    done = []
    for pid in key["items"]:
        lab = labels.get(pid) or {}
        first = lab.get("first") if isinstance(lab.get("first"), dict) else None
        src = first or lab
        if all(len(src.get(t) or []) >= 3 for t in ("mt", "fl", "pa")):
            done.append(((first or {}).get("t") or lab.get("updated") or 0, pid))
    chosen = [pid for _, pid in sorted(done)[-last:]] if last > 0 else [pid for _, pid in sorted(done)]
    sub = {**key, "items": {pid: key["items"][pid] for pid in key["items"] if pid in chosen}}
    return sub, chosen


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade the 10-image skills check against the expert answers")
    parser.add_argument("--state", type=Path, default=None,
                        help="annotator autosave.json (default: data/manual_labelling/check/autosave.json, or the "
                             "practice one with --practice)")
    parser.add_argument("--csv", type=Path, default=None,
                        help="filled answer_template.csv (FIJI), or an annotator CSV export with clicks_json")
    parser.add_argument("--rois", type=Path, default=None,
                        help="folder with the returned CHK_XX_rois.zip files: recompute and compare with the CSV")
    parser.add_argument("--key", type=Path, default=None, help="default: private/check_key.json (practice_key.json with --practice)")
    parser.add_argument("--bar", type=Path, default=KIT / "private" / "pass_bar.json")
    parser.add_argument("--name", default=None, help="label for the saved report (default: the input file stem)")
    parser.add_argument("--fix-units", action="store_true",
                        help="convert values that clearly look like cm, radians, pixels, 10x or an obtuse angle (always warned)")
    parser.add_argument("--reveal", action="store_true",
                        help="print the expert values even while the check is INCOMPLETE or for cells not measured "
                             "(the check is then used up)")
    parser.add_argument("--practice", action="store_true",
                        help="practice progress: first attempts of the last --last finished practice images")
    parser.add_argument("--last", type=int, default=5, help="with --practice: how many recent images (0 = all)")
    parser.add_argument("--reference-without", default=None, metavar="RATER",
                        help="grade against the mean of the raters WITHOUT this one (self-test / analysis)")
    parser.add_argument("--out", type=Path, default=None, help="report JSON (default: <key dir>/check_results/)")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    if args.practice:
        args.key = args.key or KIT / "private" / "practice_key.json"
        args.state = args.state or KIT / "practice" / "autosave.json"
    args.key = args.key or KIT / "private" / "check_key.json"
    args.state = args.state or KIT / "check" / "autosave.json"
    key = load_json(args.key)
    bar_file = load_json(args.bar) if (not args.practice and args.bar and Path(args.bar).exists()) else None
    if args.practice:
        if args.csv:
            raise SystemExit("--practice reads the practice autosave.json (--state), not a CSV")
        if not Path(args.state).is_file():
            raise SystemExit(f"no practice save file at {args.state} yet")
        key, chosen = practice_subset(load_json(args.state), key, args.last)
        if not chosen:
            raise SystemExit("no finished practice image yet (all 9 measurements)")
    values, notes, meta, source = load_candidate(args, key)
    raw_values = meta.pop("_raw_values", {}) if meta else {}
    min_complete = len(key["items"]) if args.practice else MIN_COMPLETE_IMAGES
    res = grade(values, key, bar_file, without=args.reference_without, min_complete=min_complete)
    res["input"] = {"source": source, "path": str(args.csv or args.state), "meta": meta}
    if not args.practice:
        res["copy_check"] = copy_check(values, key)
    if args.rois:
        if not raw_values:
            raise SystemExit("--rois needs --csv with a filled answer_template.csv")
        rc = compare_rois(args.rois, key, raw_values)
        rc["S_from_rois"] = summarize(errors(rc["values"], key))["S"] if rc["n_images"] >= MIN_COMPLETE_IMAGES else None
        res["roi_check"] = rc
    reveal = args.reveal or args.practice
    report(res, bar_file, source, notes, reveal=reveal, practice=args.practice)
    res["notes"] = notes
    if not args.no_save:
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.name or ("practice" if args.practice else Path(args.csv or args.state).stem))
        out = args.out or (Path(args.key).parent / "check_results" / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(_clean(res if reveal else redact(res)), indent=1), encoding="utf-8")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
