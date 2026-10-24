"""Compute UMUD hand labels from the annotator's raw clicks (Python mirror of app.js).

    .venv/bin/python tools/annotator/labels.py                       # autosave.json -> labels.csv
    .venv/bin/python tools/annotator/labels.py --csv export.csv      # recompute from an exported CSV
    .venv/bin/python tools/annotator/labels.py --submission out.csv \
        --fill artifacts/sub_v1_geom_clipmed.csv                     # 309-row submission-format file

The calibration tasks (calib_manifest.json, ids CAL_XX) live in the same autosave state; their rows
go to a separate file, calib_labels.csv (CALIB_HEADER), never into labels.csv or a submission.

Formulas (organiser protocol, square pixels at the VERTICAL scale):
    MT = |y_deep - y_superficial| * mm/px, FL = polyline length * mm/px,
    PA = acute angle between the fascicle line (2 clicks) and the deep-aponeurosis line (2 clicks);
    each parameter is the mean of its (up to) 3 measurements.  Clip labels (first frame) are
    copied to every frame of the clip.  Stdlib only.
    Calibration rows also carry pa_h_deg (acute angle of each PA fascicle line vs the image
    HORIZONTAL, for the dynamic benchmark convention), fl_px and mt_px (scale-free).
    A task may define ``need`` (e.g. dynamic calibration frames: mt 0, fl 3, pa 3).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "annotator"
TOOLS = ("mt", "fl", "pa")
NEED = 3
HEADER = ["image_id", "pa_deg", "fl_mm", "mt_mm", "clicks_json", "task_id", "clip", "n_pa", "n_fl",
          "n_mt", "complete", "flag", "note", "px_per_cm", "active_s", "updated"]
CALIB_HEADER = ["image_id", "kind", "pa_deg", "pa_h_deg", "fl_mm", "fl_px", "mt_mm", "mt_px", "clicks_json",
                "task_id", "n_pa", "n_fl", "n_mt", "need_mt", "complete", "flag", "note", "px_per_cm",
                "active_s", "updated"]
DEFAULT_NEED = {"mt": NEED, "fl": NEED, "pa": NEED}


def mt_value(m: list, mm_per_px: float) -> float:
    return abs(m[1][1] - m[0][1]) * mm_per_px


def fl_value(m: list, mm_per_px: float) -> float:
    return sum(math.dist(m[i - 1], m[i]) for i in range(1, len(m))) * mm_per_px


def angle_between(p0, p1, q0, q1) -> float:
    a = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    b = math.atan2(q1[1] - q0[1], q1[0] - q0[0])
    d = math.degrees(abs(a - b)) % 180.0
    return 180.0 - d if d > 90.0 else d


def pa_value(m: list) -> float:
    return angle_between(*m[:4])


def pa_h_value(m: list) -> float:
    """Acute angle of the PA fascicle line (first 2 clicks) against the image horizontal."""

    return angle_between(m[0], m[1], [0.0, 0.0], [1.0, 0.0])


def _mean(vals: list) -> float:
    # plain left-to-right sum like JavaScript (Python 3.12's sum() of floats is compensated)
    acc = 0.0
    for v in vals:
        acc += v
    return acc / len(vals) if vals else math.nan


def summarize(label: dict, px_per_cm: float, need: dict | None = None) -> dict:
    mm_per_px = 10.0 / px_per_cm
    need = need or DEFAULT_NEED
    out: dict = {}
    for tool in TOOLS:
        items = label.get(tool) or []
        if tool == "mt":
            vals = [mt_value(m, mm_per_px) for m in items]
            out["mt_px"] = _mean([mt_value(m, 1.0) for m in items])
        elif tool == "fl":
            vals = [fl_value(m, mm_per_px) for m in items]
            out["fl_px"] = _mean([fl_value(m, 1.0) for m in items])
        else:
            vals = [pa_value(m) for m in items]
            out["pa_h"] = _mean([pa_h_value(m) for m in items])
            out["vals_pa_h"] = [pa_h_value(m) for m in items]
        out[tool] = _mean(vals)
        out[f"vals_{tool}"] = vals
        out[f"n_{tool}"] = len(vals)
    out["complete"] = all(out[f"n_{t}"] >= need.get(t, NEED) for t in TOOLS)
    return out


def merge_states(disk: dict, incoming: dict) -> tuple[dict, list[str]]:
    """Per-task newest-wins merge.  Returns (merged, ids where the disk copy was newer)."""

    labels = dict(disk.get("labels") or {})
    for tid, lab in (incoming.get("labels") or {}).items():
        if tid not in labels or (lab.get("updated") or 0) >= (labels[tid].get("updated") or 0):
            labels[tid] = lab
    newer = [tid for tid, lab in (disk.get("labels") or {}).items()
             if (lab.get("updated") or 0) > ((incoming.get("labels") or {}).get(tid, {}).get("updated") or 0)]
    times = dict(disk.get("times") or {})
    for tid, ms in (incoming.get("times") or {}).items():
        times[tid] = max(ms, times.get(tid, 0))
    merged = {
        "version": 1,
        "labels": labels,
        "times": times,
        "settings": incoming.get("settings") or disk.get("settings") or {},
        "updated": max(disk.get("updated") or 0, incoming.get("updated") or 0),
    }
    return merged, newer


def _iso(ms: float) -> str:
    """Same format as JavaScript Date.toISOString()."""

    ms = int(ms or 0)
    stamp = datetime.fromtimestamp(ms // 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    return f"{stamp}.{ms % 1000:03d}Z"


def _fmt(v: float) -> str:
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.3f}"


def _js_val(v):
    """A number as JavaScript holds it, so str() and json.dumps() match String(x) / JSON.stringify
    (150.0 -> 150, 126.1 -> 126.1)."""

    return int(v) if isinstance(v, float) and v.is_integer() else v


def build_rows(state: dict, manifest: dict) -> list[dict]:
    images = {e["image_id"]: e for e in manifest["images"]}
    rows = []
    for task in manifest["tasks"]:
        lab = (state.get("labels") or {}).get(task["task_id"])
        if not lab or not (any(lab.get(t) for t in TOOLS) or lab.get("flag") or lab.get("note")):
            continue
        entry = images[task["image_id"]]
        s = summarize(lab, entry["px_per_cm"])
        clicks = json.dumps({"task_id": task["task_id"], "px_per_cm": _js_val(entry["px_per_cm"]),
                             "mt": lab.get("mt") or [], "fl": lab.get("fl") or [],
                             "pa": lab.get("pa") or []}, separators=(",", ":"))
        for member in task["members"]:
            rows.append({
                "image_id": member, "pa_deg": _fmt(s["pa"]), "fl_mm": _fmt(s["fl"]),
                "mt_mm": _fmt(s["mt"]), "clicks_json": clicks, "task_id": task["task_id"],
                "clip": "" if task["clip"] is None else task["clip"], "n_pa": s["n_pa"],
                "n_fl": s["n_fl"], "n_mt": s["n_mt"], "complete": int(s["complete"]),
                "flag": int(bool(lab.get("flag"))), "note": lab.get("note") or "",
                "px_per_cm": _js_val(entry["px_per_cm"]),
                "active_s": round(((state.get("times") or {}).get(task["task_id"]) or 0) / 1000),
                "updated": _iso(lab.get("updated") or 0),
            })
    return rows


def build_calib_rows(state: dict, calib_manifest: dict) -> list[dict]:
    """One row per calibration task that has anything recorded (CALIB_HEADER)."""

    images = {e["image_id"]: e for e in calib_manifest["images"]}
    rows = []
    for task in calib_manifest["tasks"]:
        lab = (state.get("labels") or {}).get(task["task_id"])
        if not lab or not (any(lab.get(t) for t in TOOLS) or lab.get("flag") or lab.get("note")):
            continue
        entry = images[task["image_id"]]
        need = task.get("need") or DEFAULT_NEED
        s = summarize(lab, entry["px_per_cm"], need)
        clicks = json.dumps({"task_id": task["task_id"], "px_per_cm": _js_val(entry["px_per_cm"]),
                             "mt": lab.get("mt") or [], "fl": lab.get("fl") or [],
                             "pa": lab.get("pa") or []}, separators=(",", ":"))
        rows.append({
            "image_id": task["image_id"], "kind": task.get("kind", ""), "pa_deg": _fmt(s["pa"]),
            "pa_h_deg": _fmt(s["pa_h"]), "fl_mm": _fmt(s["fl"]), "fl_px": _fmt(s["fl_px"]),
            "mt_mm": _fmt(s["mt"]), "mt_px": _fmt(s["mt_px"]), "clicks_json": clicks,
            "task_id": task["task_id"], "n_pa": s["n_pa"], "n_fl": s["n_fl"], "n_mt": s["n_mt"],
            "need_mt": need.get("mt", NEED), "complete": int(s["complete"]),
            "flag": int(bool(lab.get("flag"))), "note": lab.get("note") or "",
            "px_per_cm": _js_val(entry["px_per_cm"]),
            "active_s": round(((state.get("times") or {}).get(task["task_id"]) or 0) / 1000),
            "updated": _iso(lab.get("updated") or 0),
        })
    return rows


def rows_to_csv(rows: list[dict], header: list[str] = HEADER) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=header, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def state_from_csv(path: Path) -> dict:
    labels: dict = {}
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if not row.get("clicks_json"):
                continue
            obj = json.loads(row["clicks_json"])
            tid = obj.get("task_id") or row.get("task_id") or row["image_id"]
            labels.setdefault(tid, {"mt": obj.get("mt", []), "fl": obj.get("fl", []),
                                    "pa": obj.get("pa", []), "flag": row.get("flag") == "1",
                                    "note": row.get("note", "")})
    return {"labels": labels, "times": {}}


def write_submission(rows: list[dict], manifest: dict, out: Path, fill: Path | None) -> None:
    complete = {r["image_id"]: r for r in rows if r["complete"] == 1}
    fallback: dict = {}
    if fill is not None:
        text = fill.read_text(encoding="utf-8-sig")
        delim = ";" if text.splitlines()[0].count(";") > text.splitlines()[0].count(",") else ","
        fallback = {r["image_id"]: r for r in csv.DictReader(text.splitlines(), delimiter=delim)}
    lines = ["image_id,pa_deg,fl_mm,mt_mm"]
    missing = 0
    for entry in manifest["images"]:
        iid = entry["image_id"]
        src = complete.get(iid) or fallback.get(iid)
        if src is None:
            missing += 1
            continue
        lines.append(f"{iid},{float(src['pa_deg']):.4f},{float(src['fl_mm']):.4f},{float(src['mt_mm']):.4f}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}: {len(complete)} hand-labelled + {len(lines) - 1 - len(complete)} filled, "
          f"{missing} missing")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute annotator labels in Python")
    parser.add_argument("--manifest", type=Path, default=DATA / "manifest.json")
    parser.add_argument("--calib-manifest", type=Path, default=DATA / "calib_manifest.json")
    parser.add_argument("--calib-out", type=Path, default=DATA / "calib_labels.csv")
    parser.add_argument("--state", type=Path, default=DATA / "autosave.json")
    parser.add_argument("--csv", type=Path, default=None, help="read clicks from an exported CSV")
    parser.add_argument("--out", type=Path, default=DATA / "labels.csv")
    parser.add_argument("--submission", type=Path, default=None)
    parser.add_argument("--fill", type=Path, default=None, help="fallback CSV for unlabelled images")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    state = state_from_csv(args.csv) if args.csv else json.loads(args.state.read_text(encoding="utf-8"))
    rows = build_rows(state, manifest)
    args.out.write_text(rows_to_csv(rows), encoding="utf-8")
    n_complete = len({r["task_id"] for r in rows if r["complete"] == 1})
    print(f"wrote {args.out}: {len(rows)} rows, {n_complete}/{len(manifest['tasks'])} tasks complete")
    if args.calib_manifest.exists():
        calib = json.loads(args.calib_manifest.read_text(encoding="utf-8"))
        crows = build_calib_rows(state, calib)
        args.calib_out.write_text(rows_to_csv(crows, CALIB_HEADER), encoding="utf-8")
        n_c = sum(r["complete"] == 1 for r in crows)
        print(f"wrote {args.calib_out}: {len(crows)} rows, {n_c}/{len(calib['tasks'])} calibration tasks complete")
    if args.submission:
        write_submission(rows, manifest, args.submission, args.fill)


if __name__ == "__main__":
    main()
