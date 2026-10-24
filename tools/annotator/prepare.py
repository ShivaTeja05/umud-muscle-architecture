"""Prepare the UMUD test images for the offline annotation tool.

Converts every competition test image to PNG (one image at a time, no pools),
decodes its physical scale with ``umud.geometry.ultrasound_metadata`` and writes
``manifest.json`` describing the annotation tasks.

Clips: IMG_00056-IMG_00195 are 28 Telemed ArtUS clips of 5 near-identical frames
(clip = (n - 56) // 5).  Only the first frame of each clip is an annotation task;
its labels are copied to all 5 frames on export.  28 clip tasks + 169 singles = 197.

Optional model hints (``--hints sub.csv`` with image_id,pa_deg,fl_mm,mt_mm) are
stored in the manifest; the browser tool only reveals them after an image is
finished, to avoid anchoring.

Calibration (``--calib``): 12 benchmark images with expert answers (8 OSF static images + 4 frames
of the OSF dynamic GM calf-raise video) that the annotator measures FIRST, so that their personal
per-component error and bias can be measured (tools/annotator/score_calibration.py).  The answers go
to a separate file that serve.py refuses to serve and the page never loads.

Usage (from the project root):
    PYTHONPATH=src .venv/bin/python tools/annotator/prepare.py \
        [--hints artifacts/sub_v1_geom_clipmed.csv] [--force]
    PYTHONPATH=src .venv/bin/python tools/annotator/prepare.py --calib [--force]

Output (data/ is gitignored, never publish it):
    data/annotator/img/<stem>.png
    data/annotator/manifest.json
    data/annotator/calib/CAL_XX[_nK].png       (--calib; _nK = view-only neighbour frames)
    data/annotator/calib_manifest.json         (--calib; NO answers)
    data/annotator/calib_answers.json          (--calib; expert answers, never served)
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from umud.geometry import ultrasound_metadata  # noqa: E402

DEFAULT_TEST_DIR = (
    ROOT
    / "data"
    / "umud-challenge-muscle-architecture-in-ultrasound-data"
    / "test_images_v2"
    / "test_set_v2"
)
DEFAULT_OUT = ROOT / "data" / "annotator"
CLIP_FIRST, CLIP_LAST, CLIP_LEN = 56, 195, 5
IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
# Families whose scale decode is a heuristic; flag them in the UI.
TRUSTED_FAMILIES = {"siemens_juniper", "telemed_artus", "philips_lumify", "legacy_512", "legacy_853"}

# ----------------------------------------------------------------------------- calibration set
BENCH = ROOT / "data" / "external_umud_benchmark"
STATIC_DIR = BENCH / "static" / "benchmark_dataset_architecture_v0.1.0"
STATIC_XLSX = STATIC_DIR / "Results_benchmark_architecture_v0.1.0.xlsx"
DYN_DIR = BENCH / "dynamic" / "benchmark_dataset_architecture_GM_dynamic_v0.1.0"
DYN_XLSX = DYN_DIR / "benchmark_architecture_GM_calf_raise_v0.1.0.xlsx"
# Static images our mask models never saw in training (scripts/build_candidates.py: CLEAN).  The
# pipeline comparison is honest only on these and on the dynamic frames.
CLEAN_STATIC = {f"im_{k:02d}_arch" for k in (2, 3, 14, 17, 20, 23, 24, 31, 33, 34)}
# Fixed, interleaved order (static Echo Blaster / static 126 px/cm / dynamic), so that learning
# during the calibration does not line up with one kind of image.  Dynamic entries are frame
# numbers (only even frames were rated by the 3 experts; odd frames are interpolations).
#   660x556 @ 77 px/cm  = Telemed Echo Blaster 128 (same size as the dynamic video); clean set
#   1234x688 @ 126 px/cm = Telemed ArtUs (same px/cm as the 1088x644 Telemed test clips); in training
CALIB_ORDER = ["im_14_arch", "im_04_arch", 8, "im_03_arch", "im_17_arch", 96,
               "im_15_arch", "im_02_arch", 48, "im_31_arch", "im_20_arch", 128]
DYN_VIEW_FRAMES = 5  # the target frame + the next 4 frames, view only (like a 5-frame test clip)
DYN_NOTE = ("Dynamic frame (calf-raise video): only FL and PA are needed here; MT is optional. "
            "Measure PA exactly as on every other image.")
RATERS_STATIC = [f"R{k}" for k in range(1, 8)]
TOLERANCES = {"pa_deg": 6.0, "fl_mm": 12.0, "mt_mm": 3.0}


def dynamic_px_per_cm() -> float:
    """The project's dynamic-benchmark scale (scripts/build_candidates.py: DYN_PPC), read from the source."""

    text = (ROOT / "scripts" / "build_candidates.py").read_text(encoding="utf-8")
    match = re.search(r"^DYN_PPC\s*=\s*([0-9.]+)", text, flags=re.M)
    return float(match.group(1)) if match else 77.4


def image_number(name: str) -> int | None:
    match = re.search(r"(\d+)", Path(name).stem)
    return int(match.group(1)) if match else None


def clip_of(number: int | None) -> int | None:
    if number is None or not CLIP_FIRST <= number <= CLIP_LAST:
        return None
    return (number - CLIP_FIRST) // CLIP_LEN


def load_hints(path: Path | None) -> dict[str, dict[str, float]]:
    if path is None:
        return {}
    text = path.read_text(encoding="utf-8-sig")
    dialect = ";" if text.splitlines()[0].count(";") > text.splitlines()[0].count(",") else ","
    hints: dict[str, dict[str, float]] = {}
    for row in csv.DictReader(text.splitlines(), delimiter=dialect):
        try:
            hints[row["image_id"].strip()] = {
                "pa_deg": round(float(row["pa_deg"]), 3),
                "fl_mm": round(float(row["fl_mm"]), 3),
                "mt_mm": round(float(row["mt_mm"]), 3),
            }
        except (KeyError, ValueError):
            continue
    return hints


def is_gray(rgb: np.ndarray) -> bool:
    """True when (almost) every pixel has R == G == B, so storing L loses nothing visible."""

    if rgb.ndim == 2:
        return True
    diff = np.abs(rgb[..., 0].astype(np.int16) - rgb[..., 1]) + np.abs(
        rgb[..., 1].astype(np.int16) - rgb[..., 2]
    )
    return float(np.mean(diff > 6)) < 0.002


def save_png(image: Image.Image, rgb: np.ndarray, dst: Path, force: bool) -> None:
    if force or not dst.exists():
        if is_gray(rgb):
            Image.fromarray(rgb[..., 0]).save(dst, format="PNG", compress_level=6)
        else:
            image.save(dst, format="PNG", compress_level=6)


def convert_one(src: Path, dst: Path, force: bool) -> dict:
    with Image.open(src) as opened:
        image = opened.convert("RGB")
    rgb = np.asarray(image)
    height, width = rgb.shape[:2]
    meta = ultrasound_metadata(rgb, src)
    save_png(image, rgb, dst, force)
    info = {
        "width": int(width),
        "height": int(height),
        "px_per_cm": round(float(meta.px_per_cm), 4),
        "group": meta.scanner_family,
        "viewport": [int(v) for v in meta.viewport],
        "scale_trusted": meta.scanner_family in TRUSTED_FAMILIES,
    }
    del image, rgb
    return info


# ----------------------------------------------------------------------------- calibration
def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else round(f, 4)  # NaN -> None


def _mean(values) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def static_answers() -> dict[str, dict]:
    """Expert answers for every static image: mean over R1..R7 (im_19 R7_MT -> NaN, an 80.13 typo),
    plus the per-rater values (scripts/build_candidates.py: static_labels / static_gts)."""

    import pandas as pd  # only needed for --calib

    lab = pd.read_excel(STATIC_XLSX, sheet_name="Manual_architecture")
    lab.loc[lab.ImageID == "im_19_arch", "R7_MT"] = np.nan
    out = {}
    for row in lab.to_dict("records"):
        raters = {r: {"pa_deg": _num(row[f"{r}_PA"]), "fl_mm": _num(row[f"{r}_FL"]),
                      "mt_mm": _num(row[f"{r}_MT"])} for r in RATERS_STATIC}
        raters = {r: v for r, v in raters.items() if any(x is not None for x in v.values())}
        out[row["ImageID"]] = {
            "px_per_cm": float(row["Scale_pixel_per_cm"]),
            "expert": {c: _mean(v[c] for v in raters.values()) for c in ("pa_deg", "fl_mm", "mt_mm")},
            "raters": raters,
            "tools": {"SMA": {"pa_deg": _num(row["SMA_PA"]), "fl_mm": _num(row["SMA_FL"]), "mt_mm": _num(row["SMA_MT"])},
                      "DLTrack": {"pa_deg": _num(row["DLTrack_PA"]), "fl_mm": _num(row["DLTrack_FL"]),
                                  "mt_mm": _num(row["DLTrack_MT"])}},
        }
    return out


def dynamic_answers(frames: list[int], px_per_cm: float) -> dict[int, dict]:
    """Expert answers for dynamic frames: PA in degrees measured against the image HORIZONTAL
    (benchmark convention, see research/external_data.md) and FL in pixels; 3 raters + 'Mean'."""

    import pandas as pd

    sheets = {k: pd.read_excel(DYN_XLSX, sheet_name=k).dropna(subset=["frame"]) for k in ("PA", "FL")}
    out = {}
    for fr in frames:
        rec = {k: s[s.frame.astype(int) == fr].iloc[0] for k, s in sheets.items()}
        raters = {}
        for k in (1, 2, 3):
            fl_px = _num(rec["FL"][f"Rater_{k}_FL_GM"])
            raters[f"Rater_{k}"] = {"pa_deg": _num(rec["PA"][f"Rater_{k}_PA_GM"]), "fl_px": fl_px,
                                    "fl_mm": None if fl_px is None else round(fl_px * 10 / px_per_cm, 4)}
        fl_px = _num(rec["FL"]["Mean"])
        out[fr] = {"expert": {"pa_deg": _num(rec["PA"]["Mean"]), "fl_px": fl_px,
                              "fl_mm": round(fl_px * 10 / px_per_cm, 4), "mt_mm": None},
                   "raters": raters}
    return out


def _write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    tmp.replace(path)


def prepare_calib(out: Path, force: bool) -> None:
    calib_dir = out / "calib"
    calib_dir.mkdir(parents=True, exist_ok=True)
    dyn_ppc = dynamic_px_per_cm()
    st = static_answers()
    dyn = dynamic_answers([x for x in CALIB_ORDER if isinstance(x, int)], dyn_ppc)
    images, tasks, answers = [], [], {}
    for k, src in enumerate(CALIB_ORDER, 1):
        cid = f"CAL_{k:02d}"
        if isinstance(src, int):
            kind = "dynamic"
            paths = [DYN_DIR / "frames" / f"frame_{src + j:05d}.png" for j in range(DYN_VIEW_FRAMES)]
            paths = [p for p in paths if p.exists()]
            ppc, source = dyn_ppc, f"frame_{src:05d}"
        else:
            kind, paths, ppc, source = "static", [STATIC_DIR / f"{src}.tif"], st[src]["px_per_cm"], src
        frame_ids = []
        for j, path in enumerate(paths):
            iid = cid if j == 0 else f"{cid}_n{j}"
            with Image.open(path) as opened:
                image = opened.convert("RGB")
            rgb = np.asarray(image)
            height, width = rgb.shape[:2]
            if j == 0:  # viewport only (the scale is the expert one); view-only frames share it
                viewport = [int(v) for v in ultrasound_metadata(rgb, path).viewport]
            save_png(image, rgb, calib_dir / f"{iid}.png", force)
            images.append({"image_id": iid, "stem": iid, "file": f"calib/{iid}.png", "number": None, "clip": None,
                           "width": int(width), "height": int(height), "px_per_cm": round(float(ppc), 4),
                           "group": f"calib_{kind}", "viewport": viewport,
                           "scale_trusted": True, "kind": kind, "task_id": cid})
            frame_ids.append(iid)
            del image, rgb
            gc.collect()
        task = {"task_id": cid, "image_id": cid, "clip": None, "members": [cid], "kind": kind,
                "need": {"mt": 0 if kind == "dynamic" else 3, "fl": 3, "pa": 3}}
        if len(frame_ids) > 1:
            task["frames"] = frame_ids
        if kind == "dynamic":
            task["note"] = DYN_NOTE
        tasks.append(task)
        width_height = f"{images[-len(frame_ids)]['width']}x{images[-len(frame_ids)]['height']}"
        if kind == "static":
            a = st[src]
            answers[cid] = {"source": source, "kind": kind, "px_per_cm": ppc, "size": width_height,
                            "device_guess": "Telemed Echo Blaster 128" if ppc == 77 else "Telemed ArtUs EXT-1H",
                            "pipeline_honest": src in CLEAN_STATIC, "pa_reference": "deep_aponeurosis",
                            "fl_unit": "mm", "expert": a["expert"], "raters": a["raters"], "tools": a["tools"]}
        else:
            a = dyn[src]
            answers[cid] = {"source": source, "kind": kind, "px_per_cm": ppc, "size": width_height,
                            "device_guess": "Telemed Echo Blaster 128 (GM calf raise video)",
                            "pipeline_honest": True, "pa_reference": "horizontal", "fl_unit": "px",
                            "expert": a["expert"], "raters": a["raters"]}
        print(f"{cid}: {source:12s} {kind:7s} {ppc:6.2f} px/cm  {width_height}  frames={len(frame_ids)}", flush=True)

    manifest = {
        "version": 1, "kind": "calib", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "scale_note": "px_per_cm is the expert scale of each benchmark image (static: workbook "
                      "Scale_pixel_per_cm; dynamic: scripts/build_candidates.py DYN_PPC), applied to both axes.",
        "note": "Calibration images with known expert answers. The answers are NOT in this file.",
        "n_images": len(images), "n_tasks": len(tasks), "images": images, "tasks": tasks,
    }
    _write_json(out / "calib_manifest.json", manifest)
    _write_json(out / "calib_answers.json", {
        "version": 1, "created": manifest["created"],
        "note": "Expert answers for the calibration tasks. Never served by serve.py, never loaded by the page. "
                "static: mean of R1..R7 (im_19 R7_MT -> NaN), PA vs the deep aponeurosis, FL/MT in mm. "
                "dynamic: mean of 3 raters, PA vs the image HORIZONTAL, FL in px (fl_mm = fl_px * 10 / px_per_cm), "
                "no MT. pipeline_honest = image not in our models' training data.",
        "tolerances": TOLERANCES, "dyn_px_per_cm": dyn_ppc, "items": answers,
    })
    print(f"wrote {out / 'calib_manifest.json'} ({len(tasks)} tasks, {len(images)} images) and "
          f"{out / 'calib_answers.json'} (answers; never served)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--hints", type=Path, default=None, help="optional model CSV")
    parser.add_argument("--force", action="store_true", help="re-encode existing PNGs")
    parser.add_argument("--limit", type=int, default=0, help="debug: only the first N images")
    parser.add_argument("--calib", action="store_true",
                        help="build the 12-image calibration set (calib/, calib_manifest.json, calib_answers.json) "
                             "instead of the test manifest")
    args = parser.parse_args()
    if args.calib:
        prepare_calib(args.out, args.force)
        return

    files = sorted(
        (p for p in args.test_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda p: (image_number(p.name) or 0, p.name),
    )
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"no images found in {args.test_dir}")

    img_dir = args.out / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    hints = load_hints(args.hints)

    stems = [p.stem for p in files]
    if len(set(stems)) != len(stems):
        raise SystemExit("duplicate image stems; PNG names would collide")

    images = []
    started = time.time()
    for index, path in enumerate(files, 1):
        number = image_number(path.name)
        clip = clip_of(number)
        png_name = f"{path.stem}.png"
        info = convert_one(path, img_dir / png_name, args.force)
        entry = {
            "image_id": path.name,  # submission IDs keep their extension
            "stem": path.stem,
            "file": f"img/{png_name}",
            "number": number,
            "clip": clip,
            **info,
        }
        if path.name in hints:
            entry["hint"] = hints[path.name]
        images.append(entry)
        if index % 25 == 0 or index == len(files):
            print(f"[{index}/{len(files)}] {path.name} {info['group']} "
                  f"{info['px_per_cm']} px/cm  ({time.time() - started:.0f}s)", flush=True)
        gc.collect()

    # Build tasks: singles + first frame of each clip.
    by_clip: dict[int, list[dict]] = {}
    for entry in images:
        if entry["clip"] is not None:
            by_clip.setdefault(entry["clip"], []).append(entry)
    tasks = []
    seen_clips: set[int] = set()
    for entry in images:
        clip = entry["clip"]
        if clip is None:
            members = [entry["image_id"]]
        elif clip in seen_clips:
            entry["task_id"] = by_clip[clip][0]["image_id"]
            continue
        else:
            seen_clips.add(clip)
            members = [e["image_id"] for e in by_clip[clip]]
        entry["task_id"] = entry["image_id"]
        tasks.append({
            "task_id": entry["image_id"],
            "image_id": entry["image_id"],
            "clip": clip,
            "members": members,
        })

    groups: dict[str, int] = {}
    for entry in images:
        groups[entry["group"]] = groups.get(entry["group"], 0) + 1
    untrusted = [e["image_id"] for e in images if not e["scale_trusted"]]

    manifest = {
        "version": 1,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_dir": str(args.test_dir.relative_to(ROOT)) if args.test_dir.is_relative_to(ROOT)
        else str(args.test_dir),
        "scale_note": "px_per_cm is the VERTICAL scale, applied to both axes (square pixels).",
        "hints_source": str(args.hints) if args.hints else None,
        "n_images": len(images),
        "n_tasks": len(tasks),
        "groups": groups,
        "untrusted_scale": untrusted,
        "images": images,
        "tasks": tasks,
    }
    out_path = args.out / "manifest.json"
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    tmp.replace(out_path)
    n_clip_tasks = sum(1 for t in tasks if t["clip"] is not None)
    print(f"wrote {out_path}: {len(images)} images, {len(tasks)} tasks "
          f"({n_clip_tasks} clip tasks + {len(tasks) - n_clip_tasks} singles), "
          f"hints for {sum('hint' in e for e in images)} images")
    print("groups:", groups)
    if untrusted:
        print("WARNING: heuristic scale for", untrusted)


if __name__ == "__main__":
    main()
