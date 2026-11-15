"""Build the manual labelling kit (practice set, hidden-answer check set, expert package) in data/manual_labelling/.

Source: the public OSF UMUD static benchmark only (35 images, CC BY 4.0, https://osf.io/xbawc/).  No competition
test image is read or copied.  One image at a time, PIL + numpy + pandas, about 2 minutes, < 0.3 GB.

    # once: choose the 10 check images (a JSON {"check": {"im_XX_arch": "why", ...}}); the CHK order is drawn
    # at random and stored, with the practice order, in private/selection.json (git-ignored, never served)
    PYTHONPATH=src .venv/bin/python tools/annotator/prepare_manual_kit.py --init-selection my_choice.json
    # (re)build everything from private/selection.json
    PYTHONPATH=src .venv/bin/python tools/annotator/prepare_manual_kit.py

Output (data/ is git-ignored):
    check/          annotator data dir: calib_manifest.json (no answers), calib/CHK_XX.png (mirrored left-right),
                    manifest.json (empty shell that serve.py needs).  No answer file: the answers are in private/.
    practice/       annotator data dir: manifest.json (PRAC_01..25, hint = expert mean + range), img/PRAC_XX.png
    expert_package/ images/CHK_XX.png (byte-identical to check/calib/), scales.csv, answer_template.csv, CREDITS.txt
    private/        check_key.json, calib_answers.json (the check answers in the data/annotator format; never in
                    check/), practice_key.json, pass_bar.json, selection.md (+ selection.json, build_cache.json)

The CHK -> source mapping exists only in private/.  This script holds no selection, so it can be published.
--init-selection refuses (unless overridden) a check image that is in the old calibration set (data/annotator/,
answers on this computer, sources listed in tools/annotator/README.md) or that is a near-duplicate of a practice
image (32x32 viewport thumbnail NCC > 0.75), because then the check would not be blind.
"""

from __future__ import annotations

import argparse
import csv
import gc
import io
import json
import math
import random
import shutil
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

TOOL_DIR = Path(__file__).resolve().parent
ROOT = TOOL_DIR.parents[1]
sys.path.insert(0, str(TOOL_DIR))
sys.path.insert(0, str(ROOT / "src"))

import prepare  # noqa: E402  (is_gray, _num, the benchmark paths)
import score_check as sc  # noqa: E402

KIT = ROOT / "data" / "manual_labelling"
STATIC_DIR = prepare.STATIC_DIR
XLSX = prepare.STATIC_XLSX
RATERS = [f"R{k}" for k in range(1, 8)]
COMPS = sc.COMPS
COL = {"pa_deg": "PA", "fl_mm": "FL", "mt_mm": "MT"}
CLEAN_STATIC = prepare.CLEAN_STATIC  # static images that are NOT in the competition training data
KG0_STATIC = ROOT / "artifacts" / "notes" / "umud_v15_scratch" / "relshrink" / "work" / "bench_pred_kg0_static.csv"
OLD_CALIB = ROOT / "data" / "annotator"  # read only: C12 predictions cached for the old calibration images
# Entries treated as missing in every key (documented in selection.md and check_key.json)
EXCLUDED = [
    ("im_19_arch", "R7", "mt_mm", "80.13 mm is a typo: the other 5 raters say 19.9-20.1 mm (also excluded by "
                                  "prepare.py and scripts/build_candidates.py)"),
    ("im_26_arch", "R7", "fl_mm", "33.9 mm cannot be an apo-to-apo fascicle here: FL x sin(PA) = 33.9 x sin(19.0) = "
                                  "11.0 mm, but MT is 22.8 mm; the other 5 raters say 64.7-78.8 mm"),
]
NEAR_DUP_NCC = 0.75  # check vs practice: above this, two images are treated as the same muscle / session
EXT_TIP = 1.05  # practice tip when the mean fascicle spans more than this share of the image width
PRACTICE_NOTE = ("Tip: most fascicles here are longer than the image is wide. Extend them outside the image "
                 "in a straight line until they meet the (extended) aponeurosis.")
CREDITS = """Image credits and licence
=========================

The images in this package come from:

  UMUD benchmark muscle architecture dataset, version 0.1.0.
  Authors: the UMUD project team (Paul Ritsche and colleagues; full list on the OSF page).
  Open Science Framework, project xbawc: https://osf.io/xbawc/
  Licence: Creative Commons Attribution 4.0 International (CC BY 4.0),
           https://creativecommons.org/licenses/by/4.0/
  Related article: Ritsche P, Sarto F, et al., BMC Medical Imaging 26:139 (2026),
                   https://doi.org/10.1186/s12880-026-02170-0

Changes made by us:
  - a subset of the images was selected and given new names (CHK_01 ... CHK_10) in a new order;
{mirror}  - the images were converted from TIFF to PNG ({pixels}, same size);
  - all file metadata was removed.
No other change was made. The original authors do not endorse this use.
"""
CREDITS_MIRROR = "  - each image was mirrored left-right (this does not change MT, FL or PA);\n"


# ----------------------------------------------------------------------------- data
def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(sc._clean(obj), indent=1), encoding="utf-8")
    tmp.replace(path)


def read_workbook() -> dict[str, dict]:
    import pandas as pd

    lab = pd.read_excel(XLSX, sheet_name="Manual_architecture")
    out = {}
    for row in lab.to_dict("records"):
        iid = row["ImageID"]
        raters = {}
        for r in RATERS:
            v = {c: prepare._num(row[f"{r}_{COL[c]}"]) for c in COMPS}
            for src, rr, comp, _ in EXCLUDED:
                if src == iid and rr == r:
                    v[comp] = None
            if any(x is not None for x in v.values()):
                raters[r] = v
        tools = {t: {c: prepare._num(row[f"{t}_{COL[c]}"]) for c in COMPS} for t in ("SMA", "DLTrack")}
        out[iid] = {"px_per_cm": float(row["Scale_pixel_per_cm"]), "raters": raters, "tools": tools,
                    "raw": {f"{r}_{COL[c]}": prepare._num(row[f"{r}_{COL[c]}"]) for r in RATERS for c in COMPS}}
    return out


def device_of(w: int, h: int, ppc: float) -> tuple[str, str]:
    """(short group, description).  From the image size, scale and the on-screen UI; a guess, not metadata."""

    if (w, h) == (660, 556) and round(ppc) == 77:
        return "telemed_echoblaster", "Telemed Echo Blaster 128 (likely: Echo Wave UI, 660x556, same frame as the GM dynamic video)"
    if h == 688 and round(ppc) == 126:
        return "telemed_artus", "Telemed ArtUs EXT-1H (likely: Echo Wave UI, 126 px/cm like the ArtUs test clips)"
    if (w, h) == (800, 600):
        return "philips_hd11", "Philips HD11 (likely: L12-5 probe and Philips UI text)"
    return "unknown", "unknown (no Telemed or Philips UI; perhaps the Hitachi Aloka Alpha-10)"


def muscle_of(num: int, group: str, mt: float) -> str:
    if group.startswith("telemed"):
        if mt < 13.0:
            return "GM or soleus (unsure: thin superficial muscle)"
        return "GM (likely: superficial calf muscle over a second pennate muscle)"
    if group == "philips_hd11":
        return "unknown (GM or VL)"
    return "unknown"


def image_stats(src: str, wb: dict) -> dict:
    item = wb[src]
    st = {"expert": {}, "sd": {}, "min": {}, "max": {}, "n": {}}
    for c in COMPS:
        v = [x[c] for x in item["raters"].values() if x.get(c) is not None]
        st["expert"][c] = sum(v) / len(v)
        st["sd"][c] = statistics.stdev(v) if len(v) > 1 else math.nan
        st["min"][c], st["max"][c], st["n"][c] = min(v), max(v), len(v)
    return st


def load_rgb(src: str) -> np.ndarray:
    with Image.open(STATIC_DIR / f"{src}.tif") as opened:
        return np.asarray(opened.convert("RGB"))


def viewport_of(src: str, rgb: np.ndarray, cache: dict) -> list[int]:
    if src not in cache:
        from umud.geometry import ultrasound_metadata

        cache[src] = [int(v) for v in ultrasound_metadata(rgb, STATIC_DIR / f"{src}.tif").viewport]
    return cache[src]


def thumb(rgb: np.ndarray, vp: list[int]) -> np.ndarray:
    """32x32 z-scored grey thumbnail of the ultrasound field (for the near-duplicate check)."""

    g = Image.fromarray(np.ascontiguousarray(rgb.mean(axis=2).astype(np.uint8))).crop(tuple(vp)).resize((32, 32), Image.BILINEAR)
    a = np.asarray(g, dtype=np.float64).ravel()
    return (a - a.mean()) / (a.std() + 1e-9)


def ncc(a, b) -> float:
    return float(np.dot(np.asarray(a), np.asarray(b)) / len(a))


def near_duplicates(table: dict, check: list[str], practice: list[str]) -> list[tuple[float, str, str]]:
    """Every check x practice pair, most similar first (thumbnail NCC, either orientation)."""

    out = []
    for c in check:
        tc, tm = np.asarray(table[c]["thumb"]), np.asarray(table[c]["thumb"]).reshape(32, 32)[:, ::-1].ravel()
        for q in practice:
            out.append((max(ncc(tc, table[q]["thumb"]), ncc(tm, table[q]["thumb"])), c, q))
    return sorted(out, reverse=True)


def old_calibration_sources() -> set[str]:
    """Static source ids of the old calibration set in data/annotator/ (read only)."""

    path = OLD_CALIB / "calib_answers.json"
    if not path.exists():
        return set()
    items = json.loads(path.read_text(encoding="utf-8")).get("items", {})
    return {it["source"] for it in items.values() if it.get("kind") == "static" and it.get("source")}


def save_png(rgb: np.ndarray, dst: Path) -> None:
    """Fresh image from the pixel array: no text, ICC, EXIF or time chunks are carried over.  Stored as one
    channel only when R == G == B on every pixel (exactly the same pixels); otherwise RGB (some images have a
    faint colour tint that prepare.is_gray would drop)."""

    dst.parent.mkdir(parents=True, exist_ok=True)
    exact_gray = np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2])
    img = Image.fromarray(np.ascontiguousarray(rgb[..., 0])) if exact_gray else Image.fromarray(rgb)
    img.save(dst, format="PNG", compress_level=6)


def png_chunks(path: Path) -> list[str]:
    data = path.read_bytes()
    pos, out = 8, []
    while pos < len(data):
        n = int.from_bytes(data[pos:pos + 4], "big")
        out.append(data[pos + 4:pos + 8].decode("latin-1"))
        pos += 12 + n
    return out


# ----------------------------------------------------------------------------- per-image table
def build_table(wb: dict, cache: dict) -> dict[str, dict]:
    """Per source image: size, device / muscle guess, expert stats, LOO difficulty, extension need, viewport."""

    full_key = {"raters": RATERS, "items": {s: {"raters": wb[s]["raters"], "px_per_cm": wb[s]["px_per_cm"]} for s in wb}}
    table = {}
    for src in sorted(wb):
        num = int(src[3:5])
        rgb = load_rgb(src)
        h, w = rgb.shape[:2]
        vp = viewport_of(src, rgb, cache)
        th = thumb(rgb, vp)
        del rgb
        gc.collect()
        ppc = wb[src]["px_per_cm"]
        st = image_stats(src, wb)
        group, dev = device_of(w, h, ppc)
        field_mm = (vp[2] - vp[0]) / ppc * 10.0
        hspan = st["expert"]["fl_mm"] * math.cos(math.radians(st["expert"]["pa_deg"]))
        loo = {r: sc.summarize(sc.loo_errors({"items": {src: full_key["items"][src]}}, r))["S"]
               for r in RATERS if r in wb[src]["raters"]}
        table[src] = {"num": num, "width": w, "height": h, "px_per_cm": ppc, "viewport": vp, "group": group,
                      "device_guess": dev, "muscle_guess": muscle_of(num, group, st["expert"]["mt_mm"]), **st,
                      "n_raters": len(wb[src]["raters"]), "field_mm": field_mm, "fascicle_span_mm": hspan,
                      "extension_ratio": hspan / field_mm, "expert_loo_S_mean": statistics.mean(loo.values()),
                      "clean": src in CLEAN_STATIC, "thumb": th.tolist()}
        print(f"  {src} {w}x{h} {ppc:g} px/cm {group:19s} loo {table[src]['expert_loo_S_mean']:.3f} "
              f"ext {table[src]['extension_ratio']:.2f}", flush=True)
    return table


# ----------------------------------------------------------------------------- selection
def practice_order(sources: list[str], table: dict) -> list[str]:
    """Easy / clear first: mean expert leave-one-out score (low = experts agree), small bonus for Telemed."""

    return sorted(sources, key=lambda s: (table[s]["expert_loo_S_mean"] - (0.03 if table[s]["group"].startswith("telemed") else 0.0), s))


def init_selection(choice_path: Path, table: dict, force: bool, allow_old_calib: bool = False,
                   allow_near_duplicates: bool = False) -> None:
    out = KIT / "private" / "selection.json"
    if out.exists() and not force:
        raise SystemExit(f"{out} exists; pass --force-selection to draw a new CHK order (the old check is then void)")
    raw = json.loads(choice_path.read_text(encoding="utf-8"))
    choice = raw["check"]
    srcs = list(choice)
    if len(srcs) != 10 or any(s not in table for s in srcs):
        raise SystemExit("need exactly 10 valid source ids, e.g. im_04_arch")
    old = sorted(set(srcs) & old_calibration_sources())
    if old and not allow_old_calib:
        raise SystemExit(f"{old} are old calibration images (answers in data/annotator/, sources in the README): "
                         "the check would not be blind. Choose others or pass --allow-old-calib.")
    dups = [d for d in near_duplicates(table, srcs, [s for s in table if s not in srcs]) if d[0] > NEAR_DUP_NCC]
    if dups and not allow_near_duplicates:
        raise SystemExit("near-duplicates of practice images (their answers would be shown in practice): "
                         + ", ".join(f"{c} ~ {q} (NCC {n:.2f})" for n, c, q in dups)
                         + ". Move both to the same side or pass --allow-near-duplicates.")
    rng = random.SystemRandom()

    def bad(o: list[str]) -> bool:  # sorted, or 3 images of one device in a row
        runs = any(len({table[x]["group"] for x in o[i:i + 3]}) == 1 for i in range(len(o) - 2))
        return o == sorted(o) or o == sorted(o, reverse=True) or o == srcs or runs

    order = srcs[:]
    while bad(order):
        rng.shuffle(order)
    practice = practice_order([s for s in table if s not in srcs], table)
    write_json(out, {"created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                     "note": "PRIVATE. The CHK order was drawn with random.SystemRandom (no seed).",
                     "check": [{"chk": f"CHK_{k:02d}", "source": s, "why": choice[s]} for k, s in enumerate(order, 1)],
                     "practice": [{"prac": f"PRAC_{k:02d}", "source": s} for k, s in enumerate(practice, 1)],
                     "not_chosen": raw.get("not_chosen", {})})
    print(f"wrote {out}")


# ----------------------------------------------------------------------------- models (read only, no inference)
def model_values() -> dict[str, dict]:
    """{name: {"values": {src: {comp: v}}, "label": str, "honest": set(src)}} from cached predictions."""

    import pandas as pd

    out = {}
    if KG0_STATIC.exists():
        kg = pd.read_csv(KG0_STATIC, index_col=0)
        out["KG0 (Final A pipeline)"] = {
            "values": {s: {c: float(kg.loc[s, c]) for c in COMPS} for s in kg.index},
            "honest": set(CLEAN_STATIC),
            "label": "cached benchmark predictions (artifacts/notes/umud_v15_scratch/relshrink/work/bench_pred_kg0_static.csv, "
                     "umud.run bench_preds: static held out of its own error model and label constants = LODO). Mask "
                     "models were trained on the competition training data, which contains 25 of the 35 static images: "
                     "honest only on the 10 clean images. Its benchmark error is about 1/3 of its leaderboard error."}
    ans, pipe = OLD_CALIB / "calib_answers.json", OLD_CALIB / "calib_pipeline.json"
    if ans.exists() and pipe.exists():
        a, p = json.loads(ans.read_text()), json.loads(pipe.read_text())
        vals = {}
        for cid, rec in p.get("items", {}).items():
            it = a["items"].get(cid, {})
            if it.get("kind") == "static":
                vals[it["source"]] = {c: rec.get(c) if rec.get(c) is not None else math.nan for c in COMPS}
        out["C12 (older pipeline)"] = {"values": vals, "honest": set(CLEAN_STATIC),
                                       "label": "data/annotator/calib_pipeline.json (LODO benchmark predictions, LB 0.3767); "
                                                "only the 8 old calibration images; honest only on clean images"}
    return out


def tool_values(wb: dict) -> dict[str, dict]:
    return {f"{t} (workbook)": {"values": {s: {c: (wb[s]["tools"][t][c] if wb[s]["tools"][t][c] is not None else math.nan)
                                                for c in COMPS} for s in wb},
                                "honest": set(wb), "label": f"{t} values stored in the OSF workbook (automatic tool)"}
            for t in ("SMA", "DLTrack")}


def score_models(models: dict, key: dict, src_of: dict) -> dict:
    out = {}
    for name, m in models.items():
        vals = {cid: m["values"][src] for cid, src in src_of.items() if src in m["values"]}
        if not vals:
            continue
        res = sc.summarize(sc.errors(vals, key))
        honest_ids = {cid for cid, src in src_of.items() if src in m["honest"] and cid in vals}
        hon = sc.summarize(sc.errors({c: v for c, v in vals.items() if c in honest_ids}, key))
        out[name] = {"S": res["S"], "components": res["components"], "n_images": res["n_images"],
                     "S_honest_images": hon["S"], "n_honest_images": hon["n_images"],
                     "honest_images": sorted(honest_ids), "label": m["label"]}
    return out


# ----------------------------------------------------------------------------- pass bar
def bar_block(key: dict, raters_all: list[str], bar_raters: list[str], n_boot: int) -> dict:
    ids = list(key["items"])
    per_rater = {}
    for r in raters_all:
        res = sc.summarize(sc.loo_errors(key, r))
        per_rater[r] = {"S": res["S"], "n_images": res["n_images"], "n_cells": res["n_cells"],
                        "components": {c: {k: res["components"][c].get(k) for k in ("n", "mae", "mae_tau", "bias")}
                                       for c in COMPS}}
    table = {r: sc.summarize(sc.loo_errors(key, r)) for r in bar_raters}
    boot = sc.bootstrap({r: sc.loo_errors(key, r) for r in bar_raters}, ids, n_boot=n_boot)
    return {"n_images": len(ids), "per_rater_loo": per_rater, "bar": sc.bar_stats(table),
            "bootstrap": {k: v for k, v in boot["stats"].items()}, "n_boot": n_boot}


# ----------------------------------------------------------------------------- build
def build(table: dict, wb: dict, sel: dict, n_boot: int, mirror: bool = True) -> None:
    created = time.strftime("%Y-%m-%dT%H:%M:%S")
    check = sel["check"]
    practice = [p["source"] for p in sel["practice"]]
    src_of = {c["chk"]: c["source"] for c in check}
    assert len(src_of) == 10 and len(practice) == 25 and not set(practice) & set(src_of.values())
    d_check, d_prac, d_pkg, d_priv = KIT / "check", KIT / "practice", KIT / "expert_package", KIT / "private"

    # ---- check set
    images, tasks, answers, key_items = [], [], {}, {}
    for cid, src in src_of.items():
        t = table[src]
        rgb = load_rgb(src)
        # mirrored left-right: MT, FL and PA do not change, but a file / pixel match with the public TIFF fails
        save_png(np.ascontiguousarray(rgb[:, ::-1]) if mirror else rgb, d_check / "calib" / f"{cid}.png")
        del rgb
        gc.collect()
        x0, y0, x1, y1 = t["viewport"]
        vp = [t["width"] - x1, y0, t["width"] - x0, y1] if mirror else list(t["viewport"])
        images.append({"image_id": cid, "stem": cid, "file": f"calib/{cid}.png", "number": None, "clip": None,
                       "width": t["width"], "height": t["height"], "px_per_cm": t["px_per_cm"], "group": "calib_static",
                       "viewport": vp, "scale_trusted": True, "kind": "static", "task_id": cid})
        tasks.append({"task_id": cid, "image_id": cid, "clip": None, "members": [cid], "kind": "static",
                      "need": {"mt": 3, "fl": 3, "pa": 3}})
        common = {"kind": "static", "px_per_cm": t["px_per_cm"], "size": f"{t['width']}x{t['height']}",
                  "device_guess": t["device_guess"], "pipeline_honest": t["clean"], "pa_reference": "deep_aponeurosis",
                  "fl_unit": "mm", "expert": t["expert"], "raters": wb[src]["raters"], "tools": wb[src]["tools"]}
        answers[cid] = {"source": cid, **common}  # no original file name outside private/
        key_items[cid] = {"source": src, **common, "mirrored": mirror, "px_per_mm": t["px_per_cm"] / 10.0, "width": t["width"],
                          "height": t["height"], "muscle_guess": t["muscle_guess"], "sd": t["sd"], "min": t["min"],
                          "max": t["max"], "n": t["n"], "n_raters": t["n_raters"],
                          "extension_ratio": t["extension_ratio"], "expert_loo_S_mean": t["expert_loo_S_mean"]}
        print(f"{cid}: written", flush=True)
    write_json(d_check / "calib_manifest.json", {
        "version": 1, "kind": "calib", "created": created, "phase_name": "Check", "progress_label": "check",
        "review_note": "Check image: the expert answers stay hidden. Measure it carefully, the same way on every image, then press N.",
        "scale_note": "px_per_cm is the scale of each image, applied to both axes.",
        "note": "Skills-check images. The answers are NOT in this file.",
        "n_images": len(images), "n_tasks": len(tasks), "images": images, "tasks": tasks})
    write_json(d_check / "manifest.json", {
        "version": 1, "kind": "check_shell", "created": created, "storage_key": "umud_manual_check_v2",
        "note": "Empty on purpose: the 10 check images are the calibration tasks in calib_manifest.json. serve.py needs this file.",
        "help_order": ["Check: 10 images, CHK_01 to CHK_10. The expert answers stay hidden.",
                       "Measure every image fully: 3 x MT, 3 x FL, 3 x PA. Then press N.",
                       *(["The images are mirrored left-right on purpose. This does not change MT, FL or PA."] if mirror else []),
                       "If an image is really impossible, press X to flag it. Flag only if you must.",
                       "When all 10 are done, the score is made with tools/annotator/score_check.py."],
        "n_images": 0, "n_tasks": 0, "images": [], "tasks": []})
    tol = sc.TOL
    # The answers never go into check/ (the folder the person being checked works in): private/ only.
    stale = d_check / "calib_answers.json"
    if stale.exists():
        stale.replace(d_priv / "calib_answers.json")
    write_json(d_priv / "calib_answers.json", {
        "version": 1, "created": created,
        "note": "PRIVATE. Expert answers for the check tasks (same format as data/annotator/calib_answers.json). Never "
                "in check/, never served. Mean of R1..R7 (missing / excluded entries skipped), PA vs the deep "
                "aponeurosis, FL / MT in mm. 'source' is the CHK id; the original names are in check_key.json.",
        "tolerances": tol, "items": answers})

    # ---- practice set
    p_images, p_tasks, p_key = [], [], {}
    for k, src in enumerate(practice, 1):
        pid = f"PRAC_{k:02d}"
        t = table[src]
        rgb = load_rgb(src)
        save_png(rgb, d_prac / "img" / f"{pid}.png")
        del rgb
        gc.collect()
        entry = {"image_id": pid, "stem": pid, "file": f"img/{pid}.png", "number": None, "clip": None,
                 "width": t["width"], "height": t["height"], "px_per_cm": t["px_per_cm"], "group": t["group"],
                 "viewport": t["viewport"], "scale_trusted": True, "kind": "static", "task_id": pid,
                 "hint": {c: round(t["expert"][c], 3) for c in COMPS},
                 "hint_range": {c: [round(t["min"][c], 2), round(t["max"][c], 2)] for c in COMPS},
                 "hint_n": dict(t["n"])}
        p_images.append(entry)
        task = {"task_id": pid, "image_id": pid, "clip": None, "members": [pid], "kind": "static"}
        if t["extension_ratio"] >= EXT_TIP:
            task["note"] = PRACTICE_NOTE
        p_tasks.append(task)
        p_key[pid] = {"source": src, "px_per_cm": t["px_per_cm"], "size": f"{t['width']}x{t['height']}",
                      "device_guess": t["device_guess"], "muscle_guess": t["muscle_guess"], "expert": t["expert"],
                      "sd": t["sd"], "min": t["min"], "max": t["max"], "n": t["n"], "raters": wb[src]["raters"],
                      "extension_ratio": t["extension_ratio"], "expert_loo_S_mean": t["expert_loo_S_mean"],
                      "tip_shown": "note" in task}
        print(f"{pid}: written", flush=True)
    groups: dict[str, int] = {}
    for e in p_images:
        groups[e["group"]] = groups.get(e["group"], 0) + 1
    write_json(d_prac / "manifest.json", {
        "version": 1, "kind": "practice", "created": created, "storage_key": "umud_manual_practice_v2",
        "phase_name": "Practice", "progress_label": "practice",
        "hint_is_answer": True, "hint_label": "Expert answer", "hint_label_short": "experts",
        "help_order": ["Practice: 25 public benchmark images, PRAC_01 to PRAC_25. The easier images come first.",
                       "Measure every image fully: 3 x MT, 3 x FL, 3 x PA.",
                       "When all 9 are done, the review card shows the expert answer (the mean of 6 or 7 experts) and their range.",
                       "Compare, and find what you did differently. Do not move your points to match the answer.",
                       "You may fix a clear mistake. Your progress is judged on your first answer: the tool keeps a copy of it."],
        "scale_note": "px_per_cm is the expert scale of each image, applied to both axes.",
        "source_note": "UMUD benchmark muscle architecture dataset v0.1.0, OSF xbawc, CC BY 4.0; renamed, PNG, metadata removed.",
        "n_images": len(p_images), "n_tasks": len(p_tasks), "groups": groups, "untrusted_scale": [],
        "images": p_images, "tasks": p_tasks})

    # ---- expert package
    (d_pkg / "images").mkdir(parents=True, exist_ok=True)
    for cid in src_of:
        shutil.copyfile(d_check / "calib" / f"{cid}.png", d_pkg / "images" / f"{cid}.png")
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["image_id", "px_per_cm", "px_per_mm", "width_px", "height_px"])
    for cid in sorted(src_of):
        t = table[src_of[cid]]
        w.writerow([cid, f"{t['px_per_cm']:g}", f"{t['px_per_cm'] / 10:g}", t["width"], t["height"]])
    (d_pkg / "scales.csv").write_text(buf.getvalue(), encoding="utf-8")
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["image_id", "MT1_mm", "MT2_mm", "MT3_mm", "FL1_mm", "FL2_mm", "FL3_mm", "PA1_deg", "PA2_deg", "PA3_deg",
                "minutes", "confidence_1to5", "flag", "notes"])
    for cid in sorted(src_of):
        w.writerow([cid] + [""] * 13)
    (d_pkg / "answer_template.csv").write_text(buf.getvalue(), encoding="utf-8")
    (d_pkg / "CREDITS.txt").write_text(CREDITS.format(mirror=CREDITS_MIRROR if mirror else "",
                                                      pixels="same pixel values" if mirror else "same pixels"),
                                       encoding="utf-8")

    # ---- keys
    excluded = [{"source": s, "rater": r, "comp": c, "value": wb[s]["raw"][f"{r}_{COL[c]}"], "reason": why}
                for s, r, c, why in EXCLUDED]
    check_key = {"version": 1, "kind": "umud_manual_check_key", "created": created,
                 "note": "PRIVATE: never send, never serve. CHK -> source image, every rater's value, mean, SD, range.",
                 "tolerances": tol, "raters": RATERS, "excluded_entries": excluded, "items": key_items}
    write_json(d_priv / "check_key.json", check_key)
    write_json(d_priv / "practice_key.json", {"version": 1, "kind": "umud_manual_practice_key", "created": created,
                                              "note": "PRAC -> source image and expert values (the hints are the same means).",
                                              "tolerances": tol, "raters": RATERS, "excluded_entries": excluded,
                                              "items": p_key})

    # ---- pass bar
    key = sc.load_json(d_priv / "check_key.json")
    bar_raters = sc.default_bar_raters(key)
    full_key = {"raters": RATERS, "items": {s: {"raters": wb[s]["raters"], "px_per_cm": wb[s]["px_per_cm"]} for s in sorted(wb)}}
    full_bar_raters = sc.default_bar_raters(full_key)
    models = {**model_values(), **tool_values(wb)}
    src_full = {s: s for s in wb}
    check_block = bar_block(key, RATERS, bar_raters, n_boot)
    full_block = bar_block(full_key, RATERS, full_bar_raters, n_boot)
    pass_bar = {
        "version": 1, "created": created,
        "rule": {"score": "S = mean over PA, FL, MT of MAE / tau (tau 6 deg, 12 mm, 3 mm); competition core without "
                          "tie-breakers; cells with a missing value are skipped",
                 "candidate": "graded against the mean of ALL raters on each image",
                 "experts": "each rater graded against the mean of the OTHER raters (leave-one-out) on the same cells",
                 "GOOD": "candidate S <= median expert LOO S", "PASS": "candidate S <= worst expert LOO S",
                 "FAIL": "otherwise",
                 "INCOMPLETE": f"fewer than {sc.MIN_COMPLETE_IMAGES} images with all three parameters (verdict is provisional)"},
        "bar_raters": bar_raters,
        "bar_raters_note": f"Raters with values on >= 80% of the check cells. R5 rated only images 1-17 "
                           f"({sum('R5' in it['raters'] for it in key['items'].values())} of the 10 check images) and "
                           "is listed but not used for the bar.",
        "reference_note": "A candidate is compared with a mean of 6-7 raters, an expert with a mean of 5-6: the "
                          "reference noise favours the candidate by roughly 1-2% of S. Grading a rater against the "
                          "all-rater mean (which contains them) would shrink each of their errors by (n-1)/n.",
        "sample_note": "Only 10 images: the bootstrap SD (resampling images) of a rater's S is given below; treat "
                       "candidates within about one SD of the bar as borderline.",
        "check_10": check_block, "all_35_reference": full_block, "full_bar_raters_35": full_bar_raters,
        "models_on_check": score_models(models, key, src_of),
        "models_on_all_35": score_models(models, full_key, src_full),
    }
    write_json(d_priv / "pass_bar.json", pass_bar)
    dups = near_duplicates(table, list(src_of.values()), practice)
    write_selection_md(d_priv / "selection.md", table, sel, pass_bar, excluded, dups, mirror)

    # ---- self-checks on the outputs
    for cid in src_of:
        a, b = d_check / "calib" / f"{cid}.png", d_pkg / "images" / f"{cid}.png"
        assert a.read_bytes() == b.read_bytes(), cid
    for p in list((d_check / "calib").glob("*.png")) + list((d_prac / "img").glob("*.png")):
        assert set(png_chunks(p)) <= {"IHDR", "IDAT", "IEND"}, (p, png_chunks(p))
    assert not (d_check / "calib_answers.json").exists()
    if dups and dups[0][0] > NEAR_DUP_NCC:
        print(f"WARNING: near-duplicate check / practice pair {dups[0][1]} ~ {dups[0][2]} (NCC {dups[0][0]:.2f})")
    print(f"done: {KIT}")


# ----------------------------------------------------------------------------- selection.md
def _fmt(x, d=1):
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{d}f}"


def write_selection_md(path: Path, table: dict, sel: dict, bar: dict, excluded: list, dups: list, mirror: bool) -> None:
    src_chk = {c["source"]: c["chk"] for c in sel["check"]}
    src_prac = {p["source"]: p["prac"] for p in sel["practice"]}
    L = ["# Check / practice selection (PRIVATE: never send or serve)", "",
         f"Built {bar['created']} by tools/annotator/prepare_manual_kit.py from private/selection.json.", "",
         "## Rules used, in order",
         "0. Blind: no image of the old calibration set (data/annotator/: its answers are on this computer and its "
         "sources are listed in the committed tools/annotator/README.md), and no near-duplicate of a practice image "
         f"(32x32 viewport thumbnail NCC > {NEAR_DUP_NCC}; near-duplicate groups stay on one side).",
         "1. Prefer Telemed images (ArtUs EXT-1H first, then Echo Blaster 128) and GM / VL. The test set is Telemed / "
         "Siemens / Lumify GM / VL / RF. Avoid images that may be soleus.",
         "2. A spread of difficulty, with at least 3 images whose fascicles are longer than the image is wide "
         "(extension ratio = FL x cos(PA) / field width >= 1).",
         "3. A usable key: no image with an extreme expert spread (FL rater range > 35 mm).",
         "4. More than one device.", "",
         "Difficulty = mean over raters of the rater's score against the mean of the other raters on that image "
         "(competition core, `expert LOO S`). Lower = the experts agree more.", "",
         "## Excluded rater entries", ""]
    for e in excluded:
        L.append(f"- {e['source']} {e['rater']} {COL[e['comp']]} = {e['value']}: {e['reason']}.")
    L += ["- R5 has no values for images 18-35 (6 raters there). Not an error, just missing.", "",
          "## The 10 check images (CHK order is random)", "",
          "| CHK | source | device | muscle | raters | MT mm (sd) | FL mm (sd, range) | PA deg (sd) | ext | expert LOO S | honest model | why |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in sel["check"]:
        t = table[c["source"]]
        L.append(f"| {c['chk']} | {c['source']} | {t['group']} | {t['muscle_guess'].split(' (')[0]} | {t['n_raters']} | "
                 f"{_fmt(t['expert']['mt_mm'])} ({_fmt(t['sd']['mt_mm'], 2)}) | {_fmt(t['expert']['fl_mm'])} "
                 f"({_fmt(t['sd']['fl_mm'])}, {_fmt(t['max']['fl_mm'] - t['min']['fl_mm'])}) | {_fmt(t['expert']['pa_deg'])} "
                 f"({_fmt(t['sd']['pa_deg'])}) | {_fmt(t['extension_ratio'], 2)} | {_fmt(t['expert_loo_S_mean'], 3)} | "
                 f"{'yes' if t['clean'] else 'no'} | {c['why']} |")
    ext = sum(table[c['source']]['extension_ratio'] >= 1.0 for c in sel["check"])
    devs = sorted({table[c['source']]['group'] for c in sel["check"]})
    if sel.get("not_chosen"):
        L += ["", "Telemed images NOT chosen for the check, and why:", ""]
        L += [f"- {s} ({src_prac.get(s, '?')}): {why}" for s, why in sel["not_chosen"].items()]
    L += ["", f"Check summary: {ext} images with extension ratio >= 1.0; devices {', '.join(devs)}; "
              f"expert LOO S from {_fmt(min(table[c['source']]['expert_loo_S_mean'] for c in sel['check']), 3)} to "
              f"{_fmt(max(table[c['source']]['expert_loo_S_mean'] for c in sel['check']), 3)}.", "",
          "## Blindness and honesty", "",
          f"- Check images vs practice images: the most similar pairs (thumbnail NCC, either orientation): "
          + ", ".join(f"{c} ~ {q} {n:.2f}" for n, c, q in dups[:5]) + f". Limit {NEAR_DUP_NCC}.",
          f"- The check images are {'mirrored left-right (MT, FL and PA do not change; ' if mirror else 'NOT mirrored ('}"
          "listed under 'Changes made by us' in CREDITS.txt). So a file or pixel match with the public TIFF fails.",
          "- The user's own check is blind as long as private/ is not opened and score_check.py is run only once, "
          "when at least 8 images are complete (it shows no expert value before that, and never for cells not measured).",
          "- For an OUTSIDE expert the check is honour-based. CC BY needs the source to be named, the OSF zip holds every "
          "rater's values, and a determined person can match the images by eye. score_check.py flags values that match "
          "a published value (copy check) and recomputes the values from the returned ROI files (--rois). Treat an "
          "outside expert's GOOD / PASS as evidence, not proof: also look at the ROI files and the time per image.",
          "", "## Pass bar (private/pass_bar.json)", ""]
    cb = bar["check_10"]
    L.append(f"Bar raters: {', '.join(bar['bar_raters'])}. Leave-one-out S on the 10 check images:")
    L.append("")
    L.append("| rater | S | MT MAE/tau | FL MAE/tau | PA MAE/tau | images | bootstrap SD |")
    L.append("|---|---|---|---|---|---|---|")
    for r, v in cb["per_rater_loo"].items():
        sd = cb["bootstrap"].get(r, {}).get("sd")
        L.append(f"| {r}{'' if r in bar['bar_raters'] else ' (not in bar)'} | {_fmt(v['S'], 3)} | "
                 f"{_fmt(v['components']['mt_mm']['mae_tau'], 3)} | {_fmt(v['components']['fl_mm']['mae_tau'], 3)} | "
                 f"{_fmt(v['components']['pa_deg']['mae_tau'], 3)} | {v['n_images']} | {_fmt(sd, 3)} |")
    b = cb["bar"]
    L += ["", f"best {_fmt(b['best'], 3)} ({b['best_rater']}), median {_fmt(b['median'], 3)}, worst {_fmt(b['worst'], 3)} "
              f"({b['worst_rater']}); bootstrap SD of the median {_fmt(cb['bootstrap']['_median']['sd'], 3)}, of the worst "
              f"{_fmt(cb['bootstrap']['_worst']['sd'], 3)}.",
          f"All 35 images (reference): median {_fmt(bar['all_35_reference']['bar']['median'], 3)}, worst "
          f"{_fmt(bar['all_35_reference']['bar']['worst'], 3)} (raters {', '.join(bar['full_bar_raters_35'])}).", "",
          "Models / tools on the check images (vs the mean of all raters):", ""]
    for name, m in bar["models_on_check"].items():
        L.append(f"- {name}: S {_fmt(m['S'], 3)} on {m['n_images']} images; on honest images {_fmt(m['S_honest_images'], 3)} "
                 f"({m['n_honest_images']} images). {m['label']}")
    L += ["", "## All 35 images", "",
          "| source | role | device | muscle guess | size | px/cm | raters | MT | FL (range) | PA | ext | expert LOO S | KG0 S | clean |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for src in sorted(table):
        t = table[src]
        role = src_chk.get(src) or src_prac.get(src, "?")
        L.append(f"| {src} | {role} | {t['group']} | {t['muscle_guess']} | {t['width']}x{t['height']} | {t['px_per_cm']:g} | "
                 f"{t['n_raters']} | {_fmt(t['expert']['mt_mm'])} | {_fmt(t['expert']['fl_mm'])} "
                 f"({_fmt(t['max']['fl_mm'] - t['min']['fl_mm'])}) | {_fmt(t['expert']['pa_deg'])} | "
                 f"{_fmt(t['extension_ratio'], 2)} | {_fmt(t['expert_loo_S_mean'], 3)} | {_fmt(t.get('kg0_S', math.nan), 3)} | "
                 f"{'yes' if t['clean'] else 'no'} |")
    L += ["", "Device and muscle are guesses from the image size, the scale and the on-screen UI text; the OSF files carry "
              "no device or muscle field. 'clean' = not in the competition training data (honest model comparison).",
          "Practice order = expert LOO S minus 0.03 for Telemed images (easy, clear, test-like images first)."]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--init-selection", type=Path, default=None,
                        help='JSON {"check": {"im_XX_arch": "why", ...}} with the 10 check images')
    parser.add_argument("--force-selection", action="store_true")
    parser.add_argument("--allow-old-calib", action="store_true", help="allow old calibration images in the check")
    parser.add_argument("--allow-near-duplicates", action="store_true",
                        help="allow a check image that looks like a practice image")
    parser.add_argument("--no-mirror", action="store_true", help="do not mirror the check images left-right")
    parser.add_argument("--n-boot", type=int, default=sc.N_BOOT)
    args = parser.parse_args()
    t0 = time.time()
    wb = read_workbook()
    cache_path = KIT / "private" / "build_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    print("per-image table (viewports are cached in private/build_cache.json):", flush=True)
    table = build_table(wb, cache)
    write_json(cache_path, cache)
    if KG0_STATIC.exists():
        import pandas as pd

        kg = pd.read_csv(KG0_STATIC, index_col=0)
        for src, t in table.items():
            if src in kg.index:
                t["kg0_S"] = statistics.mean(abs(float(kg.loc[src, c]) - t["expert"][c]) / sc.TOL[c] for c in COMPS)
    if args.init_selection:
        init_selection(args.init_selection, table, args.force_selection, args.allow_old_calib, args.allow_near_duplicates)
    sel_path = KIT / "private" / "selection.json"
    if not sel_path.exists():
        raise SystemExit(f"{sel_path} missing: run with --init-selection first")
    build(table, wb, json.loads(sel_path.read_text(encoding="utf-8")), args.n_boot, mirror=not args.no_mirror)
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
