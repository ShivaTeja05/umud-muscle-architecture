"""Write the model's suggested lines ("prefill") into the work-set manifest, with blind control tasks.

    .venv/bin/python tools/annotator/apply_prefill.py \
        --suggestions artifacts/labelling_suggest_out/suggestions_v1.json            # writes (after a backup)
    .venv/bin/python tools/annotator/apply_prefill.py --suggestions ... --dry-run  # report only, writes nothing
    .venv/bin/python tools/annotator/apply_prefill.py --remove                     # strip every prefill key again

suggestions_v1.json is the output of the Kaggle kernel shivateja005/umud-labelling-suggest
(kaggle/labelling_suggest/). Keep it OUTSIDE data/manual_labelling/work/ (that folder is served to the browser and the
file also holds the suggestions of the blind tasks).

What a run does (idempotent: a second run with the same inputs changes no byte and rewrites no file):
 1. Backs up work/manifest.json to work/manifest.before_prefill.json, once. An existing backup is never overwritten.
 2. Blind control tasks: 3 per participant (24 of 96), one from each tercile of that participant's selection rank
    (ranks 0-3 / 4-7 / 8-11 from private/work_selection.json, else the queue order), drawn with a fixed seed and
    balanced over the ranks, so every block of 8 consecutive queue tasks holds exactly 2 blind tasks. An existing
    private/prefill_blind.json is reused: the blind set never
    changes once it exists. --reselect-blind redraws it, and is refused when work/autosave.json already holds labels
    (unless --force).
 3. Every main image (never a view-only *_n1 frame) of a non-blind task with a valid suggestion (and confidence >=
    --min-confidence) gets "prefill". A blind task gets "prefill_blind": true on the TASK and no prefill on its image;
    its suggestion is kept only in private/prefill_blind.json (outside every served folder), for anchoring_report.py.
 4. Sets prefill_enabled / prefill_source / prefill_blind_rule at the top of the manifest, validates the result and
    writes it atomically with the layout prepare_work.py uses (json indent 1).

Shared prefill schema (also used by app.js; native image pixels, x right, y down; the label point formats):
    "prefill": {"mt": [[[x, y], [x, y]], x3],            superficial inner edge, then deep inner edge; same x
                "fl": [[[x, y], ..., [x, y]], x3],       polyline deep -> superficial, points may lie outside
                "pa": [[[fx0, fy0], [fx1, fy1], [ax0, ay0], [ax1, ay1]], x3],   2 fascicle + 2 deep-apo points
                "apo_sup": [[x, y], ...], "apo_deep": [[x, y], ...],          inner-edge polylines, x increasing
                "confidence": 0..1}
Practice and check manifests never get prefill; the page ignores it unless manifest.prefill_enabled is true.
Stdlib only.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KIT = ROOT / "data" / "manual_labelling"
DEFAULT_WORK = KIT / "work"
DEFAULT_PRIVATE = KIT / "private"
DEFAULT_SUGGESTIONS = ROOT / "artifacts" / "labelling_suggest_out" / "suggestions_v1.json"
BACKUP_NAME = "manifest.before_prefill.json"
BLIND_NAME = "prefill_blind.json"
SEED = 20261102
PER_PARTICIPANT = 3
PREFILL_KEYS = ("mt", "fl", "pa", "apo_sup", "apo_deep", "confidence")
TOP_KEYS = ("prefill_enabled", "prefill_source", "prefill_blind_rule")
N_POINTS = {"mt": 2, "pa": 4}
EPS_VERTICAL = 0.011       # MT x of both points equal after rounding to 0.01 px


# ------------------------------------------------------------------------------------------------ validation
def _is_point(p) -> bool:
    return (isinstance(p, (list, tuple)) and len(p) == 2 and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                                                 and math.isfinite(v) for v in p))


def validate_prefill(p, width: float | None = None, height: float | None = None) -> list[str]:
    """Errors of one image's prefill against the shared schema (empty list = valid)."""
    if not isinstance(p, dict):
        return ["prefill is not an object"]
    errs = []
    missing = [k for k in PREFILL_KEYS if k not in p]
    extra = [k for k in p if k not in PREFILL_KEYS]
    if missing:
        errs.append(f"missing keys {missing}")
    if extra:
        errs.append(f"unexpected keys {extra}")
    if missing:
        return errs
    lim_x = (-5.0 * width, 6.0 * width) if width else (-1e5, 1e5)
    lim_y = (-5.0 * height, 6.0 * height) if height else (-1e5, 1e5)

    def pts_ok(name, seq, n_min=2, n_exact=None):
        if not isinstance(seq, list) or (n_exact and len(seq) != n_exact) or len(seq) < n_min:
            errs.append(f"{name}: needs {'exactly ' + str(n_exact) if n_exact else 'at least ' + str(n_min)} points")
            return False
        for q in seq:
            if not _is_point(q):
                errs.append(f"{name}: bad point {q!r}")
                return False
            if not (lim_x[0] <= q[0] <= lim_x[1] and lim_y[0] <= q[1] <= lim_y[1]):
                errs.append(f"{name}: point {q!r} far outside the image")
                return False
        return True

    for tool in ("mt", "fl", "pa"):
        items = p[tool]
        if not isinstance(items, list) or len(items) != 3:
            errs.append(f"{tool}: needs exactly 3 measurements")
            continue
        for k, m in enumerate(items):
            name = f"{tool}[{k}]"
            if not pts_ok(name, m, n_exact=N_POINTS.get(tool)):
                continue
            if tool == "mt":
                if abs(m[0][0] - m[1][0]) > EPS_VERTICAL:
                    errs.append(f"{name}: not vertical (x {m[0][0]} vs {m[1][0]})")
                if not m[1][1] > m[0][1]:
                    errs.append(f"{name}: the deep point must lie below the superficial point")
            elif tool == "fl":
                if sum(math.dist(m[i - 1], m[i]) for i in range(1, len(m))) <= 1.0:
                    errs.append(f"{name}: zero length")
                if not m[-1][1] < m[0][1]:
                    errs.append(f"{name}: must run from the deep (lower) to the superficial (upper) aponeurosis")
            else:
                if math.dist(m[0], m[1]) <= 0.5 or math.dist(m[2], m[3]) <= 0.5:
                    errs.append(f"{name}: a zero-length PA arm")
    for key in ("apo_sup", "apo_deep"):
        if pts_ok(key, p[key]):
            xs = [q[0] for q in p[key]]
            if any(b <= a for a, b in zip(xs, xs[1:])):
                errs.append(f"{key}: x must increase")
    c = p["confidence"]
    if not (isinstance(c, (int, float)) and not isinstance(c, bool) and math.isfinite(c) and 0.0 <= c <= 1.0):
        errs.append(f"confidence {c!r} not in [0, 1]")
    return errs


def normalize_prefill(p: dict) -> dict:
    """Schema key order, points rounded to 0.01 px (the annotator's precision), confidence to 0.001."""
    r = lambda q: [round(float(q[0]), 2), round(float(q[1]), 2)]  # noqa: E731
    return {"mt": [[r(q) for q in m] for m in p["mt"]], "fl": [[r(q) for q in m] for m in p["fl"]],
            "pa": [[r(q) for q in m] for m in p["pa"]], "apo_sup": [r(q) for q in p["apo_sup"]],
            "apo_deep": [r(q) for q in p["apo_deep"]], "confidence": round(float(p["confidence"]), 3)}


def strip_prefill(manifest: dict) -> dict:
    m = copy.deepcopy(manifest)
    for k in TOP_KEYS:
        m.pop(k, None)
    for e in m.get("images", []):
        e.pop("prefill", None)
    for t in m.get("tasks", []):
        t.pop("prefill_blind", None)
    return m


def main_images(manifest: dict) -> dict:
    return {e["image_id"]: e for e in manifest["images"] if not e.get("view_only")}


def validate_manifest(m: dict, blind_ids: set[str]) -> list[str]:
    errs = []
    if m.get("prefill_enabled") is not True:
        errs.append("prefill_enabled is not true")
    for k in ("prefill_source", "prefill_blind_rule"):
        if not isinstance(m.get(k), str) or not m.get(k):
            errs.append(f"{k} missing")
    images = {e["image_id"]: e for e in m["images"]}
    for e in m["images"]:
        if "prefill" in e:
            if e.get("view_only"):
                errs.append(f"{e['image_id']}: view-only frame has prefill")
            errs += [f"{e['image_id']}: {x}" for x in validate_prefill(e["prefill"], e.get("width"), e.get("height"))]
    for t in m["tasks"]:
        blind = t.get("prefill_blind") is True
        if blind != (t["task_id"] in blind_ids):
            errs.append(f"{t['task_id']}: prefill_blind flag does not match the blind list")
        if blind:
            for iid in {t["image_id"], *t.get("members", []), *t.get("frames", [])}:
                if "prefill" in images.get(iid, {}):
                    errs.append(f"{t['task_id']}: blind task but {iid} has prefill")
    return errs


# ------------------------------------------------------------------------------------------------ blind set
def task_ranks(manifest: dict, selection: Path | None) -> tuple[dict, str]:
    """task_id -> selection rank within its participant (work_selection.json), else the queue order."""
    if selection and selection.exists():
        sel = json.loads(selection.read_text(encoding="utf-8"))
        ranks = {t["task_id"]: int(t["rank"]) for t in sel.get("tasks", []) if "rank" in t}
        if all(t["task_id"] in ranks for t in manifest["tasks"]):
            return ranks, f"selection rank ({selection.name})"
    imgs = {e["image_id"]: e for e in manifest["images"]}
    count: dict[str, int] = {}
    ranks = {}
    for t in manifest["tasks"]:
        p = imgs[t["image_id"]].get("participant") or "?"
        ranks[t["task_id"]] = count.get(p, 0)
        count[p] = count.get(p, 0) + 1
    return ranks, "queue order within the participant"


def select_blind(manifest: dict, ranks: dict, seed: int = SEED, per: int = PER_PARTICIPANT) -> list[dict]:
    """`per` tasks per participant, one from each of `per` equal strata of the rank (deterministic).

    Within a stratum the participants' picks are balanced over the positions (a seeded shuffle of 0, 1, .., L-1
    repeated), so with 8 participants x 12 ranks every rank, i.e. every block of 8 consecutive queue tasks, holds
    exactly 2 blind tasks."""
    imgs = {e["image_id"]: e for e in manifest["images"]}
    by_p: dict[str, list[dict]] = {}
    for t in manifest["tasks"]:
        by_p.setdefault(imgs[t["image_id"]].get("participant") or "?", []).append(t)
    parts = sorted(by_p)
    strata = {}
    for p in parts:
        tasks = sorted(by_p[p], key=lambda t: (ranks[t["task_id"]], t["task_id"]))
        n = len(tasks)
        if n < per:
            raise ValueError(f"participant {p} has {n} tasks < {per}")
        strata[p] = [tasks[s * n // per:(s + 1) * n // per] for s in range(per)]
    out = []
    for s in range(per):
        size = min(len(strata[p][s]) for p in parts)
        offsets = [i % size for i in range(len(parts))]
        random.Random(f"umud-prefill-blind:{seed}:stratum{s}").shuffle(offsets)
        for p, o in zip(parts, offsets):
            stratum = strata[p][s]
            pick = stratum[o % len(stratum)]
            out.append({"task_id": pick["task_id"], "image_id": pick["image_id"], "participant": p,
                        "rank": ranks[pick["task_id"]], "stratum": s,
                        "stratum_ranks": [ranks[stratum[0]["task_id"]], ranks[stratum[-1]["task_id"]]]})
    return sorted(out, key=lambda b: b["task_id"])


def check_blind(blind: list[dict], manifest: dict, per: int = PER_PARTICIPANT) -> list[str]:
    errs = []
    tids = {t["task_id"]: t for t in manifest["tasks"]}
    imgs = {e["image_id"]: e for e in manifest["images"]}
    counts: dict[str, int] = {}
    for b in blind:
        t = tids.get(b["task_id"])
        if t is None:
            errs.append(f"blind task {b['task_id']} not in the manifest")
            continue
        counts[imgs[t["image_id"]].get("participant") or "?"] = counts.get(imgs[t["image_id"]].get("participant") or "?", 0) + 1
    parts = {imgs[t["image_id"]].get("participant") or "?" for t in manifest["tasks"]}
    for p in sorted(parts):
        if counts.get(p, 0) != per:
            errs.append(f"participant {p}: {counts.get(p, 0)} blind tasks, expected {per}")
    if len({b["task_id"] for b in blind}) != len(blind):
        errs.append("duplicate blind tasks")
    return errs


# ------------------------------------------------------------------------------------------------ apply
def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(manifest: dict, sugg: dict, sugg_sha: str, blind: list[dict], min_conf: float) -> tuple[dict, dict, dict]:
    """(new manifest, blind suggestions {image_id: prefill}, report). Pure: the inputs are not modified."""
    base = strip_prefill(manifest)
    mains = main_images(base)
    blind_tids = {b["task_id"] for b in blind}
    blind_iids = set()
    for t in base["tasks"]:
        if t["task_id"] in blind_tids:
            t["prefill_blind"] = True
            blind_iids |= {t["image_id"], *t.get("members", []), *t.get("frames", [])}
    task_of = {t["image_id"]: t for t in base["tasks"]}
    simg = sugg.get("images") or {}
    report = {"prefilled": [], "blind": sorted(blind_tids), "skipped": {}, "unknown": sorted(set(simg) - set(mains)),
              "confidence": []}
    blind_sugg = {}
    for iid, e in mains.items():
        s = simg.get(iid)
        reason = None
        if iid not in task_of:
            reason = "not the measured frame of a task"
        elif not s:
            reason = "no suggestion"
        elif not s.get("ok"):
            reason = f"kernel: {s.get('reason', 'failed')}"
        else:
            errs = validate_prefill(s.get("prefill"), e.get("width"), e.get("height"))
            if errs:
                reason = "invalid: " + "; ".join(errs[:3])
        if iid in blind_iids:
            if reason is None:
                blind_sugg[iid] = normalize_prefill(s["prefill"])
            else:
                report["skipped"][iid] = f"(blind) {reason}"
            continue
        if reason is None and float(s["prefill"]["confidence"]) < min_conf:
            reason = f"confidence {s['prefill']['confidence']} < {min_conf}"
        if reason:
            report["skipped"][iid] = reason
            continue
        e["prefill"] = normalize_prefill(s["prefill"])
        report["prefilled"].append(iid)
        report["confidence"].append(e["prefill"]["confidence"])
    created = str(sugg.get("created") or "")[:19]
    top = {
        "prefill_enabled": True,
        "prefill_source": (f"{sugg.get('source_short') or 'model suggestions'}; kernel {sugg.get('kernel') or '?'} "
                           f"{created}; suggestions sha256 {sugg_sha[:12]}"),
        "prefill_blind_rule": (f"{len(blind)} of {len(base['tasks'])} tasks ({PER_PARTICIPANT} per participant, one from "
                               f"each third of the selection order, fixed seed) are blind controls: they never show "
                               f"suggestions. Their labels measure how much the suggestions pull the values."),
    }
    out = {}
    for k, v in base.items():
        if k == "images":
            out.update(top)
        out[k] = v
    if "images" not in base:
        out.update(top)
    return out, blind_sugg, report


def dump_manifest(m: dict) -> str:
    return json.dumps(m, indent=1)          # prepare_work.py / prepare_manual_kit.py layout, no trailing newline


def atomic_write(path: Path, text: str) -> bool:
    """Write only when the content changes. Returns True when the file was (re)written."""
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return True


def autosave_label_count(work: Path) -> int:
    p = work / "autosave.json"
    if not p.exists():
        return 0
    try:
        labels = json.loads(p.read_text(encoding="utf-8")).get("labels") or {}
    except (OSError, ValueError):
        return -1
    return sum(1 for lab in labels.values() if any(lab.get(t) for t in ("mt", "fl", "pa")) or lab.get("flag"))


def run(args) -> int:
    work, private = Path(args.work), Path(args.private)
    man_path = work / "manifest.json"
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "work" and not args.any_kind:
        print(f"refused: {man_path} has kind {manifest.get('kind')!r}, not 'work' (practice / check never get prefill)")
        return 2
    blind_path = private / BLIND_NAME
    backup = work / BACKUP_NAME
    if args.remove:
        text = dump_manifest(strip_prefill(manifest))
        if args.dry_run:
            print(f"dry run: would strip prefill from {man_path} ({'no change' if text == man_path.read_text(encoding='utf-8') else 'changes'})")
            return 0
        if not backup.exists():
            shutil.copy2(man_path, backup)
        print(f"{'rewrote' if atomic_write(man_path, text) else 'unchanged'} {man_path} (prefill removed; blind file kept)")
        return 0

    sugg_path = Path(args.suggestions)
    sugg = json.loads(sugg_path.read_text(encoding="utf-8"))
    if sugg.get("version") != 1 or not isinstance(sugg.get("images"), dict):
        print(f"refused: {sugg_path} is not a suggestions_v1 file")
        return 2
    sugg_sha = sha256_file(sugg_path)
    n_labels = autosave_label_count(work)
    old_blind = json.loads(blind_path.read_text(encoding="utf-8")) if blind_path.exists() else None
    if old_blind and not args.reselect_blind:
        blind = old_blind["tasks"]
        seed_used, how = old_blind.get("seed"), old_blind.get("selection", "")
        print(f"blind set: reused {blind_path}")
    else:
        if old_blind and n_labels and not args.force:
            print(f"refused: --reselect-blind while {work / 'autosave.json'} has {n_labels} labelled tasks (use --force)")
            return 2
        ranks, rank_src = task_ranks(manifest, private / "work_selection.json")
        blind = select_blind(manifest, ranks, args.seed)
        seed_used, how = args.seed, f"drawn with seed {args.seed}, one per tercile of the {rank_src}"
        old_blind = None if args.reselect_blind else old_blind
    errs = check_blind(blind, manifest)
    if errs:
        print("refused: blind set invalid:\n  " + "\n  ".join(errs))
        return 2
    new, blind_sugg, rep = build(manifest, sugg, sugg_sha, blind, args.min_confidence)
    errs = validate_manifest(new, {b["task_id"] for b in blind})
    if errs:
        print("refused: the new manifest does not validate:\n  " + "\n  ".join(errs[:20]))
        return 2
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    blind_doc = {
        "version": 1,
        "note": "PRIVATE: never serve. Blind control tasks of the work set: they never show suggestions. "
                "suggestions = the model lines they WOULD have shown (for tools/annotator/anchoring_report.py).",
        "created": (old_blind or {}).get("created") or now,
        "applied": (old_blind or {}).get("applied") or now,   # first time the prefill went live
        "seed": seed_used,
        "rule": new["prefill_blind_rule"],
        "selection": how,
        "suggestions_file": sugg_path.name,
        "suggestions_sha256": sugg_sha,
        "tasks": blind,
        "suggestions": dict(sorted(blind_sugg.items())),
    }
    conf = sorted(rep["confidence"])
    q = (lambda f: conf[min(len(conf) - 1, int(f * len(conf)))] if conf else float("nan"))  # noqa: E731
    print(f"suggestions: {sugg_path} (sha256 {sugg_sha[:12]}, {sugg.get('n_ok')}/{sugg.get('n_images')} ok)")
    print(f"blind: {len(blind)} tasks ({how})")
    by_p: dict[str, list[str]] = {}
    for b in blind:
        by_p.setdefault(b["participant"], []).append(f"{b['task_id']}(r{b['rank']})")
    for p in sorted(by_p):
        print(f"  {p}: {' '.join(by_p[p])}")
    print(f"prefilled: {len(rep['prefilled'])} images; confidence min/q25/median/q75/max "
          f"{q(0):.2f} / {q(0.25):.2f} / {q(0.5):.2f} / {q(0.75):.2f} / {q(0.999):.2f}" if conf else "prefilled: 0 images")
    for iid, why in sorted(rep["skipped"].items()):
        print(f"  skipped {iid}: {why}")
    if rep["unknown"]:
        print(f"  ignored {len(rep['unknown'])} suggestions for unknown images: {rep['unknown'][:5]}")
    if n_labels:
        print(f"note: {work / 'autosave.json'} already has {n_labels} labelled tasks; anchoring_report.py excludes "
              f"tasks first completed before the prefill was applied")
    text = dump_manifest(new)
    if args.dry_run:
        same = text == man_path.read_text(encoding="utf-8")
        print(f"dry run: nothing written ({man_path.name} would {'not change' if same else 'change'}; backup "
              f"{'exists' if backup.exists() else 'would be created'})")
        return 0
    if not backup.exists():
        shutil.copy2(man_path, backup)
        print(f"backup: {backup}")
    private.mkdir(parents=True, exist_ok=True)
    wrote_b = atomic_write(blind_path, json.dumps(blind_doc, indent=1))
    wrote_m = atomic_write(man_path, text)
    print(f"{'wrote' if wrote_m else 'unchanged'} {man_path}; {'wrote' if wrote_b else 'unchanged'} {blind_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--suggestions", default=str(DEFAULT_SUGGESTIONS))
    ap.add_argument("--work", default=str(DEFAULT_WORK), help="the work data dir (manifest.json)")
    ap.add_argument("--private", default=str(DEFAULT_PRIVATE), help="where prefill_blind.json lives")
    ap.add_argument("--min-confidence", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--remove", action="store_true", help="strip every prefill key from the manifest")
    ap.add_argument("--reselect-blind", action="store_true", help="redraw the blind set (refused once labels exist)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--any-kind", action="store_true", help=argparse.SUPPRESS)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
