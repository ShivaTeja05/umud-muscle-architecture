"""Anchoring check for the pre-drawn suggestion lines of the work set.

    .venv/bin/python tools/annotator/anchoring_report.py                    # default kit paths
    .venv/bin/python tools/annotator/anchoring_report.py --use first --json /tmp/anchoring.json

Question: do your labels sit closer to the model where you SAW its lines ("shown" tasks) than where you did not
("blind" control tasks, private/prefill_blind.json)? Per parameter (MT mm, FL mm, PA deg) and task:
    d = |your value - the model's suggested value|, each the mean of 3 measurements (labels.py maths)
For each group: n, mean |d| (MAD), median |d|, mean signed difference (you - model) and MAD / tolerance
(MT 3 mm, FL 12 mm, PA 6 deg). Then shown - blind with a bootstrap 95 % CI (tasks resampled with replacement
within each group) and the anchoring index 1 - MAD_shown / MAD_blind.
Also the "kept" rate: the share of your measurements that are a suggested line left (almost) unchanged (every
point within --same-px of the corresponding point of one suggested measurement; FL compares the two ends).

Reading it: the blind group is the honest you-vs-model disagreement. A clearly smaller MAD on shown tasks (CI below
0) means the lines pulled your values toward the model - or that they fixed mistakes; without expert answers the two
cannot be told apart, so compare with the practice / check scores too. Small samples: 24 blind tasks give wide CIs.

Only complete parameters (3 measurements) of unflagged tasks count. Tasks first completed before the prefill was
applied (prefill_blind.json "applied") are excluded, they were measured without any suggestion; so are shown tasks
whose label says the ghosts were never displayed (app.js label.assist.prefill_shown === false). The shown group is
also split by label.assist.prefill_accepted (lines copied with Y vs own lines drawn next to the ghosts). Labels
without an `assist` field (older app.js) count as shown. Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import labels as L  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
KIT = ROOT / "data" / "manual_labelling"
TAU = {"mt": 3.0, "fl": 12.0, "pa": 6.0}
UNIT = {"mt": "mm", "fl": "mm", "pa": "deg"}
PARAMS = ("mt", "fl", "pa")


def _iso_ms(text: str | None) -> float | None:
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp() * 1000.0
    except ValueError:
        return None


def _mean(v: list[float]) -> float:
    return sum(v) / len(v) if v else math.nan


def _median(v: list[float]) -> float:
    s = sorted(v)
    n = len(s)
    if not n:
        return math.nan
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _pct(v: list[float], q: float) -> float:
    s = sorted(x for x in v if math.isfinite(x))
    if not s:
        return math.nan
    k = (len(s) - 1) * q
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def kept(user_m: list, sugg_ms: list, tool: str, tol: float) -> bool:
    """True when the user's measurement is one of the suggested ones moved by <= tol px at every point."""
    for s in sugg_ms:
        if tool == "fl":
            a, b = [user_m[0], user_m[-1]], [s[0], s[-1]]
        else:
            if len(user_m) != len(s):
                continue
            a, b = user_m, s
        if all(math.dist(p, q) <= tol for p, q in zip(a, b)):
            return True
    return False


def collect(manifest: dict, state: dict, blind_doc: dict | None, use: str, same_px: float) -> tuple[list[dict], dict]:
    images = {e["image_id"]: e for e in manifest["images"]}
    blind_tasks = {b["task_id"] for b in (blind_doc or {}).get("tasks", [])}
    blind_sugg = (blind_doc or {}).get("suggestions", {})
    applied = _iso_ms((blind_doc or {}).get("applied"))
    labels = state.get("labels") or {}
    rows, skipped = [], {}
    for t in manifest["tasks"]:
        tid, iid = t["task_id"], t["image_id"]
        e = images[iid]
        blind = t.get("prefill_blind") is True or tid in blind_tasks
        sugg = blind_sugg.get(iid) if blind else e.get("prefill")
        lab = labels.get(tid)
        why = None
        if sugg is None:
            why = "no suggestion"
        elif not lab:
            why = "not labelled"
        elif lab.get("flag"):
            why = "flagged"
        else:
            first = lab.get("first")
            assist = lab.get("assist") if isinstance(lab.get("assist"), dict) else {}
            if applied and first and isinstance(first.get("t"), (int, float)) and first["t"] < applied:
                why = "completed before the prefill was applied"
            elif not blind and assist.get("prefill_shown") is False:
                why = "suggestions not shown (own lines drawn first)"   # app.js assist field; absent = shown
            elif use == "first":
                lab = first
                if not lab:
                    why = "no complete first attempt"
        if why:
            skipped[why] = skipped.get(why, 0) + 1
            continue
        ppc = float(e["px_per_cm"])
        su, sm = L.summarize(lab, ppc), L.summarize(sugg, ppc)
        row = {"task_id": tid, "image_id": iid, "group": "blind" if blind else "shown",
               "participant": e.get("participant"),
               "copied": bool(not blind and assist.get("prefill_accepted") is True)}
        for p in PARAMS:
            n = su[f"n_{p}"]
            if n >= 3 and math.isfinite(su[p]) and math.isfinite(sm[p]):
                row[p] = {"user": su[p], "model": sm[p], "d": su[p] - sm[p],
                          "kept": sum(kept(m, sugg[p], p, same_px) for m in (lab.get(p) or [])), "n": n}
        rows.append(row)
    return rows, skipped


def boot_diff(a: list[float], b: list[float], n_boot: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    out = []
    for _ in range(n_boot):
        ra = [a[rng.randrange(len(a))] for _ in a]
        rb = [b[rng.randrange(len(b))] for _ in b]
        out.append(_mean(ra) - _mean(rb))
    return _pct(out, 0.025), _pct(out, 0.975)


def report(rows: list[dict], n_boot: int = 4000, seed: int = 0) -> dict:
    res = {}
    for p in PARAMS:
        g = {k: [r[p] for r in rows if r["group"] == k and p in r] for k in ("shown", "blind")}
        stats = {}
        for k, v in g.items():
            ad = [abs(x["d"]) for x in v]
            stats[k] = {"n": len(v), "mad": _mean(ad), "median_ad": _median(ad), "bias": _mean([x["d"] for x in v]),
                        "mad_over_tau": _mean(ad) / TAU[p] if ad else math.nan,
                        "kept_rate": (sum(x["kept"] for x in v) / max(1, sum(x["n"] for x in v))) if v else math.nan}
        out = {"unit": UNIT[p], "tau": TAU[p], **stats}
        for k, want in (("shown_copied", True), ("shown_own", False)):   # Y pressed (lines copied) or not
            ad = [abs(r[p]["d"]) for r in rows if r["group"] == "shown" and p in r and r.get("copied") is want]
            out[k] = {"n": len(ad), "mad": _mean(ad), "median_ad": _median(ad)}
        a, b = [abs(x["d"]) for x in g["shown"]], [abs(x["d"]) for x in g["blind"]]
        if len(a) >= 2 and len(b) >= 2:
            lo, hi = boot_diff(a, b, n_boot, seed + PARAMS.index(p))
            out["diff_shown_minus_blind"] = _mean(a) - _mean(b)
            out["diff_ci95"] = [lo, hi]
            out["anchoring_index"] = 1.0 - _mean(a) / _mean(b) if _mean(b) > 0 else math.nan
            out["verdict"] = ("closer to the model where shown (CI below 0)" if hi < 0 else
                              "farther from the model where shown (CI above 0)" if lo > 0 else "no clear difference")
        else:
            out["diff_shown_minus_blind"], out["diff_ci95"], out["anchoring_index"] = math.nan, [math.nan] * 2, math.nan
            out["verdict"] = "too few tasks (need >= 2 per group)"
        res[p] = out
    return res


def fmt(v: float, nd: int = 2) -> str:
    return "  n/a" if v is None or not math.isfinite(v) else f"{v:.{nd}f}"


def print_report(res: dict, rows: list[dict], skipped: dict, use: str) -> None:
    ns = {k: sum(r["group"] == k for r in rows) for k in ("shown", "blind")}
    print(f"anchoring report ({use} labels): {ns['shown']} shown tasks, {ns['blind']} blind tasks analysed")
    if skipped:
        print("  not analysed: " + ", ".join(f"{v} {k}" for k, v in sorted(skipped.items())))
    print(f"{'param':5} {'group':6} {'n':>3} {'MAD':>7} {'median':>7} {'bias':>7} {'MAD/tau':>7} {'kept':>6}")
    for p, r in res.items():
        for k in ("shown", "blind"):
            s = r[k]
            print(f"{p.upper():5} {k:6} {s['n']:3d} {fmt(s['mad']):>7} {fmt(s['median_ad']):>7} {fmt(s['bias']):>7} "
                  f"{fmt(s['mad_over_tau'], 3):>7} {fmt(100 * s['kept_rate'], 0):>5}%")
        lo, hi = r["diff_ci95"]
        print(f"      shown - blind = {fmt(r['diff_shown_minus_blind'])} {r['unit']} [95% CI {fmt(lo)}, {fmt(hi)}], "
              f"anchoring index {fmt(r['anchoring_index'])} -> {r['verdict']}")
        c, o = r["shown_copied"], r["shown_own"]
        print(f"      shown split: copied with Y n={c['n']} MAD {fmt(c['mad'])}, own lines n={o['n']} MAD {fmt(o['mad'])}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", default=str(KIT / "work"))
    ap.add_argument("--state", default=None, help="autosave.json (default: <work>/autosave.json)")
    ap.add_argument("--blind", default=str(KIT / "private" / "prefill_blind.json"))
    ap.add_argument("--use", choices=["final", "first"], default="final",
                    help="final labels, or the frozen first complete attempt (first)")
    ap.add_argument("--same-px", type=float, default=2.0, help="tolerance of the 'kept unchanged' test")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None, help="also write the result here")
    a = ap.parse_args(argv)
    work = Path(a.work)
    manifest = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
    state_path = Path(a.state) if a.state else work / "autosave.json"
    if not state_path.exists():
        print(f"error: {state_path} not found")
        return 2
    state = json.loads(state_path.read_text(encoding="utf-8"))
    blind_path = Path(a.blind)
    blind_doc = json.loads(blind_path.read_text(encoding="utf-8")) if blind_path.exists() else None
    if not manifest.get("prefill_enabled"):
        print("note: the manifest has no prefill_enabled flag; nothing was shown")
    if blind_doc is None:
        print(f"note: {blind_path} not found; blind tasks have no model reference")
    rows, skipped = collect(manifest, state, blind_doc, a.use, a.same_px)
    res = report(rows, a.boot, a.seed)
    print_report(res, rows, skipped, a.use)
    if a.json:
        Path(a.json).write_text(json.dumps(_clean({"use": a.use, "params": res, "skipped": skipped, "rows": rows}),
                                           indent=1, allow_nan=False), encoding="utf-8")
    return 0


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


if __name__ == "__main__":
    sys.exit(main())
