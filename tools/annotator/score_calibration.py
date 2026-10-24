"""Score the annotator's calibration labels against the hidden expert answers.

    .venv/bin/python tools/annotator/score_calibration.py                    # data/annotator/autosave.json
    .venv/bin/python tools/annotator/score_calibration.py --csv ~/Downloads/umud_calib_labels_XXXX.csv
    PYTHONPATH=src .venv/bin/python tools/annotator/score_calibration.py --refresh-pipeline

Reads the calibration labels (autosave.json, or an exported calib / labels CSV), the calibration
manifest and data/annotator/calib_answers.json, and prints:
  * per image: your value, the expert mean, the error, the error / tolerance;
  * per component (PA 6 deg, FL 12 mm, MT 3 mm): MAE, bias (mean signed error) with its t value,
    RMSE, normalised MAE, and the same split by image type (Echo Blaster static, 126 px/cm static,
    dynamic video frames);
  * you vs the expert mean AND vs single raters, next to the expert-vs-expert spread on the same
    images (each rater vs the mean of the other raters, and rater vs rater), plus your rank;
  * our current pipeline (C12, the honest leave-one-dataset-out benchmark predictions of
    scripts/build_candidates_v4.py) and the SMA / DL_Track values stored in the static workbook, on
    the same images, and a suggested weight for blending your labels with the pipeline.
Writes data/annotator/calib_score.json (never served to the browser).

Conventions: static PA is against the deep aponeurosis (your normal PA).  Dynamic PA is scored
against the image HORIZONTAL (the dynamic benchmark's convention): the fascicle line of each of your
PA measurements (first 2 clicks) vs the horizontal; your deep-aponeurosis PA is also reported there
as a diagnostic.  Dynamic FL is compared in pixels (scale-free) and converted to mm at the dynamic
scale for the normalised score.  The dynamic frames have no expert MT.

The pipeline cache (data/annotator/calib_pipeline.json) is computed on first use (about 2 s,
0.2 GB, no network, no torch); --no-pipeline skips it.  Stdlib only apart from that step.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import labels as labels_mod  # noqa: E402  (tools/annotator/labels.py)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "annotator"
COMPS = ("pa_deg", "fl_mm", "mt_mm")
SHORT = {"pa_deg": "PA", "fl_mm": "FL", "mt_mm": "MT"}
UNIT = {"pa_deg": "deg", "fl_mm": "mm", "mt_mm": "mm"}
DEFAULT_TOL = {"pa_deg": 6.0, "fl_mm": 12.0, "mt_mm": 3.0}
PIPELINE_NAME = "C12"
PIPELINE_LB_S = 0.3767  # C12 public LB score (FACTS 2026-10-21); its benchmark S is ~3x smaller


# ----------------------------------------------------------------------------- small stats
def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else math.nan


def _stats(errs: list[float], tol: float) -> dict:
    n = len(errs)
    if not n:
        return {"n": 0}
    mae = _mean([abs(e) for e in errs])
    bias = _mean(errs)
    sd = math.sqrt(sum((e - bias) ** 2 for e in errs) / (n - 1)) if n > 1 else math.nan
    se = sd / math.sqrt(n) if n > 1 else math.nan
    # leave-one-out bias correction: does subtracting your average bias help on a new image?
    loo = [abs(e - _mean(errs[:i] + errs[i + 1:])) for i, e in enumerate(errs)] if n > 2 else []
    return {"n": n, "mae": mae, "bias": bias, "sd": sd, "se_bias": se,
            "t_bias": bias / se if se and se == se and se > 0 else math.nan,
            "rmse": math.sqrt(_mean([e * e for e in errs])), "mae_norm": mae / tol,
            "mae_bias_corrected_loo": _mean(loo) if loo else math.nan}


def _clean(obj):
    """JSON-safe copy: NaN / inf -> None, floats rounded."""

    if isinstance(obj, float):
        return None if not math.isfinite(obj) else round(obj, 5)
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


# ----------------------------------------------------------------------------- user values
def user_values(state: dict, calib_manifest: dict, min_n: int = 3) -> dict[str, dict]:
    """Per calibration task: the user's values in the scoring convention of that image.

    A component counts only with at least ``min_n`` measurements; flagged tasks are skipped."""

    images = {e["image_id"]: e for e in calib_manifest["images"]}
    out = {}
    for task in calib_manifest["tasks"]:
        lab = (state.get("labels") or {}).get(task["task_id"])
        if not lab:
            continue
        entry = images[task["image_id"]]
        need = task.get("need") or labels_mod.DEFAULT_NEED
        s = labels_mod.summarize(lab, entry["px_per_cm"], need)
        rec = {"kind": task.get("kind", "static"), "flag": bool(lab.get("flag")), "complete": s["complete"],
               "n": {c: s[f"n_{SHORT[c].lower()}"] for c in COMPS}, "px_per_cm": entry["px_per_cm"],
               "pa_apo_deg": s["pa"], "pa_h_deg": s["pa_h"], "fl_px": s["fl_px"],
               "spread": {c: (max(v) - min(v)) if len(v) > 1 else math.nan
                          for c, v in (("pa_deg", s["vals_pa"]), ("fl_mm", s["vals_fl"]), ("mt_mm", s["vals_mt"]))}}
        ok = {c: s[f"n_{SHORT[c].lower()}"] >= min_n and not rec["flag"] for c in COMPS}
        scored_pa = s["pa_h"] if rec["kind"] == "dynamic" else s["pa"]
        rec["value"] = {"pa_deg": scored_pa if ok["pa_deg"] else math.nan,
                        "fl_mm": s["fl"] if ok["fl_mm"] else math.nan,
                        "mt_mm": s["mt"] if ok["mt_mm"] else math.nan}
        out[task["task_id"]] = rec
    return out


# ----------------------------------------------------------------------------- answers helpers
def _rater_values(item: dict, comp: str, ppc: float) -> dict[str, float]:
    vals = {}
    for r, v in (item.get("raters") or {}).items():
        x = v.get(comp)
        if x is None and comp == "fl_mm" and v.get("fl_px") is not None:
            x = v["fl_px"] * 10.0 / ppc
        if x is not None:
            vals[r] = float(x)
    return vals


def _expert(item: dict, comp: str) -> float:
    x = (item.get("expert") or {}).get(comp)
    return math.nan if x is None else float(x)


def subset_of(item: dict) -> str:
    if item["kind"] == "dynamic":
        return "dynamic_echoblaster"
    return "static_echoblaster_77" if round(item["px_per_cm"]) == 77 else "static_126pxcm"


def expert_spread(answers: dict, ids: list[str], tol: dict) -> dict:
    """Expert-vs-expert disagreement on the given images, per component, plus per-rater MAEs."""

    out = {}
    for comp in COMPS:
        vs_others, pairwise, per_rater = [], [], {}
        for cid in ids:
            item = answers["items"][cid]
            vals = _rater_values(item, comp, item["px_per_cm"])
            names = list(vals)
            if len(names) < 2:
                continue
            for r in names:
                others = [vals[o] for o in names if o != r]
                e = vals[r] - _mean(others)
                vs_others.append(e)
                per_rater.setdefault(f"{item['kind']}:{r}", []).append(e)
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    pairwise.append(vals[names[i]] - vals[names[j]])
        out[comp] = {"rater_vs_mean_of_others": _stats(vs_others, tol[comp]),
                     "rater_vs_rater_mae": _mean([abs(e) for e in pairwise]) if pairwise else math.nan,
                     "per_rater_mae_vs_others": {r: _mean([abs(e) for e in es]) for r, es in sorted(per_rater.items())}}
    return out


# ----------------------------------------------------------------------------- pipeline
def compute_pipeline(answers: dict) -> dict:
    """C12's honest benchmark predictions (lodo4) for the calibration images, via the research code."""

    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "src"))
    import build_candidates_v4 as b4  # noqa: E402  (pandas / numpy; no torch, no network)

    dec = json.loads(b4.DECISIONS5.read_text())
    recipe = dict(b4.c12_base(dec), offset="rec5")
    res = b4.bench_res5(recipe)
    frames = {"static": b4.b3._q(res[("lodo4", "static")]), "dynamic": b4.b3._q(res[("lodo4", "dynamic")])}
    items = {}
    for cid, a in answers["items"].items():
        q = frames[a["kind"]]
        if a["source"] not in q.index:
            continue
        row = q.loc[a["source"]]
        rec = {c: float(row[c]) for c in COMPS}
        if a["kind"] == "dynamic":
            rec["mt_mm"] = None  # no expert MT on the dynamic frames
            rec["fl_px"] = rec["fl_mm"] * a["px_per_cm"] / 10.0
        rec["honest"] = bool(a.get("pipeline_honest"))
        items[cid] = rec
    return {"name": PIPELINE_NAME, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "desc": "C12 (artifacts/candidates_v5, LB 0.3767) benchmark predictions after shrinkage and offsets, "
                    "leave-one-dataset-out constants (build_candidates_v4.bench_res5); dynamic PA is vs the "
                    "horizontal like the expert labels. 'honest' = the image was not in our mask models' training data.",
            "items": items}


def load_pipeline(path: Path, answers: dict, refresh: bool, enabled: bool) -> dict | None:
    if not enabled:
        return None
    if path.exists() and not refresh:
        cached = json.loads(path.read_text(encoding="utf-8"))
        if set(cached.get("items", {})) >= set(answers["items"]):
            return cached
    try:
        pipe = compute_pipeline(answers)
    except Exception as exc:  # research code missing / changed: score without it
        print(f"(pipeline comparison skipped: {type(exc).__name__}: {exc})")
        return None
    path.write_text(json.dumps(_clean(pipe), indent=1), encoding="utf-8")
    return pipe


# ----------------------------------------------------------------------------- scoring
def blend_weight(eu: list[float], ep: list[float], k_test: float = math.nan) -> dict:
    """Weight w on the hand label in w*user + (1-w)*pipeline from paired errors (same images).

    ``k_test`` scales the pipeline's benchmark RMSE to its test-domain RMSE (LB S / benchmark S):
    benchmark images are where the pipeline was developed and selected, so its error there is far
    below its leaderboard error; the hand-label error is assumed to transfer as measured."""

    n = len(eu)
    if n < 3:
        return {"n": n}
    su2 = _mean([e * e for e in eu])
    sp2 = _mean([e * e for e in ep])
    cov = _mean([a * b for a, b in zip(eu, ep)])
    den = su2 + sp2 - 2 * cov
    w_opt = min(1.0, max(0.0, (sp2 - cov) / den)) if den > 0 else 0.5
    w_ind = sp2 / (su2 + sp2) if su2 + sp2 > 0 else 0.5  # independent errors (more stable with small n)
    grid = {f"{w:.2f}": _mean([abs(w * a + (1 - w) * b) for a, b in zip(eu, ep)]) for w in (0, .25, .5, .75, 1)}
    corr = cov / math.sqrt(su2 * sp2) if su2 > 0 and sp2 > 0 else math.nan
    sp2_test = sp2 * k_test ** 2 if math.isfinite(k_test) else math.nan
    w_test = sp2_test / (su2 + sp2_test) if math.isfinite(sp2_test) and su2 + sp2_test > 0 else math.nan
    return {"n": n, "rmse_user": math.sqrt(su2), "rmse_pipeline": math.sqrt(sp2), "error_corr_uncentred": corr,
            "w_user_optimal": w_opt, "w_user_independent": w_ind, "mae_by_w_user": grid,
            "pipeline_test_scale_k": k_test, "w_user_test_domain": w_test}


def score(user: dict, answers: dict, pipeline: dict | None, tol: dict | None = None,
          pipeline_test_s: float = PIPELINE_LB_S) -> dict:
    tol = tol or {**DEFAULT_TOL, **(answers.get("tolerances") or {})}
    items = answers["items"]
    ids = list(items)
    per_image, errs, single, sub_errs, pipe_errs, tool_errs = [], {c: [] for c in COMPS}, {c: [] for c in COMPS}, {}, {}, {}
    pairs = {c: ([], []) for c in COMPS}
    for cid in ids:
        item = items[cid]
        u = user.get(cid)
        ppc = item["px_per_cm"]
        row = {"task_id": cid, "source": item["source"], "kind": item["kind"], "subset": subset_of(item),
               "pipeline_honest": item.get("pipeline_honest"), "measured": bool(u), "flag": bool(u and u["flag"])}
        norm = []
        for comp in COMPS:
            ex = _expert(item, comp)
            uv = u["value"][comp] if u else math.nan
            e = uv - ex if math.isfinite(uv) and math.isfinite(ex) else math.nan
            row[comp] = {"you": uv, "expert": ex, "err": e, "err_norm": abs(e) / tol[comp] if math.isfinite(e) else math.nan}
            if math.isfinite(e):
                errs[comp].append(e)
                norm.append(abs(e) / tol[comp])
                sub_errs.setdefault(row["subset"], {c: [] for c in COMPS})[comp].append(e)
                single[comp] += [uv - r for r in _rater_values(item, comp, ppc).values()]
            if pipeline and cid in pipeline["items"] and math.isfinite(ex):
                pv = pipeline["items"][cid].get(comp)
                if pv is not None and math.isfinite(pv):
                    row[comp]["pipeline"] = pv
                    pipe_errs.setdefault("all", {c: [] for c in COMPS})[comp].append(pv - ex)
                    if item.get("pipeline_honest"):
                        pipe_errs.setdefault("honest", {c: [] for c in COMPS})[comp].append(pv - ex)
                        if math.isfinite(e):
                            pairs[comp][0].append(e)
                            pairs[comp][1].append(pv - ex)
            for tname, tv in (item.get("tools") or {}).items():
                x = tv.get(comp)
                if x is not None and math.isfinite(ex):
                    tool_errs.setdefault(tname, {c: [] for c in COMPS})[comp].append(x - ex)
        if u and item["kind"] == "dynamic" and math.isfinite(u["pa_apo_deg"]) and math.isfinite(_expert(item, "pa_deg")):
            row["pa_vs_deep_apo_diagnostic"] = {"you": u["pa_apo_deg"], "err_vs_expert": u["pa_apo_deg"] - _expert(item, "pa_deg")}
        if u and item["kind"] == "dynamic":
            row["fl_px"] = {"you": u["fl_px"], "expert": (item.get("expert") or {}).get("fl_px")}
            row["mt_mm_unscored"] = u["value"]["mt_mm"]
        if u:
            row["spread_of_your_3"] = u["spread"]
            row["n"] = u["n"]
        row["err_norm_mean"] = _mean(norm) if norm else math.nan
        per_image.append(row)

    def _S(per_comp: dict) -> float:
        vals = [v["mae_norm"] for v in per_comp.values() if v.get("n")]
        return _mean(vals) if vals else math.nan

    comp_stats = {c: _stats(errs[c], tol[c]) for c in COMPS}
    for c in COMPS:
        comp_stats[c]["vs_single_rater_mae"] = _mean([abs(e) for e in single[c]]) if single[c] else math.nan
    measured_ids = [r["task_id"] for r in per_image if r["measured"]]
    spread_all = expert_spread(answers, ids, tol)
    spread_measured = expert_spread(answers, measured_ids, tol) if measured_ids else None
    ranks = {}
    if spread_measured:
        for c in COMPS:
            for kind in ("static", "dynamic"):
                mine = [abs(r[c]["err"]) for r in per_image if r["kind"] == kind and math.isfinite(r[c]["err"])]
                raters = {k: v for k, v in spread_measured[c]["per_rater_mae_vs_others"].items() if k.startswith(kind + ":")}
                if mine and raters:
                    m = _mean(mine)
                    ranks[f"{c}:{kind}"] = {"you_mae_vs_mean": m, "rank_among_raters": 1 + sum(v < m for v in raters.values()),
                                            "n_raters": len(raters) + 1, "raters": raters}
    out = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "tolerances": tol,
        "n_measured": len(measured_ids), "n_tasks": len(ids),
        "n_complete": sum(1 for u in user.values() if u["complete"] and not u["flag"]),
        "per_image": per_image, "you": {"components": comp_stats, "S": _S(comp_stats),
                                       "by_subset": {k: {c: _stats(v[c], tol[c]) for c in COMPS} for k, v in sorted(sub_errs.items())}},
        "experts_on_all_12": spread_all, "experts_on_your_images": spread_measured, "your_rank": ranks,
        "automatic_tools_static": {k: {c: _stats(v[c], tol[c]) for c in COMPS} for k, v in tool_errs.items()},
    }
    if pipeline:
        pstats = {k: {c: _stats(v[c], tol[c]) for c in COMPS} for k, v in pipe_errs.items()}
        out["pipeline"] = {"name": pipeline.get("name"), "desc": pipeline.get("desc"),
                           **{k: {"components": v, "S": _S(v)} for k, v in pstats.items()}}
        honest_ids = {r["task_id"] for r in per_image if r["pipeline_honest"]}
        you_h = {c: _stats([r[c]["err"] for r in per_image if r["task_id"] in honest_ids and math.isfinite(r[c]["err"])],
                           tol[c]) for c in COMPS}
        out["you_on_pipeline_honest_images"] = {"components": you_h, "S": _S(you_h)}
        s_h = out["pipeline"].get("honest", {}).get("S", math.nan)
        k = pipeline_test_s / s_h if pipeline_test_s and math.isfinite(s_h) and s_h > 0 else math.nan
        out["pipeline"]["test_S_assumed"] = pipeline_test_s
        out["blend"] = {c: blend_weight(*pairs[c], k_test=k) for c in COMPS}
    return out


# ----------------------------------------------------------------------------- printing
def _f(v, d=2, sign=False) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "-"
    return f"{v:+.{d}f}" if sign else f"{v:.{d}f}"


def report(res: dict) -> None:
    tol = res["tolerances"]
    print(f"\nCalibration: {res['n_measured']}/{res['n_tasks']} images measured, {res['n_complete']} complete.")
    print("\nPer image (you / expert mean / error; dynamic PA = fascicle vs horizontal, FL from px):")
    print(f"{'task':7s} {'source':12s} {'subset':22s} | {'PA you':>6s} {'exp':>6s} {'err':>6s} | {'FL you':>6s} {'exp':>6s} {'err':>6s}"
          f" | {'MT you':>6s} {'exp':>6s} {'err':>6s} | {'err/tol':>7s}")
    for r in res["per_image"]:
        cells = []
        for c in COMPS:
            x = r[c]
            cells.append(f"{_f(x['you'], 1):>6s} {_f(x['expert'], 1):>6s} {_f(x['err'], 1, True):>6s}")
        tag = " (flag)" if r["flag"] else ""
        print(f"{r['task_id']:7s} {r['source']:12s} {r['subset']:22s} | " + " | ".join(cells) + f" | {_f(r['err_norm_mean'], 3):>7s}{tag}")
    dyn = [r for r in res["per_image"] if "pa_vs_deep_apo_diagnostic" in r]
    if dyn:
        d = _mean([r["pa_vs_deep_apo_diagnostic"]["err_vs_expert"] for r in dyn])
        print(f"  dynamic diagnostic: your deep-aponeurosis PA minus the expert (horizontal) PA = {_f(d, 2, True)} deg on average "
              f"(not scored; shows the deep-aponeurosis tilt).")
    you = res["you"]["components"]
    if any(v.get("n") for v in you.values()):
        print("\nYou vs the expert mean:")
        print(f"{'':4s} {'n':>3s} {'MAE':>6s} {'bias':>7s} {'t':>6s} {'RMSE':>6s} {'MAE/tol':>8s} {'LOO bias-corr MAE':>18s} {'vs 1 rater':>10s}")
        for c in COMPS:
            s = you[c]
            if not s.get("n"):
                continue
            print(f"{SHORT[c]:4s} {s['n']:3d} {_f(s['mae']):>6s} {_f(s['bias'], 2, True):>7s} {_f(s['t_bias'], 1):>6s} "
                  f"{_f(s['rmse']):>6s} {_f(s['mae_norm'], 3):>8s} {_f(s['mae_bias_corrected_loo']):>18s} {_f(s['vs_single_rater_mae']):>10s}")
        print(f"S (mean MAE/tol, competition-style) = {_f(res['you']['S'], 4)}")
        print("\nBy image type (MAE / bias):")
        for k, v in res["you"]["by_subset"].items():
            print(f"  {k:22s} " + "  ".join(f"{SHORT[c]} n={v[c]['n']} {_f(v[c].get('mae'))}/{_f(v[c].get('bias'), 2, True)}"
                                          for c in COMPS if v[c].get("n")))
    spread = res["experts_on_your_images"] or res["experts_on_all_12"]
    label = "your images" if res["experts_on_your_images"] else "all 12 images"
    print(f"\nExpert-vs-expert spread on {label} (for comparison with the 'you' MAE above):")
    for c in COMPS:
        s = spread[c]
        r = s["rater_vs_mean_of_others"]
        if r.get("n"):
            print(f"  {SHORT[c]}: rater vs mean of the others MAE {_f(r['mae'])} {UNIT[c]} (MAE/tol {_f(r['mae_norm'], 3)}); "
                  f"rater vs rater MAE {_f(s['rater_vs_rater_mae'])} {UNIT[c]}")
    for key, v in res["your_rank"].items():
        c, kind = key.split(":")
        print(f"  {SHORT[c]} {kind}: you would rank {v['rank_among_raters']} of {v['n_raters']} "
              f"(1 = closest to the consensus; you {_f(v['you_mae_vs_mean'])} vs raters "
              + ", ".join(f"{k.split(':')[1]} {_f(x)}" for k, x in v["raters"].items()) + ")")
    if "pipeline" in res:
        p = res["pipeline"]
        print(f"\nPipeline {p['name']} on the same images (honest = not in our training data):")
        for k in ("all", "honest"):
            if k in p:
                comps = p[k]["components"]
                print(f"  {k:6s}: S {_f(p[k]['S'], 4)} | " + "  ".join(
                    f"{SHORT[c]} n={comps[c]['n']} MAE {_f(comps[c].get('mae'))} bias {_f(comps[c].get('bias'), 2, True)}"
                    for c in COMPS if comps[c].get("n")))
        yh = res["you_on_pipeline_honest_images"]
        if any(v.get("n") for v in yh["components"].values()):
            print(f"  you on the same honest images: S {_f(yh['S'], 4)} | " + "  ".join(
                f"{SHORT[c]} n={yh['components'][c]['n']} MAE {_f(yh['components'][c].get('mae'))}"
                for c in COMPS if yh["components"][c].get("n")))
        print("  blend weight on YOUR label (w*you + (1-w)*pipeline, from paired errors on honest images; small n, rough):")
        for c in COMPS:
            b = res["blend"][c]
            if b.get("n", 0) >= 3:
                print(f"    {SHORT[c]} n={b['n']}: on benchmark w_independent {_f(b['w_user_independent'])}, "
                      f"w_optimal {_f(b['w_user_optimal'])} (MAE by w " + ", ".join(f"{w}:{_f(x)}" for w, x in b["mae_by_w_user"].items())
                      + f"); test-domain w {_f(b['w_user_test_domain'])} (pipeline error x{_f(b['pipeline_test_scale_k'], 1)})")
        print(f"  The benchmark images are where the pipeline was built and selected, so it looks much better there than on "
              f"the test set (LB S {p.get('test_S_assumed')}). Use the test-domain weight, which scales the pipeline error "
              "by LB S / benchmark S, as the default.")
    tools = res["automatic_tools_static"]
    if tools:
        print("\nAutomatic tools stored in the static workbook (static images only):")
        for k, v in tools.items():
            print(f"  {k:8s} " + "  ".join(f"{SHORT[c]} MAE {_f(v[c].get('mae'))}" for c in COMPS if v[c].get("n")))
    print(f"\nTolerances: PA {tol['pa_deg']} deg, FL {tol['fl_mm']} mm, MT {tol['mt_mm']} mm.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score calibration labels against the expert answers")
    parser.add_argument("--state", type=Path, default=DATA / "autosave.json")
    parser.add_argument("--csv", type=Path, default=None, help="exported calibration (or labels) CSV instead of autosave")
    parser.add_argument("--manifest", type=Path, default=DATA / "calib_manifest.json")
    parser.add_argument("--answers", type=Path, default=DATA / "calib_answers.json")
    parser.add_argument("--pipeline-cache", type=Path, default=DATA / "calib_pipeline.json")
    parser.add_argument("--refresh-pipeline", action="store_true", help="recompute the pipeline predictions")
    parser.add_argument("--no-pipeline", action="store_true")
    parser.add_argument("--min-n", type=int, default=3, help="measurements needed for a component to count")
    parser.add_argument("--pipeline-test-s", type=float, default=PIPELINE_LB_S,
                        help="the pipeline's test-domain S (public LB) used for the test-domain blend weight")
    parser.add_argument("--out", type=Path, default=DATA / "calib_score.json")
    args = parser.parse_args()
    calib = json.loads(args.manifest.read_text(encoding="utf-8"))
    answers = json.loads(args.answers.read_text(encoding="utf-8"))
    if args.csv:
        state = labels_mod.state_from_csv(args.csv)
    elif args.state.exists():
        state = json.loads(args.state.read_text(encoding="utf-8"))
    else:
        state = {"labels": {}}
    user = user_values(state, calib, args.min_n)
    pipeline = load_pipeline(args.pipeline_cache, answers, args.refresh_pipeline, not args.no_pipeline)
    res = score(user, answers, pipeline, pipeline_test_s=args.pipeline_test_s)
    res["labels_source"] = str(args.csv or args.state)
    report(res)
    args.out.write_text(json.dumps(_clean(res), indent=1), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
