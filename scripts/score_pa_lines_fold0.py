"""Score the registered v33 fold-0 PA pilot against the frozen fusF readout.

Consumes CSVs and metadata from the Kaggle run. Prints a JSON decision and
leaves the submission and production artifact directories untouched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "artifacts/candidates_v3/cache"
BENCH = ROOT / "artifacts/candidates_v26/inputs/bench_frame.csv"
BASE_COLS = ["pa_fusion", "pa_fd_strip_g"]


def acute(angle):
    """Convert an undirected signed line difference to the positive PA convention."""
    return abs((angle + 90) % 180 - 90)


def paired_metrics(frame):
    expected = len(frame)
    frame = frame.loc[np.isfinite(frame[["truth", "baseline", "network"]]).all(axis=1)]
    result = {"expected": expected, "matched": len(frame)}
    if frame.empty:
        return result

    errors = frame[["baseline", "network"]].sub(frame.truth, axis=0)
    centered = errors - errors.median()
    median_mae = centered.abs().mean()
    corr = None
    if len(frame) > 1 and errors.nunique().min() > 1:
        corr = float(errors.baseline.corr(errors.network))
    return dict(result, raw_mae=errors.abs().mean().to_dict(),
                median_error=errors.median().to_dict(), centered_mae=median_mae.to_dict(),
                delta_deg=float(median_mae.network - median_mae.baseline), error_correlation=corr)


def train_metrics(path, baseline):
    rows = pd.read_csv(path)
    rows = rows.loc[(rows.kind == "fasc") & (rows.has_fasc == 1) & (rows.has_deep == 1)].copy()
    if not rows.fold.eq(0).all():
        raise ValueError(f"Expected only held-out fold 0 rows in {path}")
    rows["truth"] = acute(rows.gt_line_deg - rows.gt_fasc_deg)
    rows["network"] = acute(rows.pred_line_deg - rows.pred_fasc_deg)
    return paired_metrics(rows.merge(baseline, left_on="stem", right_on="fasc", how="left", validate="one_to_one"))


def vl_rows(readout):
    # The registered cache stores VL's full label ID and the Kaggle maskbank key.
    # These are both unique. Do not join on ANON_ stems, which repeat by person.
    base = pd.read_csv(CACHE / "vl__v2.csv", usecols=["image_id", "okey", *BASE_COLS])
    base["baseline"] = base[BASE_COLS].mean(axis=1)
    bench = pd.read_csv(BENCH, usecols=["dataset", "image_id", "pa_true"])
    truth = bench.loc[bench.dataset.str.startswith("vl_") & bench.pa_true.notna(), ["image_id", "pa_true"]]
    truth = truth.rename(columns={"pa_true": "truth"})
    pred = readout.loc[readout.dataset.eq("vl"), ["image_id", "pa_image_deg"]].rename(columns={"image_id": "okey"})
    pred["network"] = acute(pred.pa_image_deg)
    return truth.merge(base, on="image_id", how="left", validate="one_to_one").merge(
        pred[["okey", "network"]], on="okey", how="left", validate="one_to_one")


def score(run):
    config = json.loads((run / "pa_lines_v33_config.json").read_text())
    if config["model"] != "pa_lines_v33" or config["train"]["folds"] != [0]:
        raise ValueError("This scorer expects the v33 fold-0 pilot")
    train = pd.read_csv(CACHE / "train142__oof.csv", usecols=["fasc", *BASE_COLS])
    train["baseline"] = train[BASE_COLS].mean(axis=1)
    baseline = train[["fasc", "baseline"]]
    s1 = train_metrics(run / "oof_readout.csv", baseline)
    trajectory = {}
    for epoch in (4, 8):
        path = run / "metrics" / f"early_fold0_ep{epoch}.csv"
        if path.is_file():
            trajectory[str(epoch)] = train_metrics(path, baseline)
    vl = vl_rows(pd.read_csv(run / "readout.csv"))
    vl_missing_baseline = vl.loc[~np.isfinite(vl.baseline), "image_id"].tolist()
    vl = vl.loc[np.isfinite(vl.baseline)].copy()
    s2 = paired_metrics(vl)
    s1_pass = s1["matched"] >= 15 and s1["delta_deg"] < 0
    s1_status = "uninformative" if s1["matched"] < 15 else "pass" if s1_pass else "fail"
    s2_pass = s2["matched"] > 0 and s2["matched"] == s2["expected"] and s2["delta_deg"] <= 0.05
    result = dict(mode=config["mode"], S1=dict(s1, status=s1_status),
                  S2=dict(s2, passed=s2_pass, missing_baseline_ids=vl_missing_baseline),
                  S3_vl_error_correlation=s2.get("error_correlation"), trajectory=trajectory)

    if config["mode"] != "full":
        verdict = "SMOKE_ONLY"
    elif s1["expected"] != s1["matched"] or s2["expected"] != s2["matched"] or not s2["matched"]:
        verdict = "INCOMPLETE_PAIRING"
        result["unpaired_vl_ids"] = vl.loc[~np.isfinite(vl[["truth", "baseline", "network"]]).all(axis=1), "image_id"].tolist()
    elif s1_status != "fail" and s2_pass:
        verdict = "MORE_FOLDS_WARRANTED"
    elif not s2_pass and s2["delta_deg"] <= 0.30 and s2["error_correlation"] is not None and s2["error_correlation"] < 0.9:
        verdict = "FIXED_BLEND_DIAGNOSTIC_WARRANTED"
        vl["network"] = 0.5 * (vl.network + vl.baseline)
        result["vl_equal_blend"] = paired_metrics(vl)
    else:
        verdict = "CLOSED"
    result["stage_decision"] = verdict
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Downloaded Kaggle pilot output directory")
    print(json.dumps(score(parser.parse_args().run), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
