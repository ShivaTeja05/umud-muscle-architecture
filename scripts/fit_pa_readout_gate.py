"""Test a bounded, learned PA readout on frozen geometry caches (CPU).

Fit only on out-of-fold predictions for competition training images with both
masks. Evaluate expert VL and clean static without fitting to either panel.
The experiment passes only if VL improves by at least 0.10 degree with a
one-sided participant-bootstrap 90% upper bound below zero, and clean static
does not worsen. Even a pass requires a full production-pipeline comparison.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import expit

from score_pa_lines_fold0 import BASE_COLS, BENCH, CACHE, ROOT, paired_metrics


QUALITY = ["d_fd_strip_conc", "d_fd_strip_inl", "d_fd_seg_frac"]
CLEAN = {f"im_{i}_arch" for i in ("02", "03", "14", "17", "20", "23", "24", "31", "33", "34")}


def features(rows, center, scale):
    quality = rows[QUALITY].to_numpy(float)
    standardized = np.clip((quality - center) / scale, -3, 3)
    return np.column_stack([np.ones(len(rows)), np.nan_to_num(standardized, nan=0)])


def gate_weight(x, coef):
    return 0.15 + 0.70 * expit(x @ coef)


def model_pa(rows, weight):
    fusion, strip = (rows[c].to_numpy(float) for c in BASE_COLS)
    baseline = rows[BASE_COLS].mean(axis=1).to_numpy()
    return np.where(np.isfinite(fusion) & np.isfinite(strip), (1 - weight) * fusion + weight * strip, baseline)


def bootstrap_vl(rows):
    clusters = rows.image_id.str.split("__").str[:2].str.join("__")
    errors = rows[["baseline", "network"]].sub(rows.truth, axis=0).to_numpy()
    parts = [errors[idx] for idx in clusters.groupby(clusters).indices.values()]
    rng = np.random.default_rng(20261110)
    deltas = []
    for indices in rng.integers(0, len(parts), (3000, len(parts))):
        sample = np.concatenate([parts[i] for i in indices])
        mae = np.abs(sample - np.median(sample, axis=0)).mean(axis=0)
        deltas.append(mae[1] - mae[0])
    return len(parts), float(np.quantile(deltas, 0.9))


def main():
    keys = ["image", "fasc"]
    train = pd.read_csv(CACHE / "train142__oof.csv", usecols=[*keys, *BASE_COLS, *QUALITY])
    gt_path = ROOT / "artifacts/eval_v3/orient/cache/train142.csv"
    gt = pd.read_csv(gt_path, usecols=[*keys, "run", "pa_global"])
    gt = gt.loc[gt.run.eq("gt"), [*keys, "pa_global"]].rename(columns={"pa_global": "truth"})
    train = train.merge(gt, on=keys, validate="one_to_one").dropna(subset=["truth", *BASE_COLS, *QUALITY])
    center = train[QUALITY].median().to_numpy()
    scale = np.maximum(train[QUALITY].std().to_numpy(), 1e-6)
    x = features(train, center, scale)

    def residual(coef):
        prediction = model_pa(train, gate_weight(x, coef))
        return np.r_[prediction - train.truth, np.sqrt(len(train) * 0.05) * coef]

    fit = least_squares(residual, np.zeros(x.shape[1]), loss="soft_l1", f_scale=1)
    if not fit.success:
        raise RuntimeError(fit.message)

    bench = pd.read_csv(BENCH, usecols=["image_id", "dataset", "pa_true"])
    result = {"training_pairs": len(train), "coefficients": dict(zip(["intercept", *QUALITY], fit.x))}
    for panel in ("vl", "static"):
        selected = bench.dataset.str.startswith("vl_") if panel == "vl" else bench.image_id.isin(CLEAN) & bench.dataset.eq("static")
        truth = bench.loc[selected & bench.pa_true.notna(), ["image_id", "pa_true"]].rename(columns={"pa_true": "truth"})
        cached = pd.read_csv(CACHE / f"{panel}__v2.csv", usecols=["image_id", *BASE_COLS, *QUALITY])
        rows = truth.merge(cached, on="image_id", how="left", validate="one_to_one")
        rows["baseline"] = rows[BASE_COLS].mean(axis=1)
        missing_baseline = rows.loc[~np.isfinite(rows.baseline), "image_id"].tolist()
        rows = rows.loc[np.isfinite(rows.baseline)].reset_index(drop=True)
        weight = gate_weight(features(rows, center, scale), fit.x)
        rows["network"] = model_pa(rows, weight)
        metrics = paired_metrics(rows)
        if metrics["matched"] != len(rows):
            raise ValueError(f"Incomplete evaluation coverage for {panel}")
        metrics["missing_baseline_ids"] = missing_baseline
        metrics["strip_weight_range"] = [float(weight.min()), float(weight.max())]
        if panel == "vl":
            metrics["participants"], metrics["delta_upper90_deg"] = bootstrap_vl(rows)
        result[panel] = metrics

    vl, static = result["vl"], result["static"]
    result["decision"] = "PASS_TO_PRODUCTION_GATE" if (
        vl["delta_deg"] <= -0.10 and vl["delta_upper90_deg"] < 0 and static["delta_deg"] <= 0
    ) else "CLOSED"
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
