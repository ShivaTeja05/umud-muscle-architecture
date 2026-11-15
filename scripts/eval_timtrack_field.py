"""Evaluate the frozen fascicle-orientation field on the public TimTrack labels.

This is a diagnostic-only external benchmark.  It does not read competition test
labels, does not create a submission, and it does not choose an inference
constant.  TimTrack's multi-rater ``alpha`` is the fascicle angle to the image
horizontal, which is directly comparable to the model's pixel-coordinate
orientation field.

Inputs are the already downloaded, immutable public TimTrack labels and the
previously generated 339-image ``fasc_orient_v1`` maskbank.  The estimates are
global doubled-angle means, weighted either by the learned fascicle probability
or field agreement.  This answers a narrow first question: does the existing
learned field transfer to a second labelled image family well enough to justify
training an external-data specialist?

Run:
  .venv/bin/python scripts/eval_timtrack_field.py
  .venv/bin/python scripts/eval_timtrack_field.py \
      --bank artifacts/kaggle/tt_bench/extra/tt_out/maskbank --model fasc_ins_v16 \
      --out artifacts/external/timtrack_field_eval_insfield
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BANK = ROOT / "artifacts/kaggle/tt_bench/orient/tt_out/maskbank"
DEFAULT_LABELS = ROOT / "artifacts/external/timtrack_bench/labels_mean.csv"
DEFAULT_OUT = ROOT / "artifacts/external/timtrack_field_eval"
SUBSETS = ("tt1", "tt2", "tt3", "tt4a", "tt4b")


def decode(a: np.ndarray) -> np.ndarray:
    return a.astype(np.float32) / 127.5 - 1.0


def wrap90(x: float) -> float:
    """Signed orientation difference, where a line has period 180 degrees."""
    return float((x + 90.0) % 180.0 - 90.0)


def subject_of(subset: str, key: str) -> str:
    """Conservative cluster IDs; never combines different named subjects."""
    if subset == "tt1":
        return key.split("_", 1)[0]  # S1 ... S11
    if subset == "tt3":
        return "p6"                 # all 100 frames are one person
    # Dataset 2 lacks stable people in the public workbook.  Dataset 4 has one
    # image per person/protocol.  Treat each image independently rather than
    # fabricating a group structure.
    return f"{subset}:{key}"


def estimate(seg: np.ndarray, cos2: np.ndarray, sin2: np.ndarray, mode: str) -> tuple[float, float, int]:
    mag = np.hypot(cos2, sin2)
    if mode == "field":
        w = mag
    elif mode == "seg":
        w = np.clip(seg, 0.0, 1.0) * mag
    elif mode == "seg30":
        w = np.where(seg >= 0.30, seg * mag, 0.0)
    else:  # pragma: no cover - argparse restricts this
        raise ValueError(mode)
    good = np.isfinite(w) & np.isfinite(cos2) & np.isfinite(sin2) & (w > 1e-6)
    if not good.any():
        return float("nan"), float("nan"), 0
    c = float(np.sum(w[good] * cos2[good]))
    s = float(np.sum(w[good] * sin2[good]))
    strength = math.hypot(c, s) / max(float(np.sum(w[good])), 1e-12)
    if strength < 1e-4:
        return float("nan"), strength, int(good.sum())
    # TimTrack's ``alpha`` is the acute, unsigned fascicle angle.  Our field
    # retains its image-slope sign (and test scanners contain both directions),
    # hence the physically comparable quantity is its acute magnitude.
    return abs(math.degrees(0.5 * math.atan2(s, c))), strength, int(good.sum())


def summary(frame: pd.DataFrame) -> dict:
    out: dict[str, dict] = {}
    for name, q in list(frame.groupby("subset")) + [("pooled", frame)]:
        r: dict[str, float | int] = {"n": int(len(q)), "n_subjects": int(q.subject.nunique())}
        for mode, a in q.groupby("mode"):
            ok = a.alpha.notna() & a.pred_alpha.notna()
            e = a.loc[ok, "error_deg"].to_numpy(float)
            r[str(mode)] = {
                "n": int(ok.sum()),
                "mae_deg": float(np.mean(np.abs(e))) if len(e) else float("nan"),
                "bias_deg": float(np.mean(e)) if len(e) else float("nan"),
                "medae_deg": float(np.median(np.abs(e))) if len(e) else float("nan"),
                "rmse_deg": float(np.sqrt(np.mean(e ** 2))) if len(e) else float("nan"),
                "median_vector_strength": float(a.loc[ok, "vector_strength"].median()) if len(e) else float("nan"),
            }
        out[str(name)] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    ap.add_argument("--model", default="fasc_orient_v1",
                    help="Maskbank model stem, e.g. fasc_orient_v1 or fasc_ins_v16")
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(args.labels)
    labels = labels[["subset", "key", "alpha", "file"]].copy()
    rows: list[dict] = []
    missing: list[str] = []
    for subset in SUBSETS:
        paths = {ch: args.bank / f"{subset}__{args.model}_{ch}.npz" for ch in ("seg", "cos2", "sin2")}
        if not all(p.exists() for p in paths.values()):
            missing.append(subset)
            continue
        with np.load(paths["seg"]) as zs, np.load(paths["cos2"]) as zc, np.load(paths["sin2"]) as zi:
            keys = sorted(set(zs.files) & set(zc.files) & set(zi.files))
            for key in keys:
                seg, c, s = decode(zs[key]), decode(zc[key]), decode(zi[key])
                for mode in ("field", "seg", "seg30"):
                    pred, strength, n_pixels = estimate(seg, c, s, mode)
                    rows.append({"subset": subset, "key": key, "subject": subject_of(subset, key), "mode": mode,
                                 "pred_alpha": pred, "vector_strength": strength, "n_weighted_pixels": n_pixels})
    estimates = pd.DataFrame(rows)
    if estimates.empty:
        raise RuntimeError(f"no matching maskbank files under {args.bank}")
    joined = estimates.merge(labels, on=["subset", "key"], how="left", validate="many_to_one")
    joined["error_deg"] = [wrap90(p - t) if np.isfinite(p) and np.isfinite(t) else np.nan
                           for p, t in zip(joined.pred_alpha, joined.alpha)]
    report = {
        "purpose": "diagnostic external orientation evaluation; no submission candidate",
        "model": args.model,
        "labels": str(args.labels.relative_to(ROOT)) if args.labels.is_relative_to(ROOT) else str(args.labels),
        "bank": str(args.bank.relative_to(ROOT)) if args.bank.is_relative_to(ROOT) else str(args.bank),
        "n_rows": int(len(joined)),
        "n_with_label": int(joined.alpha.notna().sum()),
        "missing_subsets": missing,
        "summary": summary(joined),
        "notes": [
            "TimTrack alpha is an acute image-horizontal fascicle orientation, not a UMUD PA target.",
            "No train/test decision may be based on a single pooled number; tt3 is one person.",
            "The three weight modes are diagnostic readouts, not leaderboard-tuned variants.",
        ],
    }
    joined.sort_values(["subset", "key", "mode"]).to_csv(args.out / "per_image.csv", index=False)
    (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    print(json.dumps(report["summary"], indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
