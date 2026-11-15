"""v16 insfield: build the small embedded payload (laptop, light: one label mask at a time, < 0.3 GB).

    .venv/bin/python kaggle/v16_insfield/build_payload.py

Writes kaggle/v16_insfield/payload/
    vl_lines.json   one record per VL image (OSF xbawc "Example Annotated Images", CC BY 4.0): the expert's
                    superficial / deep aponeurosis lines and fascicle tracks parsed from the pure-red ImageJ lines
                    (label_masks/, same rules as scripts/vl_labels.py), converted from the annotated canvas to the
                    RAW image (canvas - offset), plus the subject id and a subject-grouped fold (5 folds, stratified
                    young / old, balanced by image count, seed fixed).
    fasc_folds.csv  the fasc_orient_v1 reference folds (stem, representative, group, fold) so that the competition
                    folds are identical to the control model (KG0's field).
    payload_summary.json
Each track: pts = smoothed traced pixels every STEP px (raw coords, superficial end first), ps / pd = intersections
of the end directions with the expert's superficial / deep line, pa = deep-end angle vs the deep line (12 % of the
track, >= 40 px; the VL label convention of artifacts/candidates/bench/vl_labels.csv), pa_all = whole-track angle vs
the deep line, fl_mm, side. GPL-3.0-only.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import vl_labels as V  # noqa: E402  (tracked parser, read-only use)
from umud import geometry_v2 as g2  # noqa: E402

VL = ROOT / "artifacts/maskbank/vl_data"
OUT = Path(__file__).resolve().parent / "payload"
STEP = 8.0
N_FOLDS = 5
SEED = 20261031


def tracks(m: np.ndarray):
    """vl_labels.parse() with the per-track smoothed pixels kept (same acceptance and dedupe rules)."""
    al = V.find_apo_lines(m)
    if al is None or "fail" in al:
        return None
    S, D = al["sup"], al["deep"]
    ys, xs = np.nonzero(m)
    ds = V.dist_to_line(xs, ys, S["c"], S["d"])
    dd = V.dist_to_line(xs, ys, D["c"], D["d"])
    keep = (np.abs(ds) > 2.5) & (np.abs(dd) > 2.5) & (ds > 0) & (dd < 0)
    F = np.zeros_like(m)
    F[ys[keep], xs[keep]] = True
    out = []
    for t in V.column_tracks(F):
        x = np.asarray(t["x"], float)
        y = np.asarray(t["y"], float)
        xg = np.arange(x.min(), x.max() + 1)
        yg = np.interp(xg, x, y)
        k = 9
        if len(yg) > k:
            ypad = np.r_[np.full(k // 2, yg[0]), yg, np.full(k // 2, yg[-1])]
            ys_ = np.convolve(ypad, np.ones(k) / k, mode="valid")
        else:
            ys_ = yg
        n = len(xg)
        e = min(max(40, int(0.12 * n)), n)
        _, dl = V.line_from_pts(xg[:e], ys_[:e])
        _, dr = V.line_from_pts(xg[-e:], ys_[-e:])
        pL, pR = np.array([xg[0], ys_[0]]), np.array([xg[-1], ys_[-1]])
        dLs = abs(V.dist_to_line([pL[0]], [pL[1]], S["c"], S["d"])[0])
        dLd = abs(V.dist_to_line([pL[0]], [pL[1]], D["c"], D["d"])[0])
        dRs = abs(V.dist_to_line([pR[0]], [pR[1]], S["c"], S["d"])[0])
        dRd = abs(V.dist_to_line([pR[0]], [pR[1]], D["c"], D["d"])[0])
        if dLs + dRd <= dLd + dRs:
            sup_end, sup_dir, deep_end, deep_dir, gs, gd, side = pL, dl, pR, dr, dLs, dRd, "right"
        else:
            sup_end, sup_dir, deep_end, deep_dir, gs, gd, side = pR, dr, pL, dl, dRs, dLd, "left"
        P_s = V.intersect(S["c"], S["d"], sup_end, sup_dir)
        P_d = V.intersect(D["c"], D["d"], deep_end, deep_dir)
        if P_s is None or P_d is None:
            continue
        arc = float(np.sum(np.hypot(np.diff(xg), np.diff(ys_))))
        ext = float(np.hypot(*(P_s - sup_end)) + np.hypot(*(P_d - deep_end)))
        fl_px = arc + ext
        ang_d = math.degrees(math.atan2(D["d"][1], D["d"][0]))
        ang_f = math.degrees(math.atan2(deep_dir[1], deep_dir[0]))
        _, d_all = V.line_from_pts(xg, ys_)
        ang_all = math.degrees(math.atan2(d_all[1], d_all[0]))
        P = np.c_[xg, ys_]
        if side == "left":
            P = P[::-1]  # superficial end first
        # resample every STEP px of arc length
        seg = np.hypot(*np.diff(P, axis=0).T)
        cum = np.r_[0, np.cumsum(seg)]
        tq = np.arange(0, cum[-1] + 1e-9, STEP)
        if tq[-1] < cum[-1]:
            tq = np.r_[tq, cum[-1]]
        Q = np.c_[np.interp(tq, cum, P[:, 0]), np.interp(tq, cum, P[:, 1])]
        out.append(dict(pts=Q, ps=P_s, pd=P_d, fl_px=fl_px, arc_px=arc, ext_px=ext, gap_s=gs, gap_d=gd,
                        pa=abs(float(g2.wrap90(ang_f - ang_d))), pa_all=abs(float(g2.wrap90(ang_all - ang_d))),
                        side=side))
    good = [f for f in out if f["gap_s"] < 25 and f["gap_d"] < 25 and f["arc_px"] > 0.35 * f["fl_px"]
            and f["ext_px"] < 0.12 * f["fl_px"]]
    uniq = []
    for f in sorted(good, key=lambda f: -f["arc_px"]):
        if any(np.hypot(*(f["ps"] - u["ps"])) < 6 and np.hypot(*(f["pd"] - u["pd"])) < 6 for u in uniq):
            continue
        uniq.append(f)
    return dict(S=S, D=D, fas=uniq)


def subject_folds(subj: pd.Series, sets: pd.Series) -> dict:
    rng = np.random.default_rng(SEED)
    out = {}
    for s in sorted(sets.unique()):
        cnt = subj[sets == s].value_counts()
        order = sorted(cnt.index, key=lambda k: (-cnt[k], rng.random()))
        load = np.zeros(N_FOLDS)
        for k in order:
            f = int(np.argmin(load + rng.random(N_FOLDS) * 1e-3))
            out[k] = f
            load[f] += cnt[k]
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    idx = pd.read_csv(VL / "vl_index.csv")
    idx["subject"] = idx["key"].str.split("__").apply(lambda p: f"{p[0]}/{p[1]}")
    folds = subject_folds(idx["subject"], idx["set"])
    recs, n_fail = [], 0
    for _, row in idx.iterrows():
        m = np.asarray(Image.open(VL / "label_masks" / f"{row['key']}.png")) > 0
        res = tracks(m)
        if res is None or not res["fas"]:
            n_fail += 1
            continue
        off = np.array([float(row["offset_x"]), float(row["offset_y"])])
        S, D = res["S"], res["D"]
        rel = row["raw_source"].split("/", 1)[1]
        mt_px = None
        recs.append(dict(
            key=row["key"], rel=rel, set=row["set"], subject=row["subject"], fold=int(folds[row["subject"]]),
            H=int(row["raw_height"]), W=int(row["raw_width"]), ppc=float(row["ann_xres"]),
            sup=[*(np.asarray(S["c"]) - off).round(3).tolist(), *np.asarray(S["d"]).round(6).tolist()],
            deep=[*(np.asarray(D["c"]) - off).round(3).tolist(), *np.asarray(D["d"]).round(6).tolist()],
            tracks=[dict(pts=(f["pts"] - off).round(1).tolist(), ps=(f["ps"] - off).round(2).tolist(),
                         pd=(f["pd"] - off).round(2).tolist(), pa=round(f["pa"], 4), pa_all=round(f["pa_all"], 4),
                         fl_mm=round(f["fl_px"] * 10 / float(row["ann_xres"]), 4), side=f["side"])
                    for f in res["fas"]],
        ))
        del m, res, mt_px
    data = json.dumps(recs, separators=(",", ":")).encode()
    (OUT / "vl_lines.json").write_bytes(data)
    fsrc = ROOT / "artifacts/kaggle/fasc_orient_v1/metrics/folds.csv"
    fb = fsrc.read_bytes()
    (OUT / "fasc_folds.csv").write_bytes(fb)
    n_tr = sum(len(r["tracks"]) for r in recs)
    fold_imgs = pd.Series([r["fold"] for r in recs]).value_counts().sort_index().to_dict()
    summ = dict(n_images=len(recs), n_failed=n_fail, n_tracks=n_tr, n_subjects=int(idx["subject"].nunique()),
                images_per_fold={int(k): int(v) for k, v in fold_imgs.items()},
                sha256={"vl_lines.json": hashlib.sha256(data).hexdigest(), "fasc_folds.csv": hashlib.sha256(fb).hexdigest()},
                bytes={"vl_lines.json": len(data), "fasc_folds.csv": len(fb)},
                source="OSF xbawc 'Example Annotated Images' (CC BY 4.0) label masks, parsed with scripts/vl_labels.py rules",
                step_px=STEP, seed=SEED)
    (OUT / "payload_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
