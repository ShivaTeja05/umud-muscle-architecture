"""Laptop self-test of the torch-free parts of train_template.py (targets, augmentation, FALLMUD groups); no torch.

    cd <root> && lockf -k <SP>/laptop.lock env PYTHONPATH=src .venv/bin/python kaggle/longfasc_v26/test_template_targets.py

The template has no torch import at module level, so it is imported as a module. Checks:
  1. competition sample (v1 path): keys / dtypes / shapes identical to the FALLMUD sample (collation), path_t = 0;
  2. FALLMUD sample of a synthetic 30 deg path field: ridge max ~1 on the skeleton, ~0 far away; IDW orientation
     target = the path angle (doubled-angle error < 1 deg) where supervised, coverage within 16 px;
  3. the FALLMUD FOV crop draws only for FALLMUD samples (competition RNG stream == v1's);
  4. fm_groups: contiguous synthetic sequences are recovered as blocks; hold-out >= 20 % of the frames.
GPL-3.0-only.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lf26_train", HERE / "train_template.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
assert "torch" not in sys.modules, "the template must not import torch at module level"


def synth_fm(h=436, w=528, ang=30.0, spacing=40.0):
    """Parallel straight paths at `ang` deg (image coords), skeleton points every 1 px, v3-style pieces every 4 px."""
    th = math.radians(ang)
    d = np.array([math.cos(th), math.sin(th)])
    nrm = np.array([-d[1], d[0]])
    skel, segs = [], []
    for off in np.arange(-800, 800, spacing):
        c = np.array([w / 2, h / 2]) + off * nrm
        t = np.arange(-900, 900, 1.0)
        P = c[None] + t[:, None] * d[None]
        P = P[(P[:, 0] >= 0) & (P[:, 0] < w - 1) & (P[:, 1] >= 0) & (P[:, 1] < h - 1)]
        skel.append(P)
        for q in P[::4]:
            segs.append([q[0] - 4 * d[0], q[1] - 4 * d[1], q[0] + 4 * d[0], q[1] + 4 * d[1]])
    rng = np.random.default_rng(0)
    crop = (rng.random((h, w)) * 255).astype(np.uint8)
    return crop, np.array(segs, np.float32), np.concatenate(skel).astype(np.float32)


def main() -> int:
    out, ok = {}, True
    crop, segs, skel = synth_fm()
    comp = T.make_sample(crop, segs[:50], np.random.default_rng(1))
    fm = T.make_sample(crop, segs, np.random.default_rng(1), skel)
    same = {k: (comp[k].dtype == fm[k].dtype and comp[k].shape == fm[k].shape) for k in comp}
    out["collate_compatible"] = all(same.values()) and set(comp) == set(fm)
    out["comp_path_zero"] = bool(float(np.abs(comp["path_t"].astype(np.float32)).max()) == 0.0)
    # un-augmented FALLMUD sample: exact geometry
    fm0 = T.make_sample(crop, segs, None, skel)
    pt = fm0["path_t"].astype(np.float32)
    out["ridge_max"] = float(pt.max())
    out["ridge_frac_gt_0.5"] = float((pt > 0.5).mean())
    om = fm0["om"] > 0
    th = 0.5 * np.degrees(np.arctan2(fm0["sin2"].astype(np.float32), fm0["cos2"].astype(np.float32)))
    err = np.abs(((th - 30.0) + 90) % 180 - 90)[om]
    from scipy import ndimage
    v = fm0["valid"] > 0
    dist = ndimage.distance_transform_edt(pt < 0.99)          # distance to the rasterised skeleton (ridge == 1)
    out["ori_supervised_frac"] = float(om[v].mean())
    out["ori_expected_frac_within_R"] = float((dist[v] <= T.FM_IDW_R).mean())   # paths 40 px apart, R 16 -> ~0.8
    out["ori_err_p99_deg"] = float(np.percentile(err, 99)) if err.size else None
    ok &= out["collate_compatible"] and out["comp_path_zero"] and out["ridge_max"] > 0.9
    ok &= out["ori_err_p99_deg"] is not None and out["ori_err_p99_deg"] < 1.0
    ok &= abs(out["ori_supervised_frac"] - out["ori_expected_frac_within_R"]) < 0.05
    # competition samples are bit-identical to the template origin (kaggle/fasc_orient_v3/train.py = v1 recipe)
    spec3 = importlib.util.spec_from_file_location("fo3_train", HERE.parent / "fasc_orient_v3/train.py")
    V3 = importlib.util.module_from_spec(spec3)
    spec3.loader.exec_module(V3)
    same_v3 = []
    for seed in range(12):
        a = T.make_sample(crop, segs[:50], np.random.default_rng(seed))
        b = V3.make_sample(crop, segs[:50], np.random.default_rng(seed))
        same_v3.append(all(np.array_equal(a[k], b[k]) for k in b))
    out["comp_samples_equal_v3_12_seeds"] = bool(all(same_v3))
    ok &= out["comp_samples_equal_v3_12_seeds"]
    # fm_groups on synthetic sequences: 6 subjects x 80 frames (consecutive frames highly correlated)
    rng = np.random.default_rng(3)
    fms = []
    for s in range(6):
        base = rng.standard_normal(T.THUMB * T.THUMB)
        for i in range(80):
            v = base + 0.05 * i / 80 * rng.standard_normal(base.size) + 0.02 * rng.standard_normal(base.size)
            v = (v - v.mean()) / np.linalg.norm(v - v.mean())
            fms.append(dict(author="Cunningham", key=f"Cunningham_{s * 80 + i}", fm_index=s * 80 + i, thumb_full=v))
    g = T.fm_groups(fms)
    out["fm_groups"] = {k: g[k] for k in ("method", "n_groups", "n_holdout", "n_train", "holdout_groups")}
    ok &= g["method"] == "index_blocks" and g["n_groups"] == 6 and g["n_holdout"] >= 0.2 * len(fms)
    out["passed"] = bool(ok)
    print(json.dumps(out, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
