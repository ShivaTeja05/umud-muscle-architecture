# UMUD Challenge solution code -- v30 S5 direct regressor: payload builder (laptop, light)
# Copyright (C) 2026 Gaddam Shivateja
# SPDX-License-Identifier: GPL-3.0-only
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 (GPL-3.0-only). It is distributed in the hope that
# it will be useful, but WITHOUT ANY WARRANTY; see <https://www.gnu.org/licenses/gpl-3.0.txt>.
"""Build the small embedded payload of the S5 kernel (numpy/pandas/PIL only, one file at a time, < 0.3 GB RSS).

    lockf -k <scratchpad>/laptop.lock env PYTHONPATH=src .venv/bin/python kaggle/s5_direct_v30/build_payload.py

Writes kaggle/s5_direct_v30/payload/:
  vl_lines.json   byte copy of kaggle/v16_insfield/payload/vl_lines.json (OSF xbawc 'Example Annotated Images', CC BY 4.0;
                  expert sup/deep aponeurosis lines + fascicle tracks, subject folds); sha256 checked against the v16
                  payload_summary.json.
  tt_labels.csv   TimTrack paper datasets 3 and 4 (CC BY 4.0): one row per (image, observer) with BOTH the fascicle
                  angle alpha and the deep-aponeurosis angle gamma (deg vs the image horizontal, + = rising to the right),
                  from artifacts/external/timtrack_bench/labels.csv; plus the image file's sha256 / bytes / size from the
                  local copy of repo timvanderzee/ultrasound-automated-algorithm @ d3ba5e2b (the kernel downloads the same
                  files from that commit and checks the sha256).
  dyn_labels.csv  UMUD GM dynamic calf-raise benchmark (OSF xbawc, CC BY 4.0): FL only (rater mean, px -> mm with
                  77.4 px/cm, as scripts/build_candidates.py dynamic_gt). PA is NOT exported (masked by design).
  geom.csv        inference geometry only (no label): static35 px_per_cm (bench_frame), test309 group / px_per_cm
                  (artifacts/candidates_v26/inputs/test_state.csv, read from the test images by our pipeline) and the
                  label-space anisotropy c (src/umud/native_geometry.NATIVE_TABLE: Siemens / Lumify native, others 1).
  ids.json        static35 / clean10 / copies25 / test309 ids, expected counts, source URLs, and the laptop guard result
                  (artifacts/candidates_v30/s5/out/i05_guard_local.json): exact mirror copies of static35 inside
                  TimTrack ds4 and the dynamic near-copy window of im_35_arch that the kernel guard must drop.
  payload_summary.json  sha256 + bytes of every payload file.
No competition image or label, and no static35 label, is exported.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "payload"
V16 = ROOT / "kaggle/v16_insfield/payload"
TT_DIR = ROOT / "artifacts/external/timtrack_repo/Example images/raw/TimTrack paper"
TT_SHA = "d3ba5e2b75fc57870e7a75b2fa68b5b7f53b10e9"
DYN_X = (ROOT / "data/external_umud_benchmark/dynamic/benchmark_dataset_architecture_GM_dynamic_v0.1.0/"
         "benchmark_architecture_GM_calf_raise_v0.1.0.xlsx")
DYN_PPC = 77.4
CLEAN10 = [f"im_{k:02d}_arch" for k in (2, 3, 14, 17, 20, 23, 24, 31, 33, 34)]
NATIVE_C = {"siemens_juniper": (1200 / 1152) / (800 / 864), "philips_lumify": (1200 / 800) / (16 / 9)}  # = NATIVE_TABLE


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files: dict[str, bytes] = {}
    # ---- VL expert lines (byte copy, sha-checked)
    vl = (V16 / "vl_lines.json").read_bytes()
    want = json.loads((V16 / "payload_summary.json").read_text())["sha256"]["vl_lines.json"]
    if sha(vl) != want:
        raise SystemExit("v16 vl_lines.json sha256 mismatch")
    files["vl_lines.json"] = vl
    recs = json.loads(vl)
    n_vl2 = sum(len(r["tracks"]) >= 2 for r in recs)
    # ---- TimTrack tt3 / tt4 observer angles + file hashes
    L = pd.read_csv(ROOT / "artifacts/external/timtrack_bench/labels.csv")
    T = L[L.subset.isin(["tt3", "tt4a", "tt4b"]) & L.exists.astype(bool)
          & np.isfinite(L.alpha) & np.isfinite(L.gamma)].copy()
    meta = {}
    for f in sorted(T.file.unique()):
        b = (TT_DIR / f).read_bytes()
        with Image.open(TT_DIR / f) as im:
            w, h = im.size
        meta[f] = (sha(b), len(b), w, h)
    T["file_sha256"] = T.file.map(lambda f: meta[f][0])
    T["file_bytes"] = T.file.map(lambda f: meta[f][1])
    T["W"] = T.file.map(lambda f: meta[f][2])
    T["H"] = T.file.map(lambda f: meta[f][3])
    T = T[["subset", "device", "file", "key", "observer", "alpha", "gamma", "file_sha256", "file_bytes", "W", "H"]]
    T = T.sort_values(["file", "observer"]).reset_index(drop=True)
    files["tt_labels.csv"] = T.to_csv(index=False, float_format="%.6f").encode()
    # ---- dynamic FL only
    fl = pd.read_excel(DYN_X, sheet_name="FL")[["frame", "Mean"]].dropna(subset=["frame", "Mean"])
    D = pd.DataFrame({"image_id": fl.frame.astype(int).map(lambda k: f"frame_{k:05d}"), "fl_px": fl["Mean"].astype(float)})
    D["px_per_cm"] = DYN_PPC
    D["fl_mm"] = D.fl_px * 10 / DYN_PPC
    files["dyn_labels.csv"] = D.to_csv(index=False, float_format="%.6f").encode()
    # ---- inference geometry (no labels)
    B = pd.read_csv(ROOT / "artifacts/candidates_v26/inputs/bench_frame.csv")
    S = B[B.dataset == "static"][["image_id", "px_per_cm", "group"]].copy()
    S["dataset"], S["c_label"] = "static", 1.0
    TS = pd.read_csv(ROOT / "artifacts/candidates_v26/inputs/test_state.csv")[["image_id", "px_per_cm", "group", "clip"]]
    TS["dataset"] = "test"
    TS["c_label"] = TS.group.map(lambda g: NATIVE_C.get(g, 1.0))
    G = pd.concat([S.assign(clip=-1), TS], ignore_index=True)[["dataset", "image_id", "group", "clip", "px_per_cm", "c_label"]]
    files["geom.csv"] = G.to_csv(index=False, float_format="%.9f").encode()
    # ---- ids / expected counts / sources
    A = pd.read_csv(ROOT / "artifacts/ready/APLUS_v2_FinalA0.21637_LUMPA+0.5_LB0.21529.csv")
    static35 = sorted(S.image_id)
    g5 = json.loads((ROOT / "artifacts/candidates_v30/s5/out/i05_guard_local.json").read_text())  # laptop guard run
    ids = {
        "static35": static35, "clean10": CLEAN10, "copies25": sorted(set(static35) - set(CLEAN10)),
        "test309": sorted(A.image_id.astype(str)),
        "expected_drop_exact": g5["expected_drop_exact"], "expected_drop_all_local": g5["dropped_ids"],
        "dup_ncc_local": g5["DUP_NCC"],
        "expected": {"vl_payload": len(recs), "vl_used_ntracks_ge2": int(n_vl2), "tt_images": int(T.file.nunique()),
                     "tt_rows": int(len(T)), "tt_by_subset": {k: int(v) for k, v in T.groupby("subset").file.nunique().items()},
                     "dyn_frames_zip": 168, "dyn_fl_labels": int(len(D)), "static": 35, "test": 309},
        "sources": {
            "vl": ["https://osf.io/download/zcuwk/", "https://osf.io/download/6ga94/"],
            "static": ["https://osf.io/download/hb5xj/"], "dynamic": ["https://osf.io/download/vqb7h/"],
            "timtrack_raw_base": f"https://raw.githubusercontent.com/timvanderzee/ultrasound-automated-algorithm/{TT_SHA}/"
                                 "Example%20images/raw/TimTrack%20paper/",
            "timtrack_commit": TT_SHA},
    }
    assert len(ids["test309"]) == 309 and sorted(TS.image_id.astype(str)) == ids["test309"]
    assert len(static35) == 35 and len(ids["copies25"]) == 25 and set(CLEAN10) <= set(static35)
    files["ids.json"] = json.dumps(ids, indent=1).encode()
    for k, v in files.items():
        (OUT / k).write_bytes(v)
    summ = {"files": {k: {"sha256": sha(v), "bytes": len(v)} for k, v in files.items()},
            "inputs_sha256": {
                "kaggle/v16_insfield/payload/vl_lines.json": sha(vl),
                "artifacts/external/timtrack_bench/labels.csv": sha((ROOT / "artifacts/external/timtrack_bench/labels.csv").read_bytes()),
                "dynamic xlsx": sha(DYN_X.read_bytes()),
                "artifacts/candidates_v26/inputs/bench_frame.csv": sha((ROOT / "artifacts/candidates_v26/inputs/bench_frame.csv").read_bytes()),
                "artifacts/candidates_v26/inputs/test_state.csv": sha((ROOT / "artifacts/candidates_v26/inputs/test_state.csv").read_bytes()),
                "artifacts/ready/APLUS_v2 (ids only)": sha((ROOT / "artifacts/ready/APLUS_v2_FinalA0.21637_LUMPA+0.5_LB0.21529.csv").read_bytes())},
            "expected": ids["expected"]}
    (OUT / "payload_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
