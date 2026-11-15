"""LF-26 kernel payload: production geometry of every evaluation image, exported on the laptop (no labels, no LF).

    cd <root> && lockf -k <SP>/laptop.lock env PYTHONPATH=src .venv/bin/python kaggle/longfasc_v26/export_payload.py \
        --set test        # then static, dynamic, vl (4 chunks, each < 2 min)
    ... export_payload.py --summary   # payload_summary.json (sha256, counts) after the 4 chunks

Per image (test 309, static 35, dynamic 168, VL 249): dataset, image_id, orient key (MASKBANK key of the v1 / LFNet
banks), the lf_readout geometry state (EdgeCurve evaluation state of the superficial / deep inner edges incl. the
straight-line fit of the band points, s_lo, s_hi, mt_px, mm, H, W, viewport x-range, production fragments cx, cy, vx,
vy, length) -> payload/geom_<set>.json.gz. VL additionally: the expert-aponeurosis geometry (lines only, see env_lf)
-> payload/geom_vl_expert.json.gz, and the K0d sample points (x, y along the expert fascicle tracks in the deepest 30 %
of MT) -> payload/k0d_points_vl.json.gz; the track angles of those points stay on the laptop
(artifacts/candidates_v26/runs/k0d_points_vl.csv). No FL / PA / MT value is exported.
Exactness: every exported curve (r1: incl. the VL expert-aponeurosis curves) is rebuilt (lf_readout.curve_from_state)
and compared bit-for-bit on a grid here.
GPL-3.0-only.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CODE = REPO / "artifacts/candidates_v26/code"
sys.path.insert(0, str(CODE))
import env_lf as E  # noqa: E402
import lf_readout as lf  # noqa: E402

PAYLOAD = HERE / "payload"
EXPECTED = {"test": 309, "static": 35, "dynamic": 168, "vl": 249}


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def write_gz(path: Path, obj) -> str:
    raw = json.dumps(obj, separators=(",", ":")).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        with gzip.GzipFile(fileobj=fh, mode="wb", mtime=0, compresslevel=9) as gz:
            gz.write(raw)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_curve(c) -> bool:
    c2 = lf.curve_from_state(json.loads(json.dumps(lf.curve_state(c))))
    grid = np.linspace(-6000, 8000, 4001)
    return bool(np.array_equal(c(grid), c2(grid)) and np.array_equal(c.slope(grid), c2.slope(grid))
                and lf.deep_line_slope(c) == lf.deep_line_slope(c2))


def export_set(ds: str, ids: set | None, out_dir: Path, limit: int = 0) -> dict:
    t0 = time.time()
    env = E.LFEnv("v1")
    recs, exp_recs, kpts, kpts_local, bad, bad_exp = [], [], {}, [], [], []
    k = 0
    try:
        for rec in env.iterate(ds, ids=ids):
            if limit and k >= limit:
                break
            k += 1
            row = dict(set=ds, image_id=rec["iid"], okey=rec["okey"], status=rec["status"])
            g = rec["geom"]
            if rec["status"] == "ok" and g is not None:
                if not (exact_curve(g["sup"]) and exact_curve(g["deep"])):
                    bad.append(rec["iid"])
                row["geom"] = lf.geom_state(g)
                if ds == "vl":
                    ge = env.expert_geom(rec["iid"], g)
                    if ge is not None:
                        if not (exact_curve(ge["sup"]) and exact_curve(ge["deep"])):  # r1: expert curves too
                            bad_exp.append(rec["iid"])
                        exp_recs.append(dict(set=ds, image_id=rec["iid"], okey=rec["okey"], status="ok",
                                             geom=lf.geom_state(ge)))
                    else:
                        exp_recs.append(dict(set=ds, image_id=rec["iid"], okey=rec["okey"], status="no_expert_apo"))
                    for j, (trk, x, y, ang) in enumerate(env.k0d_points(rec["iid"])):
                        pid = f"{trk}_{j}"
                        kpts.setdefault(rec["okey"], []).append([pid, x, y])
                        kpts_local.append(dict(image_id=rec["iid"], okey=rec["okey"], pid=pid, track=trk, x=x, y=y,
                                               track_angle=ang))
            recs.append(row)
    finally:
        env.close()
    out = dict(set=ds, n=len(recs), n_ok=sum(r["status"] == "ok" for r in recs), inexact_curves=bad,
               sec=round(time.time() - t0, 1), rss_gb=round(rss_gb(), 2))
    out["sha256"] = {f"geom_{ds}.json.gz": write_gz(out_dir / f"geom_{ds}.json.gz", recs)}
    if ds == "vl":
        out["sha256"]["geom_vl_expert.json.gz"] = write_gz(out_dir / "geom_vl_expert.json.gz", exp_recs)
        out["sha256"]["k0d_points_vl.json.gz"] = write_gz(out_dir / "k0d_points_vl.json.gz", kpts)
        out["n_expert_ok"] = sum(r["status"] == "ok" for r in exp_recs)
        out["inexact_expert_curves"] = bad_exp
        out["n_expert_curves_checked"] = 2 * out["n_expert_ok"]
        out["n_k0d_points"] = int(sum(len(v) for v in kpts.values()))
        local = REPO / "artifacts/candidates_v26/runs" / ("k0d_points_vl.csv" if out_dir == PAYLOAD else
                                                           "k0d_points_vl_smoke.csv")
        local.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(kpts_local).to_csv(local, index=False)
        out["k0d_local"] = str(local)
    (out_dir / f"export_{ds}.json").write_text(json.dumps(out, indent=1))
    return out


def summary(out_dir: Path) -> dict:
    files = sorted(p.name for p in out_dir.glob("*.json.gz"))
    rep = dict(written_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), readout_version=lf.VERSION,
               files={f: dict(bytes=(out_dir / f).stat().st_size, sha256=hashlib.sha256((out_dir / f).read_bytes()).hexdigest())
                      for f in files},
               code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                            (REPO / "src/umud/geometry_v2.py", CODE / "lf_readout.py", CODE / "env_lf.py", Path(__file__))},
               exports={})
    for ds in EXPECTED:
        p = out_dir / f"export_{ds}.json"
        if p.exists():
            rep["exports"][ds] = json.loads(p.read_text())
    v1b = REPO / "artifacts/kaggle/fasc_orient_v1/maskbank"   # the kernel checks its attached copy against these
    rep["v1_bank_sha256"] = {f"{ds}__fasc_orient_v1_{ch}.npz": hashlib.sha256((v1b / f"{ds}__fasc_orient_v1_{ch}.npz")
                                                                             .read_bytes()).hexdigest()
                             for ds in EXPECTED for ch in ("seg", "cos2", "sin2")}
    rep["complete"] = bool(all(rep["exports"].get(ds, {}).get("n") == n for ds, n in EXPECTED.items())
                           and rep["exports"].get("vl", {}).get("n_expert_ok", 0) > 200
                           and not any(rep["exports"][ds]["inexact_curves"] for ds in rep["exports"])
                           and "inexact_expert_curves" in rep["exports"].get("vl", {})  # r1: expert curves checked
                           and not rep["exports"]["vl"]["inexact_expert_curves"])
    rep["bytes_total"] = int(sum(v["bytes"] for v in rep["files"].values()))
    (out_dir / "payload_summary.json").write_text(json.dumps(rep, indent=1))
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=list(EXPECTED))
    ap.add_argument("--ids", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default=str(PAYLOAD))
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args(argv)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.set:
        ids = set(x for x in a.ids.split(",") if x) or None
        print(json.dumps(export_set(a.set, ids, out_dir, a.limit), indent=1))
    if a.summary or not a.set:
        rep = summary(out_dir)
        print(json.dumps({k: rep[k] for k in ("complete", "bytes_total", "files")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
