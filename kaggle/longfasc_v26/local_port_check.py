"""Laptop simulation of the kernel's L0port path (no torch, no Kaggle): unpack a generated kernel.py exactly as its
runner does, import the EMBEDDED geometry_v2.py / lf_readout.py (the package copy is blocked), rebuild the geometry
from the embedded payload, read the v1 orientation maps from the local maskbank as the kernel does, run the readout,
and compare with the laptop L0 run (fl_vwmed, fl_fixed; |d| <= 1e-6 mm, identical NaN pattern: same platform, so
exact). r1: --set vl --geom vl_expert --laptop runs/L0_expert_vl.csv checks the expert-aponeurosis path (L0portexp).

    cd <root> && lockf -k <SP>/laptop.lock env PYTHONPATH=src .venv/bin/python kaggle/longfasc_v26/local_port_check.py \
        --kernel kaggle/longfasc_v26/gpu/kernel.py --laptop artifacts/candidates_v26/runs/L0_prod_test.csv \
        --unpack <SP>/longfasc_v26/unpack [--limit 20]
GPL-3.0-only.
"""
from __future__ import annotations

import argparse
import ast
import base64
import gzip
import hashlib
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
V1_BANK = REPO / "artifacts/kaggle/fasc_orient_v1/maskbank"


def unpack(kernel: Path, dest: Path) -> dict:
    tree = ast.parse(kernel.read_text())
    vals = {t.targets[0].id: ast.literal_eval(t.value) for t in tree.body
            if isinstance(t, ast.Assign) and isinstance(t.targets[0], ast.Name) and t.targets[0].id in ("SHA256", "BLOBS")}
    out = {}
    for name, blob in vals["BLOBS"].items():
        data = zlib.decompress(base64.b64decode(blob))
        ok = hashlib.sha256(data).hexdigest() == vals["SHA256"][name]
        p = dest / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        out[name] = dict(bytes=len(data), sha_ok=ok)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--laptop", required=True, help="laptop L0 run CSV (run_lf.py)")
    ap.add_argument("--unpack", required=True)
    ap.add_argument("--set", default="test")
    ap.add_argument("--geom", default="", help="payload geometry name (default geom_<set>; r1: vl_expert for the expert path)")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    dest = Path(a.unpack)
    files = unpack(Path(a.kernel), dest)
    if not all(v["sha_ok"] for v in files.values()):
        raise SystemExit(f"sha256 mismatch after unpacking: {files}")
    sys.path = [p for p in sys.path if not p.rstrip("/").endswith("/src") and p != "src"]
    sys.modules["umud"] = None  # block the package copy: lf_readout must fall back to the embedded geometry_v2
    sys.path.insert(0, str(dest))
    import geometry_v2 as g2
    import lf_readout as lf
    assert Path(g2.__file__).resolve().parent == dest.resolve(), g2.__file__
    assert lf.g2 is g2
    with gzip.open(dest / f"payload/geom_{a.geom or a.set}.json.gz", "rt") as fh:
        recs = json.load(fh)
    lap = pd.read_csv(a.laptop).drop_duplicates("image_id", keep="last").set_index("image_id")
    bank = g2.MaskBank(V1_BANK)
    rows = []
    for k, rec in enumerate(recs):
        if a.limit and k >= a.limit:
            break
        if rec["status"] != "ok" or rec["image_id"] not in lap.index:
            continue
        seg, c, s = (bank.raw(a.set, f"fasc_orient_v1_{ch}", rec["okey"]) for ch in ("seg", "cos2", "sin2"))
        orient = (g2.as_prob(seg), g2.decode_orient(c), g2.decode_orient(s))
        geom = lf.geom_from_state(rec["geom"])
        r, _ = lf.readout_row(geom, orient)
        L = lap.loc[rec["image_id"]]
        rows.append(dict(image_id=rec["image_id"], kernel_fl_vwmed=r["fl_vwmed"], laptop_fl_vwmed=float(L["fl_vwmed"]),
                         kernel_fl_fixed=r["fl_fixed"], laptop_fl_fixed=float(L["fl_fixed"])))
    bank.close()
    D = pd.DataFrame(rows)
    res = dict(files=files, n=len(D))
    for c in ("fl_vwmed", "fl_fixed"):
        x, y = D[f"kernel_{c}"], D[f"laptop_{c}"]
        m = x.notna() & y.notna()
        res[c] = dict(max_abs=float((x[m] - y[m]).abs().max()) if m.any() else None, nan_mismatch=int((x.isna() != y.isna()).sum()))
    res["passed"] = bool(len(D) > 0 and all(res[c]["max_abs"] is not None and res[c]["max_abs"] <= 1e-6
                                            and res[c]["nan_mismatch"] == 0 for c in ("fl_vwmed", "fl_fixed")))
    print(json.dumps(res, indent=1))
    print(D.to_string())
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
