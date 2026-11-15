"""Rebuild the two selected submissions from the bundled code and per-image measurement tables. GPL-3.0-only."""
import hashlib, json, shutil, sys
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["pa_deg", "fl_mm", "mt_mm"]
BOUNDS = {"pa_deg": (5.0, 45.0), "fl_mm": (30.0, 200.0), "mt_mm": (10.0, 50.0)}
CELLS_A = {"telemed_artus_1200x800": {"pa_deg": -0.75, "mt_mm": 0.325},
           "telemed_artus_1088x644": {"pa_deg": 1.1},
           "siemens_juniper": {"pa_deg": 0.85}}
CELLS_APLUS = {**CELLS_A, "philips_lumify": {"pa_deg": 0.5}}
PINS = {"final_A": "5d216fdaf1ed6b6e9da95c671e943a75465d4bb8c44f6d4f7ec6ba754ec332bf",
        "final_A_lb": "1bf9fe4ab3b199f4a5200a919e6ceb5c7e9f203d85a1c1fa23923f0c07f2486e",
        "A_plus_v2": "8fd252872b4f03dfa8e5e8fccb76087668d33a33c20e29d6b3965ac0318a619d"}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def cell_constants(raw, group_of, cells):
    """Add one constant per (scanner group, target) to the written 4-decimal file."""
    lines = raw[:-2].split(b"\r\n")
    for i in range(1, len(lines)):
        p = lines[i].decode().split(",")
        for c, x in cells.get(group_of[p[0]], {}).items():
            j = 1 + COLS.index(c)
            lo, hi = BOUNDS[c]
            p[j] = repr(float(np.round(np.clip(float(p[j]) + x, lo + 0.001, hi - 0.001), 4)))
        lines[i] = ",".join(p).encode()
    return b"\r\n".join(lines) + b"\r\n"


def rebuild(src_repo, work, test_images):
    repo = Path(work) / "repo"
    if not repo.exists():
        shutil.copytree(src_repo, repo)  # the build writes next to its inputs; the input mount is read-only
    sys.path.insert(0, str(repo / "src"))
    from umud import layout as lm
    from umud import run as ur
    out = Path(work) / "out"
    lay_path = out / "test_layout.csv"
    out.mkdir(parents=True, exist_ok=True)  # the layout (scanner groups, clips) is recomputed from the images
    d = out / "final_A"
    code = ur.main(["--config", str(repo / "configs/final_A.yaml"), "--stage", "build", "--out", str(d),
                    "--set", f"paths.out_dir={json.dumps(str(d))}",
                    "--set", f"paths.test_layout={json.dumps(str(lay_path))}",
                    "--set", f"paths.test_images={json.dumps(str(test_images))}",
                    "--set", "output.validate=false"])
    raw = (d / "final_A.csv").read_bytes()
    lay = lm.read_layout(lay_path)
    group_of = {str(k): str(v) for k, v in lay["group"].items()}
    files = {"final_A": raw, "final_A_lb": cell_constants(raw, group_of, CELLS_A),
             "A_plus_v2": cell_constants(raw, group_of, CELLS_APLUS)}
    rep = {}
    for k, b in files.items():
        (out / f"{k}.csv").write_bytes(b)
        new = pd.read_csv(out / f"{k}.csv").set_index("image_id")
        ref = pd.read_csv(repo / "reference" / f"{k}.csv").set_index("image_id").reindex(new.index)
        diff = (new - ref).abs()
        rep[k] = {"byte_identical_to_scored_file": sha(b) == PINS[k],
                  "values_that_differ": int((diff > 0).sum().sum()), "of": int(diff.size),
                  "largest_difference": float(diff.max().max())}
    return out, lay, rep


if __name__ == "__main__":
    o, _, r = rebuild(sys.argv[1], sys.argv[2], sys.argv[3])
    print(json.dumps(r, indent=1))
