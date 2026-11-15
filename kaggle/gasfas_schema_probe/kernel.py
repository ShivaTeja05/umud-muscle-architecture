"""CPU-only schema audit of the public GasFasTrack validation archive.

The preceding metadata audit established that this CC-BY archive contains five
participants, three sessions and three examiners.  This script answers the
remaining eligibility question: whether ``TrackedData.mat`` has independently
usable, documented reference architecture fields.  It emits variable names,
types and shapes only; it does not train, infer, create a competition CSV or
access any competition data.
"""
from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from scipy.io import loadmat


URL = "https://zenodo.org/api/records/2598553/files/validation_data.zip/content"
WORK = Path("/kaggle/working")
ARCHIVE = WORK / "validation_data.zip"
OUT = WORK / "gasfas_schema_report.json"
CHUNK = 8 * 1024 * 1024


def download() -> dict:
    t0 = time.time()
    req = Request(URL, headers={"User-Agent": "UMUD-GasFasTrack-schema-audit/1.0"})
    with urlopen(req, timeout=120) as response, ARCHIVE.open("wb") as fh:
        while True:
            block = response.read(CHUNK)
            if not block:
                break
            fh.write(block)
    return {"bytes": ARCHIVE.stat().st_size, "elapsed_s": round(time.time() - t0, 1)}


def describe(value: object, depth: int = 0) -> dict:
    """Return an intentionally value-free structural description."""
    result = {"type": type(value).__name__}
    if isinstance(value, np.ndarray):
        result["shape"] = list(value.shape)
        result["dtype"] = str(value.dtype)
        if value.dtype.names:
            result["fields"] = list(value.dtype.names)
        if value.dtype == object and depth < 7 and value.size:
            result["first_item"] = describe(value.flat[0], depth + 1)
    elif hasattr(value, "_fieldnames"):
        result["fields"] = list(value._fieldnames or [])
        if depth < 7:
            result["children"] = {k: describe(getattr(value, k), depth + 1) for k in value._fieldnames or []}
    return result


def inspect_mat(raw: bytes) -> dict:
    try:
        data = loadmat(io.BytesIO(raw), squeeze_me=True, struct_as_record=False)
        return {
            "reader": "scipy.io.loadmat",
            "variables": {k: describe(v) for k, v in data.items() if not k.startswith("__")},
        }
    except Exception as exc:
        # MATLAB v7.3 files are HDF5 and need a separate parser later.  Naming
        # that fact is valuable; do not silently claim labels were recovered.
        return {"reader": "unreadable_by_scipy", "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    cached = sorted(Path('/kaggle/input').rglob('validation_data.zip'))
    archive = cached[0] if cached else ARCHIVE
    dl = {"reused_input": str(archive)} if cached else download()
    with zipfile.ZipFile(archive) as zf:
        names = [info.filename for info in zf.infolist() if not info.is_dir()]
        root_mats = [
            name for name in names
            if name.endswith("/TrackedData.mat") and name.count("/") == 4
        ]
        with zipfile.ZipFile(WORK / 'gasfas_mat_files.zip', 'w', zipfile.ZIP_DEFLATED) as exported:
            for name in names:
                if name.lower().endswith(('.mat', '.txt')):
                    exported.writestr(name, zf.read(name))
        # One root TrackedData file for each subject × day × examiner.  The
        # parent path is preserved to prove held-out-participant splits later.
        report = {
            "purpose": "schema-only external-data eligibility audit; no model training or UMUD submission",
            "source": "https://zenodo.org/records/2598553",
            "license": "CC-BY-4.0",
            "download": dl,
            "readme": zf.read("validation_data/readme.txt").decode("utf-8", errors="replace")[:12000],
            "n_root_trackeddata": len(root_mats),
            "files": {name: inspect_mat(zf.read(name)) for name in sorted(root_mats)},
        }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"n_root_trackeddata": report["n_root_trackeddata"], "download": dl}, indent=2))
    for name, schema in report["files"].items():
        print(name)
        print(json.dumps(schema, indent=2)[:3000])


if __name__ == "__main__":
    main()
