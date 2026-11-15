"""No-GPU data-structure probe for the public GasFasTrack validation archive.

This private Kaggle CPU kernel downloads the CC-BY-4.0 Zenodo record's
``validation_data.zip`` (not competition data), lists only metadata and likely
manual-label paths, then writes a small JSON manifest.  It deliberately does
not train, infer, generate a competition CSV, or inspect any competition test
label.

The point is to verify that GasFasTrack has participant IDs and separately
available manual PA/FL values before allocating a GPU experiment.  The archive
is ~1.9 GB, so this runs in Kaggle rather than consuming paid infrastructure.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen


URL = "https://zenodo.org/api/records/2598553/files/validation_data.zip/content"
WORK = Path("/kaggle/working")
ARCHIVE = WORK / "validation_data.zip"
OUT = WORK / "gasfas_probe_manifest.json"
CHUNK = 8 * 1024 * 1024


def download() -> dict:
    start = ARCHIVE.stat().st_size if ARCHIVE.exists() else 0
    headers = {"User-Agent": "UMUD-GasFasTrack-metadata-probe/1.0"}
    if start:
        headers["Range"] = f"bytes={start}-"
    req = Request(URL, headers=headers)
    t0 = time.time()
    with urlopen(req, timeout=120) as response:
        status = getattr(response, "status", None)
        # A server that ignores Range returns a whole archive.  Start afresh to
        # avoid silently appending it to an existing partial file.
        mode = "ab" if start and status == 206 else "wb"
        if mode == "wb":
            start = 0
        with ARCHIVE.open(mode) as fh:
            written = start
            while True:
                block = response.read(CHUNK)
                if not block:
                    break
                fh.write(block)
                written += len(block)
                if written == len(block) or written // (256 * 1024**2) != (written - len(block)) // (256 * 1024**2):
                    print(f"downloaded {written / 1024**3:.2f} GiB", flush=True)
    return {"bytes": ARCHIVE.stat().st_size, "elapsed_s": round(time.time() - t0, 1), "http_status": status}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> None:
    dl = download()
    with zipfile.ZipFile(ARCHIVE) as zf:
        infos = zf.infolist()
    names = [x.filename for x in infos if not x.is_dir()]
    suffixes = (".mat", ".csv", ".xlsx", ".json", ".txt", ".tsv")
    label_words = ("manual", "valid", "track", "fasc", "angle", "length", "architect", "result")
    structured = [x for x in names if x.lower().endswith(suffixes)]
    likely = [x for x in names if any(w in x.lower() for w in label_words)]
    report = {
        "purpose": "metadata-only eligibility and structure probe; no training or UMUD submission",
        "source": {
            "record": "https://zenodo.org/records/2598553",
            "file": "validation_data.zip",
            "license": "CC-BY-4.0 (record metadata verified before this run)",
        },
        "download": dl,
        "archive_sha256": sha256(ARCHIVE),
        "n_files": len(names),
        "uncompressed_bytes": sum(x.file_size for x in infos),
        "structured_paths": structured,
        "likely_label_paths": likely,
        "top_level": sorted({x.split("/", 1)[0] for x in names}),
        "next_step": "Only parse labelled files after confirming distinct participant IDs and manual PA/FL fields.",
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("download", "archive_sha256", "n_files", "top_level")}, indent=2))
    print("\nLIKELY LABEL PATHS")
    print("\n".join(likely[:500]))


if __name__ == "__main__":
    main()
