"""Read-only check that no WORK frame is (or looks like) a competition test image.  Run it after the work set is
unzipped and before anyone labels it.

    .venv/bin/python tools/annotator/check_work_overlap.py          # data/manual_labelling/work vs the test set

For every work PNG against every test image (the competition test_set_v2 files and their PNG copies in
data/annotator/img/):
    * file hash (sha256 of the bytes) and pixel hash (sha256 of the grey pixels + size);
    * same height x width (reported only: different recordings can share a frame size);
    * 32x32 grey thumbnail NCC, test image cropped to its ultrasound field when data/annotator/manifest.json gives
      the viewport, and the full test image; work frame as it is and mirrored.  The highest value counts.
Any hash hit, or an NCC above --block (0.95), BLOCKS the work set: work/manifest.json is renamed to
work/manifest.BLOCKED.json, so serve.py cannot open it.  Above --warn (0.85) it only warns.

Read only for every test file: nothing about them is copied or written into the kit.  The report
(private/work_overlap_check.json) holds counts and per-WRK scores only, no test file names; the closest pairs are
printed to the terminal.  One process, one image at a time, about 0.1 GB, no network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
KIT = ROOT / "data" / "manual_labelling"
TEST_DIRS = [ROOT / "data" / "umud-challenge-muscle-architecture-in-ultrasound-data" / "test_images_v2",
             ROOT / "data" / "annotator" / "img"]
TEST_MANIFEST = ROOT / "data" / "annotator" / "manifest.json"  # read only: viewports of the test images
EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}


def _z(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64).ravel()
    return (a - a.mean()) / (a.std() + 1e-9)


def _thumb(gray: Image.Image, box=None) -> np.ndarray:
    g = gray.crop(tuple(box)) if box else gray
    return _z(np.asarray(g.resize((32, 32), Image.BILINEAR)))


def describe(path: Path, viewport=None) -> dict:
    data = path.read_bytes()
    with Image.open(path) as im:
        gray = im.convert("L")
        arr = np.asarray(gray)
        thumbs = [_thumb(gray)]
        if viewport:
            x0, y0, x1, y1 = (int(v) for v in viewport)
            if 0 <= x0 < x1 <= gray.width and 0 <= y0 < y1 <= gray.height:
                thumbs.append(_thumb(gray, (x0, y0, x1, y1)))
    pix = hashlib.sha256(f"{arr.shape}".encode() + arr.tobytes()).hexdigest()
    return {"file": hashlib.sha256(data).hexdigest(), "pixel": pix, "hw": arr.shape[:2], "thumbs": thumbs}


def test_viewports() -> dict[str, list]:
    if not TEST_MANIFEST.exists():
        return {}
    man = json.loads(TEST_MANIFEST.read_text(encoding="utf-8"))
    out = {}
    for e in man.get("images", []):
        if e.get("viewport"):
            out[Path(str(e.get("image_id", ""))).stem] = e["viewport"]
            out[Path(str(e.get("file", ""))).stem] = e["viewport"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--work", type=Path, default=KIT / "work", help="work folder (its PNGs are searched recursively)")
    parser.add_argument("--block", type=float, default=0.95, help="NCC above this blocks the work set")
    parser.add_argument("--warn", type=float, default=0.85, help="NCC above this is reported as a warning")
    parser.add_argument("--no-block", action="store_true", help="report only; never rename the manifest (dry run)")
    parser.add_argument("--report", type=Path, default=KIT / "private" / "work_overlap_check.json")
    args = parser.parse_args()
    t0 = time.time()
    work_files = sorted(p for p in args.work.rglob("*.png") if p.is_file())
    if not work_files:
        raise SystemExit(f"no PNG in {args.work}: unzip the work set first")
    vps = test_viewports()
    test = []
    for d in TEST_DIRS:
        if d.is_dir():
            for p in sorted(d.rglob("*")):
                if p.is_file() and p.suffix.lower() in EXTS:
                    t = describe(p, vps.get(p.stem))
                    t["name"] = p.name
                    test.append(t)
    if not test:
        raise SystemExit("no competition test image found to compare with (expected under data/)")
    print(f"{len(work_files)} work PNGs vs {len(test)} test files", flush=True)
    files = {t["file"] for t in test}
    pixels = {t["pixel"] for t in test}
    sizes = {t["hw"] for t in test}
    bank = np.stack([th for t in test for th in t["thumbs"]])  # rows: test thumbnails
    owner = [t["name"] for t in test for _ in t["thumbs"]]
    per, pairs = {}, []
    hits_file = hits_pixel = same_hw = 0
    for p in work_files:
        w = describe(p)
        hits_file += w["file"] in files
        hits_pixel += w["pixel"] in pixels
        same_hw += w["hw"] in sizes
        th = w["thumbs"][0]
        mirrored = _z(th.reshape(32, 32)[:, ::-1])
        scores = np.maximum(bank @ th, bank @ mirrored) / th.size
        k = int(np.argmax(scores))
        per[p.stem] = round(float(scores[k]), 4)
        pairs.append((float(scores[k]), p.name, owner[k]))
    pairs.sort(reverse=True)
    max_ncc = pairs[0][0]
    blocked = hits_file or hits_pixel or max_ncc > args.block
    status = "BLOCKED" if blocked else ("WARN" if max_ncc > args.warn else "PASS")
    print("closest pairs (work ~ test, NCC):")
    for n, a, b in pairs[:5]:
        print(f"  {a} ~ {b}  {n:.3f}")
    print(f"file-hash hits {hits_file}, pixel-hash hits {hits_pixel}, same HxW as a test image {same_hw}, "
          f"max NCC {max_ncc:.3f} (block > {args.block}, warn > {args.warn})")
    man = args.work / "manifest.json"
    if blocked and not args.no_block and man.exists():
        man.replace(args.work / "manifest.BLOCKED.json")
        print(f"BLOCKED: renamed {man.name} to manifest.BLOCKED.json. Do not label this set; ask Claude to check it.")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "work": str(args.work), "status": status,
        "n_work": len(work_files), "n_test_files": len(test), "file_hash_hits": hits_file, "pixel_hash_hits": hits_pixel,
        "same_hw_as_a_test_image": same_hw, "max_ncc": round(max_ncc, 4), "block_above": args.block,
        "warn_above": args.warn, "dry_run": args.no_block, "max_ncc_per_work_image": per,
        "note": "No test file names are stored here on purpose."}, indent=1), encoding="utf-8")
    print(f"STATUS: {status}   ({time.time() - t0:.0f} s; report {args.report})")


if __name__ == "__main__":
    main()
