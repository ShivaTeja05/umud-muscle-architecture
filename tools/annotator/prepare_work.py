"""Build the WORK labelling set from UltraTimTrack videos that are already on this computer.

No network.  One process, one video at a time, frame by frame (well under 1 GB RAM).  The logic is
in kaggle/labelling_frames/kernel.py (the Kaggle kernel runs the same code after it downloads the
files); this script only finds the local files and calls it.

Usage (from the project root):
    .venv/bin/python tools/annotator/prepare_work.py --videos <folder with the pas_005 videos> \
        [--estimates <fascicle_tracking_estimates.zip>] [--out data/manual_labelling] [--force]

Files are recognised by size + md5 (figshare computed_md5), so their names do not matter: every
participant has a file called pas_005_01.mp4.  If --estimates is not given, a zip with the right md5
inside --videos is used.  --map p3=/path/video.mp4 forces a file for one participant.

Writes (data/ is git-ignored):
    <out>/work/img/WRK_XXX.png (+ WRK_XXX_n1.png, WRK_XXX_n2.png view-only neighbours)
    <out>/work/manifest.json, <out>/work/CREDITS.txt
    <out>/private/work_selection.json   (never serve or send)
Then: .venv/bin/python tools/annotator/serve.py --data data/manual_labelling/work --port 8767
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KERNEL = ROOT / "kaggle" / "labelling_frames" / "kernel.py"
DEFAULT_OUT = ROOT / "data" / "manual_labelling"
# Never write the work set into the old test-image annotator folder or the competition data.
FORBIDDEN = [ROOT / "data" / "annotator", ROOT / "data" / "umud-challenge-muscle-architecture-in-ultrasound-data"]


def load_kernel():
    spec = importlib.util.spec_from_file_location("umud_labelling_frames_kernel", KERNEL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_files(folder: Path, kernel) -> tuple[dict[str, Path], Path | None]:
    """{participant: video path} and the estimates zip, by exact size, then md5."""

    by_size: dict[int, list[dict]] = {}
    for entry in kernel.VIDEOS + [kernel.ESTIMATES]:
        by_size.setdefault(entry["size"], []).append(entry)
    videos, zip_path = {}, None
    for path in sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in {".mp4", ".zip"}):
        size = path.stat().st_size
        if size not in by_size:
            continue
        md5 = kernel.md5_file(path)
        for entry in by_size[size]:
            if entry["md5"] != md5:
                continue
            if entry is kernel.ESTIMATES:
                zip_path = path
            else:
                videos.setdefault(entry["participant"], path)
            print(f"  found {entry['participant'] or 'estimates'}: {path} (md5 ok)")
    return videos, zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--videos", type=Path, help="folder with the downloaded pas_005 videos (searched recursively)")
    parser.add_argument("--estimates", type=Path, default=None, help="fascicle_tracking_estimates.zip (scale check)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="kit root (work/ and private/ go inside)")
    parser.add_argument("--map", action="append", default=[], metavar="pN=PATH", help="force a video for one participant")
    parser.add_argument("--trust-files", action="store_true", help="accept --map files whose md5 does not match")
    parser.add_argument("--n-per-video", type=int, default=None, help="frames per video (default 12)")
    parser.add_argument("--budget-mb", type=float, default=78.0, help="drop neighbour frames above this (0 = keep all)")
    parser.add_argument("--force", action="store_true", help="replace an existing work set")
    args = parser.parse_args()

    out = args.out.resolve()
    for bad in FORBIDDEN:
        if out == bad.resolve() or bad.resolve() in out.parents:
            raise SystemExit(f"refusing to write into {bad}")
    if (out / "work" / "manifest.json").exists() and not args.force:
        raise SystemExit(f"{out / 'work' / 'manifest.json'} exists; use --force to rebuild")
    if not args.videos and not args.map:
        raise SystemExit("give --videos <folder> or --map pN=<file>")

    kernel = load_kernel()
    found: dict[str, Path] = {}
    zip_path = args.estimates
    if args.videos:
        if not args.videos.is_dir():
            raise SystemExit(f"not a folder: {args.videos}")
        found, auto_zip = find_files(args.videos, kernel)
        zip_path = zip_path or auto_zip
    pinned = {v["participant"]: v for v in kernel.VIDEOS}
    for item in args.map:
        part, _, raw = item.partition("=")
        path = Path(raw)
        if part not in pinned or not path.is_file():
            raise SystemExit(f"bad --map {item!r}")
        if kernel.md5_file(path) != pinned[part]["md5"] and not args.trust_files:
            raise SystemExit(f"{path}: md5 is not the figshare md5 of {part} {pinned[part]['name']} (use --trust-files)")
        found[part] = path
    missing = [p for p in pinned if p not in found]
    if not found:
        raise SystemExit("no pas_005 video found (they are recognised by size + md5)")
    if missing:
        print(f"WARNING: no video for {missing}; building the set without them")

    estimates = None
    if zip_path:
        print(f"reading scale and estimates from {zip_path}")
        estimates = kernel.read_estimates(zip_path, {p: Path(pinned[p]["name"]).stem for p in found})
    else:
        print("no fascicle_tracking_estimates.zip: scale = nominal 50 mm depth (marked heuristic)")

    videos = [dict(v, path=str(found[v["participant"]])) for v in kernel.VIDEOS if v["participant"] in found]
    summary = kernel.build_work_set(
        videos, out, estimates=estimates, delete_after=False,
        n_per_video=args.n_per_video or kernel.N_PER_VIDEO,
        budget_bytes=None if args.budget_mb <= 0 else int(args.budget_mb * 1e6), clean=args.force)
    print(json.dumps(summary, indent=1))
    if summary["problems"]:
        sys.exit(1)
    print(f"\nDone. Open it with:\n  .venv/bin/python tools/annotator/serve.py --data {out / 'work'} --port 8767\n"
          "  then http://127.0.0.1:8767/")


if __name__ == "__main__":
    main()
