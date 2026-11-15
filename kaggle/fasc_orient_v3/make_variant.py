"""Write a pushable Kaggle kernel directory that is a variant of kaggle/fasc_orient_v3 (adapted from
kaggle/fasc_orient_v2/make_variant.py).

Only top-level `NAME = value` constants of train.py are rewritten (exactly one match required), plus the kernel id,
title, kernel_sources and GPU flags in kernel-metadata.json. The source directory is never modified.

v21 track A (run from the project root; python3 is enough, no third-party imports):
  python3 kaggle/fasc_orient_v3/make_variant.py kaggle/fasc_orient_v3 kaggle/fasc_orient_v3/variants/cnx_fm \
      --id shivateja005/umud-fasc-orient-v3-cnx-fm --set BACKBONE='"convnext"' --set MODEL_NAME='"fasc_orient_v3cfm"'
  python3 kaggle/fasc_orient_v3/make_variant.py kaggle/fasc_orient_v3 kaggle/fasc_orient_v3/variants/dino_fm_cpusmoke \
      --cpu --id shivateja005/umud-fasc-orient-v3-dino-fm-cpusmoke
  python3 kaggle/fasc_orient_v3/make_variant.py kaggle/fasc_orient_v3 kaggle/fasc_orient_v3/variants/cnx_fm_cpusmoke \
      --cpu --id shivateja005/umud-fasc-orient-v3-cnx-fm-cpusmoke --set BACKBONE='"convnext"' \
      --set MODEL_NAME='"fasc_orient_v3cfm"'
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--id", required=True, help="new kernel id owner/slug")
    ap.add_argument("--title", default=None, help="kernel title (default: derived from the slug)")
    ap.add_argument("--set", action="append", default=[], help="NAME=python_literal (top-level constant)")
    ap.add_argument("--sources", nargs="*", default=None, help="replace kernel_sources")
    ap.add_argument("--cpu", action="store_true", help='CPU smoke: MODE="cpusmoke", enable_gpu false')
    a = ap.parse_args()
    code = (a.src / "train.py").read_text()
    meta = json.loads((a.src / "kernel-metadata.json").read_text())
    sets = list(a.set)
    if a.cpu:
        sets.append('MODE="cpusmoke"')
    for kv in sets:
        name, val = kv.split("=", 1)
        pat = re.compile(rf"^{re.escape(name)} = [^#\n]*?(\s*#.*)?$", re.M)
        hits = pat.findall(code)
        if len(hits) != 1:
            raise SystemExit(f"constant {name}: expected exactly one top-level assignment, found {len(hits)}")
        code = pat.sub(lambda m: f"{name} = {val}" + (m.group(1) or ""), code, count=1)
    slug = a.id.split("/", 1)[1]
    meta["id"] = a.id
    meta["title"] = a.title or " ".join(w.upper() if w == "umud" else w.capitalize() for w in slug.split("-"))
    if a.sources is not None:
        meta["kernel_sources"] = a.sources
    if a.cpu:
        meta["enable_gpu"] = "false"
        meta.pop("machine_shape", None)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "train.py").write_text(code)
    (a.out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {a.out} id={meta['id']} title={meta['title']!r} sources={meta['kernel_sources']} "
          f"gpu={meta['enable_gpu']} sets={sets}")


if __name__ == "__main__":
    main()
