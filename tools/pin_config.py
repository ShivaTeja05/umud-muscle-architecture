#!/usr/bin/env python
"""Re-pin the sha256 values of a umud-run config in place.

    python tools/pin_config.py configs/C18.yaml [--check]

Every line of the form  `- {path: "<path>", ..., sha256: "<hex or @sha>"}`  (the `inputs`, `code` and `maps.files`
lists) gets the sha256 of the file named by its path (${repo} expands to $UMUD_REPO or this repository). With
--check nothing is written and the exit code is 1 when a pin is stale or a file is missing.
Re-run after an INTENDED change of an input or a pinned code file; the tests then pass again. Streams files through
sha256 only (laptop-light). GPL-3.0-only.
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path

REPO = Path(os.environ.get("UMUD_REPO", Path(__file__).resolve().parents[1]))
LINE = re.compile(r'path:\s*"([^"]+)"(.*?)sha256:\s*"([^"]*)"')


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv: list[str]) -> int:
    check = "--check" in argv
    paths = [a for a in argv if a != "--check"]
    if not paths:
        print(__doc__)
        return 2
    bad = 0
    for cfg in paths:
        lines = Path(cfg).read_text().splitlines(keepends=True)
        out, n, changed = [], 0, 0
        for line in lines:
            m = LINE.search(line)
            if m:
                p = Path(m.group(1).replace("${repo}", str(REPO)))
                if not p.exists():
                    print(f"MISSING {p}")
                    bad += 1
                    out.append(line)
                    continue
                h = sha256_file(p)
                n += 1
                if h != m.group(3):
                    changed += 1
                    print(f"{'STALE' if check else 'pinned'} {m.group(1)}")
                    line = line[:m.start(3)] + h + line[m.end(3):]
            out.append(line)
        if not check and changed:
            Path(cfg).write_text("".join(out))
        bad += changed if check else 0
        print(f"{cfg}: {n} pins, {changed} {'stale' if check else 'updated'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
