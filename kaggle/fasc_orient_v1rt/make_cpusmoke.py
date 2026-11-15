"""Write the CPU smoke copy of this kernel to ./cpusmoke (MODE="cpusmoke", enable_gpu false, id *-cpusmoke).

Run from this directory: python3 make_cpusmoke.py. Only the MODE constant and the kernel metadata change.
"""
import json
import re
from pathlib import Path

here = Path(__file__).resolve().parent
code = (here / "infer.py").read_text()
pat = re.compile(r'^MODE = "full"(\s*#.*)?$', re.M)
assert len(pat.findall(code)) == 1, "expected exactly one MODE assignment"
code = pat.sub(lambda m: 'MODE = "cpusmoke"' + (m.group(1) or ""), code, count=1)
meta = json.loads((here / "kernel-metadata.json").read_text())
meta["id"] = "shivateja005/umud-fasc-orient-v1rt-cpusmoke"
meta["title"] = "UMUD Fasc Orient V1rt Cpusmoke"
meta["enable_gpu"] = "false"
meta.pop("machine_shape", None)
out = here / "cpusmoke"
out.mkdir(exist_ok=True)
(out / "infer.py").write_text(code)
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print(f"wrote {out} id={meta['id']}")
