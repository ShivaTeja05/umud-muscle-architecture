"""Checks for the annotator's assists (run: python tools/annotator/test_assist.py [--app-js PATH]).

- known answers for the band / FL-extension / PA-arm maths in assist_geom.py: horizontal and sloped bands,
  a fascicle drawn top-to-bottom, overshoot trimming, curved polylines, parallel / too-far failures;
- parity: the GEOM block of app.js (between ASSIST-GEOM-BEGIN / ASSIST-GEOM-END) runs in node on the same
  cases plus 300 random ones and must match assist_geom.py (skipped with a note if node is missing);
- the page: every element id app.js looks up exists in index.html, and the geometry assist defaults to on;
- labels.py still reads labels that carry the new assist fields (same rows as without them).
"""

from __future__ import annotations

import json
import math
import random
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import assist_geom as g  # noqa: E402
import labels  # noqa: E402

APP_JS = HERE / "app.js"
INDEX = HERE / "index.html"
W = 1172  # work-set image width; max extension = 3 widths as in the tool
MAX = 3 * W
PPC = 110.4

# 3 MT lines at x 300 / 500 / 700, superficial inner edge y 100, deep inner edge y 300 (horizontal bands)
MT_FLAT = [[[300, 100], [300, 300]], [[500, 100], [500, 300]], [[700, 100], [700, 300]]]
# sloped: superficial y = 100 + 0.1 (x - 300), deep y = 300 - 0.1 (x - 300); the deep point clicked first on one
MT_SLOPED = [[[300, 100], [300, 300]], [[500, 280], [500, 120]], [[700, 140], [700, 260]]]
# kinked deep band: 300 -> 300, 500 -> 320, 700 -> 300
MT_KINK = [[[300, 100], [300, 300]], [[500, 100], [500, 320]], [[700, 100], [700, 300]]]


def close(a, b, tol=1e-6) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def pclose(p, q, tol=1e-6) -> bool:
    return close(p[0], q[0], tol) and close(p[1], q[1], tol)


def on_line(p, a, b, tol=1e-6) -> bool:
    """p lies on the infinite line through a and b."""

    cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    return abs(cross) <= tol * max(1.0, math.dist(a, b) * math.dist(a, p))


def length(pts) -> float:
    return sum(math.dist(pts[i - 1], pts[i]) for i in range(1, len(pts)))


def test_bands() -> None:
    b = g.bands_from_mt(MT_FLAT)
    assert b == {"sup": [[300.0, 100.0], [500.0, 100.0], [700.0, 100.0]],
                 "deep": [[300.0, 300.0], [500.0, 300.0], [700.0, 300.0]]}
    s = g.bands_from_mt(MT_SLOPED)  # the reversed MT line gives the same bands
    assert s["sup"] == [[300.0, 100.0], [500.0, 120.0], [700.0, 140.0]]
    assert s["deep"] == [[300.0, 300.0], [500.0, 280.0], [700.0, 260.0]]
    # unsorted input is sorted by x; outside the outer points the band continues with the nearest slope
    s2 = g.bands_from_mt(list(reversed(MT_SLOPED)))
    assert s2 == s
    assert close(g.band_y(s["sup"], 0), 70.0) and close(g.band_y(s["sup"], 1000), 170.0)
    k = g.bands_from_mt(MT_KINK)["deep"]
    assert close(g.band_y(k, 400), 310.0) and close(g.band_y(k, 600), 310.0)
    assert close(g.band_y(k, 100), 280.0) and close(g.band_y(k, 900), 280.0)  # nearest-segment slope
    # 2 MT lines: the straight line through them; fewer than 2 (or 2 at the same x): no band
    two = g.bands_from_mt(MT_SLOPED[:2])
    assert two["deep"] == [[300.0, 300.0], [500.0, 280.0]] and close(g.band_y(two["deep"], 900), 240.0)
    assert g.bands_from_mt(MT_FLAT[:1]) is None and g.bands_from_mt([]) is None
    assert g.bands_from_mt([[[300, 100], [300, 300]], [[300.2, 104], [300.2, 296]]]) is None
    merged = g.bands_from_mt([[[300, 100], [300, 300]], [[300.2, 104], [300.2, 296]], [[600, 100], [600, 300]]])
    assert len(merged["sup"]) == 2 and close(merged["sup"][0][1], 102.0) and close(merged["deep"][0][0], 300.1)


def test_extend_horizontal_bands() -> None:
    bands = g.bands_from_mt(MT_FLAT)
    # a short fascicle mid-muscle, drawn bottom-to-top: dx 50, dy -100
    r = g.extend_fascicle([[400, 250], [450, 150]], bands, MAX)
    assert r["ok"] and r["deep"] == "extend" and r["sup"] == "extend"
    assert len(r["pts"]) == 2 and pclose(r["pts"][0], [375, 300]) and pclose(r["pts"][1], [475, 100])
    assert close(length(r["pts"]), math.hypot(100, 200))
    # the same fascicle drawn top-to-bottom: stored deep -> superficial all the same
    r2 = g.extend_fascicle([[450, 150], [400, 250]], bands, MAX)
    assert r2["ok"] and all(pclose(p, q) for p, q in zip(r2["pts"], r["pts"]))


def test_extend_sloped_bands() -> None:
    bands = g.bands_from_mt(MT_SLOPED)
    raw = [[420, 230], [480, 170]]
    r = g.extend_fascicle(raw, bands, MAX)
    assert r["ok"] and len(r["pts"]) == 2
    d, s = r["pts"]
    assert close(d[1], g.band_y(bands["deep"], d[0])) and close(s[1], g.band_y(bands["sup"], s[0]))
    assert on_line(d, *raw) and on_line(s, *raw)
    # the line is y = 650 - x; deep band y = 330 - 0.1 x -> x = 320 / 0.9; superficial y = 70 + 0.1 x -> x = 580 / 1.1
    assert close(d[0], 320 / 0.9) and close(s[0], 580 / 1.1)


def test_overshoot_trimming() -> None:
    bands = g.bands_from_mt(MT_FLAT)
    # clicked into the bright bands at both ends (y 330 and y 70): trimmed back to the inner edges
    r = g.extend_fascicle([[360, 330], [520, 70]], bands, MAX)
    assert r["ok"] and r["deep"] == "trim" and r["sup"] == "trim"
    assert close(r["pts"][0][1], 300) and close(r["pts"][1][1], 100)
    assert on_line(r["pts"][0], [360, 330], [520, 70]) and on_line(r["pts"][1], [360, 330], [520, 70])
    # two clicks beyond the deep band: both dropped, the crossing is on the segment that enters the muscle
    raw = [[300, 360], [330, 320], [400, 250], [450, 150]]
    r = g.extend_fascicle(raw, bands, MAX)
    assert r["ok"] and r["deep"] == "trim" and r["sup"] == "extend"
    assert pclose(r["pts"][0], [350, 300]) and pclose(r["pts"][1], [400, 250])
    assert pclose(r["pts"][2], [475, 100]) and len(r["pts"]) == 3


def test_curved_polyline_keeps_inner_clicks() -> None:
    bands = g.bands_from_mt(MT_FLAT)
    raw = [[380, 270], [410, 230], [445, 190], [470, 160]]  # gently curved, drawn bottom-to-top
    r = g.extend_fascicle(raw, bands, MAX)
    assert r["ok"] and len(r["pts"]) == 4
    assert pclose(r["pts"][1], raw[1]) and pclose(r["pts"][2], raw[2])  # inner clicks kept
    assert close(r["pts"][0][1], 300) and on_line(r["pts"][0], raw[0], raw[1])  # along the end segments
    assert close(r["pts"][3][1], 100) and on_line(r["pts"][3], raw[2], raw[3])
    # the ends sit beyond the clicked ends, so the length only grows
    assert length(r["pts"]) > length(raw)


def test_kinked_band() -> None:
    bands = g.bands_from_mt(MT_KINK)
    r = g.extend_fascicle([[560, 250], [600, 200]], bands, MAX)  # meets the deep band on its 500-700 piece
    d = r["pts"][0]
    assert r["ok"] and 500 <= d[0] <= 700 and close(d[1], g.band_y(bands["deep"], d[0]))


def test_failures() -> None:
    bands = g.bands_from_mt(MT_FLAT)
    r = g.extend_fascicle([[400, 200], [500, 200]], bands, MAX)  # parallel to the bands
    assert not r["ok"] and "never meets" in r["why"]
    r = g.extend_fascicle([[400, 200], [500, 200.01]], bands, MAX)  # nearly parallel: far beyond 3 widths
    assert not r["ok"] and "too far" in r["why"]
    r = g.extend_fascicle([[400, 320], [450, 340]], bands, MAX)  # entirely below the deep band
    assert not r["ok"] and "between the two bands" in r["why"]
    r = g.extend_fascicle([[400, 200], [400, 200]], bands, MAX)
    assert not r["ok"] and "no length" in r["why"]
    # a hooked deep end that points back up never meets the deep band
    r = g.extend_fascicle([[420, 230], [400, 250], [440, 150]], bands, MAX)
    assert not r["ok"] and r["why"].startswith("deep end") and "never meets" in r["why"]


def test_pa_arm() -> None:
    deep = g.bands_from_mt(MT_FLAT)["deep"]
    half = 0.5 * PPC  # 5 mm
    r = g.pa_arm([400, 280], [440, 240], deep, half, MAX)
    # the line y = 280 - (x - 400) meets y = 300 at x = 380 (left of the clicks: the fascicle rises to the right)
    assert pclose(r["X"], [380, 300]) and pclose(r["arm"][0], [380 - half, 300]) and pclose(r["arm"][1], [380 + half, 300])
    assert close(labels.pa_value([[400, 280], [440, 240], *r["arm"]]), 45.0)
    # clicked far from the insertion, either direction: same crossing
    r2 = g.pa_arm([480, 180], [460, 200], deep, half, MAX)
    assert pclose(r2["X"], [360, 300])
    # sloped deep band (slope -0.1): the arm follows it, so PA is measured against the band
    sdeep = g.bands_from_mt(MT_SLOPED)["deep"]
    r3 = g.pa_arm([400, 280], [440, 240], sdeep, half, MAX)
    assert close(r3["arm"][0][1], g.band_y(sdeep, r3["arm"][0][0]))
    expected = 45.0 - math.degrees(math.atan(0.1))
    assert close(labels.pa_value([[400, 280], [440, 240], *r3["arm"]]), expected)
    assert g.pa_arm([400, 200], [500, 200], deep, half, MAX) is None  # parallel
    assert g.pa_arm([400, 200], [500, 200.01], deep, half, MAX) is None  # too far
    assert g.pa_arm([400, 200], [400, 200], deep, half, MAX) is None


# ---------- parity with app.js (node) ----------

def _cases() -> list[dict]:
    rnd = random.Random(1234)
    cases: list[dict] = []
    for mt in (MT_FLAT, MT_SLOPED, MT_KINK, MT_SLOPED[:2], MT_FLAT[:1],
               [[[300, 100], [300, 300]], [[300.2, 104], [300.2, 296]], [[600, 100], [600, 300]]]):
        cases.append({"fn": "bands", "mt": mt})
    fixed = [
        (MT_FLAT, [[400, 250], [450, 150]]), (MT_FLAT, [[450, 150], [400, 250]]), (MT_SLOPED, [[420, 230], [480, 170]]),
        (MT_FLAT, [[360, 330], [520, 70]]), (MT_FLAT, [[300, 360], [330, 320], [400, 250], [450, 150]]),
        (MT_FLAT, [[380, 270], [410, 230], [445, 190], [470, 160]]), (MT_KINK, [[560, 250], [600, 200]]),
        (MT_FLAT, [[400, 200], [500, 200]]), (MT_FLAT, [[400, 200], [500, 200.01]]), (MT_FLAT, [[400, 320], [450, 340]]),
        (MT_FLAT, [[420, 230], [400, 250], [440, 150]]), (MT_FLAT, [[400, 200], [400, 200]]),
    ]
    for mt, raw in fixed:
        cases.append({"fn": "extend", "raw": raw, "bands": g.bands_from_mt(mt), "max": MAX})
    for mt in (MT_FLAT, MT_SLOPED, MT_KINK):
        deep = g.bands_from_mt(mt)["deep"]
        for f0, f1 in (([400, 280], [440, 240]), ([480, 180], [460, 200]), ([400, 200], [500, 200]), ([620, 290], [590, 250])):
            cases.append({"fn": "arm", "f0": f0, "f1": f1, "deep": deep, "half": 0.5 * PPC, "max": MAX})
    for _ in range(300):
        n_mt = rnd.choice([2, 3, 3])
        xs = sorted(rnd.uniform(240, 940) for _ in range(n_mt))
        y0, slope, thick = rnd.uniform(60, 200), rnd.uniform(-0.15, 0.15), rnd.uniform(120, 320)
        mt = []
        for x in xs:
            top = [round(x, 2), round(y0 + slope * (x - 500) + rnd.uniform(-6, 6), 2)]
            bot = [top[0], round(top[1] + thick + rnd.uniform(-10, 10), 2)]
            mt.append([top, bot] if rnd.random() < 0.8 else [bot, top])
        bands = g.bands_from_mt(mt)
        if bands is None:
            continue
        ang = math.radians(rnd.uniform(8, 40))
        lean = rnd.choice([1, -1])  # fascicles leaning either way
        cx, cy = rnd.uniform(350, 800), y0 + thick * rnd.uniform(0.1, 0.9)
        k = rnd.choice([2, 2, 3, 4])
        span = rnd.uniform(20, thick * 1.6)
        raw = []
        for i in range(k):
            s = -span / 2 + span * i / (k - 1)
            raw.append([round(cx + lean * s * math.cos(ang) + rnd.uniform(-2, 2), 2), round(cy - s * math.sin(ang) + rnd.uniform(-2, 2), 2)])
        if rnd.random() < 0.5:
            raw.reverse()
        cases.append({"fn": "extend", "raw": raw, "bands": bands, "max": MAX})
        cases.append({"fn": "arm", "f0": raw[0], "f1": raw[-1], "deep": bands["deep"], "half": 0.5 * PPC, "max": MAX})
        cases.append({"fn": "y", "band": bands["deep"], "x": rnd.uniform(-600, 1800)})
    return cases


def _py(case: dict):
    fn = case["fn"]
    if fn == "bands":
        return g.bands_from_mt(case["mt"])
    if fn == "extend":
        return g.extend_fascicle(case["raw"], case["bands"], case["max"])
    if fn == "arm":
        return g.pa_arm(case["f0"], case["f1"], case["deep"], case["half"], case["max"])
    return g.band_y(case["band"], case["x"])


def _same(a, b, path="") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return [f"{path}: keys {sorted(a)} vs {sorted(b)}"]
        return [m for k in a for m in _same(a[k], b[k], f"{path}.{k}")]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: len {len(a)} vs {len(b)}"]
        return [m for i, (x, y) in enumerate(zip(a, b)) for m in _same(x, y, f"{path}[{i}]")]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        return [] if close(float(a), float(b), 1e-7) else [f"{path}: {a} vs {b}"]
    return [] if a == b else [f"{path}: {a!r} vs {b!r}"]


def js_block(app_js: Path = APP_JS) -> str:
    text = app_js.read_text(encoding="utf-8")
    m = re.search(r"// ASSIST-GEOM-BEGIN\n(.*?)// ASSIST-GEOM-END", text, re.S)
    assert m, "GEOM block markers missing in app.js"
    return m.group(1)


def test_js_parity(app_js: Path = APP_JS) -> None:
    node = shutil.which("node")
    if not node:
        print("   (skip: node not found, JS/Python parity not checked)")
        return
    cases = _cases()
    runner = js_block(app_js) + """
const fs = require('fs');
const cases = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const out = cases.map((c) => {
  if (c.fn === 'bands') return GEOM.bandsFromMT(c.mt);
  if (c.fn === 'extend') return GEOM.extendFascicle(c.raw, c.bands, c.max);
  if (c.fn === 'arm') return GEOM.paArm(c.f0, c.f1, c.deep, c.half, c.max);
  return GEOM.bandY(c.band, c.x);
});
process.stdout.write(JSON.stringify(out));
"""
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "run.js").write_text(runner, encoding="utf-8")
        (Path(tmp) / "cases.json").write_text(json.dumps(cases), encoding="utf-8")
        res = subprocess.run([node, str(Path(tmp) / "run.js"), str(Path(tmp) / "cases.json")],
                             capture_output=True, text=True, timeout=60, check=False)
    assert res.returncode == 0, res.stderr
    js = json.loads(res.stdout)
    assert len(js) == len(cases)
    bad = []
    for i, (c, out) in enumerate(zip(cases, js)):
        bad += [f"case {i} ({c['fn']}){m}" for m in _same(_py(c), out)]
    assert not bad, "\n".join(bad[:10])
    n_ok = sum(1 for c, o in zip(cases, js) if c["fn"] == "extend" and o["ok"])
    print(f"   node parity: {len(cases)} cases identical ({n_ok} successful extensions)")


# ---------- page wiring and backward compatibility ----------

def test_page_ids_exist(app_js: Path = APP_JS) -> None:
    js = app_js.read_text(encoding="utf-8")
    html = (app_js.parent / "index.html").read_text(encoding="utf-8")
    ids = set(re.findall(r"\$\('([\w-]+)'\)", js))
    html_ids = set(re.findall(r'id="([\w-]+)"', html))
    missing = sorted(ids - html_ids)
    assert not missing, f"app.js looks up ids missing from index.html: {missing}"
    assert {"ck-geom", "assist-banner", "assist-status"} <= html_ids
    assert re.search(r"geomAssist: true", js), "the geometry assist must default to on"
    assert "manifest.prefill_enabled !== true" in js, "suggestions must need manifest.prefill_enabled"


def test_labels_ignore_assist_fields() -> None:
    manifest = {"images": [{"image_id": "WRK_001", "px_per_cm": 110.4}],
                "tasks": [{"task_id": "WRK_001", "image_id": "WRK_001", "clip": None, "members": ["WRK_001"]}]}
    base = {"mt": MT_FLAT, "fl": [[[375, 300], [475, 100]]] * 3,
            "pa": [[[400, 280], [440, 240], [364.8, 300], [475.2, 300]]] * 3, "updated": 5, "flag": False, "note": ""}
    extra = dict(base, fl_raw=[[[400, 250], [450, 150]]] * 3,
                 assist={"mt": ["hand"] * 3, "fl": ["ext"] * 3, "pa": ["arm"] * 3, "prefill_shown": True,
                         "prefill_accepted": False, "prefill_blind": False},
                 prefill_start={"mt": [], "fl": [], "pa": [], "t": 1})
    rows_a = labels.build_rows({"labels": {"WRK_001": base}, "times": {}}, manifest)
    rows_b = labels.build_rows({"labels": {"WRK_001": extra}, "times": {}}, manifest)
    assert rows_a == rows_b and rows_a[0]["complete"] == 1
    merged, _ = labels.merge_states({"labels": {}}, {"labels": {"WRK_001": extra}})
    assert merged["labels"]["WRK_001"]["assist"]["fl"] == ["ext"] * 3


if __name__ == "__main__":
    args = sys.argv[1:]
    app = Path(args[args.index("--app-js") + 1]).resolve() if "--app-js" in args else APP_JS
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            if "app_js" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
                fn(app)
            else:
                fn()
            print("ok", name)
