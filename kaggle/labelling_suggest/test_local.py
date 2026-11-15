"""Local check of the labelling-suggest kernel WITHOUT torch: data + geometry stages on synthetic frames.

    .venv/bin/python kaggle/labelling_suggest/test_local.py [--keep DIR]

Builds a synthetic labelling_work_v1.zip (two frames with the real layouts: a 1172x608 screen capture with viewport
[232, 43, 940, 596] at 110.4 px/cm and a 978x780 crop at 152.6 px/cm, plus one frame without a deep aponeurosis), mounts
it under a fake /kaggle/input, runs stage_data, writes synthetic maps in place of the infer stage (two straight bands,
fascicle fragments and a constant orientation field at a known angle), runs stage_geometry and checks the suggestions
against the analytic MT / PA / FL and the shared prefill schema (tools/annotator/apply_prefill.validate_prefill).
One process, < 0.3 GB, a few seconds. GPL-3.0-only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "tools" / "annotator"))

import apply_prefill as ap  # noqa: E402

FRAMES = {
    # id: full size, corrected viewport, px/cm, participant, band geometry in crop px (sup top y0, slope, thickness;
    # deep top y0, slope, thickness), fascicle angle (deg, image coords; negative = rising to the right)
    "WRK_001": dict(size=(1172, 608), vp=(232, 43, 940, 596), ppc=110.4, p="p1",
                    sup=(36.0, 0.010, 10.0), deep=(312.0, -0.020, 14.0), theta=-20.0),
    "WRK_006": dict(size=(978, 780), vp=(1, 0, 978, 780), ppc=152.6, p="p6",
                    sup=(70.0, 0.060, 12.0), deep=(345.0, 0.050, 16.0), theta=-14.0),
    "WRK_011": dict(size=(1172, 608), vp=(232, 43, 940, 596), ppc=110.4, p="p3",
                    sup=(36.0, 0.0, 10.0), deep=None, theta=-20.0),
    # a thin skin line 17 px above the superficial aponeurosis (closer than pair_min_sep): the guard must use the
    # aponeurosis, not the skin line, as the superficial inner edge
    "WRK_016": dict(size=(978, 780), vp=(1, 0, 978, 780), ppc=152.6, p="p8", skin=(48.0, 0.010, 5.0),
                    sup=(70.0, 0.010, 12.0), deep=(360.0, -0.030, 16.0), theta=16.0),
}


def band_mask(W: int, H: int, y0: float, slope: float, th: float) -> np.ndarray:
    yc = np.arange(H)[:, None] + 0.5
    top = y0 + slope * (np.arange(W)[None, :] + 0.5)
    return (yc >= top) & (yc < top + th)


def synth(iid: str, rng: np.random.Generator):
    f = FRAMES[iid]
    x1, y1, x2, y2 = f["vp"]
    W, H = x2 - x1, y2 - y1
    crop = rng.normal(28, 9, (H, W))
    sup = band_mask(W, H, *f["sup"])
    deep = band_mask(W, H, *f["deep"]) if f["deep"] else np.zeros_like(sup)
    xs = np.arange(W)[None, :] + 0.5
    yy = np.arange(H)[:, None] + 0.5
    sup_bot = f["sup"][0] + f["sup"][1] * xs + f["sup"][2]
    deep_top = (f["deep"][0] + f["deep"][1] * xs) if f["deep"] else np.full_like(xs, H + 50.0)
    comp = (yy > sup_bot + 2) & (yy < deep_top - 2)
    t = math.radians(f["theta"])
    # fascicles: parallel lines at theta every 22 px (perpendicular offset), 2 px wide, broken into fragments
    perp = -(xs - 0) * math.sin(t) + yy * math.cos(t)
    along = xs * math.cos(t) + yy * math.sin(t)
    lines = (np.mod(perp, 22.0) < 2.0) & (np.mod(along, 70.0) < 50.0) & comp
    skin = band_mask(W, H, *f["skin"]) if f.get("skin") else np.zeros_like(sup)
    crop[lines] = 120
    crop[sup | deep | skin] = 235
    crop = np.clip(crop, 0, 255).astype(np.uint8)
    full = np.full((f["size"][1], f["size"][0]), 50, np.uint8)
    full[y1:y2, x1:x2] = crop
    apo = ((sup | deep | skin) * 255).astype(np.uint8)
    fasc = (lines * 255).astype(np.uint8)
    seg = (comp * 200).astype(np.uint8)
    c2 = np.full((H, W), np.uint8(round((math.cos(2 * t) + 1) * 127.5)))
    s2 = np.full((H, W), np.uint8(round((math.sin(2 * t) + 1) * 127.5)))
    c2[~comp], s2[~comp] = 128, 128
    return full, dict(apo=apo, fasc=fasc, seg=seg, cos2=c2, sin2=s2)


def png_bytes(a: np.ndarray) -> bytes:
    b = io.BytesIO()
    Image.fromarray(a).save(b, format="PNG")
    return b.getvalue()


def expected(iid: str) -> dict:
    """Analytic MT at the guides, PA vs the deep line, FL of the midpoint-placed straight fascicles (crop px)."""
    f = FRAMES[iid]
    W = f["vp"][2] - f["vp"][0]
    ys = lambda x: f["sup"][0] + f["sup"][1] * x + f["sup"][2]  # noqa: E731
    yd = lambda x: f["deep"][0] + f["deep"][1] * x  # noqa: E731
    mm = 10.0 / f["ppc"]
    mt = [(yd(fr * W) - ys(fr * W)) * mm for fr in (0.25, 0.5, 0.75)]
    pa = abs(f["theta"] - math.degrees(math.atan(f["deep"][1])))
    t = math.radians(f["theta"])
    d = np.array([math.cos(t), math.sin(t)])
    d = -d if d[1] > 0 else d
    fls = []
    for fr in (0.35, 0.5, 0.65):
        # line (xm, ym) + s d meets y = A + b x at s = (A + b xm - ym) / (dy - b dx); the chord midpoint is at x = xm
        # when s_deep + s_sup = 0, which fixes ym (linear in ym)
        xm = fr * W
        a_d, k_d = f["deep"][0] + f["deep"][1] * xm, d[1] - f["deep"][1] * d[0]
        a_s, k_s = f["sup"][0] + f["sup"][2] + f["sup"][1] * xm, d[1] - f["sup"][1] * d[0]
        ym = (a_d / k_d + a_s / k_s) / (1.0 / k_d + 1.0 / k_s)
        fls.append(abs((a_s - ym) / k_s - (a_d - ym) / k_d) * mm)
    return dict(mt=mt, pa=pa, fl=fls)


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--keep", default=None, help="work in this folder and keep it (default: a temp dir)")
    args = a.parse_args()
    tmpdir = tempfile.TemporaryDirectory() if not args.keep else None
    base = Path(args.keep or tmpdir.name)
    base.mkdir(parents=True, exist_ok=True)
    inp, tmp, out = base / "input", base / "tmp", base / "working"
    rng = np.random.default_rng(0)
    zdir = inp / "notebooks" / "shivateja005" / "umud-labelling-frames"
    zdir.mkdir(parents=True, exist_ok=True)
    maps, table, images = {}, {}, []
    with zipfile.ZipFile(zdir / "labelling_work_v1.zip", "w") as zf:
        for iid, f in FRAMES.items():
            full, m = synth(iid, rng)
            maps[iid] = m
            data = png_bytes(full)
            zf.writestr(f"work/img/{iid}.png", data)
            zf.writestr(f"work/img/{iid}_n1.png", data)
            # the zip manifest carries the UNcorrected scale, like the real one
            images += [{"image_id": iid, "file": f"img/{iid}.png", "width": f["size"][0], "height": f["size"][1],
                        "px_per_cm": 119.9211, "viewport": [0, 0, *f["size"]], "participant": f["p"]},
                       {"image_id": f"{iid}_n1", "file": f"img/{iid}_n1.png", "width": f["size"][0],
                        "height": f["size"][1], "px_per_cm": 119.9211, "viewport": [0, 0, *f["size"]],
                        "participant": f["p"], "view_only": True}]
            table[iid] = {"width": f["size"][0], "height": f["size"][1], "viewport": list(f["vp"]),
                          "px_per_cm": f["ppc"], "participant": f["p"], "sha256": hashlib.sha256(data).hexdigest()}
        zf.writestr("work/manifest.json", json.dumps({"kind": "work", "images": images, "tasks": []}))
        zf.writestr("private/work_selection.json", "{}")
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "frames_table.json").write_text(json.dumps(table))
    os.environ.update(SUGGEST_INPUT=str(inp), SUGGEST_TMP=str(tmp), SUGGEST_OUT=str(out),
                      SUGGEST_G2=str(REPO / "src" / "umud" / "geometry_v2.py"))
    spec = importlib.util.spec_from_file_location("suggest_stages", HERE / "suggest_stages.py")
    st = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(st)

    # ---- data stage: zip found, corrected table applied, crops = viewport
    st.stage_data()
    fr = {f["image_id"]: f for f in st.read_frames()}
    assert sorted(fr) == sorted(FRAMES), fr.keys()
    for iid, f in FRAMES.items():
        x1, y1, x2, y2 = f["vp"]
        assert fr[iid]["viewport"] == [x1, y1, x2, y2] and fr[iid]["px_per_cm"] == f["ppc"], fr[iid]
        assert fr[iid]["scale_source"] == "embedded_local_manifest" and fr[iid]["sha_ok"] is True, fr[iid]
        assert (fr[iid]["crop_w"], fr[iid]["crop_h"]) == (x2 - x1, y2 - y1)
    print("ok data stage (zip found under notebooks/, corrected viewport + scale, sha256 match)")

    # ---- the extracted-folder layout is found too
    alt = base / "input_dir" / "umud-labelling-frames" / "labelling_work_v1" / "work"
    (alt / "img").mkdir(parents=True, exist_ok=True)
    (alt / "manifest.json").write_text("{}")
    (alt / "img" / "WRK_001.png").write_bytes(b"x")
    old = st.INPUT
    st.INPUT = base / "input_dir"
    kind, src = st.find_work_source()
    st.INPUT = old
    assert kind == "dir" and src == alt, (kind, src)
    print("ok extracted-folder fallback")

    # ---- synthetic maps in place of the infer stage
    st.MAPS.mkdir(parents=True, exist_ok=True)
    names = {"ours_apo384": "apo", "ours_apo384_noflip": "apo", "ours_fasc384_noflip": "fasc",
             "fasc_orient_v1_seg": "seg", "fasc_orient_v1_cos2": "cos2", "fasc_orient_v1_sin2": "sin2"}
    for model, key in names.items():
        np.savez_compressed(st.MAPS / f"{model}.npz", **{iid: m[key] for iid, m in maps.items()})
    st.stage_geometry()
    res = json.loads((out / "suggestions_v1.json").read_text())
    assert res["version"] == 1 and res["n_images"] == 4 and res["n_ok"] == 3, (res["n_images"], res["n_ok"])
    assert not res["images"]["WRK_011"]["ok"], res["images"]["WRK_011"]
    print(f"ok no-deep-band frame fails cleanly: {res['images']['WRK_011']['reason']}")
    d16 = res["images"]["WRK_016"]["diag"]
    assert d16["sup_refined"] is True and res["images"]["WRK_001"]["diag"]["sup_refined"] is False, d16
    print(f"ok skin-line guard: pair {d16['pair']} -> superficial band {d16['sup_band_to']}, "
          f"shift {d16['sup_shift_px']:.1f} px; pipeline MT {res['images']['WRK_016']['values']['pipeline']['mt_mm']:.2f}")
    for iid in ("WRK_001", "WRK_006", "WRK_016"):
        e, f = res["images"][iid], FRAMES[iid]
        pf, v = e["prefill"], e["values"]
        errs = ap.validate_prefill(pf, f["size"][0], f["size"][1])
        assert not errs, (iid, errs)
        x1, y1, x2, y2 = f["vp"]
        for m, frac in zip(pf["mt"], (0.25, 0.5, 0.75)):   # MT on the annotator guides, native frame
            assert abs(m[0][0] - (x1 + frac * (x2 - x1))) < 0.011, (m, frac)
        ex = expected(iid)
        for got, want in zip(v["mt_mm"], ex["mt"]):
            assert abs(got - want) < 0.15, (iid, "mt", got, want)
        for got in v["pa_deg"]:
            assert abs(got - ex["pa"]) < 0.8, (iid, "pa", got, ex["pa"])
        for got, want in zip(v["fl_mm"], ex["fl"]):
            assert abs(got - want) < 0.03 * want + 0.5, (iid, "fl", got, want)
        for line in pf["fl"]:   # deep end on the deep inner edge, superficial end on the superficial one
            xd, yd = line[0][0] - x1, line[0][1] - y1
            xs, ys = line[-1][0] - x1, line[-1][1] - y1
            assert abs(yd - (f["deep"][0] + f["deep"][1] * xd)) < 1.2, (iid, line[0])
            assert abs(ys - (f["sup"][0] + f["sup"][1] * xs + f["sup"][2])) < 1.2, (iid, line[-1])
        for m in pf["pa"]:      # PA fascicle arm within ~1 cm of the insertion, apo arm 1 cm long
            assert abs(math.dist(m[2], m[3]) * 10 / f["ppc"] - 10.0) < 0.05, m
            assert math.dist(m[0], m[1]) * 10 / f["ppc"] <= 9.05, m
        assert 0.0 <= pf["confidence"] <= 1.0
        print(f"ok {iid}: MT {v['mt_mean']:.2f} mm (want {np.mean(ex['mt']):.2f}), PA {v['pa_mean']:.2f} deg "
              f"(want {ex['pa']:.2f}), FL {v['fl_mean']:.1f} mm (want {np.mean(ex['fl']):.1f}), "
              f"extrap {['%.2f' % x for x in v['fl_extrap_frac']]}, conf {pf['confidence']}")
    prev = sorted((out / "previews").glob("*.png"))
    assert prev and sum(p.stat().st_size for p in prev) < st.PREVIEW_BUDGET, prev
    print(f"ok previews: {[p.name for p in prev]} {sum(p.stat().st_size for p in prev) / 1e3:.0f} kB")
    print(f"suggestions_v1.json {(out / 'suggestions_v1.json').stat().st_size / 1e3:.1f} kB")
    if tmpdir:
        tmpdir.cleanup()
    print("all local kernel checks passed")


if __name__ == "__main__":
    main()
