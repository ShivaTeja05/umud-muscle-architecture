"""Checks for score_check.py and the manual labelling kit (run: python tools/annotator/test_score_check.py).

Synthetic tests always run.  Tests on the real kit (data/manual_labelling/, git-ignored) skip when it is absent.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import struct
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import score_check as sc  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
KIT = ROOT / "data" / "manual_labelling"
HEADER = ["image_id", "MT1_mm", "MT2_mm", "MT3_mm", "FL1_mm", "FL2_mm", "FL3_mm", "PA1_deg", "PA2_deg", "PA3_deg",
          "minutes", "confidence_1to5", "flag", "notes"]


def synthetic_key(n_images: int = 10) -> dict:
    """10 images, 4 raters with fixed, uneven offsets; R4 misses image 1 (like R5 in the real key)."""

    items = {}
    for i in range(1, n_images + 1):
        base = {"pa_deg": 15.0 + i, "fl_mm": 50.0 + 2 * i, "mt_mm": 15.0 + 0.5 * i}
        raters = {}
        for k, r in enumerate(("R1", "R2", "R3", "R4")):
            off = (k - 1.5) * (1 + (i % 3))
            raters[r] = {"pa_deg": base["pa_deg"] + 0.5 * off, "fl_mm": base["fl_mm"] + 2.0 * off,
                         "mt_mm": base["mt_mm"] + 0.2 * off}
        if i == 1:
            raters["R4"] = {"pa_deg": None, "fl_mm": None, "mt_mm": None}
        items[f"CHK_{i:02d}"] = {"px_per_cm": 100.0, "raters": raters,
                                 "expert": {c: sc._mean([v[c] for v in raters.values()]) for c in sc.COMPS}}
    return {"raters": ["R1", "R2", "R3", "R4"], "items": items}


def write_template(path: Path, values: dict, delim: str = ",", decimal_comma: bool = False, edit=None) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter=delim)
        w.writerow(HEADER)
        for cid, v in values.items():
            row = [cid] + [v["mt_mm"]] * 3 + [v["fl_mm"]] * 3 + [v["pa_deg"]] * 3 + ["4", "3", "", ""]
            row = [f"{x:.6f}".replace(".", ",") if decimal_comma and isinstance(x, float) else x for x in row]
            if edit:
                edit(cid, row)
            w.writerow(row)


def test_expert_mean_scores_zero() -> None:
    key = synthetic_key()
    vals = {cid: dict(it["expert"]) for cid, it in key["items"].items()}
    res = sc.grade(vals, key, n_boot=200)
    assert abs(res["you"]["S"]) < 1e-12 and res["verdict"] == "GOOD" and res["rank"] == 1
    assert res["n_images_complete"] == 10 and res["bar_raters"] == ["R1", "R2", "R3", "R4"]


def test_rater_loo_identity_and_shrink() -> None:
    key = synthetic_key()
    for r in key["raters"]:
        own = sc.rater_values(key, r)
        loo = sc.summarize(sc.loo_errors(key, r))
        res = sc.grade(own, key, without=r, n_boot=100)
        assert abs(res["you"]["S"] - loo["S"]) < 1e-12, r
        # graded against the mean of ALL raters, which contains r: each error shrinks by (n - 1) / n
        e_all, e_loo = sc.errors(own, key), sc.loo_errors(key, r)
        assert set(e_all) == set(e_loo)
        for (cid, comp), e in e_all.items():
            n = sum(sc._finite(v.get(comp)) for v in key["items"][cid]["raters"].values())
            assert abs(e - e_loo[(cid, comp)] * (n - 1) / n) < 1e-9, (r, cid, comp)


def test_verdicts() -> None:
    key = synthetic_key()
    table = sc.expert_table(key, key["raters"])
    worst = max(v["S"] for v in table.values())
    vals = {cid: {c: it["expert"][c] + (0.0 if c != "pa_deg" else 20.0) for c in sc.COMPS}
            for cid, it in key["items"].items()}
    assert sc.grade(vals, key, n_boot=50)["verdict"] == "FAIL"  # 20 deg PA error everywhere
    few = {cid: v for cid, v in list({cid: dict(it["expert"]) for cid, it in key["items"].items()}.items())[:5]}
    res = sc.grade(few, key, n_boot=50)
    assert res["verdict"].startswith("INCOMPLETE") and res["n_images_complete"] == 5
    # the bar is recomputed on the candidate's cells only
    assert all(v["n_images"] <= 5 for v in res["experts_loo_same_cells"].values())
    assert worst > 0


def test_template_parsing_and_units() -> None:
    key = synthetic_key()
    vals = {cid: dict(it["expert"]) for cid, it in key["items"].items()}
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "a.csv"
        write_template(p, vals, delim=";", decimal_comma=True)
        got, notes, _ = sc.values_from_template(p, key)
        assert all(abs(got[c][k] - vals[c][k]) < 1e-5 for c in vals for k in sc.COMPS), notes

        def slip(cid, row):
            if cid == "CHK_02":
                row[1] = vals[cid]["mt_mm"] / 10  # MT1 in cm
                row[7] = math.radians(vals[cid]["pa_deg"])  # PA1 in radians
                row[4] = vals[cid]["fl_mm"] * 10  # FL1 in pixels (100 px/cm = 10 px/mm)
                row[6] = ""  # a blank
            if cid == "CHK_03":
                row[1:4] = ["", "", ""]  # MT missing
        write_template(p, vals, edit=slip)
        got, notes, _ = sc.values_from_template(p, key)
        text = "\n".join(notes)
        assert "CENTIMETRES" in text and "RADIANS" in text and "PIXELS" in text, text
        assert math.isnan(got["CHK_03"]["mt_mm"]) and "CHK_02 FL: 2 of 3 values" in text
        fixed, _, _ = sc.values_from_template(p, key, fix_units=True)
        for comp in sc.COMPS:
            assert abs(fixed["CHK_02"][comp] - vals["CHK_02"][comp]) < 1e-6, comp
        res = sc.grade(fixed, key, n_boot=50)
        assert res["n_images_complete"] == 9 and res["n_cells"] == 29
    assert sc.norm_id("chk_1.png") == "CHK_01" and sc.norm_id("CHK-10") == "CHK_10"


def test_units_ambiguous_x10_and_obtuse_pa() -> None:
    key = synthetic_key()
    for it in key["items"].values():
        it["px_per_cm"] = 77.0  # 7.7 px/mm: pixels and a 10x slip give different values
    vals = {cid: dict(it["expert"]) for cid, it in key["items"].items()}
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "a.csv"

        def slip(cid, row):
            if cid == "CHK_04":  # Set Scale with px_per_mm and Known distance 10: every length 10x too large
                row[1:7] = [x * 10 for x in row[1:7]]
            if cid == "CHK_02":  # FIJI angle with the arms the other way round: 180 - x
                row[7] = 180.0 - vals[cid]["pa_deg"]
        write_template(p, vals, edit=slip)
        got, notes, _ = sc.values_from_template(p, key, fix_units=True)
        text = "\n".join(notes)
        assert "AMBIGUOUS" in text and "OBTUSE" in text, text
        assert abs(got["CHK_04"]["mt_mm"] - 10 * vals["CHK_04"]["mt_mm"]) < 1e-6  # not "fixed" to a wrong value
        assert abs(got["CHK_02"]["pa_deg"] - vals["CHK_02"]["pa_deg"]) < 1e-6  # obtuse angle fixed


def test_csv_encodings() -> None:
    key = synthetic_key()
    vals = {cid: dict(it["expert"]) for cid, it in key["items"].items()}
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "a.csv"

        def note(cid, row):
            row[-1] = "PA ~20\u00b0, M\u00fcller"
        write_template(p, vals, edit=note)
        text = p.read_text(encoding="utf-8")
        for enc in ("cp1252", "mac_roman"):
            q = Path(tmp) / f"{enc}.csv"
            q.write_bytes(text.encode(enc))
            got, notes, meta = sc.values_from_template(q, key)
            assert all(abs(got[c]["pa_deg"] - vals[c]["pa_deg"]) < 1e-5 for c in vals), enc
            assert any("not UTF-8" in n for n in notes), notes


def test_copy_check() -> None:
    key = synthetic_key()
    copied = {cid: {c: round(v[c], 2) for c in sc.COMPS} for cid, v in sc.rater_values(key, "R2").items()}
    cc = sc.copy_check(copied, key)
    assert cc["suspect"] and cc["n_images"] == 10, cc
    honest = {cid: {"pa_deg": it["expert"]["pa_deg"] + 0.7, "fl_mm": it["expert"]["fl_mm"] - 1.3,
                    "mt_mm": it["expert"]["mt_mm"] + 0.17} for cid, it in key["items"].items()}
    assert not sc.copy_check(honest, key)["suspect"]


def _roi_bytes(kind: str, pts: list, subpixel: bool = False) -> bytes:
    """A minimal ImageJ .roi encoder (the RoiDecoder layout) for the round trip."""

    h = bytearray(64)
    h[0:4] = b"Iout"
    struct.pack_into(">h", h, 4, 228)
    h[6] = {"line": 3, "polyline": 5, "angle": 9}[kind]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    left, top = int(math.floor(min(xs))), int(math.floor(min(ys)))
    struct.pack_into(">hhhh", h, 8, top, left, int(max(ys)) + 1, int(max(xs)) + 1)
    if kind == "line":
        struct.pack_into(">ffff", h, 18, xs[0], ys[0], xs[1], ys[1])
        return bytes(h)
    n = len(pts)
    struct.pack_into(">H", h, 16, n)
    if subpixel:
        struct.pack_into(">h", h, 50, 128)
    body = struct.pack(f">{n}h", *[int(x) - left for x in xs]) + struct.pack(f">{n}h", *[int(y) - top for y in ys])
    if subpixel:
        body += struct.pack(f">{n}f", *xs) + struct.pack(f">{n}f", *ys)
    return bytes(h) + body


def test_roi_recompute() -> None:
    key = synthetic_key()  # 100 px/cm = 10 px/mm
    vals = {cid: dict(it["expert"]) for cid, it in key["items"].items()}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for k, (cid, v) in enumerate(vals.items()):
            a = math.radians(v["pa_deg"])
            rois = []
            for j in range(3):  # MT: vertical lines (+150 px offset, as after Canvas Size)
                rois.append(_roi_bytes("line", [(200.0 + 60 * j, 150.0), (200.0 + 60 * j, 150.0 + v["mt_mm"] * 10)]))
            for j in range(3):  # FL: 3-point segmented lines along one straight path
                L = v["fl_mm"] * 10
                p0 = (300.0, 500.0 + j)
                pts = [p0, (p0[0] + 0.4 * L * math.cos(a), p0[1] - 0.4 * L * math.sin(a)),
                       (p0[0] + L * math.cos(a), p0[1] - L * math.sin(a))]
                rois.append(_roi_bytes("polyline", pts, subpixel=True))
            for j in range(3):  # PA: arm 1 along the fascicle, vertex, arm 2 along the aponeurosis
                vx, vy = 400.0 + 30 * j, 600.0
                rois.append(_roi_bytes("angle", [(vx + 80 * math.cos(a), vy - 80 * math.sin(a)), (vx, vy), (vx + 90.0, vy)],
                                       subpixel=True))
            with zipfile.ZipFile(tmp / f"{cid}_rois.zip", "w") as zf:
                for j, b in enumerate(rois):
                    zf.writestr(f"{j:04d}-0000-0000.roi", b)
        p = tmp / "a.csv"
        write_template(p, vals, edit=lambda cid, row: row.__setitem__(1, vals[cid]["mt_mm"] + 2.0) if cid == "CHK_05" else None)
        got, notes, meta = sc.values_from_template(p, key)
        rc = sc.compare_rois(tmp, key, meta["_raw_values"])
        assert rc["n_images"] == 10 and rc["n_mismatch"] == 1, rc["notes"]
        assert any(n.startswith("MISMATCH CHK_05 MT") for n in rc["notes"]), rc["notes"]
        for cid, v in vals.items():
            for c in sc.COMPS:
                assert abs(rc["values"][cid][c] - v[c]) < 0.02, (cid, c, rc["values"][cid][c], v[c])


def test_report_hides_answers() -> None:
    key = synthetic_key()
    for it in key["items"].values():  # make the expert means easy to spot in the printout
        for r in it["raters"].values():
            if r["mt_mm"] is not None:
                r["mt_mm"] += 0.123
    exp = {cid: {c: sc.reference(it, c) for c in sc.COMPS} for cid, it in key["items"].items()}
    one = {"CHK_01": {c: exp["CHK_01"][c] + 0.5 for c in sc.COMPS}}
    res = sc.grade(one, key, n_boot=20)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sc.report(res, None, "test", [])
    out = buf.getvalue()
    assert "INCOMPLETE" in out and "Score S" not in out, out
    for cid in ("CHK_01", "CHK_02"):
        assert f"{exp[cid]['mt_mm']:.1f}" not in out.split("Still to finish")[0], out
    red = sc.redact(res)
    assert "per_image" not in red and red["verdict"] == "INCOMPLETE"
    nine = {cid: {c: exp[cid][c] + 0.3 for c in sc.COMPS} for cid in list(exp)[:9]}
    res = sc.grade(nine, key, n_boot=20)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sc.report(res, None, "test", [])
    row10 = [ln for ln in buf.getvalue().splitlines() if ln.startswith("CHK_10")][0]
    assert f"{exp['CHK_10']['mt_mm']:.1f}" not in row10 and f"{exp['CHK_10']['fl_mm']:.1f}" not in row10, row10
    assert sc.redact(res)["per_image"][9]["mt_mm"]["expert"] is None


def test_missing_or_empty_input_exits() -> None:
    key = synthetic_key()
    with tempfile.TemporaryDirectory() as tmp:
        empty = Path(tmp) / "autosave.json"
        empty.write_text(json.dumps({"labels": {}}))
        for path in (Path(tmp) / "missing.json", empty):
            args = argparse.Namespace(csv=None, state=path, practice=False, fix_units=False)
            try:
                sc.load_candidate(args, key)
            except SystemExit:
                continue
            raise AssertionError(f"no error for {path}")


def test_practice_first_attempt() -> None:
    key = synthetic_key()
    labels = {}
    for k, (cid, it) in enumerate(key["items"].items()):
        good = _clicks_for(it["expert"], it["px_per_cm"])
        moved = _clicks_for({c: v + 5.0 for c, v in it["expert"].items()}, it["px_per_cm"])
        labels[cid] = dict(moved, first={"mt": good["mt"], "fl": good["fl"], "pa": good["pa"], "t": 1000 + k})
    state = {"labels": labels}
    sub, chosen = sc.practice_subset(state, key, 5)
    assert chosen == [f"CHK_{k:02d}" for k in range(6, 11)], chosen
    vals, notes = sc.values_from_state(state, sub, use_first=True)
    res = sc.grade(vals, sub, n_boot=20, min_complete=len(sub["items"]))
    assert res["you"]["S"] < 1e-9 and not sc.hidden(res), res["you"]
    vals, _ = sc.values_from_state(state, sub, use_first=False)
    assert sc.grade(vals, sub, n_boot=20)["you"]["S"] > 0.3


def _clicks_for(v: dict, px_per_cm: float) -> dict:
    ppmm = px_per_cm / 10.0
    mt = [[[100.0 + 50 * k, 50.0], [100.0 + 50 * k, 50.0 + v["mt_mm"] * ppmm]] for k in range(3)]
    a = math.radians(v["pa_deg"])
    fl = [[[0.0, 0.0], [v["fl_mm"] * ppmm * math.cos(a), -v["fl_mm"] * ppmm * math.sin(a)]] for _ in range(3)]
    pa = [[[300.0, 300.0], [300.0 + 80 * math.cos(a), 300.0 - 80 * math.sin(a)], [250.0, 300.0], [400.0, 300.0]]
          for _ in range(3)]
    return {"mt": mt, "fl": fl, "pa": pa, "flag": False, "note": "", "updated": 1}


def test_annotator_state_path() -> None:
    key = synthetic_key()
    state = {"labels": {cid: _clicks_for(it["expert"], it["px_per_cm"]) for cid, it in key["items"].items()}}
    vals, notes = sc.values_from_state(state, key)
    res = sc.grade(vals, key, n_boot=50)
    assert res["you"]["S"] < 1e-9 and res["n_images_complete"] == 10, (res["you"], notes)
    del state["labels"]["CHK_04"]["pa"][2]
    vals, notes = sc.values_from_state(state, key)
    assert any("CHK_04 PA: only 2 of 3" in n for n in notes)


def test_serve_denies_kit_answers() -> None:
    import serve
    with tempfile.TemporaryDirectory() as tmp:
        kit = Path(tmp)
        check, private = kit / "check", kit / "private"
        (check / "calib").mkdir(parents=True)
        private.mkdir()
        for name in ("manifest.json", "calib_manifest.json", "calib_answers.json", "check_key.json", "pass_bar.json"):
            (check / name).write_text("{}")
        (check / "calib" / "CHK_01.png").write_bytes(b"x")
        (private / "check_key.json").write_text("{}")
        assert serve.servable(check, "calib_manifest.json") is not None
        assert serve.servable(check, "calib/CHK_01.png") is not None
        for d in ("private", "PRIVATE", "Private", "BACKUPS", "Check_Results"):
            (check / d).mkdir(exist_ok=True)
            (check / d / "x.json").write_text("{}")
        for name in ("selection.json", "work_selection.json", "build_cache.json"):
            (check / name).write_text("{}")
        for bad in ("calib_answers.json", "../private/check_key.json", "calib/../../private/check_key.json",
                    "check_key.json", "pass_bar.json", "private/check_key.json", "private/x.json", "PRIVATE/x.json",
                    "Private/x.json", "BACKUPS/x.json", "Check_Results/x.json", "selection.json",
                    "work_selection.json", "build_cache.json", "Selection.JSON"):
            assert serve.servable(check, bad) is None, bad


# ----------------------------------------------------------------------------- the real kit (skips if absent)
def _kit() -> tuple[dict, dict] | None:
    kp, bp = KIT / "private" / "check_key.json", KIT / "private" / "pass_bar.json"
    if not (kp.exists() and bp.exists()):
        print("  (skipped: data/manual_labelling/private/ not built)")
        return None
    return sc.load_json(kp), sc.load_json(bp)


def test_real_kit_bar_identity() -> None:
    kit = _kit()
    if not kit:
        return
    key, bar = kit
    assert len(key["items"]) == 10 and sorted(key["items"]) == [f"CHK_{k:02d}" for k in range(1, 11)]
    mean = {cid: dict(it["expert"]) for cid, it in key["items"].items()}
    res = sc.grade(mean, key, bar, n_boot=100)
    assert res["you"]["S"] < 1e-5 and res["verdict"] == "GOOD"  # key means are stored rounded to 1e-5
    for r in bar["bar_raters"]:
        res = sc.grade(sc.rater_values(key, r), key, bar, without=r, n_boot=50)
        assert abs(res["you"]["S"] - bar["check_10"]["per_rater_loo"][r]["S"]) < 1e-4, r
    with tempfile.TemporaryDirectory() as tmp:  # the same through the CSV path (3 equal values per parameter)
        p = Path(tmp) / "r3.csv"
        write_template(p, sc.rater_values(key, "R3"))
        vals, notes, _ = sc.values_from_template(p, key)
        res = sc.grade(vals, key, bar, without="R3", n_boot=50)
        assert abs(res["you"]["S"] - bar["check_10"]["per_rater_loo"]["R3"]["S"]) < 1e-4 and not notes, notes
    b = bar["check_10"]["bar"]
    assert b["best"] <= b["median"] <= b["worst"] and "R5" not in bar["bar_raters"]


def test_real_kit_no_leaks() -> None:
    if not _kit():
        return
    import re

    key = sc.load_json(KIT / "private" / "check_key.json")
    sources = {it["source"] for it in key["items"].values()}
    pattern = re.compile(r"im_\d\d|_arch")
    for d in ("check", "expert_package", "practice"):
        for p in (KIT / d).rglob("*"):
            if p.is_file() and p.suffix.lower() in {".json", ".csv", ".txt", ".html", ".md"}:
                assert not pattern.search(p.read_text(encoding="utf-8", errors="ignore")), p
            assert not pattern.search(p.name), p
    assert not (KIT / "check" / "calib_answers.json").exists()  # the answers live in private/ only
    answers = sc.load_json(KIT / "private" / "calib_answers.json")
    assert all(v["source"] == k for k, v in answers["items"].items())
    for p in (KIT / "check").rglob("*"):  # no answer-like file anywhere in the folder the candidate works in
        if p.is_file() and p.suffix == ".json":
            text = p.read_text(encoding="utf-8")
            assert '"raters"' not in text and '"expert"' not in text, p
    manifest = sc.load_json(KIT / "check" / "calib_manifest.json")
    text = json.dumps(manifest)
    assert '"expert":' not in text and '"raters":' not in text and not any(s in text for s in sources)
    for k in range(1, 11):
        a = (KIT / "check" / "calib" / f"CHK_{k:02d}.png").read_bytes()
        b = (KIT / "expert_package" / "images" / f"CHK_{k:02d}.png").read_bytes()
        assert a == b, k
        pos, chunks = 8, set()
        while pos < len(a):
            n = int.from_bytes(a[pos:pos + 4], "big")
            chunks.add(a[pos + 4:pos + 8].decode("latin-1"))
            pos += 12 + n
        assert chunks <= {"IHDR", "IDAT", "IEND"}, chunks
    prac = sc.load_json(KIT / "practice" / "manifest.json")
    assert prac["n_tasks"] == 25 and prac["hint_is_answer"] is True
    assert all(t["clip"] is None for t in prac["tasks"]) and all(e["clip"] is None for e in prac["images"])
    assert not (KIT / "practice" / "calib_manifest.json").exists()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
