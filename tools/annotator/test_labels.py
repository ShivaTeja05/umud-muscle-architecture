"""Unit checks for the annotator's label maths (run: python tools/annotator/test_labels.py)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import labels  # noqa: E402

MANIFEST = {
    "images": [
        {"image_id": f"IMG_{n:05d}.tif", "px_per_cm": 126.1} for n in range(56, 61)
    ] + [{"image_id": "IMG_00001.tif", "px_per_cm": 135.3333}],
    "tasks": [
        {"task_id": "IMG_00001.tif", "image_id": "IMG_00001.tif", "clip": None,
         "members": ["IMG_00001.tif"]},
        {"task_id": "IMG_00056.tif", "image_id": "IMG_00056.tif", "clip": 0,
         "members": [f"IMG_{n:05d}.tif" for n in range(56, 61)]},
    ],
}
CLIP_LABEL = {
    "mt": [[[300, 100], [300, 300]], [[500, 90], [500, 300]], [[700, 110], [700, 300]]],
    "fl": [[[400, 300], [500, 150], [560, 60]]] * 3,
    "pa": [[[400, 300], [450, 260], [350, 310], [500, 305]]] * 3,
    "updated": 2,
}


def test_values() -> None:
    mmpp = 10 / 126.1
    assert math.isclose(labels.mt_value([[0, 10], [0, 210]], mmpp), 200 * mmpp)
    assert math.isclose(labels.mt_value([[0, 210], [0, 10]], mmpp), 200 * mmpp)  # order-free
    assert math.isclose(labels.fl_value([[0, 0], [3, 4], [3, 10]], 1.0), 11.0)
    # acute angle between the lines, independent of click direction
    assert math.isclose(labels.angle_between([0, 0], [1, -1], [0, 0], [1, 0]), 45.0)
    assert math.isclose(labels.angle_between([1, -1], [0, 0], [0, 0], [1, 0]), 45.0)
    assert math.isclose(labels.angle_between([0, 0], [1, 1], [1, 0], [0, 0]), 45.0)
    assert math.isclose(labels.angle_between([0, 0], [0, 1], [0, 0], [1, 0]), 90.0)


def test_summary_and_clip_expansion() -> None:
    state = {"labels": {"IMG_00056.tif": CLIP_LABEL}, "times": {}}
    rows = labels.build_rows(state, MANIFEST)
    assert [r["image_id"] for r in rows] == [f"IMG_{n:05d}.tif" for n in range(56, 61)]
    assert {(r["pa_deg"], r["fl_mm"], r["mt_mm"]) for r in rows} == {("36.751", "22.874", "15.860")}
    assert all(r["complete"] == 1 for r in rows)
    clicks = json.loads(rows[0]["clicks_json"])
    assert clicks["task_id"] == "IMG_00056.tif" and len(clicks["pa"]) == 3


def test_merge_newest_wins() -> None:
    disk = {"labels": {"A": {"mt": [1], "updated": 5}, "B": {"mt": [2], "updated": 1}}}
    incoming = {"labels": {"A": {"mt": [], "updated": 3}, "B": {"mt": [3], "updated": 9}}}
    merged, newer = labels.merge_states(disk, incoming)
    assert merged["labels"]["A"]["mt"] == [1] and merged["labels"]["B"]["mt"] == [3]
    assert newer == ["A"]


def _pa_clicks(fas_deg: float, apo_deg: float, x0: float = 400.0, y0: float = 300.0) -> list:
    """4 PA clicks: fascicle rising to the right at fas_deg vs horizontal, apo at apo_deg (y down)."""
    f = [x0 + 80 * math.cos(math.radians(fas_deg)), y0 - 80 * math.sin(math.radians(fas_deg))]
    a = [x0 + 120 * math.cos(math.radians(apo_deg)), y0 - 120 * math.sin(math.radians(apo_deg))]
    return [[x0, y0], f, [x0 - 30, y0], a] if apo_deg == 0 else [[x0, y0], f, [x0, y0], a]


def _fl_clicks(length_px: float, deg: float) -> list:
    """3-point straight polyline of the given length."""
    dx, dy = math.cos(math.radians(deg)), -math.sin(math.radians(deg))
    return [[200, 400], [200 + dx * length_px / 2, 400 + dy * length_px / 2],
            [200 + dx * length_px, 400 + dy * length_px]]


CALIB = {
    "kind": "calib",
    "images": [{"image_id": "CAL_01", "px_per_cm": 150.0, "kind": "static"},
               {"image_id": "CAL_02", "px_per_cm": 77.4, "kind": "dynamic"}],
    "tasks": [{"task_id": "CAL_01", "image_id": "CAL_01", "clip": None, "members": ["CAL_01"], "kind": "static",
               "need": {"mt": 3, "fl": 3, "pa": 3}},
              {"task_id": "CAL_02", "image_id": "CAL_02", "clip": None, "members": ["CAL_02"], "kind": "dynamic",
               "need": {"mt": 0, "fl": 3, "pa": 3}}],
}
ANSWERS = {"tolerances": {"pa_deg": 6.0, "fl_mm": 12.0, "mt_mm": 3.0}, "items": {
    "CAL_01": {"source": "im_x", "kind": "static", "px_per_cm": 150.0, "pipeline_honest": True,
               "expert": {"pa_deg": 20.0, "fl_mm": 60.0, "mt_mm": 15.0},
               "raters": {"R1": {"pa_deg": 19.0, "fl_mm": 58.0, "mt_mm": 14.8},
                          "R2": {"pa_deg": 21.0, "fl_mm": 62.0, "mt_mm": 15.2}}},
    "CAL_02": {"source": "frame_x", "kind": "dynamic", "px_per_cm": 77.4, "pipeline_honest": True,
               "expert": {"pa_deg": 12.0, "fl_px": 300.0, "fl_mm": 3000 / 77.4, "mt_mm": None},
               "raters": {"Rater_1": {"pa_deg": 11.0, "fl_px": 290.0}, "Rater_2": {"pa_deg": 13.0, "fl_px": 310.0}}},
}}


def _perfect_state() -> dict:
    static = {"mt": [[[300 + 100 * k, 100], [300 + 100 * k, 100 + 15 * 15.0]] for k in range(3)],
              "fl": [_fl_clicks(60 * 15.0, 20)] * 3, "pa": [_pa_clicks(20, 0)] * 3, "updated": 1}
    # dynamic: fascicle 12 deg vs the horizontal, deep apo tilted -5 deg -> 17 deg vs the apo
    dynamic = {"mt": [], "fl": [_fl_clicks(300.0, 12)] * 3, "pa": [_pa_clicks(12, -5)] * 3, "updated": 1}
    return {"labels": {"CAL_01": static, "CAL_02": dynamic}, "times": {}}


def test_calib_rows_need_and_horizontal() -> None:
    rows = labels.build_calib_rows(_perfect_state(), CALIB)
    by = {r["task_id"]: r for r in rows}
    assert by["CAL_01"]["complete"] == 1 and by["CAL_01"]["mt_mm"] == "15.000" and by["CAL_01"]["px_per_cm"] == 150
    assert labels.rows_to_csv(rows, labels.CALIB_HEADER).splitlines()[1].split(",")[-3] == "150"  # like JS String()
    assert by["CAL_02"]["complete"] == 1 and by["CAL_02"]["n_mt"] == 0 and by["CAL_02"]["need_mt"] == 0
    assert by["CAL_02"]["pa_h_deg"] == "12.000" and by["CAL_02"]["pa_deg"] == "17.000"
    assert by["CAL_02"]["fl_px"] == "300.000" and by["CAL_02"]["mt_mm"] == ""
    # without the need override the dynamic task would not be complete
    assert not labels.summarize(_perfect_state()["labels"]["CAL_02"], 77.4)["complete"]


def test_score_perfect_labels() -> None:
    import score_calibration as sc
    user = sc.user_values(_perfect_state(), CALIB)
    res = sc.score(user, ANSWERS, None)
    you = res["you"]["components"]
    assert you["pa_deg"]["n"] == 2 and you["fl_mm"]["n"] == 2 and you["mt_mm"]["n"] == 1
    for c in ("pa_deg", "fl_mm", "mt_mm"):
        assert abs(you[c]["mae"]) < 1e-6, (c, you[c])
    dyn = [r for r in res["per_image"] if r["task_id"] == "CAL_02"][0]
    assert abs(dyn["pa_vs_deep_apo_diagnostic"]["err_vs_expert"] - 5.0) < 1e-6
    # expert spread: rater vs the other rater (2 raters) on PA = |19 - 21| for both images
    assert abs(res["experts_on_all_12"]["pa_deg"]["rater_vs_mean_of_others"]["mae"] - 2.0) < 1e-9
    assert abs(you["pa_deg"]["vs_single_rater_mae"] - 1.0) < 1e-6
    pipe = {"name": "P", "items": {"CAL_01": {"pa_deg": 22.0, "fl_mm": 60.0, "mt_mm": 15.0},
                                   "CAL_02": {"pa_deg": 10.0, "fl_mm": 3000 / 77.4, "mt_mm": None}}}
    res = sc.score(user, ANSWERS, pipe)
    assert abs(res["pipeline"]["all"]["components"]["pa_deg"]["mae"] - 2.0) < 1e-9


def test_serve_denies_answers() -> None:
    import tempfile
    import serve
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        for name in ("calib_manifest.json", "calib_answers.json", "calib_score.json", "calib_pipeline.json",
                     "manifest.json"):
            (d / name).write_text("{}")
        (d / "backups").mkdir()
        (d / "backups" / "autosave_x.json").write_text("{}")
        assert serve.servable(d, "calib_manifest.json") is not None and serve.servable(d, "manifest.json") is not None
        for bad in ("calib_answers.json", "CALIB_ANSWERS.json", "calib/../calib_answers.json", "calib_score.json",
                    "calib_pipeline.json", "backups/autosave_x.json", "../x.json", "manifest.py",
                    "calib_answers.json\x00", "x\x00.json"):
            assert serve.servable(d, bad) is None, bad


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
