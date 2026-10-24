"""Reproducibility tests of the final package (umud-run, configs/). GPL-3.0-only.

Run (light: one process, CSV caches + a few images, < 0.5 GB RSS, about 2 minutes):
    uv run --with pytest pytest -q tests/test_final_package.py
The data-dependent tests skip when data/ or artifacts/ (not in git) are absent.

What is proven
--------------
* configs/final_A.yaml rebuilds the best public file 02_K25b_B1J4.csv BYTE-FOR-BYTE from the cached per-image tables
  (layout and clips from the images, native-geometry factors confirmed by the image detectors, rater offsets
  re-estimated from the OSF xlsx); configs/C18.yaml rebuilds C18 byte-for-byte; the intermediate scored files B1 and
  B1_J4 are byte-identical rebuilds of config variants;
* configs/final_B.yaml is Final A without the LB-only Telemed MT constant: it differs from Final A only in the
  Telemed MT and PA cells, J4 and K25b touch disjoint cells, and the exact-additive prediction is checked;
* clips are found from image content and are exactly the 28 Telemed clips of 5 frames; the layout table equals the
  research-era table; the factor detectors recover synthetic factors and confirm the declared ones on the images;
* the measure stage re-measures images from the maps identically to the caches; the maps stage writes the kernels;
* pinned inputs / code, the LB-constant registry and the factor detectors refuse silent drift.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import resource
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from umud import clips as cm  # noqa: E402
from umud import native_geometry as ng  # noqa: E402
from umud import run as ur  # noqa: E402

CFG = REPO / "configs"
ART = REPO / "artifacts"
TEST_DIR = REPO / "data/umud-challenge-muscle-architecture-in-ultrasound-data/test_images_v2/test_set_v2"
LEGACY = ART / "legacy_research"
SCORED = {
    "C18": (ART / "candidates_v6/C18_C12_telemedFLraw.csv", 0.33526),
    "B1": (ART / "candidates_v8/submit_2026-10-23b/B1_C18_J1_J2.csv", 0.23931),
    "B1_J4": (ART / "candidates_v8/submit_2026-10-23b/B1_J4.csv", 0.23693),
    "K25b": (ART / "candidates_v9/submit_2026-10-24/02_K25b_B1J4.csv", 0.23297),
}
K25B_SHA256 = "364e9cc33823fec031a58854a20ff54ba6e71c72921657664926095583c3afaa"
TOL = 1e-4
HAVE_DATA = TEST_DIR.exists() and (ART / "candidates_v3/cache/test__v2.csv").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="competition data / per-image caches not present")
C3 = ur.C3


def variant(path: Path, overrides: dict | None = None) -> dict:
    """A config with changes; registered LB constants follow their targets when the new value is on the grid."""
    cfg = ur.load_config(path, overrides=overrides)
    for e in cfg["lb_constants"]:
        ts = e["target"] if isinstance(e["target"], list) else [e["target"]]
        vals = [ur.get_dotted(cfg, t) for t in ts]
        if all(ur._same(v, vals[0]) for v in vals) and any(ur._same(vals[0], g) for g in e["grid"]):
            e["value"] = vals[0]
    return cfg


def build_text(cfg: dict) -> tuple[str, dict]:
    return ur.Assembler(cfg, ur.bind(cfg)).submission()


def frame(text: str) -> pd.DataFrame:
    return ur.from_text(text)


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).set_index("image_id")[list(C3)].astype(float)


def cells(a: pd.DataFrame, b: pd.DataFrame, lay: pd.DataFrame, tol: float = 0.0) -> dict:
    out = {}
    g = lay.loc[a.index, "group"]
    for c in C3:
        d = (a[c] - b.reindex(a.index)[c]).abs() > tol
        for grp, n in d.groupby(g).sum().items():
            if n:
                out[f"{grp}|{ur.SHORT[c]}"] = int(n)
    return out


@pytest.fixture(scope="module")
def layout():
    return ur.Assembler(ur.load_config(CFG / "C18.yaml"), None).layout() if HAVE_DATA else None


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """C18, Final B and Final A through stage_build (reports, validation)."""
    out = {}
    for name in ("C18", "final_B", "final_A"):
        d = tmp_path_factory.mktemp(name)
        rep = ur.stage_build(ur.load_config(CFG / f"{name}.yaml"), d)
        out[name] = (rep, Path(rep["file"]))
    return out


# =================================================================================== unit: clip detection
def _texture(seed: int, h: int = 240, w: int = 320) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 3)
    return np.clip(128 + 60 * t / t.std(), 0, 255).astype(np.uint8)


def _frames(spec):
    rng = np.random.default_rng(99)
    for i, (seed, size) in enumerate(spec):
        g = _texture(seed, *size)
        g = np.clip(g.astype(np.float32) + rng.normal(0, 4, g.shape), 0, 255).astype(np.uint8)
        yield cm.FrameInfo(f"IMG_{i:05d}.tif", (g.shape[1], g.shape[0]), cm.thumbnail(g, None))


def test_clip_detection_synthetic():
    s, s2 = (240, 320), (200, 300)
    spec = [(1, s), (1, s), (1, s), (2, s), (3, s), (3, s), (4, s), (5, s), (5, s2), (6, s2), (6, s2)]
    r = cm.detect_clips(_frames(spec))
    #        A  A  A  B   C  C  D   E   E' (size change breaks the link)  F  F
    assert r.clip == [0, 0, 0, -1, 1, 1, -1, -1, -1, 2, 2]
    assert r.n_clips() == 3
    m = r.margins()
    assert m["min_within"] > 0.95 and m["max_across"] < 0.3
    assert np.isnan(r.ncc_prev[0]) and np.isnan(r.ncc_prev[8])          # first image; size change


def test_clip_detection_min_frames_and_threshold():
    s = (240, 320)
    spec = [(1, s), (1, s), (2, s), (2, s), (2, s)]
    assert cm.detect_clips(_frames(spec), min_frames=3).clip == [-1, -1, 0, 0, 0]
    assert cm.detect_clips(_frames(spec), threshold=1.01).clip == [-1] * 5


def test_thumbnail_is_viewport_crop_and_normalised():
    g = _texture(7, 400, 600)
    t = cm.thumbnail(g, (100, 50, 500, 350))
    assert t.shape == (120, 160) and abs(float(t.mean())) < 1e-4 and abs(float(t.std()) - 1) < 1e-3
    t2 = cm.thumbnail(g[50:350, 100:500], None)
    assert cm.ncc(t, t2) > 0.999


# =================================================================================== unit: factor detectors
def _lumify_like(c: float, r: float = 19.0) -> np.ndarray:
    """A Lumify-like frame top: a filled grey ellipse (native circle resampled by c) with a white letter."""
    S = 8
    big = np.zeros((100 * S, 300 * S), np.uint8)
    cv2.ellipse(big, (150 * S, 50 * S), (int(round(c * r * S)), int(round(r * S))), 0, 0, 360, 120, -1)
    cv2.rectangle(big, (146 * S, 44 * S), (153 * S, 57 * S), 255, -1)
    top = cv2.resize(big, (300, 100), interpolation=cv2.INTER_AREA)
    g = np.zeros((300, 600), np.uint8)
    g[:100, :300] = top
    return g


@pytest.mark.parametrize("c", [0.844, 0.9, 1.0])
def test_lumify_disc_recovers_synthetic_factor(c):
    d = ng.lumify_disc(_lumify_like(c))
    assert d["disc_found"]
    assert abs(d["c_ellipse"] - c) < 0.006, d["c_ellipse"]


@pytest.mark.parametrize("c", [1.0, 1.125, 0.9067])
def test_triangle_slope_recovers_synthetic_factor(c):
    S = 16
    h = 12                                   # native 45-degree triangle pointing left: half-height h px
    big = np.zeros((2 * h * S + 2 * S, 40 * S), np.uint8)
    tip, base = 30 * S - int(round(c * h * S)), 30 * S
    pts = np.array([[base, S], [base, S + 2 * h * S], [tip, S + h * S]], np.int32)
    cv2.fillPoly(big, [pts], 255)
    blk = cv2.resize(big, (40, 2 * h + 2), interpolation=cv2.INTER_AREA).astype(float)
    cov = blk.sum(1) / 255.0
    assert abs(ng.triangle_slope(cov) - c) < 0.01


def test_posthoc_transform_properties():
    pa, fl, dl = pd.Series([20.0, 45.0, 10.0]), pd.Series([80.0, 60.0, 100.0]), pd.Series([0.0, 0.0, 3.0])
    p1, f1 = ng.posthoc_transform(pa, fl, dl, 1.0)
    assert np.allclose(p1, pa) and np.allclose(f1, fl)
    p, f = ng.posthoc_transform(pa, fl, dl, 0.844)
    assert abs(p[1] - np.degrees(np.arctan(0.844))) < 1e-12               # 45 deg fascicle, flat aponeurosis
    assert (p < pa).all() and (f > fl).all()                               # c < 1: lower PA, longer FL
    p, f = ng.posthoc_transform(pa, fl, dl, 1.125)
    assert (p > pa).all() and (f < fl).all()                               # c > 1: higher PA, shorter FL
    _, f90 = ng.posthoc_transform(pd.Series([90.0]), pd.Series([50.0]), pd.Series([0.0]), 0.8)
    assert abs(f90[0] - 50.0) < 1e-9                                       # vertical lengths are unchanged


@needs_data
@pytest.mark.parametrize("group,detector,lo,hi", [("philips_lumify", "lumify_p_disc", 0.83, 0.85),
                                                  ("siemens_juniper", "siemens_ui_triangles", 1.115, 1.135),
                                                  ("telemed_artus_1200x800", "telemed_ticks", 0.905, 0.910),
                                                  ("telemed_artus_1088x644", "telemed_ticks", 0.997, 1.003)])
def test_detectors_on_test_images(layout, group, detector, lo, hi):
    ids = sorted(layout.index[layout["group"] == group])
    r = ng.confirm_factor([TEST_DIR / i for i in ids], detector, (lo + hi) / 2, (hi - lo) / 2)
    assert r["confirmed"] and not r["failed"] and r["n"] == len(ids)
    assert lo <= r["min"] and r["max"] <= hi, (r["min"], r["max"])


@needs_data
def test_detector_refuses_wrong_declared_factor(layout):
    ids = sorted(layout.index[layout["group"] == "philips_lumify"])[:5]
    with pytest.raises(ValueError, match="does not confirm"):
        ng.confirm_factor([TEST_DIR / i for i in ids], "lumify_p_disc", 0.80, 0.01)


# =================================================================================== layout + clips on the test set
@needs_data
def test_layout_from_images_equals_research_table_and_clips(tmp_path):
    from umud import layout as lm
    lay, s = lm.build_layout(TEST_DIR)
    assert len(lay) == 309 and s["n_clips"] == 28 and s["frames_per_clip"] == [5]
    assert s["min_within"] > 0.9 and s["max_across"] < 0.6
    assert set(s["clip_groups"]) == {"telemed_artus_1200x800", "telemed_artus_1088x644"}
    n = lay.index.str.extract(r"IMG_(\d+)", expand=False).astype(int).to_numpy()
    assert (lay["clip"].to_numpy() == np.where((n >= 56) & (n <= 195), (n - 56) // 5, -1)).all()
    old = LEGACY / "data-labels-geometry/out/test_layout.csv"
    if old.exists():
        b = pd.read_csv(old).set_index("image_id")
        assert list(lay.index) == list(b.index)
        for c in ("size", "ext", "family"):
            assert (lay[c].astype(str) == b[c].astype(str)).all(), c
        assert np.allclose(lay["px_per_cm"].to_numpy(), b["px_per_cm"].to_numpy(), rtol=0, atol=1e-9)
    lm.write_layout(lay, tmp_path / "l.csv")
    back = lm.read_layout(tmp_path / "l.csv")
    assert back["clip"].equals(lay["clip"])
    if old.exists():     # as read by the pipeline (CSV), px_per_cm is bit-identical to the research table
        assert (back["px_per_cm"].to_numpy() == b["px_per_cm"].to_numpy()).all()


# =================================================================================== config structure
@pytest.mark.parametrize("name", ["C18", "final_A", "final_B"])
def test_config_blocks(name):
    cfg = ur.load_config(CFG / f"{name}.yaml")
    for b in ur.REQUIRED_BLOCKS:
        assert b in cfg, b
    assert set(cfg["geometry_space"]["scanners"]) == set(cfg["groups"]["all"])
    for e in cfg["lb_constants"]:
        assert e["grid"] and str(e.get("rationale", "")).strip() and e.get("kind"), e["name"]
    assert "/private/tmp" not in json.dumps(cfg) and "scratchpad" not in json.dumps(cfg)


def test_final_A_is_final_B_plus_the_telemed_mt_constant_only():
    a, b = ur.strip_docs(ur.load_config(CFG / "final_A.yaml")), ur.strip_docs(ur.load_config(CFG / "final_B.yaml"))
    for d in (a, b):
        for k in ("name", "reference", "lb_constants", "output", "paths"):
            d.pop(k, None)
    assert a.pop("mt_edge")["constant_mm"] == {"default": 0.0, "telemed_artus_1200x800": -0.5,
                                               "telemed_artus_1088x644": -0.5}
    assert b.pop("mt_edge")["constant_mm"] == {"default": 0.0, "telemed_artus_1200x800": 0.0,
                                               "telemed_artus_1088x644": 0.0}
    assert a == b
    la = {e["name"]: e for e in ur.load_config(CFG / "final_A.yaml")["lb_constants"]}
    lb = {e["name"]: e for e in ur.load_config(CFG / "final_B.yaml")["lb_constants"]}
    assert set(la) == set(lb)
    assert [n for n in la if la[n]["value"] != lb[n]["value"]] == ["telemed_mt_offset"]
    assert lb["telemed_mt_offset"]["value"] == 0.0 and la["telemed_mt_offset"]["value"] == -0.5


@needs_data
def test_pinned_inputs_and_code_unchanged():
    for name in ("C18", "final_A", "final_B"):
        cfg = ur.load_config(CFG / f"{name}.yaml")
        rows = ur.check_hashes(cfg["inputs"]) + ur.check_hashes(cfg["code"])
        assert all(r["ok"] for r in rows) and len(cfg["inputs"]) >= 12 and len(cfg["code"]) >= 14


# =================================================================================== the reproductions
@needs_data
def test_C18_rebuilt_byte_for_byte(built):
    rep, path = built["C18"]
    assert path.read_bytes() == SCORED["C18"][0].read_bytes()
    assert rep["reference"]["bytes_identical"] and rep["validate"].startswith("Valid submission")
    assert rep["pins_ok"] and rep["peak_rss_gb"] < 1.5


@needs_data
def test_final_A_rebuilds_K25b_byte_for_byte(built):
    rep, path = built["final_A"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == K25B_SHA256
    assert path.read_bytes() == SCORED["K25b"][0].read_bytes()
    assert rep["reference"]["bytes_identical"] and rep["reference"]["official_distance"] == 0.0
    assert rep["reference_sha256"]["ok"] and rep["validate"].startswith("Valid submission")
    assert rep["layout"]["n_clips"] == 28 and rep["layout"]["frames_per_clip"] == [5]
    # factors confirmed by the detectors on every image of the group; offsets re-estimated from the xlsx
    gf = rep["geometry_factors"]
    assert gf["philips_lumify"]["confirmed"] and gf["philips_lumify"]["n"] == 58
    assert gf["siemens_juniper"]["confirmed"] and gf["siemens_juniper"]["n"] == 91
    assert rep["rater_style_reestimated"]["applied"]["pa_add"] == 0.3285798847465733
    assert rep["rater_style_reestimated"]["applied"]["mt_add"] == 0.03876938123115803
    steps = [s["rule"] for s in rep["post_rules"]["steps"]]
    assert steps == ["native_geometry", "native_geometry", "mt_constant", "coupling_pull"]


@needs_data
@pytest.mark.parametrize("name,base,ov", [
    ("B1", "final_B", {"pa_head": {"coupling_pull": {"weight": 0.0}}}),
    ("B1_J4", "final_A", {"pa_head": {"coupling_pull": {"weight": 0.0}}}),
])
def test_intermediate_scored_files_rebuilt_byte_for_byte(name, base, ov):
    text, _ = build_text(variant(CFG / f"{base}.yaml", ov))
    assert text.encode() == SCORED[name][0].read_bytes()


@needs_data
def test_final_B_cells_and_additivity(built, layout):
    A, B = read(built["final_A"][1]), read(built["final_B"][1])
    B1, B1J4, C18 = (read(SCORED[k][0]) for k in ("B1", "B1_J4", "C18"))
    tel = layout.loc[A.index, "group"].str.startswith("telemed")
    j4 = cells(B1J4, B1, layout)
    k25 = cells(A, B1J4, layout)
    assert set(j4) == {"telemed_artus_1200x800|mt", "telemed_artus_1088x644|mt"}
    assert set(k25) == {"telemed_artus_1200x800|pa", "telemed_artus_1088x644|pa"}
    assert not set(j4) & set(k25)                                        # J4 and K25b touch disjoint cells
    assert set(cells(B, B1, layout)) == set(k25)                         # Final B = B1 + its own Telemed PA
    assert set(cells(B, A, layout)) == set(j4) | set(k25)
    assert (B.loc[tel, "mt_mm"] == C18.loc[tel, "mt_mm"]).all()          # no MT constant in Final B
    # the exact-additive variant (B1 + Final A's Telemed PA): public score 0.23297 + (0.23931 - 0.23693)
    Bii = B1.copy()
    Bii.loc[tel, "pa_deg"] = A.loc[tel, "pa_deg"]
    assert set(cells(Bii, A, layout)) == set(j4)
    pred = SCORED["K25b"][1] + (SCORED["B1"][1] - SCORED["B1_J4"][1])
    assert round(pred, 5) == 0.23535
    # Final B differs from that variant on TEL.pa only (coupling with MT = C18 MT instead of C18 - 0.5)
    d = (B.loc[tel, "pa_deg"] - Bii.loc[tel, "pa_deg"])
    assert (d > 0).all() and 0.05 < d.mean() < 0.12
    per = np.mean(np.stack([(B[c] - Bii[c]).abs().to_numpy() / ur.TAU[c] for c in C3], 1), 1)
    assert abs(per.mean() - 0.00204) < 5e-5
    assert np.sort(per)[::-1][:102].mean() < 0.005                      # |LB(Final B) - 0.23535| bound
    rep = built["final_B"][0]
    assert rep["validate"].startswith("Valid submission") and "reference" not in rep
    assert rep["reference_sha256"]["ok"]                                # Final B is reproduced exactly


@needs_data
def test_finals_keep_clips_constant(built, layout):
    m = layout["clip"] >= 0
    for name in ("final_A", "final_B", "C18"):
        X = read(built[name][1])
        for c in C3:
            assert (X.loc[m, c].groupby(layout.loc[m, "clip"]).nunique() == 1).all(), (name, c)


LEDGER = {   # scored or built ledger files that are single config changes of C18 (4-decimal tolerance)
    "C12": (ART / "candidates_v5/C12_hyb_med_ens_pp_muscle5_emclean_off.csv",
            {"fl_rules": {"shrink_lambda": {"telemed_artus_1200x800": 1.0, "telemed_artus_1088x644": 1.0}}}),
    "C17": (ART / "candidates_v6/C17_C12_nooff.csv",
            {"fl_rules": {"shrink_lambda": {"telemed_artus_1200x800": 1.0, "telemed_artus_1088x644": 1.0}},
             "rater_style": {"mode": "frozen", "pa_add": 0.0, "mt_add": 0.0}}),
    "C21": (ART / "candidates_v7/C21_C18_telemedPAraw.csv", {"shrink": {"pa": {"lambda": {"default": 1.0, "telemed": 0.0}}}}),
    "C23": (ART / "candidates_v7/C23_C18_FLgate225.csv", {"fl_rules": {"eb_gate_fl_fov": {"enabled": True}}}),
    "J1": (LEGACY / "v8/judge/J1_C18_LUMphys084.csv", {"geometry_space": {"scanners": {"philips_lumify": {"space": "physical"}}}}),
    "J2": (LEGACY / "v8/judge/J2_C18_SIEphys1125.csv", {"geometry_space": {"scanners": {"siemens_juniper": {"space": "physical"}}}}),
    "J4": (LEGACY / "v8/judge/J4_C18_TELmt_m05.csv", {"mt_edge": {"constant_mm": {"default": 0.0, "telemed": -0.5}}}),
}


@needs_data
@pytest.mark.parametrize("name", sorted(LEDGER))
def test_config_variants_reproduce_ledger_files(name):
    path, ov = LEDGER[name]
    if not path.exists():
        pytest.skip(f"{path} not present")
    text, _ = build_text(variant(CFG / "C18.yaml", ov))
    got, ref = frame(text), read(path)
    d = (got - ref.reindex(got.index)).abs()
    assert float(d.to_numpy().max()) <= TOL, d.max().to_dict()
    assert ur.official_distance(got, ref) < 1e-6


# =================================================================================== rater style, measure, maps
@needs_data
def test_rater_style_rule_identical_to_recorded_rule(tmp_path):
    cfg = ur.load_config(CFG / "C18.yaml")
    est = ur.Assembler(cfg, ur.bind(cfg)).estimate_rater_style(tmp_path)
    rec = json.loads((ART / "candidates_v5/rater/recommended_offsets_C12.json").read_text())
    assert est["rule"] == rec["offsets"]
    assert len(est["decision"]) == len(rec["decision"]) == 12
    for a, b in zip(est["decision"], rec["decision"]):
        for k, v in b.items():
            assert (abs(a[k] - v) < 1e-12) if isinstance(v, float) else a[k] == v, (b["comp"], b["part"], k)
    assert abs(est["applied"]["pa_add"] - 0.5 * est["point"]["pa_deg_total_add_clean10"]) < 1e-12


@needs_data
def test_measure_stage_remeasures_from_maps(tmp_path, layout):
    cfg = ur.load_config(CFG / "final_A.yaml")
    rep = ur.stage_measure(cfg, tmp_path)
    s = rep["smoke"]
    assert set(layout.loc[s["ids"], "group"]) == set(cfg["groups"]["all"])       # every scanner group
    assert len(s["ids"]) >= 7 and s["n_columns"] > 200 and s["nan_mismatch"] == 0
    assert s["max_abs_diff"] <= 1e-9, s
    assert all(r["ok"] for r in rep["caches"]) and rep["peak_rss_gb"] < 1.5


@needs_data
def test_maps_stage_writes_kaggle_kernels(tmp_path):
    rep = ur.stage_maps(ur.load_config(CFG / "final_A.yaml"), tmp_path)
    assert rep["map_files_ok"] and len(rep["kernels"]) == 7
    for k in rep["kernels"]:
        assert k["in_repo"] and k["id"].startswith("shivateja005/")
        d = tmp_path / "kaggle" / k["slug"]
        meta = json.loads((d / "kernel-metadata.json").read_text())
        assert (d / meta["code_file"]).is_file()
        run = (d / "RUN.md").read_text()
        assert "kernels push" in run and "kernels output" in run
        assert "artifacts/final/maps/" in run.split("kernels output", 1)[1]    # never over the frozen maps


# =================================================================================== drift protection, CLI
@needs_data
def test_lb_registry_rejects_unregistered_or_off_grid_values():
    cfg = ur.load_config(CFG / "final_B.yaml")
    cfg["pa_head"]["coupling_pull"]["weight"] = 0.5          # on the grid but not the registered value
    with pytest.raises(ValueError, match="lb_constants"):
        ur.Assembler(cfg, ur.bind(cfg)).check_lb_constants()
    cfg = variant(CFG / "final_B.yaml", {"pa_head": {"coupling_pull": {"weight": 0.3}}})     # off the grid
    with pytest.raises(ValueError, match="grid"):
        ur.Assembler(cfg, ur.bind(cfg)).check_lb_constants()


@needs_data
def test_wrong_declared_factor_stops_the_build():
    cfg = variant(CFG / "final_B.yaml", {"geometry_space": {"scanners": {"philips_lumify": {"factor": 0.80}}}})
    with pytest.raises(ValueError, match="does not confirm"):
        build_text(cfg)


@needs_data
def test_code_and_input_drift_are_detected(tmp_path):
    cfg = ur.load_config(CFG / "C18.yaml")
    cfg["shrink"]["pinned_code_defaults"]["fit_group_prior"]["prior_sd_logtau"] = 0.5
    with pytest.raises(RuntimeError, match="code drift"):
        ur.bind(cfg)
    f = tmp_path / "x.csv"
    f.write_text("a\n1\n")
    with pytest.raises(RuntimeError, match="changed"):
        ur.check_hashes([{"path": str(f), "sha256": "0" * 64}])


@needs_data
def test_cli_entry_point(tmp_path):
    r = subprocess.run([sys.executable, str(REPO / "scripts/umud_run.py"), "--config", str(CFG / "final_A.yaml"),
                        "--stage", "build", "--out", str(tmp_path)], capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=str(REPO / "src")))
    assert r.returncode == 0, r.stdout + r.stderr
    assert hashlib.sha256((tmp_path / "final_A.csv").read_bytes()).hexdigest() == K25B_SHA256
    rep = json.loads((tmp_path / "final_A_report.json").read_text())
    assert rep["reference"]["bytes_identical"] and rep["pins_ok"]


def test_process_stayed_light():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    gb = r / 1e9 if sys.platform == "darwin" else r / 1e6
    assert gb < 1.5, gb
