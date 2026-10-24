"""Synthetic-mask tests for umud.geometry_v2 and umud.metric (known MT / PA / FL).

Runs with ``python -m unittest`` or pytest. Images are small; nothing heavy.
"""
import math
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from umud import geometry_v2 as g2
from umud.metric import TOL, per_image_score, score_arrays, umud_score

H, W, PPC = 400, 600, 100.0  # 100 px/cm -> 1 px = 0.1 mm


def band(mask, y0, y1, slope=0.0, x0=0, x1=W, curve=0.0, xc=W / 2):
    """Fill a band whose top edge is y0 + slope*(x-xc) + curve*(x-xc)^2 (thickness y1-y0)."""
    for x in range(x0, x1):
        yt = y0 + slope * (x - xc) + curve * (x - xc) ** 2
        mask[int(round(yt)):int(round(yt + (y1 - y0))), x] = 255


def fragments(mask, angle_deg, centers, length=80, thickness=1):
    t = math.radians(angle_deg)
    sh = 8  # sub-pixel endpoints (coordinates * 2**sh) so the drawn angle is exact
    for cx, cy in centers:
        dx, dy = 0.5 * length * math.cos(t), 0.5 * length * math.sin(t)
        p0 = (int(round((cx - dx) * 2 ** sh)), int(round((cy - dy) * 2 ** sh)))
        p1 = (int(round((cx + dx) * 2 ** sh)), int(round((cy + dy) * 2 ** sh)))
        cv2.line(mask, p0, p1, 255, thickness, cv2.LINE_8, sh)


def simple_scene(angle=-20.0, sup=(60, 71), deep=(260, 271), deep_slope=0.0):
    A = np.zeros((H, W), np.uint8)
    F = np.zeros((H, W), np.uint8)
    band(A, *sup)
    band(A, *deep, slope=deep_slope)
    ys = 0.5 * (sup[1] + deep[0])
    centers = [(x, ys + d) for x in (170, 250, 330, 410) for d in (-35, 0, 35)]
    fragments(F, angle + math.degrees(math.atan(deep_slope)), centers, length=140)
    return A, F


def lean(**kw):
    return g2.GeoConfig(compute_structure=False, compute_streamline=False, **kw)


class MetricTests(unittest.TestCase):
    def test_perfect_and_known_errors(self):
        t = {"pa_deg": np.array([10.0, 20.0]), "fl_mm": np.array([50.0, 60.0]), "mt_mm": np.array([15.0, 20.0])}
        self.assertAlmostEqual(score_arrays(t, t), 0.0, places=12)
        p = {"pa_deg": t["pa_deg"] + 6, "fl_mm": t["fl_mm"] - 12, "mt_mm": t["mt_mm"] + 3}
        # every component is exactly one tolerance off -> main term 1, tie-breakers add 1e-6 + 1e-9
        self.assertAlmostEqual(score_arrays(p, t), 1.0 + 1e-6 + 1e-9, places=9)
        np.testing.assert_allclose(per_image_score(p, t), [1.0, 1.0])

    def test_dataframe_interface_and_nan_policy(self):
        sol = pd.DataFrame({"image_id": ["a", "b"], "pa_deg": [10, 20], "fl_mm": [50, 60], "mt_mm": [15, 20]})
        sub = sol.copy()
        sub.loc[0, "mt_mm"] = 18.0
        sc, rep = umud_score(sol, sub, report=True)
        self.assertAlmostEqual(rep["mt_mm"]["mae"], 1.5)
        self.assertAlmostEqual(sc, (1.5 / TOL["mt_mm"]) / 3 + 1e-6 * (1.5 / 3) / 3 + 1e-9 * (math.sqrt(4.5) / 3) / 3, places=9)
        sub.loc[1, "pa_deg"] = np.nan
        with self.assertRaises(ValueError):
            umud_score(sol, sub)
        sc2 = umud_score(sol, sub, nan_policy="omit")
        self.assertTrue(np.isfinite(sc2))


class ConversionTests(unittest.TestCase):
    def test_as_prob_and_orient_decoding(self):
        self.assertEqual(g2.as_prob(np.array([[0, 255]], np.uint8)).tolist(), [[0.0, 1.0]])
        self.assertEqual(g2.as_prob(np.array([[False, True]])).tolist(), [[0.0, 1.0]])
        self.assertAlmostEqual(float(g2.as_prob(np.array([[127.5]]))[0, 0]), 0.5, places=5)
        enc = np.array([[0, 255, 128]], np.uint8)
        np.testing.assert_allclose(g2.decode_orient(enc), [[-1.0, 1.0, 128 / 255 * 2 - 1]], atol=1e-6)


class ParallelGeometryTests(unittest.TestCase):
    def test_known_mt_pa_fl_flat(self):
        A, F = simple_scene(-20.0)
        r = g2.measure(A, F, None, PPC, lean())
        mt_true = (260 - 71) * 0.1  # inner edges: bottom of superficial (71), top of deep (260)
        self.assertAlmostEqual(r.mt_mm, mt_true, delta=0.05)
        self.assertAlmostEqual(r.pa_deg, 20.0, delta=0.6)
        fl_true = mt_true / math.sin(math.radians(20.0))
        self.assertAlmostEqual(r.variants["fl_straight"], fl_true, delta=0.02 * fl_true)
        self.assertAlmostEqual(r.variants["fl_central3"], fl_true, delta=0.02 * fl_true)
        self.assertAlmostEqual(r.variants["fl_constructed"], fl_true, delta=0.02 * fl_true)
        self.assertAlmostEqual(r.variants["fl_triangle"], 1.065 * fl_true, delta=0.03 * fl_true)
        self.assertEqual(r.diag["pair"], "0,1")

    def test_pa_is_measured_against_the_deep_aponeurosis(self):
        # deep aponeurosis tilted by +4 deg, fascicles at 20 deg to it
        slope = math.tan(math.radians(4.0))
        A, F = simple_scene(-20.0, deep_slope=slope)
        r = g2.measure(A, F, None, PPC, lean())
        self.assertAlmostEqual(r.diag["deep_angle"], 4.0, delta=0.3)
        self.assertAlmostEqual(r.pa_deg, 20.0, delta=0.7)
        rh = g2.measure(A, F, None, PPC, lean(pa_reference="horizontal"))
        self.assertAlmostEqual(rh.pa_deg, 16.0, delta=0.7)

    def test_mirrored_image_gives_same_values(self):
        A, F = simple_scene(-18.0)
        r1 = g2.measure(A, F, None, PPC, lean())
        r2 = g2.measure(A[:, ::-1].copy(), F[:, ::-1].copy(), None, PPC, lean())
        self.assertAlmostEqual(r1.pa_deg, r2.pa_deg, delta=0.3)
        self.assertAlmostEqual(r1.mt_mm, r2.mt_mm, delta=0.05)
        self.assertAlmostEqual(r1.variants["fl_straight"], r2.variants["fl_straight"], delta=0.5)

    def test_px_units_without_scale(self):
        A, F = simple_scene(-20.0)
        r = g2.measure(A, F, None, None, lean())
        self.assertEqual(r.diag["units"], "px")
        self.assertAlmostEqual(r.mt_mm, 189.0, delta=0.5)

    def test_probability_maps_edge_level(self):
        # soft band edges: sub-pixel inner edge at the 0.5 level
        A, F = simple_scene(-20.0)
        Ap = cv2.GaussianBlur(A.astype(np.float32) / 255.0, (0, 0), 1.0)
        r = g2.measure(Ap, F, None, PPC, lean(apo_thr=0.5))
        self.assertAlmostEqual(r.mt_mm, 18.9, delta=0.1)

    def test_v1_preset_runs(self):
        A, F = simple_scene(-20.0)
        r = g2.measure(A, F, None, PPC, g2.GeoConfig.v1())
        self.assertAlmostEqual(r.pa_deg, 20.0, delta=0.6)
        self.assertAlmostEqual(r.mt_mm, 18.9, delta=0.1)

    def test_mt_positions(self):
        # converging aponeuroses: MT at 25/50/75 % of the shared range differs from 10/50/90 %
        A = np.zeros((H, W), np.uint8)
        F = np.zeros((H, W), np.uint8)
        band(A, 60, 70)
        band(A, 260, 270, slope=0.1)
        fragments(F, -20 + math.degrees(math.atan(0.1)), [(300, 170), (250, 160), (350, 175)])
        r = g2.measure(A, F, None, PPC, lean(mt_xpos=(0.5,)))
        self.assertAlmostEqual(r.mt_mm, (260 - 70) * 0.1, delta=0.1)  # centre column 300 (+0.5 px)
        self.assertAlmostEqual(r.variants["mt_q"], r.variants["mt_c"], delta=0.05)  # linear -> same mean


class CurvedGeometryTests(unittest.TestCase):
    def test_curved_deep_aponeurosis(self):
        A = np.zeros((H, W), np.uint8)
        F = np.zeros((H, W), np.uint8)
        band(A, 60, 70)
        c = 2e-4
        band(A, 250, 260, curve=c)
        fragments(F, -20.0, [(300, 170), (260, 160), (340, 180)], length=60)
        r = g2.measure(A, F, None, PPC, lean(edge_fit="poly2"), keep_details=True)
        # analytic: deep top edge y = 250 + c (x-300)^2, superficial bottom edge y = 70
        for f in r.fragments:
            if f["p_deep"] is None:
                continue
            xd, yd = f["p_deep"]
            self.assertAlmostEqual(yd, 250 + c * (xd - 300.5) ** 2, delta=1.0)
        line = g2.measure(A, F, None, PPC, lean(edge_fit="poly1"))
        # a straight fit of a curved aponeurosis shifts the MT (mean of the curve at 3 positions)
        xs = np.array([0.25, 0.5, 0.75]) * (W - 1)
        mt_true = np.mean(250 + c * (xs - 300) ** 2 - 70) * 0.1
        self.assertAlmostEqual(r.mt_mm, mt_true, delta=0.15)
        self.assertGreater(abs(line.mt_mm - mt_true), abs(r.mt_mm - mt_true) - 1e-9)

    def test_tangent_extrapolation_beyond_band_end(self):
        x = np.arange(100, 300, dtype=float)
        y = 200 + 1e-3 * (x - 200) ** 2
        tan = g2.EdgeCurve(x, y, "poly2", "tangent")
        pol = g2.EdgeCurve(x, y, "poly2", "poly")
        self.assertAlmostEqual(float(tan(250)), float(pol(250)), places=6)  # identical inside
        s_end = 2e-3 * (299 - 200)
        self.assertAlmostEqual(float(tan(400)), float(pol(299)) + s_end * (400 - 299), places=4)
        self.assertGreater(float(pol(400)), float(tan(400)))  # quadratic keeps bending
        lin = g2.EdgeCurve(x, y, "poly2", "end_linear")
        self.assertTrue(np.isfinite(float(lin(500))))


class PairSelectionTests(unittest.TestCase):
    def test_two_muscles_superficial_pair(self):
        A = np.zeros((H, W), np.uint8)
        F = np.zeros((H, W), np.uint8)
        band(A, 40, 50)
        band(A, 170, 180)
        band(A, 330, 340)
        fragments(F, -20, [(x, 110) for x in (150, 300, 450)])   # superficial muscle
        fragments(F, -15, [(x, 255) for x in (150, 300, 450)])   # deeper muscle, also has fascicles
        r = g2.measure(A, F, None, PPC, lean(pair_mode="superficial"))
        self.assertEqual(r.diag["pair"], "0,1")
        self.assertAlmostEqual(r.mt_mm, (170 - 50) * 0.1, delta=0.1)
        v1 = g2.measure(A, F, None, PPC, lean(pair_mode="coverage"))
        self.assertEqual(v1.diag["pair"], "0,2")  # documents the v1 failure mode

    def test_spurious_band_crossed_by_fascicles_is_skipped(self):
        A = np.zeros((H, W), np.uint8)
        F = np.zeros((H, W), np.uint8)
        band(A, 40, 50)
        band(A, 300, 310)
        band(A, 165, 175, x0=100, x1=500)  # bright septum inside the muscle
        fragments(F, -20, [(x, y) for x in (160, 260, 360, 460) for y in (110, 170, 230)], length=110)
        r = g2.measure(A, F, None, PPC, lean(pair_mode="superficial"))
        self.assertEqual(r.diag["pair"], "0,2")
        self.assertAlmostEqual(r.mt_mm, (300 - 50) * 0.1, delta=0.1)

    def test_single_band_returns_nan_with_reason(self):
        A = np.zeros((H, W), np.uint8)
        F = np.zeros((H, W), np.uint8)
        band(A, 40, 50)
        r = g2.measure(A, F, None, PPC, lean())
        self.assertTrue(math.isnan(r.mt_mm))
        self.assertEqual(r.diag["fail"], "no_aponeurosis_pair")

    def test_narrow_viewport_span(self):
        # Lumify-like layout: the ultrasound image occupies only columns 250-350 of a 600-px frame
        rng = np.random.default_rng(1)
        img = np.zeros((H, W), np.uint8)
        img[:, 250:350] = np.clip(80 + 30 * rng.normal(size=(H, 100)), 0, 255).astype(np.uint8)
        A = np.zeros((H, W), np.uint8)
        F = np.zeros((H, W), np.uint8)
        band(A, 60, 70, x0=250, x1=350)
        band(A, 260, 270, x0=250, x1=350)
        fragments(F, -20, [(300, 140), (300, 190)], length=60)
        self.assertTrue(math.isnan(g2.measure(A, F, img, PPC, lean()).mt_mm))  # 17 % of the width
        r = g2.measure(A, F, img, PPC, lean(apo_span_ref="viewport"))
        self.assertAlmostEqual(r.mt_mm, 19.0, delta=0.1)
        self.assertAlmostEqual(r.diag["viewport_w"], 100, delta=8)

    def test_fallback_apo_map_supplies_missing_band(self):
        A, F = simple_scene(-20.0)
        prim = A.copy()
        prim[:200] = 0  # primary model misses the superficial aponeurosis
        fb = A.copy()
        fb[200:] = 0
        self.assertTrue(math.isnan(g2.measure(prim, F, None, PPC, lean()).mt_mm))
        r = g2.measure(prim, F, None, PPC, lean(), apo_fallback=fb)
        self.assertAlmostEqual(r.mt_mm, 18.9, delta=0.05)
        self.assertIn("apo_fallback", r.flags)

    def test_broken_band_is_merged(self):
        A, F = simple_scene(-20.0)
        A[255:275, 280:320] = 0  # gap in the deep aponeurosis
        bands = g2.extract_bands(A, g2.GeoConfig(apo_merge=True, apo_min_span=0.6))
        self.assertEqual(len(bands), 2)


class OrientationTests(unittest.TestCase):
    def test_learned_field_pa_and_streamline(self):
        A, F = simple_scene(-20.0)
        th = math.radians(-20.0)
        c2 = np.full((H, W), math.cos(2 * th), np.float32)
        s2 = np.full((H, W), math.sin(2 * th), np.float32)
        enc = lambda v: np.round((v + 1) / 2 * 255).astype(np.uint8)  # noqa: E731
        r = g2.measure(A, F, None, PPC, g2.GeoConfig(compute_structure=False, sl_field="learned",
                                                     pa_method="field", fl_method="streamline"),
                       orient=(enc(c2), enc(s2)))
        self.assertAlmostEqual(r.variants["pa_field"], 20.0, delta=0.5)
        fl_true = 18.9 / math.sin(math.radians(20.0))
        self.assertAlmostEqual(r.fl_mm, fl_true, delta=0.02 * fl_true)
        self.assertEqual(r.diag["sl_field"], "learned")

    def test_fragment_field_streamline(self):
        A, F = simple_scene(-20.0)
        r = g2.measure(A, F, None, PPC, g2.GeoConfig(compute_structure=False, sl_field="fragments"))
        fl_true = 18.9 / math.sin(math.radians(20.0))
        self.assertAlmostEqual(r.variants["fl_streamline"], fl_true, delta=0.03 * fl_true)

    def test_classical_directional_pa_on_texture(self):
        rng = np.random.default_rng(0)
        A, F = simple_scene(-20.0)
        speckle = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), sigmaX=3.0, sigmaY=1.0)
        img = 80 + 25 * speckle
        lines = np.zeros((H, W), np.uint8)
        t = math.radians(-20.0)
        for k in range(-60, 60):
            x0 = 300 + k * 14 / math.sin(-t)
            cv2.line(lines, (int(x0 - 900 * math.cos(t)), int(200 - 900 * math.sin(t))),
                     (int(x0 + 900 * math.cos(t)), int(200 + 900 * math.sin(t))), 255, 2)
        img = img + 0.25 * cv2.GaussianBlur(lines.astype(np.float32), (0, 0), 1.0)
        img = img + (A > 0) * 120
        img = np.clip(img, 0, 255).astype(np.uint8)
        r = g2.measure(A, np.zeros_like(F), img, PPC, g2.GeoConfig(compute_streamline=False, pa_method="structure"))
        self.assertAlmostEqual(r.variants["pa_structure"], 20.0, delta=1.5)
        self.assertEqual(r.diag["pa_used"], "structure")  # works without any fascicle map


class MaskBankTests(unittest.TestCase):
    def test_roundtrip_lazy_npz(self):
        A, F = simple_scene(-20.0)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            np.savez_compressed(d / "test__apo.npz", IMG_00001=A)
            np.savez_compressed(d / "test__fasc.npz", IMG_00001=F)
            pd.DataFrame([dict(dataset="test", image_id="IMG_00001", source_filename="IMG_00001.tif",
                               height=H, width=W)]).to_csv(d / "manifest.csv", index=False)
            bank = g2.MaskBank(d)
            self.assertEqual(bank.models("test"), ["apo", "fasc"])
            r1 = g2.measure_from_bank(bank, "test", "IMG_00001.tif", "apo", "fasc", None, PPC, lean())
            r2 = g2.measure(A, F, None, PPC, lean())
            self.assertAlmostEqual(r1.pa_deg, r2.pa_deg, places=9)
            self.assertTrue(math.isnan(g2.measure_from_bank(bank, "test", "IMG_00002", "apo", "fasc").mt_mm))
            bank.close()


class ResultTests(unittest.TestCase):
    def test_as_row_and_flags(self):
        A, F = simple_scene(-20.0)
        r = g2.measure(A, F, None, PPC, lean())
        row = r.as_row()
        for k in ("pa_deg", "fl_mm", "mt_mm", "confidence", "flags", "pa_global", "fl_ensemble", "d_n_frag"):
            self.assertIn(k, row)
        self.assertGreater(r.confidence, 0.0)
        self.assertLessEqual(r.confidence, 1.0)
        F2 = np.zeros_like(F)
        fragments(F2, -20.0, [(300, 165)])
        r2 = g2.measure(A, F2, None, PPC, lean())
        self.assertIn("few_fragments", r2.flags)
        self.assertLess(r2.confidence, r.confidence)


if __name__ == "__main__":
    unittest.main()
