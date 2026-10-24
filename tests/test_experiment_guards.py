"""Prevent leakage in the curved-track pilot; no torch, model inference or GPU."""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


engine = load_file("curve_engine", ROOT / "kaggle/fasc_orient_v2/train.py")
prep = load_file("curve_preparation", ROOT / "scripts/prepare_curve_experiment.py")
fov = load_file("fov_benchmark", ROOT / "scripts/benchmark_fov.py")


class ExperimentIsolationTests(unittest.TestCase):
    @staticmethod
    def complete_vl_fixture(supervised=True):
        tracks = engine.decode_vl_tracks()
        vf = engine.subject_folds([rec["subject"] for rec in tracks.values()])
        items = [dict(ds="vl", src=key, tracks={"subject": rec["subject"]},
                      vfold=vf[rec["subject"]], train_pl=supervised) for key, rec in tracks.items()]
        return tracks, items

    def test_complete_embedded_coverage_has_expected_subject_split(self):
        tracks, items = self.complete_vl_fixture()
        supervised = engine.validate_complete_vl(items, tracks, [0], True)
        self.assertEqual(supervised["n_expected"], 245)
        self.assertEqual(supervised["n_matched"], 245)
        self.assertEqual(supervised["folds"]["0"], dict(n_val_items=50, n_val_subjects=25,
                                                       n_train_items=195, n_train_subjects=99))
        _, control_items = self.complete_vl_fixture(supervised=False)
        control = engine.validate_complete_vl(control_items, tracks, [0], False)
        self.assertEqual(control["folds"]["0"], dict(n_val_items=50, n_val_subjects=25))

    def test_failed_or_partial_external_download_cannot_pass_coverage(self):
        tracks, items = self.complete_vl_fixture()
        for incomplete in ([], items[:-1]):
            with self.subTest(n_images=len(incomplete)), self.assertRaisesRegex(RuntimeError, "coverage"):
                engine.validate_complete_vl(incomplete, tracks, [0], True)
        with self.assertRaisesRegex(RuntimeError, "duplicate matched source key"):
            engine.validate_complete_vl(items[:-1] + [items[0]], tracks, [0], True)

    def test_matched_source_with_wrong_subject_or_fold_is_rejected(self):
        tracks, items = self.complete_vl_fixture()
        wrong_subject = dict(items[0], tracks={"subject": "wrong subject"})
        wrong_fold = dict(items[0], vfold=(items[0]["vfold"] + 1) % engine.N_FOLDS)
        for wrong in (wrong_subject, wrong_fold):
            with self.subTest(item=wrong), self.assertRaisesRegex(RuntimeError, "metadata mismatch"):
                engine.validate_complete_vl([wrong] + items[1:], tracks, [0], True)

    def test_supervised_pilot_requires_all_other_subjects_in_training(self):
        tracks, items = self.complete_vl_fixture()
        for u in items:
            u["train_pl"] = False
        with self.assertRaisesRegex(RuntimeError, "no training subjects"):
            engine.validate_complete_vl(items, tracks, [0], True)
        for u in items:
            u["train_pl"] = True
        next(u for u in items if u["vfold"] != 0)["train_pl"] = False
        with self.assertRaisesRegex(RuntimeError, "incomplete VL training coverage"):
            engine.validate_complete_vl(items, tracks, [0], True)

    def test_pilot_requires_nonempty_held_out_subjects(self):
        tracks = {"set/subject/image.tiff": {"subject": "only subject"}}
        items = [dict(ds="vl", src=key, tracks={"subject": "only subject"}, vfold=0, train_pl=True)
                 for key in tracks]
        with self.assertRaisesRegex(RuntimeError, "no expected held-out records"):
            engine.validate_complete_vl(items, tracks, [1], True)
        with self.assertRaisesRegex(RuntimeError, "requires subject-fold validation"):
            engine.validate_complete_vl(items, tracks, [0], False, validation_by_fold=False)

    def test_held_out_vl_pseudo_copy_cannot_enter_training(self):
        items = [dict(ds="vl", vfold=0, train_pl=True, pseudo="teacher"),
                 dict(ds="vl", vfold=1, train_pl=True),
                 dict(ds="vl", vfold=-1, train_pl=True),
                 dict(ds="test", train_pl=True), dict(ds="static", train_pl=False)]
        pool = engine.fold_training_pool(items, 0, vl_supervised=True)
        self.assertEqual(pool, [items[1], items[3]])
        self.assertEqual(engine.fold_training_pool(items, 0, vl_supervised=False), items[:4])

    def test_numpy_only_training_targets_contain_curved_orientations(self):
        crop = np.zeros((160, 200), np.uint8)
        tracks = dict(fas=[[(20, 125), (95, 80), (180, 55)]],
                      sup=[0, 20, 1, 0], deep=[0, 140, 1, 0])
        sample = engine.make_sample(crop, None, tracks=tracks, track_train=True)
        gate = sample["ow"] > 0
        theta = np.degrees(0.5 * np.arctan2(sample["sin2"][gate], sample["cos2"][gate]))
        self.assertGreater(int(gate.sum()), 10)
        self.assertGreater(float(np.ptp(theta)), 10.0)
        self.assertEqual(int(sample["segm"].sum()), 0)  # tracks supervise angle, not fake dense masks

    def test_config_preparation_changes_literals_without_execution(self):
        code = 'PSEUDO = True  # guard\nVL_SUP = False\n'
        new = prep.rewrite_constants(code, {"PSEUDO": False, "VL_SUP": True})
        self.assertIn("PSEUDO = False", new)
        with self.assertRaises(ValueError):
            prep.rewrite_constants(code, {"UNKNOWN": True})

    def test_fov_crop_preserves_pixel_coordinates(self):
        x = fov.central_window(601, 0.35)
        a = np.tile(np.arange(601), (10, 1))
        np.testing.assert_array_equal(np.diff(a[:, x], axis=1), 1)
        self.assertEqual(x.stop-x.start, round(601*0.35))
        with self.assertRaises(ValueError):
            fov.central_window(601, 0)


if __name__ == "__main__":
    unittest.main()
