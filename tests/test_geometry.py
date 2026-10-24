import unittest

import cv2
import numpy as np

from umud.geometry import (
    Line,
    architecture_from_masks,
    binarize_mask,
    fit_aponeuroses,
    infer_pixel_spacing_mm,
    is_dynamic_ultrasound,
    ultrasound_metadata,
    vertical_line_distance_px,
)


class GeometryTests(unittest.TestCase):
    def test_dense_binary_mask_is_not_erased(self) -> None:
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[:25] = 255
        self.assertEqual(int(binarize_mask(mask).sum()), 2500)

    def test_aponeurosis_pair_encloses_fascicles(self) -> None:
        apo = np.zeros((220, 300), dtype=np.uint8)
        fasc = np.zeros_like(apo)
        for y in (40, 140, 195):
            cv2.line(apo, (10, y), (290, y), 255, 3)
        cv2.line(fasc, (60, 138), (230, 42), 255, 4)

        upper, lower = fit_aponeuroses(apo, fasc)

        self.assertIsNotNone(upper)
        self.assertIsNotNone(lower)
        assert upper is not None and lower is not None
        self.assertAlmostEqual(float(upper.y(150)), 40.0, delta=1.0)
        self.assertAlmostEqual(float(lower.y(150)), 140.0, delta=1.0)

    def test_known_architecture_is_recovered(self) -> None:
        image = np.zeros((220, 300), dtype=np.uint8)
        apo = np.zeros_like(image)
        fasc = np.zeros_like(image)
        cv2.line(apo, (5, 50), (295, 50), 255, 3)
        cv2.line(apo, (5, 150), (295, 150), 255, 3)
        cv2.line(fasc, (45, 150), (218, 50), 255, 4)

        result = architecture_from_masks(image, apo, fasc, pixel_spacing_mm=0.1)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result.pa_deg, 30.0, delta=0.5)
        self.assertAlmostEqual(result.mt_mm, 10.0, delta=0.2)
        self.assertAlmostEqual(result.fl_mm, 20.0, delta=0.4)

    def test_known_scanner_spacing(self) -> None:
        image = np.zeros((800, 1200), dtype=np.uint8)
        image[42, 67:74] = 180
        self.assertAlmostEqual(
            infer_pixel_spacing_mm(image, image_path="frame.tif"),
            10.0 / ((783.0 - 42.0) / 5.0),
        )
        image = np.zeros((644, 1088), dtype=np.uint8)
        self.assertAlmostEqual(
            infer_pixel_spacing_mm(image, image_path="frame.tif"),
            10.0 / (630.5 / 5.0),
        )

    def test_known_scanner_viewports(self) -> None:
        image = np.zeros((644, 1088), dtype=np.uint8)
        metadata = ultrasound_metadata(image, "frame.tif")
        self.assertEqual(metadata.viewport, (140, 0, 947, 643))
        self.assertEqual(metadata.scanner_family, "telemed_artus")

        image = np.zeros((513, 465), dtype=np.uint8)
        metadata = ultrasound_metadata(image, "frame.tif")
        self.assertEqual(metadata.viewport, (0, 0, 465, 503))
        self.assertAlmostEqual(metadata.px_per_cm, 77.8)

    def test_vertical_thickness_is_not_perpendicular_thickness(self) -> None:
        upper = Line(slope=0.5, intercept=10.0)
        lower = Line(slope=0.5, intercept=110.0)
        self.assertAlmostEqual(vertical_line_distance_px(upper, lower, [25, 50, 75]), 100.0)

    def test_video_layout_detection(self) -> None:
        dynamic = np.zeros((800, 1200), dtype=np.uint8)
        dynamic[50:650, 200:1000] = 80
        static = np.zeros((800, 1200), dtype=np.uint8)
        static[88:697, 200:1000] = 80
        self.assertTrue(is_dynamic_ultrasound(dynamic, "frame.tif"))
        self.assertFalse(is_dynamic_ultrasound(static, "frame.tif"))


if __name__ == "__main__":
    unittest.main()
