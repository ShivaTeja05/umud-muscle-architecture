import unittest

import numpy as np
import pandas as pd

from umud.postprocess import apply_benchmark_calibration, finalize_predictions


class PostprocessTests(unittest.TestCase):
    def test_static_and_dynamic_calibrations_are_distinct(self) -> None:
        frame = pd.DataFrame(
            {
                "image_id": ["static.tif", "video.tif"],
                "pa_deg": [20.0, 20.0],
                "fl_mm": [80.0, 80.0],
                "mt_mm": [20.0, 20.0],
                "pixel_spacing_mm": [0.08, 0.08],
                "is_dynamic": [False, True],
            }
        )
        result = apply_benchmark_calibration(frame)
        self.assertAlmostEqual(result.loc[0, "pa_deg"], 20.452870, places=5)
        self.assertAlmostEqual(result.loc[1, "pa_deg"], 20.0, places=5)
        self.assertAlmostEqual(result.loc[1, "fl_mm"], 72.822508, places=5)

    def test_missing_values_are_filled_and_bounded(self) -> None:
        frame = pd.DataFrame(
            {
                "image_id": [f"frame_{index}.tif" for index in range(3)],
                "pa_deg": [12.0, np.nan, 100.0],
                "fl_mm": [60.0, np.nan, 500.0],
                "mt_mm": [20.0, np.nan, 80.0],
                "pixel_spacing_mm": [0.08, 0.08, 0.08],
                "is_dynamic": [False, False, False],
            }
        )
        result = finalize_predictions(frame, calibrated=False)
        self.assertFalse(result[["pa_deg", "fl_mm", "mt_mm"]].isna().any().any())
        self.assertLessEqual(result.pa_deg.max(), 45.0)
        self.assertLessEqual(result.fl_mm.max(), 200.0)
        self.assertLessEqual(result.mt_mm.max(), 50.0)


if __name__ == "__main__":
    unittest.main()
