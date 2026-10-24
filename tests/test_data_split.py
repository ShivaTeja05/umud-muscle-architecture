import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from umud.data_split import deduplicate_pairs, grouped_split_pairs, image_fingerprint


class DataSplitTests(unittest.TestCase):
    def test_decoded_duplicate_pairs_are_removed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.arange(64, dtype=np.uint8).reshape(8, 8)
            mask = (image > 30).astype(np.uint8) * 255
            pairs = []
            for index in range(2):
                image_path = root / f"image_{index}.png"
                mask_path = root / f"mask_{index}.png"
                Image.fromarray(image).save(image_path)
                Image.fromarray(mask).save(mask_path)
                pairs.append((image_path, mask_path))

            unique, removed = deduplicate_pairs(pairs)

            self.assertEqual(len(unique), 1)
            self.assertEqual(removed, 1)

    def test_identical_images_cannot_cross_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pairs = []
            for index, value in enumerate((10, 10, 20, 30)):
                image_path = root / f"image_{index}.png"
                mask_path = root / f"mask_{index}.png"
                Image.fromarray(np.full((8, 8), value, np.uint8)).save(image_path)
                Image.fromarray(np.full((8, 8), index, np.uint8)).save(mask_path)
                pairs.append((image_path, mask_path))

            train, validation = grouped_split_pairs(pairs, val_fraction=0.5, seed=42)
            train_hashes = {image_fingerprint(pair[0]) for pair in train}
            validation_hashes = {image_fingerprint(pair[0]) for pair in validation}

            self.assertFalse(train_hashes & validation_hashes)
            self.assertEqual(len(train) + len(validation), len(pairs))


if __name__ == "__main__":
    unittest.main()
