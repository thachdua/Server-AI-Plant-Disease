import io
import unittest

from fastapi import HTTPException
from PIL import Image

from deploy.routers.predict import (
    _validate_image,
    infer_plant_from_disease_label,
    parse_confidence_percent,
)


class PredictHelperTests(unittest.TestCase):
    def test_infer_plant_from_common_model_labels(self):
        self.assertEqual(infer_plant_from_disease_label("Tomato___Late_blight"), "Tomato")
        self.assertEqual(infer_plant_from_disease_label("Apple_Black_rot"), "Apple")
        self.assertEqual(infer_plant_from_disease_label("Grape Black rot"), "Grape")
        self.assertIsNone(infer_plant_from_disease_label(None))

    def test_parse_confidence_percent(self):
        self.assertEqual(parse_confidence_percent(0.8234), "82.34%")
        self.assertEqual(parse_confidence_percent(82.345), "82.34%")
        self.assertEqual(parse_confidence_percent("91.2%"), "91.20%")
        self.assertIsNone(parse_confidence_percent("unknown"))

    def test_validate_image_accepts_jpeg(self):
        buf = io.BytesIO()
        Image.new("RGB", (8, 8), color=(0, 128, 0)).save(buf, format="JPEG")

        self.assertEqual(_validate_image(buf.getvalue()), "image/jpeg")

    def test_validate_image_rejects_non_image(self):
        with self.assertRaises(HTTPException) as ctx:
            _validate_image(b"not an image")

        self.assertEqual(ctx.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
