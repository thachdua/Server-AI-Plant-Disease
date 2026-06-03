import io
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from PIL import Image

from deploy.main import app
from deploy.rate_limit import reset_rate_limits


def jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (12, 12), color=(40, 160, 80)).save(buf, format="JPEG")
    return buf.getvalue()


class PredictEndpointTests(unittest.TestCase):
    def setUp(self):
        reset_rate_limits()
        self.client = TestClient(app)

    def test_predict_success_normalizes_response(self):
        hf_response = Mock()
        hf_response.status_code = 200
        hf_response.json.return_value = {
            "status": "success",
            "plant": "Tomato",
            "disease": "Tomato___Late_blight",
            "confidence": 0.9234,
        }

        with patch("deploy.routers.predict.requests.post", return_value=hf_response):
            with patch("deploy.routers.predict.print"):
                with patch(
                    "deploy.routers.predict._upload_image_to_supabase",
                    return_value="https://example.com/predictions/image.jpg",
                ):
                    response = self.client.post(
                        "/predict",
                        data={"selected_plant": "Tomato"},
                        files={"file": ("leaf.jpg", jpeg_bytes(), "image/jpeg")},
                    )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "status": "success",
                "plant": "Tomato",
                "disease": "Tomato___Late_blight",
                "confidence": "92.34%",
                "image_url": "https://example.com/predictions/image.jpg",
            },
        )

    def test_predict_rejects_incomplete_model_result(self):
        hf_response = Mock()
        hf_response.status_code = 200
        hf_response.json.return_value = {
            "status": "success",
            "plant": "Tomato",
            "confidence": 0.9,
        }

        with patch("deploy.routers.predict.requests.post", return_value=hf_response):
            with patch("deploy.routers.predict.print"):
                response = self.client.post(
                    "/predict",
                    data={"selected_plant": "Tomato"},
                    files={"file": ("leaf.jpg", jpeg_bytes(), "image/jpeg")},
                )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.json()["detail"],
            "Prediction service returned incomplete result",
        )


if __name__ == "__main__":
    unittest.main()
