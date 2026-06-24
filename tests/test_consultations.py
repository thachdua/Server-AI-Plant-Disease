import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from deploy.main import app
from deploy.rate_limit import reset_rate_limits


class ConsultationRouterTests(unittest.TestCase):
    def setUp(self):
        reset_rate_limits()
        self.client = TestClient(app)

    def _image_file(self, name="plant.jpg"):
        return (name, b"fake image bytes", "image/jpeg")

    def test_expert_request_rejects_invalid_metadata_json(self):
        with patch("deploy.routers.consultations.require_authenticated_user", return_value="user-1"):
            response = self.client.post(
                "/consultations/expert-request",
                data={"metadata": "{not-json"},
                files={"primary_photo": self._image_file()},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("metadata must be valid JSON", response.text)

    def test_expert_request_rejects_more_than_four_total_photos(self):
        metadata = {"question": "Lá bị đốm nâu, cần chuyên gia kiểm tra."}
        files = [
            ("primary_photo", self._image_file("primary.jpg")),
            ("symptom_photos", self._image_file("symptom-1.jpg")),
            ("symptom_photos", self._image_file("symptom-2.jpg")),
            ("symptom_photos", self._image_file("symptom-3.jpg")),
            ("symptom_photos", self._image_file("symptom-4.jpg")),
        ]

        with patch("deploy.routers.consultations.require_authenticated_user", return_value="user-1"):
            response = self.client.post(
                "/consultations/expert-request",
                data={"metadata": json.dumps(metadata)},
                files=files,
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Upload at most 1 primary photo and 3 symptom photos", response.text)

    def test_expert_request_rejects_missing_question(self):
        metadata = {"title": "Cây cần kiểm tra"}

        with patch("deploy.routers.consultations.require_authenticated_user", return_value="user-1"):
            with patch("deploy.routers.consultations._validate_upload_metadata"):
                with patch(
                    "deploy.routers.consultations._sanitize_image",
                    return_value=(b"clean jpeg", "image/jpeg"),
                ):
                    with patch(
                        "deploy.routers.consultations._upload_consultation_image",
                        return_value="https://example.com/plant.jpg",
                    ):
                        response = self.client.post(
                            "/consultations/expert-request",
                            data={"metadata": json.dumps(metadata)},
                            files={"primary_photo": self._image_file()},
                        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("question is required", response.text)


if __name__ == "__main__":
    unittest.main()
