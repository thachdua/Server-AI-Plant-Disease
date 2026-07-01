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

    def test_expert_request_from_url_rejects_missing_question(self):
        payload = {"image_url": "https://example.com/recovery.jpg", "question": "   "}

        with patch("deploy.routers.consultations.require_authenticated_user", return_value="user-1"):
            response = self.client.post("/consultations/expert-request-from-url", json=payload)

        self.assertEqual(response.status_code, 400)
        self.assertIn("question is required", response.text)

    def test_expert_request_from_url_rejects_invalid_image_url(self):
        payload = {"image_url": "ftp://example.com/recovery.jpg", "question": "Cây nặng hơn."}

        with patch("deploy.routers.consultations.require_authenticated_user", return_value="user-1"):
            response = self.client.post("/consultations/expert-request-from-url", json=payload)

        self.assertEqual(response.status_code, 400)
        self.assertIn("image_url must be an http or https URL", response.text)

    def test_expert_request_from_url_creates_pending_request(self):
        captured = {}

        def fake_insert(row):
            captured["row"] = row
            return {
                **row,
                "id": "consult-1",
            }

        payload = {
            "image_url": "https://example.com/recovery.jpg",
            "title": "Cây cà chua gửi lại",
            "question": "Cây có dấu hiệu nặng hơn sau theo dõi.",
            "contact_phone": "0900000000",
            "user_plant_id": "plant-1",
            "storage_consent": True,
            "questionnaire_json": {"Triệu chứng": "Đốm lá"},
            "profile_snapshot": {"name": "Nguyen"},
            "diagnosis_context": {"source": "recovery_check_in"},
            "notify_email": False,
            "notify_local": True,
            "user_email": "user@example.com",
        }

        with patch("deploy.routers.consultations.require_authenticated_user", return_value="user-1"):
            with patch("deploy.routers.consultations._upload_consultation_image") as upload:
                with patch("deploy.routers.consultations._insert_consultation_request", side_effect=fake_insert):
                    response = self.client.post("/consultations/expert-request-from-url", json=payload)

        self.assertEqual(response.status_code, 200)
        upload.assert_not_called()
        row = captured["row"]
        self.assertEqual(row["created_by"], "user-1")
        self.assertEqual(row["image_url"], payload["image_url"])
        self.assertEqual(row["primary_photo_url"], payload["image_url"])
        self.assertEqual(row["photo_urls"], [payload["image_url"]])
        self.assertEqual(row["status"], "pending")
        self.assertEqual(row["diagnostic_flow_version"], "recovery_followup_v1")

        body = response.json()
        self.assertEqual(body["id"], "consult-1")
        self.assertEqual(body["status"], "pending")
        self.assertEqual(body["photo_urls"], [payload["image_url"]])
        self.assertEqual(body["primary_photo_url"], payload["image_url"])
        self.assertTrue(body["expected_reply_at"])


if __name__ == "__main__":
    unittest.main()
