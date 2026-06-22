import unittest
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from deploy.main import app
from deploy.rate_limit import reset_rate_limits


class LLMCarePlanTests(unittest.TestCase):
    def setUp(self):
        reset_rate_limits()
        self.client = TestClient(app)

    def test_weather_advice_includes_plant_context_in_hash_payload(self):
        captured = {}

        def fake_json(_, payload):
            captured.update(payload)
            return {
                "summary_vi": "Theo dõi cà chua sau mưa.",
                "symptoms": [],
                "causes": [],
                "treatments": [],
                "prevention": [],
                "when_to_seek_expert": "Khi bệnh lan nhanh.",
            }

        with patch("deploy.routers.llm.llm_cache_get", return_value=None):
            with patch("deploy.routers.llm.llm_cache_upsert"):
                with patch("deploy.routers.llm.call_gemini_json", side_effect=fake_json):
                    response = self.client.post(
                        "/llm/advice/weather",
                        json={
                            "lat": 10.0,
                            "lng": 106.0,
                            "weather_snapshot": {"temp": 30},
                            "plant": "Tomato",
                            "disease": "Late blight",
                        },
                    )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured["plant"], "Tomato")
        self.assertEqual(captured["disease"], "Late blight")

    def test_care_plan_endpoint_validates_and_normalizes_tasks(self):
        with patch("deploy.routers.llm.llm_cache_get", return_value=None):
            with patch("deploy.routers.llm.llm_cache_upsert"):
                with patch(
                    "deploy.routers.llm.call_gemini_json",
                    return_value={
                        "summary_vi": "Lịch chăm sóc",
                        "tasks": [
                            {
                                "title": "Tưới gốc",
                                "detail": "Tránh ướt lá.",
                                "category": "watering",
                                "due_in_days": 1,
                                "repeat_rule": "weekly",
                                "reminder_hour": 7,
                            }
                        ],
                        "checklist": ["Cắt lá bệnh"],
                        "safety_note": "Không tự ý dùng hoá chất.",
                    },
                ):
                    response = self.client.post(
                        "/llm/care-plan/diagnosis",
                        json={"plant": "Tomato", "disease": "Late blight", "confidence": 66},
                    )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["care_plan"]["tasks"][0]["category"], "watering")
        self.assertEqual(body["care_plan"]["checklist"], ["Cắt lá bệnh"])

    def test_care_plan_endpoint_falls_back_when_gemini_json_invalid(self):
        with patch("deploy.routers.llm.llm_cache_get", return_value=None):
            with patch("deploy.routers.llm.llm_cache_upsert") as upsert:
                with patch(
                    "deploy.routers.llm.call_gemini_json",
                    side_effect=HTTPException(
                        status_code=502,
                        detail="Gemini returned invalid JSON",
                    ),
                ):
                    response = self.client.post(
                        "/llm/care-plan/diagnosis",
                        json={
                            "plant": "Bell pepper",
                            "disease": "Bacterial spot",
                            "confidence": 72,
                        },
                    )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["fallback"])
        self.assertEqual(body["model"], "local-fallback")
        self.assertGreaterEqual(len(body["care_plan"]["tasks"]), 1)
        upsert.assert_not_called()


if __name__ == "__main__":
    unittest.main()
