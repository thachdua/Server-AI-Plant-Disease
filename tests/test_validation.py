import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from deploy.main import app
from deploy.rate_limit import reset_rate_limits


class ValidationTests(unittest.TestCase):
    def setUp(self):
        reset_rate_limits()
        self.client = TestClient(app)

    def test_weather_rejects_invalid_coordinates_before_openweather(self):
        with patch("deploy.routers.weather.requests.get") as get:
            response = self.client.get("/weather", params={"lat": 120, "lng": 106})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "lat must be between -90 and 90")
        get.assert_not_called()

    def test_llm_weather_rejects_invalid_coordinates_before_openweather(self):
        with patch("deploy.routers.llm.requests.get") as get:
            response = self.client.post(
                "/llm/advice/weather",
                json={"lat": 10.0, "lng": 999.0},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "lng must be between -180 and 180")
        get.assert_not_called()

    def test_outbreaks_rejects_invalid_severity_before_supabase(self):
        response = self.client.get("/outbreaks", params={"severity": 9})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "severity must be between 1 and 5")

    def test_outbreak_areas_rejects_invalid_since_days_before_supabase(self):
        response = self.client.get("/outbreaks/areas", params={"since_days": 3651})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "since_days must be between 1 and 3650")

    def test_outbreak_areas_accepts_ten_year_window_before_querying(self):
        with patch("deploy.routers.outbreaks.requests.get") as get:
            get.return_value.status_code = 502
            response = self.client.get("/outbreaks/areas", params={"since_days": 3650})

        self.assertNotEqual(response.status_code, 400)

    def test_llm_diagnosis_requires_disease(self):
        response = self.client.post(
            "/llm/advice/diagnosis",
            json={"disease": "   "},
        )

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
