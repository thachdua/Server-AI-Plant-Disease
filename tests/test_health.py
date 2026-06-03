import unittest

from fastapi.testclient import TestClient

from deploy.main import app
from deploy.config import config_status


class HealthTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_is_lightweight(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_ready_reports_missing_required_without_secrets(self):
        response = self.client.get("/health/ready")
        body = response.json()

        self.assertIn(response.status_code, {200, 503})
        self.assertIn(body["status"], {"ok", "degraded"})
        self.assertIn("config", body)
        self.assertIn("missing_required", body["config"])
        self.assertIn("warnings", body["config"])
        serialized = str(body)
        self.assertNotIn("your_service_role_key", serialized)
        self.assertNotIn("DB_PASSWORD=", serialized)

    def test_config_status_shape(self):
        status = config_status()

        self.assertIn("ok", status)
        self.assertIn("missing_required", status)
        self.assertIn("warnings", status)
        self.assertIn("supabase_key_source", status)
        self.assertIn("optional", status)
        self.assertIn("predict", status)
        self.assertIn("limits", status)
        self.assertIn("database", status)


if __name__ == "__main__":
    unittest.main()
