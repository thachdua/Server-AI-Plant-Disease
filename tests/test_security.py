import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from deploy.main import app
from deploy.rate_limit import reset_rate_limits


class SecurityMiddlewareTests(unittest.TestCase):
    def setUp(self):
        reset_rate_limits()
        self.client = TestClient(app)

    def test_security_headers_are_added(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")
        self.assertEqual(response.headers["x-frame-options"], "DENY")
        self.assertEqual(response.headers["referrer-policy"], "no-referrer")

    def test_json_endpoint_rejects_wrong_content_type(self):
        response = self.client.post(
            "/llm/chat",
            content='{"messages":[{"role":"user","text":"hi"}]}',
            headers={"content-type": "text/plain"},
        )

        self.assertEqual(response.status_code, 415)
        self.assertEqual(response.json()["detail"], "Expected application/json")

    def test_json_endpoint_rejects_large_declared_body(self):
        with patch("deploy.security.SECURITY_MAX_JSON_BYTES", 16):
            response = self.client.post(
                "/llm/chat",
                json={
                    "mode": "agriculture",
                    "messages": [{"role": "user", "text": "x" * 40}],
                },
            )

        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["detail"], "JSON body is larger than allowed")

    def test_global_rate_limit_blocks_repeated_requests(self):
        with patch("deploy.security.SECURITY_GLOBAL_RATE_LIMIT_PER_MINUTE", 1):
            first = self.client.get("/does-not-exist")
            second = self.client.get("/does-not-exist")

        self.assertEqual(first.status_code, 404)
        self.assertEqual(second.status_code, 429)

    def test_chat_rejects_extra_fields_before_router_logic(self):
        response = self.client.post(
            "/llm/chat",
            json={
                "mode": "agriculture",
                "messages": [{"role": "user", "text": "hi"}],
                "unexpected": "<script>alert(1)</script>",
            },
        )

        self.assertEqual(response.status_code, 422)

    def test_predict_rejects_non_image_upload_metadata(self):
        response = self.client.post(
            "/predict",
            data={"selected_plant": "Tomato"},
            files={"file": ("payload.js", b"alert(1)", "application/javascript")},
        )

        self.assertEqual(response.status_code, 415)
        self.assertEqual(
            response.json()["detail"],
            "Only JPEG, PNG, or WebP uploads are accepted",
        )


if __name__ == "__main__":
    unittest.main()
