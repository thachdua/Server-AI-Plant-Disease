import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from deploy.main import app
from deploy.rate_limit import reset_rate_limits


class RateLimitTests(unittest.TestCase):
    def setUp(self):
        reset_rate_limits()
        self.client = TestClient(app)

    def test_llm_chat_rejects_large_prompt_before_gemini_call(self):
        with patch("deploy.routers.llm.LLM_CHAT_MAX_CHARS", 10):
            response = self.client.post(
                "/llm/chat",
                json={
                    "mode": "agriculture",
                    "messages": [{"role": "user", "text": "x" * 20}],
                },
            )

        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["detail"], "Chat prompt is too long")

    def test_llm_rate_limit_blocks_repeated_calls_before_gemini_call(self):
        with patch("deploy.routers.llm.LLM_RATE_LIMIT_PER_MINUTE", 1):
            with patch("deploy.routers.llm.call_gemini_text", return_value="ok"):
                first = self.client.post(
                    "/llm/chat",
                    json={
                        "mode": "agriculture",
                        "messages": [{"role": "user", "text": "xin chào"}],
                    },
                )
                second = self.client.post(
                    "/llm/chat",
                    json={
                        "mode": "agriculture",
                        "messages": [{"role": "user", "text": "xin chào"}],
                    },
                )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 429)


if __name__ == "__main__":
    unittest.main()
