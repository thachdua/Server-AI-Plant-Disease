import unittest
from unittest.mock import Mock, patch

from fastapi import HTTPException

from deploy.models import SaveHistoryRequest
from deploy.routers.history import save_history


class HistoryRouterTests(unittest.IsolatedAsyncioTestCase):
    async def test_save_history_uses_authenticated_user(self):
        req = SaveHistoryRequest(
            plant="Tomato",
            disease="Late blight",
            confidence=91.2,
            image_url="https://example.com/image.jpg",
        )

        with patch("deploy.routers.history.require_authenticated_user", return_value="user-1"):
            with patch("deploy.routers.history.save_to_db") as save_to_db:
                response = await save_history(req, Mock())

        self.assertEqual(response, {"status": "success"})
        save_to_db.assert_called_once_with(
            "Tomato",
            "Late blight",
            91.2,
            "https://example.com/image.jpg",
            "user-1",
        )

    async def test_save_history_reports_database_failure(self):
        req = SaveHistoryRequest(
            plant="Tomato",
            disease="Late blight",
            confidence=91.2,
            image_url="https://example.com/image.jpg",
        )

        with patch("deploy.routers.history.require_authenticated_user", return_value="user-1"):
            with patch("deploy.routers.history.save_to_db", side_effect=Exception("db down")):
                with patch("deploy.routers.history.print"):
                    with self.assertRaises(HTTPException) as ctx:
                        await save_history(req, Mock())

        self.assertEqual(ctx.exception.status_code, 500)


if __name__ == "__main__":
    unittest.main()
