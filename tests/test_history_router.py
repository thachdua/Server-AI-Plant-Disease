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
            with patch("deploy.routers.history.save_to_db", return_value="history-1") as save_to_db:
                with patch("deploy.routers.history.save_outbreak_case") as save_outbreak_case:
                    response = await save_history(req, Mock())

        self.assertEqual(response, {"status": "success", "outbreak_saved": False, "history_id": "history-1"})
        save_to_db.assert_called_once_with(
            "Tomato",
            "Late blight",
            91.2,
            "https://example.com/image.jpg",
            "user-1",
        )
        save_outbreak_case.assert_not_called()

    async def test_save_history_creates_outbreak_for_confident_unhealthy_diagnosis_with_location(self):
        req = SaveHistoryRequest(
            plant="Tomato",
            disease="Late blight",
            confidence=60.0,
            image_url="https://example.com/image.jpg",
            lat=16.0471,
            lng=108.2068,
            location_label="Hải Châu, Đà Nẵng, Vietnam",
            province_name="Đà Nẵng",
        )

        with patch("deploy.routers.history.require_authenticated_user", return_value="user-1"):
            with patch("deploy.routers.history.save_to_db", return_value="history-1"):
                with patch("deploy.routers.history.save_outbreak_case", return_value="outbreak-1") as save_outbreak_case:
                    response = await save_history(req, Mock())

        self.assertEqual(response, {"status": "success", "outbreak_saved": True, "history_id": "history-1"})
        save_outbreak_case.assert_called_once_with(
            lat=16.0471,
            lng=108.2068,
            plant="Tomato",
            disease="Late blight",
            confidence=60.0,
            image_url="https://example.com/image.jpg",
            history_id="history-1",
            created_by="user-1",
            location_label="Hải Châu, Đà Nẵng, Vietnam",
            province_name="Đà Nẵng",
        )

    async def test_save_history_skips_outbreak_below_confidence_threshold(self):
        req = SaveHistoryRequest(
            plant="Tomato",
            disease="Late blight",
            confidence=59.99,
            image_url="https://example.com/image.jpg",
            lat=16.0471,
            lng=108.2068,
        )

        with patch("deploy.routers.history.require_authenticated_user", return_value="user-1"):
            with patch("deploy.routers.history.save_to_db", return_value="history-1"):
                with patch("deploy.routers.history.save_outbreak_case") as save_outbreak_case:
                    response = await save_history(req, Mock())

        self.assertEqual(response, {"status": "success", "outbreak_saved": False, "history_id": "history-1"})
        save_outbreak_case.assert_not_called()

    async def test_save_history_skips_outbreak_for_healthy_diagnosis(self):
        req = SaveHistoryRequest(
            plant="Tomato",
            disease="Healthy",
            confidence=98.0,
            image_url="https://example.com/image.jpg",
            lat=16.0471,
            lng=108.2068,
        )

        with patch("deploy.routers.history.require_authenticated_user", return_value="user-1"):
            with patch("deploy.routers.history.save_to_db", return_value="history-1"):
                with patch("deploy.routers.history.save_outbreak_case") as save_outbreak_case:
                    response = await save_history(req, Mock())

        self.assertEqual(response, {"status": "success", "outbreak_saved": False, "history_id": "history-1"})
        save_outbreak_case.assert_not_called()

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
