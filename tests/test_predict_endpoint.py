import io
import unittest
from unittest.mock import Mock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from deploy.main import app
from deploy.rate_limit import reset_rate_limits
from deploy.routers.predict import process_prediction_job


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

    def test_prediction_job_create_returns_accepted_and_starts_background(self):
        with patch("deploy.routers.predict.optional_authenticated_user", return_value="user-1"):
            with patch(
                "deploy.routers.predict._upload_image_to_supabase",
                return_value="https://example.com/predictions/job.jpg",
            ):
                with patch(
                    "deploy.routers.predict._insert_prediction_job",
                    return_value={
                        "id": "job-1",
                        "created_by": "user-1",
                        "selected_plant": "Tomato",
                        "image_url": "https://example.com/predictions/job.jpg",
                        "status": "pending",
                    },
                ) as insert_job:
                    with patch("deploy.routers.predict.process_prediction_job") as process_job:
                        response = self.client.post(
                            "/prediction-jobs",
                            data={"selected_plant": "Tomato"},
                            files={"file": ("leaf.jpg", jpeg_bytes(), "image/jpeg")},
                        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "status": "accepted",
                "job_id": "job-1",
                "image_url": "https://example.com/predictions/job.jpg",
            },
        )
        payload = insert_job.call_args.args[0]
        self.assertEqual(payload["created_by"], "user-1")
        self.assertEqual(payload["selected_plant"], "Tomato")
        self.assertEqual(payload["status"], "pending")
        process_job.assert_called_once_with("job-1")

    def test_prediction_job_get_returns_pending_and_done(self):
        pending = {
            "id": "job-1",
            "created_by": "user-1",
            "status": "pending",
            "image_url": "https://example.com/predictions/job.jpg",
        }
        done = {
            **pending,
            "status": "done",
            "predicted_plant": "Tomato",
            "predicted_disease": "Tomato___Late_blight",
            "confidence": 92.34,
            "confidence_text": "92.34%",
            "result_image_url": "https://example.com/predictions/job.jpg",
        }

        with patch("deploy.routers.predict.require_authenticated_user", return_value="user-1"):
            with patch("deploy.routers.predict._get_prediction_job", return_value=pending):
                pending_response = self.client.get("/prediction-jobs/job-1")
            with patch("deploy.routers.predict._get_prediction_job", return_value=done):
                done_response = self.client.get("/prediction-jobs/job-1")

        self.assertEqual(pending_response.status_code, 200)
        self.assertEqual(pending_response.json()["status"], "pending")
        self.assertIsNone(pending_response.json()["result"])
        self.assertEqual(done_response.status_code, 200)
        result = done_response.json()["result"]
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["disease"], "Tomato___Late_blight")

    def test_prediction_job_get_rejects_other_user(self):
        job = {
            "id": "job-1",
            "created_by": "owner-1",
            "status": "pending",
            "image_url": "https://example.com/predictions/job.jpg",
        }

        with patch("deploy.routers.predict._get_prediction_job", return_value=job):
            with patch("deploy.routers.predict.require_authenticated_user", return_value="other-user"):
                response = self.client.get("/prediction-jobs/job-1")

        self.assertEqual(response.status_code, 403)

    def test_prediction_job_failed_when_hugging_face_fails(self):
        updates = []
        job = {
            "id": "job-1",
            "created_by": "user-1",
            "selected_plant": "Tomato",
            "status": "pending",
            "image_url": "https://example.com/predictions/job.jpg",
        }

        with patch("deploy.routers.predict._get_prediction_job", return_value=job):
            with patch("deploy.routers.predict._update_prediction_job", side_effect=lambda _, payload: updates.append(payload)):
                with patch("deploy.routers.predict._download_prediction_image", return_value=(jpeg_bytes(), "image/jpeg")):
                    with patch(
                        "deploy.routers.predict._call_hugging_face",
                        side_effect=HTTPException(status_code=502, detail="Could not reach Hugging Face"),
                    ):
                        process_prediction_job("job-1")

        self.assertEqual(updates[0]["status"], "processing")
        self.assertEqual(updates[-1]["status"], "failed")
        self.assertEqual(updates[-1]["error_message"], "Could not reach Hugging Face")

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

    def test_predict_under_unrecognized_threshold_does_not_auto_log(self):
        hf_response = Mock()
        hf_response.status_code = 200
        hf_response.json.return_value = {
            "status": "success",
            "plant": "Tomato",
            "disease": "Tomato___Late_blight",
            "confidence": 0.42,
        }
        table = Mock()
        insert_result = Mock()
        table.insert.return_value = insert_result
        fake_supabase = Mock()
        fake_supabase.table.return_value = table

        with patch("deploy.routers.predict.requests.post", return_value=hf_response):
            with patch("deploy.routers.predict.optional_authenticated_user", return_value="user-1"):
                with patch("deploy.routers.predict.print"):
                    with patch(
                        "deploy.routers.predict._upload_image_to_supabase",
                        return_value="https://example.com/predictions/image.jpg",
                    ):
                        with patch("deploy.routers.predict.supabase", fake_supabase):
                            response = self.client.post(
                                "/predict",
                                data={"selected_plant": "Tomato"},
                                files={"file": ("leaf.jpg", jpeg_bytes(), "image/jpeg")},
                            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "unrecognized")
        self.assertEqual(
            response.json()["message"],
            "xin lỗi, hiện tại ứng dụng của chúng tôi không nhận diện được loại cây này",
        )
        fake_supabase.table.assert_not_called()

    def test_predict_logs_low_confidence_above_unrecognized_threshold_when_authenticated(self):
        hf_response = Mock()
        hf_response.status_code = 200
        hf_response.json.return_value = {
            "status": "success",
            "plant": "Tomato",
            "disease": "Tomato___Late_blight",
            "confidence": 0.65,
        }
        table = Mock()
        insert_result = Mock()
        table.insert.return_value = insert_result
        fake_supabase = Mock()
        fake_supabase.table.return_value = table

        with patch("deploy.routers.predict.requests.post", return_value=hf_response):
            with patch("deploy.routers.predict.optional_authenticated_user", return_value="user-1"):
                with patch("deploy.routers.predict.print"):
                    with patch(
                        "deploy.routers.predict._upload_image_to_supabase",
                        return_value="https://example.com/predictions/image.jpg",
                    ):
                        with patch("deploy.routers.predict.supabase", fake_supabase):
                            response = self.client.post(
                                "/predict",
                                data={"selected_plant": "Tomato"},
                                files={"file": ("leaf.jpg", jpeg_bytes(), "image/jpeg")},
                            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "success")
        fake_supabase.table.assert_called_once_with("ai_feedback_cases")
        table.insert.assert_called_once()
        payload = table.insert.call_args.args[0]
        self.assertEqual(payload["reason"], "low_confidence")
        self.assertEqual(payload["confidence"], 65.0)

    def test_predict_low_confidence_logs_client_quality_metadata(self):
        hf_response = Mock()
        hf_response.status_code = 200
        hf_response.json.return_value = {
            "status": "success",
            "plant": "Tomato",
            "disease": "Tomato___Late_blight",
            "confidence": 0.65,
        }
        table = Mock()
        table.insert.return_value = Mock()
        fake_supabase = Mock()
        fake_supabase.table.return_value = table

        with patch("deploy.routers.predict.requests.post", return_value=hf_response):
            with patch("deploy.routers.predict.optional_authenticated_user", return_value="user-1"):
                with patch("deploy.routers.predict.print"):
                    with patch(
                        "deploy.routers.predict._upload_image_to_supabase",
                        return_value="https://example.com/predictions/image.jpg",
                    ):
                        with patch("deploy.routers.predict.supabase", fake_supabase):
                            response = self.client.post(
                                "/predict",
                                data={
                                    "selected_plant": "Tomato",
                                    "client_flow_version": "scanner_v2",
                                    "client_quality_json": '{"score":72,"warnings":["Ảnh hơi tối"]}',
                                },
                                files={"file": ("leaf.jpg", jpeg_bytes(), "image/jpeg")},
                            )

        self.assertEqual(response.status_code, 200)
        payload = table.insert.call_args.args[0]
        self.assertEqual(payload["client_flow_version"], "scanner_v2")
        self.assertEqual(payload["quality_json"]["score"], 72)

    def test_predict_ignores_invalid_client_quality_metadata(self):
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
                        data={
                            "selected_plant": "Tomato",
                            "client_quality_json": "{bad json",
                        },
                        files={"file": ("leaf.jpg", jpeg_bytes(), "image/jpeg")},
                    )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "success")

    def test_low_confidence_feedback_endpoint_saves_after_consent(self):
        table = Mock()
        insert_result = Mock()
        table.insert.return_value = insert_result
        fake_supabase = Mock()
        fake_supabase.table.return_value = table

        with patch("deploy.routers.predict.require_authenticated_user", return_value="user-1"):
            with patch("deploy.routers.predict.print"):
                with patch(
                    "deploy.routers.predict._upload_image_to_supabase",
                    return_value="https://example.com/predictions/unclear.jpg",
                ):
                    with patch("deploy.routers.predict.supabase", fake_supabase):
                        response = self.client.post(
                            "/ai-feedback/low-confidence",
                            data={
                                "selected_plant": "Tomato",
                                "predicted_plant": "Apple",
                                "predicted_disease": "Apple___Black_rot",
                                "confidence": "1.00%",
                                "user_note": "User agreed to save unrecognized image",
                            },
                            files={"file": ("leaf.jpg", jpeg_bytes(), "image/jpeg")},
                        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "success")
        fake_supabase.table.assert_called_once_with("ai_feedback_cases")
        payload = table.insert.call_args.args[0]
        self.assertEqual(payload["created_by"], "user-1")
        self.assertEqual(payload["plant"], "Apple")
        self.assertEqual(payload["predicted_disease"], "Apple___Black_rot")
        self.assertEqual(payload["confidence"], 1.0)
        self.assertEqual(payload["reason"], "low_confidence")
        self.assertEqual(payload["image_url"], "https://example.com/predictions/unclear.jpg")

    def test_low_confidence_feedback_saves_client_quality_metadata(self):
        table = Mock()
        table.insert.return_value = Mock()
        fake_supabase = Mock()
        fake_supabase.table.return_value = table

        with patch("deploy.routers.predict.require_authenticated_user", return_value="user-1"):
            with patch("deploy.routers.predict.print"):
                with patch(
                    "deploy.routers.predict._upload_image_to_supabase",
                    return_value="https://example.com/predictions/unclear.jpg",
                ):
                    with patch("deploy.routers.predict.supabase", fake_supabase):
                        response = self.client.post(
                            "/ai-feedback/low-confidence",
                            data={
                                "selected_plant": "Tomato",
                                "confidence": "62%",
                                "client_flow_version": "scanner_v2",
                                "client_quality_json": '{"score":41,"minDimension":220}',
                            },
                            files={"file": ("leaf.jpg", jpeg_bytes(), "image/jpeg")},
                        )

        self.assertEqual(response.status_code, 200)
        payload = table.insert.call_args.args[0]
        self.assertEqual(payload["client_flow_version"], "scanner_v2")
        self.assertEqual(payload["quality_json"]["minDimension"], 220)


if __name__ == "__main__":
    unittest.main()
