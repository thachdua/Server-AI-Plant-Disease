import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from deploy.cache import _cache
from deploy.main import app
from deploy.routers.outbreaks import _nearby_alerts, _nearby_recommendation_summary, _rows_to_areas


class OutbreakRouterTests(unittest.TestCase):
    def setUp(self):
        _cache.clear()
        self.client = TestClient(app)

    def test_ward_areas_require_parent_id(self):
        response = self.client.get("/outbreaks/areas", params={"level": "ward"})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "parent_id is required when level=ward")

    def test_province_areas_use_internal_rows_not_external_http(self):
        row = {
            "area_id": "01",
            "name": "Hà Nội",
            "full_name": "Thành phố Hà Nội",
            "parent_id": None,
            "parent_name": None,
            "province_id": "01",
            "geojson": {
                "type": "MultiPolygon",
                "coordinates": [[[[105.0, 20.0], [106.0, 20.0], [106.0, 21.0], [105.0, 20.0]]]],
            },
            "bbox_geojson": {
                "type": "Polygon",
                "coordinates": [[[105.0, 20.0], [106.0, 20.0], [106.0, 21.0], [105.0, 20.0]]],
            },
            "centroid_geojson": {"type": "Point", "coordinates": [105.5, 20.5]},
            "case_count": 6,
            "max_severity": 2,
            "top_disease": "Late blight",
            "recent_cases": [],
            "ward_count": 126,
        }

        with patch("deploy.routers.outbreaks._fetch_area_rows", return_value=[row]) as fetch:
            response = self.client.get("/outbreaks/areas", params={"level": "province"})

        self.assertEqual(response.status_code, 200)
        fetch.assert_called_once()
        item = response.json()["items"][0]
        self.assertEqual(item["area_id"], "01")
        self.assertEqual(item["risk_level"], 3)
        self.assertEqual(item["ward_count"], 126)
        self.assertEqual(item["type"], "MultiPolygon")

    def test_rows_to_areas_serializes_geojson_and_recent_cases(self):
        row = {
            "area_id": "00004",
            "name": "Ba Đình",
            "full_name": "Phường Ba Đình",
            "parent_id": "01",
            "parent_name": "Hà Nội",
            "province_id": "01",
            "geojson": '{"type":"MultiPolygon","coordinates":[[[[105.0,20.0],[106.0,20.0],[106.0,21.0],[105.0,20.0]]]]}',
            "bbox_geojson": '{"type":"Polygon","coordinates":[[[105.0,20.0],[106.0,20.0],[106.0,21.0],[105.0,20.0]]]}',
            "centroid_geojson": '{"type":"Point","coordinates":[105.5,20.5]}',
            "case_count": 1,
            "max_severity": 5,
            "top_disease": "Rust",
            "recent_cases": [
                {
                    "id": "case-1",
                    "lat": 20.5,
                    "lng": 105.5,
                    "disease": "Rust",
                    "severity": 5,
                    "province_name": "Hà Nội",
                    "ward_name": "Phường Ba Đình",
                }
            ],
            "ward_count": 0,
        }

        area = _rows_to_areas([row])[0]

        self.assertEqual(area["bbox"], [105.0, 20.0, 106.0, 21.0])
        self.assertEqual(area["centroid"], [105.5, 20.5])
        self.assertEqual(area["risk_level"], 4)
        self.assertEqual(area["recent_cases"][0]["location_label"], "Phường Ba Đình, Hà Nội, Vietnam")

    def test_ward_id_selected_only_passes_to_area_query(self):
        row = {
            "area_id": "00070",
            "name": "Hoàn Kiếm",
            "full_name": "Phường Hoàn Kiếm",
            "parent_id": "01",
            "parent_name": "Hà Nội",
            "province_id": "01",
            "unit_type": "Phường",
            "area_km2": 5.1,
            "selected": True,
            "geojson": None,
            "bbox_geojson": {"type": "Polygon", "coordinates": [[[105.8, 21.0], [105.9, 21.0], [105.9, 21.1], [105.8, 21.0]]]},
            "centroid_geojson": {"type": "Point", "coordinates": [105.85, 21.03]},
            "case_count": 3,
            "max_severity": 4,
            "top_disease": "Rust",
            "recent_cases": [],
            "ward_count": 0,
        }

        with patch("deploy.routers.outbreaks._fetch_area_rows", return_value=[row]) as fetch:
            response = self.client.get(
                "/outbreaks/areas",
                params={
                    "level": "ward",
                    "parent_id": "01",
                    "ward_id": "00070",
                    "selected_only": "true",
                    "include_geometry": "false",
                },
            )

        self.assertEqual(response.status_code, 200)
        item = response.json()["items"][0]
        self.assertEqual(item["area_id"], "00070")
        self.assertTrue(item["selected"])
        self.assertIsNone(item["coordinates"])
        self.assertEqual(item["unit_type"], "Phường")
        self.assertEqual(fetch.call_args.kwargs["ward_id"], "00070")
        self.assertTrue(fetch.call_args.kwargs["selected_only"])

    def test_nearby_alerts_rank_matching_disease_and_plant(self):
        cases = [
            {
                "id": "far",
                "distance_km": 2.0,
                "severity": 2,
                "plant": "Corn",
                "disease": "Rust",
                "ward_name": "A",
            },
            {
                "id": "match",
                "distance_km": 3.0,
                "severity": 4,
                "plant": "Tomato",
                "disease": "Late Blight",
                "ward_name": "B",
            },
        ]

        alerts = _nearby_alerts(cases, plant="Tomato", disease="Late Blight")

        self.assertEqual(alerts[0]["id"], "match")
        self.assertTrue(alerts[0]["same_plant"])
        self.assertTrue(alerts[0]["same_disease"])
        self.assertNotIn("recommended_action", alerts[0])
        self.assertEqual(alerts[0]["urgency"], "urgent")
        self.assertGreater(alerts[0]["recommendation_score"], alerts[1]["recommendation_score"])
        self.assertIn("trùng bệnh đang theo dõi", alerts[0]["explain_reasons"])
        self.assertTrue(alerts[0]["next_steps"])

    def test_nearby_recommendations_deduplicate_group_actions(self):
        alerts = _nearby_alerts(
            [
                {
                    "id": "a",
                    "distance_km": 2.0,
                    "severity": 4,
                    "plant": "Tomato",
                    "disease": "Powdery Mildew",
                    "ward_name": "A",
                },
                {
                    "id": "b",
                    "distance_km": 4.0,
                    "severity": 3,
                    "plant": "Tomato",
                    "disease": "Downy Mildew",
                    "ward_name": "B",
                },
            ],
            plant="Tomato",
            disease="Powdery Mildew",
        )

        summary = _nearby_recommendation_summary(alerts)
        action_ids = [item["id"] for item in summary["recommended_actions"]]

        self.assertEqual(action_ids.count("fungal-humidity-check"), 1)
        self.assertIn("high-risk-24h", action_ids)
        self.assertIn("very-near-radius", action_ids)
        self.assertEqual(summary["risk_context"]["matched_plant_count"], 2)
        self.assertGreaterEqual(summary["risk_context"]["matched_disease_count"], 1)
        self.assertEqual(summary["urgency"], "urgent")
        self.assertTrue(summary["next_steps"])

    def test_summary_applies_filters_and_aggregates(self):
        cases = [
            {
                "plant": "Tomato",
                "disease": "Late blight",
                "severity": 5,
                "reported_at": "2026-06-25T08:00:00+00:00",
                "province_id": "01",
                "province_name": "Hà Nội",
                "ward_id": "00004",
                "ward_name": "Ba Đình",
            },
            {
                "plant": "Tomato",
                "disease": "Rust",
                "severity": 2,
                "reported_at": "2026-06-25T09:00:00+00:00",
                "province_id": "01",
                "province_name": "Hà Nội",
                "ward_id": "00004",
                "ward_name": "Ba Đình",
            },
        ]

        with patch("deploy.routers.outbreaks._summary_cases", return_value=cases) as summary_cases:
            response = self.client.get(
                "/outbreaks/summary",
                params={"since_days": 7, "province_id": "01", "min_severity": 2},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["totals"]["case_count"], 2)
        self.assertEqual(body["totals"]["province_count"], 1)
        self.assertEqual(body["totals"]["ward_count"], 1)
        self.assertEqual(body["top_plants"][0], {"name": "Tomato", "count": 2})
        self.assertEqual(body["risky_areas"][0]["risk_level"], 4)
        summary_cases.assert_called_once()


if __name__ == "__main__":
    unittest.main()
