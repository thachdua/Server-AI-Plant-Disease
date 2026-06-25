import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from deploy.cache import _cache
from deploy.main import app
from deploy.routers.outbreaks import _rows_to_areas


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
