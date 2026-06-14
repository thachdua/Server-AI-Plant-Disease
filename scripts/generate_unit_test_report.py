from __future__ import annotations

import html
import pathlib
import sys
import time
import unittest
from dataclasses import dataclass
from datetime import datetime


ROOT = pathlib.Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "tests"
OUTPUT = ROOT / "unit_test_report.html"


GROUP_TITLES = {
    "test_predict_endpoint.PredictEndpointTests": "Predict API - Chẩn đoán bệnh cây từ ảnh",
    "test_predict_helpers.PredictHelperTests": "Predict Helpers - Xử lý ảnh và chuẩn hóa kết quả",
    "test_validation.ValidationTests": "Validation - Kiểm tra dữ liệu đầu vào",
    "test_security.SecurityTests": "Security - Bảo mật API và giới hạn request",
    "test_history_router.HistoryRouterTests": "History API - Lưu lịch sử chẩn đoán",
    "test_health.HealthTests": "Health API - Kiểm tra trạng thái backend",
    "test_rate_limit.RateLimitTests": "Rate Limit - Giới hạn tần suất request",
    "test_llm_care_plan.LLMCarePlanTests": "LLM Care Plan - Tư vấn AI và lịch chăm sóc",
    "test_database.DatabaseTests": "Database - Kết nối và thao tác cơ sở dữ liệu",
    "test_config.ConfigTests": "Config - Cấu hình backend",
    "test_supabase_sql.SupabaseSQLTests": "Supabase SQL - Migration và phân quyền dữ liệu",
    "test_geo.GeoTests": "Geo - Xử lý dữ liệu bản đồ vùng dịch",
}


TEST_TITLES = {
    "test_predict_success_normalizes_response": "API chẩn đoán trả kết quả thành công và chuẩn hóa độ tin cậy",
    "test_predict_rejects_incomplete_model_result": "Từ chối kết quả AI thiếu dữ liệu bắt buộc",
    "test_predict_under_unrecognized_threshold_does_not_auto_log": "Không tự lưu ca bệnh khi độ tin cậy dưới ngưỡng nhận diện",
    "test_predict_logs_low_confidence_above_unrecognized_threshold_when_authenticated": "Lưu ca độ tin cậy thấp khi người dùng đã đăng nhập",
    "test_low_confidence_feedback_endpoint_saves_after_consent": "Lưu phản hồi ảnh độ tin cậy thấp sau khi người dùng đồng ý",
    "test_infer_plant_from_common_model_labels": "Tách tên cây từ nhãn mô hình AI",
    "test_parse_confidence_percent": "Chuyển đổi độ tin cậy sang định dạng phần trăm",
    "test_validate_image_accepts_jpeg": "Chấp nhận ảnh JPEG hợp lệ",
    "test_validate_image_rejects_non_image": "Từ chối file không phải hình ảnh",
    "test_weather_rejects_invalid_coordinates_before_openweather": "Từ chối tọa độ thời tiết không hợp lệ trước khi gọi OpenWeather",
    "test_llm_weather_rejects_invalid_coordinates_before_openweather": "Từ chối tọa độ tư vấn thời tiết không hợp lệ",
    "test_outbreaks_rejects_invalid_severity_before_supabase": "Từ chối mức độ ổ dịch nằm ngoài khoảng cho phép",
    "test_outbreak_areas_rejects_invalid_since_days_before_supabase": "Từ chối khoảng thời gian vùng dịch quá lớn",
    "test_outbreak_areas_accepts_ten_year_window_before_querying": "Chấp nhận truy vấn vùng dịch trong giới hạn mười năm",
    "test_llm_diagnosis_requires_disease": "Yêu cầu có tên bệnh khi tạo tư vấn AI",
    "test_save_history_uses_authenticated_user": "Lưu lịch sử theo đúng người dùng đã xác thực",
    "test_save_history_creates_outbreak_for_confident_unhealthy_diagnosis_with_location": "Tạo ca vùng dịch khi chẩn đoán bệnh có độ tin cậy cao và có vị trí",
    "test_save_history_skips_outbreak_below_confidence_threshold": "Không tạo vùng dịch khi độ tin cậy thấp",
    "test_save_history_skips_outbreak_for_healthy_diagnosis": "Không tạo vùng dịch khi cây khỏe mạnh",
    "test_save_history_reports_database_failure": "Trả lỗi phù hợp khi lưu lịch sử thất bại",
    "test_security_headers_are_added": "Thêm security headers cho response",
    "test_json_endpoint_rejects_wrong_content_type": "Từ chối content-type sai ở endpoint JSON",
    "test_json_endpoint_rejects_large_declared_body": "Từ chối request có body vượt giới hạn",
    "test_global_rate_limit_blocks_repeated_requests": "Chặn request lặp lại vượt giới hạn",
    "test_chat_rejects_extra_fields_before_router_logic": "Từ chối field lạ trong request chat",
    "test_predict_rejects_non_image_upload_metadata": "Từ chối upload không phải ảnh ở endpoint dự đoán",
    "test_health_is_lightweight": "Endpoint health hoạt động nhẹ và ổn định",
    "test_ready_reports_missing_required_without_secrets": "Ready check báo thiếu cấu hình nhưng không lộ secret",
    "test_config_status_shape": "Trả về cấu trúc trạng thái cấu hình đúng",
    "test_llm_chat_rejects_large_prompt_before_gemini_call": "Từ chối prompt quá dài trước khi gọi Gemini",
    "test_llm_rate_limit_blocks_repeated_calls_before_gemini_call": "Chặn gọi tư vấn AI liên tục vượt giới hạn",
    "test_weather_advice_includes_plant_context_in_hash_payload": "Tư vấn thời tiết có chứa ngữ cảnh cây trồng",
    "test_care_plan_endpoint_validates_and_normalizes_tasks": "Chuẩn hóa danh sách việc chăm sóc cây từ AI",
    "test_save_to_db_commits_and_closes": "Ghi dữ liệu database và đóng kết nối đúng cách",
    "test_save_to_db_falls_back_when_created_by_missing": "Lưu dữ liệu dự phòng khi thiếu created_by",
    "test_llm_cache_get_closes_connection": "Đóng kết nối sau khi đọc cache tư vấn AI",
    "test_prefers_explicit_service_role_key": "Ưu tiên dùng Supabase service role key phía backend",
    "test_warns_for_publishable_backend_key": "Cảnh báo khi cấu hình nhầm publishable key",
    "test_profile_policies_prevent_self_role_escalation": "Policy ngăn người dùng tự nâng quyền thành chuyên gia",
    "test_profile_hardening_migration_exists": "Migration hardening profile tồn tại",
    "test_workflow_updates_are_expert_only": "Chỉ chuyên gia được cập nhật workflow tư vấn",
    "test_workflow_hardening_migration_exists": "Migration hardening workflow tồn tại",
    "test_new_feature_migrations_exist_with_rls": "Migration tính năng mới có bật RLS",
    "test_compute_level_uses_case_count_not_severity": "Mức vùng dịch được tính theo số ca thay vì severity",
}


@dataclass
class CaseResult:
    test_id: str
    module_class: str
    method: str
    status: str
    message: str = ""


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.case_results: list[CaseResult] = []

    def _record(self, test, status: str, message: str = "") -> None:
        test_id = test.id()
        parts = test_id.split(".")
        method = parts[-1]
        module_class = ".".join(parts[-3:-1]) if len(parts) >= 3 else test_id
        self.case_results.append(
            CaseResult(
                test_id=test_id,
                module_class=module_class,
                method=method,
                status=status,
                message=message,
            )
        )

    def addSuccess(self, test):
        super().addSuccess(test)
        self._record(test, "PASSED")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self._record(test, "FAILED", self._exc_info_to_string(err, test))

    def addError(self, test, err):
        super().addError(test, err)
        self._record(test, "ERROR", self._exc_info_to_string(err, test))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self._record(test, "SKIPPED", reason)


class RecordingRunner(unittest.TextTestRunner):
    resultclass = RecordingResult


def display_group(module_class: str) -> str:
    return GROUP_TITLES.get(module_class, module_class)


def display_case(method: str) -> str:
    return TEST_TITLES.get(method, method.replace("test_", "").replace("_", " "))


def render_report(results: list[CaseResult], elapsed: float) -> str:
    grouped: dict[str, list[CaseResult]] = {}
    for item in results:
        grouped.setdefault(item.module_class, []).append(item)

    total = len(results)
    passed = sum(1 for item in results if item.status == "PASSED")
    failed = total - passed
    generated_at = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    group_html = []
    for index, (group, items) in enumerate(sorted(grouped.items()), start=1):
        group_passed = sum(1 for item in items if item.status == "PASSED")
        group_total = len(items)
        rows = []
        for item in items:
            status_class = item.status.lower()
            message = ""
            if item.message and item.status != "PASSED":
                message = f"<pre>{html.escape(item.message)}</pre>"
            rows.append(
                f"""
                <div class="case {status_class}">
                    <span class="badge {status_class}">{html.escape(item.status)}</span>
                    <span class="case-title">{html.escape(display_case(item.method))}</span>
                    {message}
                </div>
                """
            )

        group_html.append(
            f"""
            <section class="test-card">
                <h2><span class="diamond">◆</span> {html.escape(display_group(group))}
                    <span class="count">({group_passed}/{group_total} passed)</span>
                </h2>
                <div class="divider"></div>
                {''.join(rows)}
                <p class="caption">Hình {index}. Kết quả kiểm thử {html.escape(display_group(group).lower())}</p>
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Kết quả Unit Test - Plant Disease Detector</title>
  <style>
    body {{
      margin: 0;
      background: #262626;
      color: #1f2937;
      font-family: Arial, Helvetica, sans-serif;
    }}
    .page {{
      max-width: 1040px;
      margin: 0 auto;
      padding: 32px 22px 56px;
    }}
    .summary {{
      background: #ffffff;
      border-left: 5px solid #5276ff;
      border-radius: 4px;
      padding: 18px 22px;
      margin-bottom: 24px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.12);
    }}
    .summary h1 {{
      margin: 0 0 12px;
      font-size: 22px;
      color: #111827;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }}
    .metric {{
      background: #f8fafc;
      border: 1px solid #e5e7eb;
      border-radius: 7px;
      padding: 12px;
    }}
    .metric strong {{
      display: block;
      font-size: 22px;
      margin-bottom: 4px;
    }}
    .metric span {{
      color: #6b7280;
      font-size: 13px;
    }}
    .test-card {{
      background: #ffffff;
      border-left: 5px solid #5276ff;
      border-radius: 4px;
      padding: 14px 16px 8px;
      margin: 26px 0 34px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.12);
    }}
    h2 {{
      margin: 0;
      font-size: 15px;
      font-weight: 700;
      color: #111827;
    }}
    .diamond {{
      color: #2563eb;
      font-size: 12px;
      margin-right: 8px;
    }}
    .count {{
      font-weight: 600;
      color: #374151;
    }}
    .divider {{
      height: 1px;
      background: #e5e7eb;
      margin: 14px 0 12px;
    }}
    .case {{
      min-height: 42px;
      display: flex;
      align-items: center;
      gap: 14px;
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-left: 3px solid #22c55e;
      border-radius: 6px;
      margin: 8px 0;
      padding: 8px 12px;
      box-sizing: border-box;
    }}
    .case.failed, .case.error {{
      border-left-color: #ef4444;
    }}
    .case.skipped {{
      border-left-color: #f59e0b;
    }}
    .badge {{
      min-width: 62px;
      text-align: center;
      border-radius: 999px;
      padding: 5px 8px;
      font-size: 10px;
      font-weight: 700;
      color: #166534;
      background: #dcfce7;
    }}
    .badge.failed, .badge.error {{
      color: #991b1b;
      background: #fee2e2;
    }}
    .badge.skipped {{
      color: #92400e;
      background: #fef3c7;
    }}
    .case-title {{
      color: #4b5563;
      font-size: 13px;
      line-height: 1.35;
    }}
    pre {{
      width: 100%;
      max-height: 220px;
      overflow: auto;
      background: #111827;
      color: #f9fafb;
      border-radius: 6px;
      padding: 12px;
      font-size: 12px;
      white-space: pre-wrap;
    }}
    .caption {{
      margin: 18px 0 4px;
      text-align: center;
      color: #ffffff;
      font-family: "Times New Roman", Times, serif;
      font-size: 18px;
      font-weight: 700;
      transform: translateY(34px);
    }}
    @media print {{
      body {{ background: #ffffff; }}
      .page {{ max-width: none; padding: 0; }}
      .summary, .test-card {{ box-shadow: none; break-inside: avoid; }}
      .caption {{ color: #111827; transform: none; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="summary">
      <h1>Kết quả Unit Test - Plant Disease Detector</h1>
      <div class="summary-grid">
        <div class="metric"><strong>{total}</strong><span>Tổng test case</span></div>
        <div class="metric"><strong>{passed}</strong><span>Passed</span></div>
        <div class="metric"><strong>{failed}</strong><span>Failed/Error/Skipped</span></div>
        <div class="metric"><strong>{elapsed:.2f}s</strong><span>Thời gian chạy</span></div>
      </div>
      <p>Thời gian tạo báo cáo: {html.escape(generated_at)}</p>
    </section>
    {''.join(group_html)}
  </main>
</body>
</html>
"""


def main() -> int:
    sys.path.insert(0, str(ROOT))
    suite = unittest.defaultTestLoader.discover(str(TEST_DIR))
    start = time.time()
    runner = RecordingRunner(stream=sys.stdout, verbosity=1)
    result: RecordingResult = runner.run(suite)
    elapsed = time.time() - start
    OUTPUT.write_text(render_report(result.case_results, elapsed), encoding="utf-8")
    print(f"\nReport written to: {OUTPUT}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())

