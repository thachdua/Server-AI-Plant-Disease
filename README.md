# Plant Disease Detector

Đồ án gồm một ứng dụng iOS SwiftUI và backend FastAPI để nhận diện bệnh cây từ ảnh, lưu lịch sử, hiển thị thời tiết/ổ dịch và cung cấp tư vấn AI bằng tiếng Việt.

## Chức năng chính

- Chụp hoặc chọn ảnh lá/cây trên iOS.
- Gửi ảnh và loại cây đã chọn lên backend `/predict`.
- Backend gọi Hugging Face Space để chạy model nhận diện bệnh.
- Upload ảnh chẩn đoán lên Supabase Storage và trả về cây, bệnh, độ tin cậy, URL ảnh.
- Đăng nhập Supabase email/password, Google OAuth, chế độ khách.
- Lưu lịch sử chẩn đoán theo user đăng nhập.
- Xem từ điển bệnh, thời tiết nông nghiệp, bản đồ ổ dịch và tư vấn Gemini.
- Giao diện user/expert dựa trên role trong bảng `profiles`.

## Kiến trúc

```text
iOS SwiftUI
  -> FastAPI backend on Render
    -> Hugging Face Space /predict
    -> Supabase Auth, Storage, Postgres
    -> OpenWeather API
    -> Gemini API
```

Các phần quan trọng:

- `PlantDiseaseDetector(IOS)/`: app iOS.
- `deploy/`: backend FastAPI.
- `main.py`: entrypoint tương thích Render/local dev.
- `supabase/sql/`: schema/policy Supabase.
- `.env.example`: biến môi trường cần cấu hình.

## Chạy backend local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Nếu deploy Render, có thể dùng `render.yaml` làm blueprint và điền các biến `sync: false` trong dashboard.

`requirements.txt` dùng version range có upper bound để tránh deploy tự nhảy major version. Khi muốn nâng dependency, chạy test backend trước khi deploy.

Backend cần các biến môi trường trong `.env`, tối thiểu:

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`: service role key cho backend, không đưa key này vào iOS
- `DB_USER`, `DB_PASSWORD`
- `HF_API_URL`

Các chức năng thời tiết/AI cần thêm:

- `OPENWEATHER_API_KEY`
- `GEMINI_API_KEY` hoặc `GEMINI_API_KEYS`

## Bảo vệ API dự đoán

Backend hỗ trợ cấu hình:

- `SECURITY_GLOBAL_RATE_LIMIT_PER_MINUTE=180`: giới hạn tổng request/phút theo IP ở middleware. Đây là lớp giảm spam/app-level abuse; DDoS lớn vẫn nên chặn thêm bằng Cloudflare/Render/WAF.
- `SECURITY_MAX_REQUEST_BYTES=5767168`: giới hạn body request tổng thể.
- `SECURITY_MAX_JSON_BYTES=262144`: giới hạn body JSON cho các endpoint LLM/history.
- `SECURITY_MAX_IMAGE_PIXELS=20000000`: chặn ảnh quá lớn/decompression bomb trước khi xử lý.
- `PREDICT_REQUIRE_AUTH=false`: cho phép khách scan, nhưng nếu app gửi token thì token vẫn được xác thực.
- `PREDICT_REQUIRE_AUTH=true`: chỉ user đăng nhập Supabase mới được gọi `/predict`.
- `PREDICT_MAX_UPLOAD_BYTES=5242880`: giới hạn ảnh 5 MB.
- `PREDICT_RATE_LIMIT_PER_MINUTE=20`: giới hạn số request scan theo IP.
- `LLM_RATE_LIMIT_PER_MINUTE=10`: giới hạn request Gemini theo IP.
- `LLM_CHAT_MAX_CHARS=4000`: giới hạn tổng độ dài hội thoại gửi sang Gemini.
- `WEATHER_RATE_LIMIT_PER_MINUTE=60`: giới hạn request thời tiết theo IP.

Khi nộp/demo public, nên bật `PREDICT_REQUIRE_AUTH=true` nếu không cần chế độ khách scan.

Middleware backend cũng chặn `Content-Type` sai (`application/json` cho LLM/history, `multipart/form-data` cho upload ảnh), chặn path bất thường, thêm security headers, và re-encode ảnh upload thành JPEG sạch trước khi gửi sang Hugging Face/Supabase để loại metadata/payload lạ.

## Health check

- `GET /health`: kiểm tra backend còn sống, không phụ thuộc dịch vụ ngoài.
- `GET /health/ready`: kiểm tra cấu hình bắt buộc như Supabase/Postgres/Hugging Face URL. Endpoint này không trả secret.

Trên Render có thể dùng `/health` cho uptime check. Trước khi demo, mở `/health/ready` để xem còn thiếu biến môi trường nào không.

## Input validation

Backend trả `400/422` trước khi gọi dịch vụ ngoài nếu tọa độ không hợp lệ, `severity` ngoài khoảng 1-5, `since_days` ngoài khoảng cho phép, schema JSON sai, field lạ, hoặc prompt/tư vấn thiếu dữ liệu bắt buộc. Backend trả `413` khi payload quá lớn, `415` khi content-type/upload type sai, và `429` khi vượt rate limit theo IP.

## Cấu hình iOS

Trong `Info.plist`:

- `BACKEND_BASE_URL`: URL backend, ví dụ `https://server-ai-plant-disease.onrender.com` hoặc `http://localhost:8000`.
- `SUPABASE_URL`: URL project Supabase.
- `SUPABASE_ANON_KEY`: anon/publishable key của Supabase.

Các service iOS lấy URL backend qua `SupabaseConfig`, nên không cần sửa nhiều file khi đổi môi trường.

## Supabase

Chạy các script trong `supabase/sql/` theo thứ tự phù hợp:

1. Chọn một policy cho bảng `history`: `001` hoặc `002`.
2. Chạy `003_profiles_roles.sql` để tạo profile/role.
3. Chạy các script còn lại cho reports, avatars, outbreak cases và LLM cache nếu dùng đầy đủ tính năng.
4. Nếu đã từng chạy bản cũ của `003_profiles_roles.sql`, chạy thêm `009_harden_profile_roles.sql` để chặn user tự nâng quyền thành expert.
5. Nếu đã từng chạy bản cũ của `004_reports_and_consultations.sql`, chạy thêm `010_harden_workflow_updates.sql` để chỉ expert được cập nhật workflow/review/reply.

Xem thêm `supabase/README.md` và `docs/supabase-auth-setup.md`.

## Test backend

```bash
python -m unittest discover -s tests
```

Các test hiện tập trung vào helper của `/predict`: parse confidence, suy cây từ label model và xác thực file ảnh.
