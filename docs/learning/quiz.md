# Quiz khóa học Plant Disease Detector

File này chứa câu hỏi kiểm tra nhanh cho từng module. Khi học tương tác với giảng viên, hãy trả lời trước; sau đó mới xem phần đáp án ở cuối file.

## Module 1. Tổng quan kiến trúc

1. Vì sao đồ án tách iOS app và FastAPI backend thay vì gọi thẳng mô hình AI từ iOS?
2. Supabase trong hệ thống đảm nhiệm những vai trò nào?
3. Endpoint backend nào xử lý ảnh chẩn đoán?
4. Khi user đã đăng nhập, token được gửi từ iOS lên backend bằng header nào?
5. Vì sao service role key không được đưa vào iOS app?

## Module 2. Backend FastAPI

1. `deploy/main.py` có nhiệm vụ gì?
2. Middleware bảo mật trong `deploy/security.py` chặn những lỗi phổ biến nào?
3. `Pydantic ConfigDict(extra="forbid")` giúp ích gì?
4. Khi thiếu biến môi trường quan trọng, backend trả lỗi qua cơ chế nào?
5. Vì sao backend dùng `run_in_threadpool` khi gọi hàm blocking?

## Module 3. AI predict và xử lý ảnh

1. iOS gửi ảnh lên `/predict` theo content-type nào?
2. Vì sao backend phải kiểm tra cả content-type, đuôi file và đọc ảnh thật bằng Pillow?
3. `PREDICT_UNRECOGNIZED_THRESHOLD` khác gì `PREDICT_LOW_CONFIDENCE_THRESHOLD`?
4. Khi confidence dưới ngưỡng unrecognized, backend có upload ảnh lên Storage không?
5. Hàm `infer_plant_from_disease_label` dùng để làm gì?

## Module 4. Supabase Auth, RLS, Storage

1. Bảng `profiles` liên hệ với `auth.users` như thế nào?
2. RLS giải quyết vấn đề bảo mật gì?
3. Policy nào giúp user không tự nâng quyền thành expert?
4. Bucket `plant-images` dùng để lưu những loại ảnh nào?
5. Backend cần service role key trong trường hợp nào?

## Module 5. iOS SwiftUI App

1. `RootView` quyết định màn hình nào được hiển thị dựa trên những trạng thái nào?
2. `AuthStore` lưu session ở đâu?
3. `ScannerViewModel` kiểm tra gì trước khi gọi API predict?
4. `APIService.saveHistory` gửi dữ liệu đến endpoint nào?
5. Vì sao app có cả `APIService` và `SupabaseDataService`?

## Module 6. History, Cây của tôi, Discover, Bookmark

1. `history/save` tạo thêm `outbreak_cases` khi nào?
2. `user_plants` dùng để lưu dữ liệu gì?
3. `care_tasks` liên kết với những bảng nào?
4. `plant_resources` và `bookmarks` khác nhau thế nào?
5. Public select trong `plant_resources` có nghĩa là gì?

## Module 7. Weather, Outbreak Map

1. `/weather` gọi dịch vụ ngoài nào?
2. Vì sao phải validate lat/lng trước khi gọi OpenWeather?
3. `/outbreaks/areas` tính level vùng dịch dựa trên dữ liệu gì?
4. `source_display` trong outbreak case được tạo từ đâu?
5. Khi boundary GeoJSON lỗi, backend trả loại lỗi nào?

## Module 8. Chatbot, LLM, Care Plan

1. Vì sao hệ thống dùng `input_hash` cho LLM cache?
2. Endpoint nào tạo tư vấn theo kết quả chẩn đoán?
3. Endpoint nào tạo lịch chăm sóc?
4. Khi Gemini lỗi 503, backend có cơ chế gì?
5. Chat được lưu lên Supabase trong điều kiện nào?

## Module 9. Low-confidence feedback và chuyên gia

1. `ai_feedback_cases` lưu những trường hợp nào?
2. `report_cases` khác gì với `ai_feedback_cases`?
3. Expert được quyền update workflow nhờ điều kiện nào?
4. Endpoint `/ai-feedback/low-confidence` yêu cầu user đã đăng nhập không?
5. Dữ liệu feedback có thể dùng để cải thiện AI thế nào?

## Module 10. Security, validation, rate limit, deploy, test

1. Rate limit trong hệ thống có mấy lớp chính?
2. Vì sao test dùng mock thay vì gọi thật Hugging Face/Supabase/Gemini?
3. Lệnh chạy toàn bộ backend unit test là gì?
4. Render chạy backend bằng command nào?
5. `/health` và `/health/ready` khác nhau thế nào?

---

# Đáp án tham khảo

Chỉ xem phần này sau khi đã tự trả lời.

## Đáp án Module 1

1. Vì iOS không nên giữ secret, không nên xử lý model nặng, và backend giúp kiểm soát upload, auth, logging, storage, nâng cấp model dễ hơn.
2. Supabase đảm nhiệm Auth, Postgres database, Storage bucket và RLS/policies.
3. `/predict`.
4. `Authorization: Bearer <access_token>`.
5. Vì service role key bỏ qua RLS như chìa khóa tổng; nếu lộ trong app, người khác có thể ghi/xóa dữ liệu trái phép.

## Đáp án Module 2

1. Tạo app FastAPI, gắn middleware, exception handler và include các router.
2. Chặn path đáng ngờ, content-type sai, body quá lớn, request vượt global rate limit; thêm security headers.
3. Từ chối field lạ trong JSON, giảm nguy cơ client gửi dữ liệu ngoài schema.
4. `runtime_error_handler` bắt `RuntimeError` và trả JSON 503 có hint.
5. `requests`, Supabase client, psycopg2 là blocking; chạy trong threadpool để không làm nghẽn event loop async.

## Đáp án Module 3

1. `multipart/form-data`.
2. Vì metadata có thể giả; cần đọc ảnh thật để tránh file độc hại, sai định dạng hoặc decompression bomb.
3. `unrecognized` là ngưỡng quá thấp nên không tin kết quả; `low_confidence` là ngưỡng thấp nhưng vẫn có kết quả để log retrain.
4. Không. Trả `image_url: None`.
5. Tách tên cây từ nhãn kiểu `Tomato___Late_blight` khi model không trả riêng `plant`.

## Đáp án Module 4

1. `profiles.id` là khóa chính và foreign key đến `auth.users(id)`.
2. Đảm bảo user chỉ đọc/ghi dữ liệu được phép ngay ở tầng database.
3. Policy insert/update trong `003_profiles_roles.sql` và bản harden `009` chỉ cho role ban đầu là user, update role phải giữ nguyên role hiện tại.
4. Ảnh chẩn đoán, ảnh history, ảnh feedback/retrain, ảnh tư vấn/chuyên gia.
5. Khi backend cần upload Storage, insert DB, ghi cache hoặc thao tác vượt quyền client nhưng vẫn trong logic server tin cậy.

## Đáp án Module 5

1. `authStore.isLoading`, `session`, `role`, `isGuest`.
2. Keychain qua `KeychainStore`.
3. Đã chọn cây, ảnh chuyển được JPEG, dung lượng dưới 5MB, không đang phân tích.
4. `/history/save`.
5. `APIService` gọi backend FastAPI; `SupabaseDataService` gọi trực tiếp Supabase REST cho các bảng có RLS.

## Đáp án Module 6

1. Khi có lat/lng, confidence >= 60 và disease không phải healthy.
2. Bộ sưu tập cây cá nhân, thông tin chăm sóc, kích thước chậu/cây, bệnh hiện tại.
3. `user_plants` và `care_plans`.
4. `plant_resources` là tài liệu công khai; `bookmarks` là tài nguyên mà user đã lưu.
5. Ai cũng có thể đọc tài nguyên, kể cả khách.

## Đáp án Module 7

1. OpenWeather API.
2. Để chặn dữ liệu sai trước khi tốn request ngoài và tránh lỗi không cần thiết.
3. Số ca trong tỉnh/khu vực, max severity và hàm `compute_level`.
4. Backend lấy `created_by`, join/lookup `profiles`, rồi định dạng tên user/expert.
5. 502 vì lỗi phụ thuộc dịch vụ ngoài/boundary fetch.

## Đáp án Module 8

1. Để nhận diện cùng một input và trả cache, giảm chi phí/độ trễ gọi Gemini.
2. `/llm/advice/diagnosis`.
3. `/llm/care-plan/diagnosis`.
4. Dùng fallback advice hoặc stale cache nếu có.
5. Khi user đồng ý lưu (`storage_consent = true`) và có access token hợp lệ.

## Đáp án Module 9

1. Dự đoán confidence thấp, ảnh không rõ, user báo sai hoặc ca manual để retrain.
2. `report_cases` là workflow user báo sai/chuyên gia xử lý; `ai_feedback_cases` thiên về dataset cải thiện AI/retrain.
3. Policy kiểm `public.is_expert(auth.uid())`.
4. Có. Endpoint dùng `require_authenticated_user`.
5. Expert gán nhãn đúng, lọc ảnh lỗi, sau đó dùng làm dữ liệu huấn luyện/đánh giá model.

## Đáp án Module 10

1. Global middleware rate limit, predict rate limit, LLM rate limit, weather rate limit.
2. Để test nhanh, ổn định, không tốn quota, không phụ thuộc mạng.
3. `python3 -m unittest discover -s tests`.
4. `uvicorn main:app --host 0.0.0.0 --port $PORT`.
5. `/health` chỉ kiểm backend sống; `/health/ready` kiểm cấu hình bắt buộc và không lộ secret.

