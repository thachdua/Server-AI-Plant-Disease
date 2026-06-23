# Khóa học đọc hiểu đồ án Plant Disease Detector

Vai trò: giảng viên hướng dẫn đồ án.  
Nguyên tắc học: đi từ tổng quan đến chi tiết, mỗi module học xong sẽ dừng lại để kiểm tra nhanh trước khi sang module tiếp theo.

## 0. Tình trạng repo khi bắt đầu học

Đã đọc cấu trúc repo, git log/diff và các nhóm file chính.

Git log gần nhất:

- `aa3586a Q&A`
- `ef81703 Fix Toolkit`
- `f99ee87 Fix Checklists`
- `b745945 Fix location`
- `03cee28 Link Outbreak cases`
- `290bf0e Harden backend security`
- `f6929f3 Add low confidence feedback storage`
- `e50334b Handle low confidence predictions and AI feedback upload`

Git status khi đọc:

- `PlantDiseaseDetector(IOS)` đang modified theo git parent.
- `supabase/sql/025_chat_message_attachments.sql` bị xóa 3 dòng comment.
- `supabase/sql/026_consultation_photos_and_feedback_admin.sql` bị xóa 2 dòng comment.
- `supabase/sql/027_profiles_account_metadata.sql` là file mới chưa commit.

Không sửa code ứng dụng trong quá trình lập khóa học này. Chỉ tạo tài liệu Markdown trong `docs/learning/`.

## 1. Bản đồ tổng quan kiến trúc hệ thống

Hệ thống có 4 khối lớn:

1. iOS SwiftUI App: giao diện, camera, đăng nhập, hiển thị kết quả, lịch sử, cây của tôi, discover/bookmark, weather, outbreak, chat.
2. FastAPI Backend: API trung gian để xử lý ảnh, gọi AI model, bảo vệ request, gọi Gemini/OpenWeather, ghi DB.
3. Supabase: Auth, Postgres, Storage, RLS/policies.
4. Dịch vụ ngoài: Hugging Face, Gemini API, OpenWeather API, GeoJSON boundary.

Luồng dễ nhớ:

```text
iOS App
  -> FastAPI Backend
    -> Hugging Face model
    -> Supabase Auth/Postgres/Storage
    -> Gemini API
    -> OpenWeather API
```

Ví dụ đời thường:

- iOS là quầy tiếp nhận người dùng.
- FastAPI là nhân viên xử lý hồ sơ.
- Hugging Face là chuyên viên nhìn ảnh đoán bệnh.
- Supabase là tủ hồ sơ và kho ảnh.
- RLS là bảo vệ kiểm tra ai được mở ngăn hồ sơ nào.
- Render là nơi đặt văn phòng backend lên Internet.

Chi tiết sơ đồ nằm trong `docs/learning/system_flow.md`.

---

# Lộ trình học từ dễ đến khó

## Module 1. Tổng quan kiến trúc và cách đọc repo

### Mục tiêu cần hiểu

- Biết đồ án gồm những phần nào.
- Biết request đi từ màn hình iOS đến backend, model, database rồi quay về UI như thế nào.
- Biết file nào là điểm vào của backend và iOS.

### File liên quan

- `README.md`
- `main.py`
- `deploy/main.py`
- `PlantDiseaseDetector(IOS)/PlantDiseaseDetector/PlantDiseaseDetectorApp.swift`
- `PlantDiseaseDetector(IOS)/PlantDiseaseDetector/Features/Auth/RootView.swift`
- `PlantDiseaseDetector(IOS)/PlantDiseaseDetector/Services/SupabaseConfig.swift`
- `render.yaml`
- `.env.example`

### Tính năng giải quyết vấn đề gì

Module này trả lời câu hỏi: “Toàn bộ hệ thống hoạt động như một sản phẩm thật ra sao?” Nếu không nắm được kiến trúc, khi lỗi xảy ra sẽ không biết lỗi nằm ở app, backend, model hay Supabase.

### Luồng UI → API → database/model → kết quả

```text
RootView chọn giao diện theo session/role
ScannerView cho user chụp ảnh
ScannerViewModel nén ảnh và gọi APIService
APIService gửi /predict đến FastAPI
FastAPI gọi Hugging Face và Supabase Storage
FastAPI trả JSON
iOS decode PredictResponse và hiển thị
```

### Giải thích code quan trọng

- `PlantDiseaseDetectorApp.swift`: điểm khởi động app, tạo `AuthStore` và `ChatSessionStore`.
- `RootView.swift`: router lớn nhất phía iOS. Nếu đang loading thì hiện splash; nếu có session và role expert thì vào giao diện expert; nếu user thường thì vào giao diện user; nếu guest thì vào guest tabs; nếu chưa đăng nhập thì vào login.
- `deploy/main.py`: điểm khởi động backend. Tạo `FastAPI`, gắn middleware bảo mật, include các router.
- `SupabaseConfig.swift`: lấy `BACKEND_BASE_URL`, `SUPABASE_URL`, `SUPABASE_ANON_KEY` từ `Info.plist`.
- `render.yaml`: khai báo cách Render build/start backend và các biến môi trường cần nhập.

### Lỗi thường gặp và cách debug

- App không gọi được backend: kiểm tra `BACKEND_BASE_URL` trong `Info.plist`, thử mở `/health`.
- Backend báo thiếu env: mở `/health/ready`, kiểm tra Render env vars.
- App crash khi mở: kiểm tra `SUPABASE_URL` và `SUPABASE_ANON_KEY` trong `Info.plist`.
- Không rõ request đi đâu: tìm service tương ứng trong thư mục `Services`.

### 5 câu hỏi kiểm tra nhanh

Xem `docs/learning/quiz.md`, Module 1. Khi học tương tác, hãy trả lời 5 câu đó trước khi sang Module 2.

---

## Module 2. Backend FastAPI, cấu hình và middleware

### Mục tiêu cần hiểu

- Hiểu cách FastAPI tổ chức router.
- Hiểu config lấy biến môi trường thế nào.
- Hiểu middleware bảo vệ API trước khi request vào business logic.

### File liên quan

- `deploy/main.py`
- `deploy/config.py`
- `deploy/security.py`
- `deploy/models.py`
- `deploy/validation.py`
- `deploy/rate_limit.py`
- `deploy/routers/health.py`

### Tính năng giải quyết vấn đề gì

Backend là lớp trung gian giúp app không giữ secret, không gọi model/API ngoài trực tiếp, và có thể kiểm soát bảo mật, giới hạn request, logging, fallback.

### Luồng hoạt động

```text
Request vào FastAPI
  -> security_middleware
  -> validate path/content-type/content-length/rate limit
  -> router xử lý nghiệp vụ
  -> response thêm security headers
```

### Giải thích code quan trọng

- `config.py`: gom biến môi trường như `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `HF_API_URL`, `GEMINI_API_KEY`, `OPENWEATHER_API_KEY`.
- `_SupabaseProxy`: lazy client, chỉ tạo Supabase client khi dùng lần đầu. Cách này giúp import app không chết ngay khi test thiếu env.
- `security.py`: có 3 lớp chặn sớm: path đáng ngờ, content-type sai, body quá lớn. Sau đó thêm headers như `X-Frame-Options`, `nosniff`.
- `models.py`: Pydantic schema đặt giới hạn field, kiểu dữ liệu, và `extra="forbid"` để từ chối field lạ.
- `rate_limit.py`: giữ bộ đếm request theo IP/kind trong bộ nhớ.

### Lỗi thường gặp và cách debug

- 415: sai `Content-Type`; `/predict` phải multipart, `/llm/chat` phải JSON.
- 413: body hoặc ảnh vượt giới hạn.
- 429: vượt rate limit.
- 422: JSON sai schema hoặc có field lạ.
- 503 `/health/ready`: thiếu env bắt buộc.

### 5 câu hỏi kiểm tra nhanh

Xem `docs/learning/quiz.md`, Module 2.

---

## Module 3. AI Predict Plant Disease và xử lý ảnh

### Mục tiêu cần hiểu

- Hiểu luồng upload ảnh từ iOS lên backend.
- Hiểu backend validate/sanitize ảnh bằng Pillow.
- Hiểu cách gọi Hugging Face và xử lý confidence.
- Hiểu low-confidence/unrecognized khác nhau thế nào.

### File liên quan

- `deploy/routers/predict.py`
- `deploy/config.py`
- `PlantDiseaseDetector(IOS)/PlantDiseaseDetector/Features/Scanner/ScannerView.swift`
- `PlantDiseaseDetector(IOS)/PlantDiseaseDetector/Features/Scanner/ScannerViewModel.swift`
- `PlantDiseaseDetector(IOS)/PlantDiseaseDetector/Services/APIService.swift`
- `PlantDiseaseDetector(IOS)/PlantDiseaseDetector/Utilities/ImageUtils.swift`
- `PlantDiseaseDetector(IOS)/PlantDiseaseDetector/Utilities/DiseaseLocalizer.swift`
- `tests/test_predict_endpoint.py`
- `tests/test_predict_helpers.py`

### Tính năng giải quyết vấn đề gì

Người dùng không cần tự biết bệnh cây. Họ chỉ cần chụp/chọn ảnh; hệ thống trả cây, bệnh, độ tin cậy và ảnh đã lưu.

### Luồng hoạt động

```text
ScannerView
  -> ScannerViewModel.analyzeImage
  -> ImageUtils tạo JPEG < 5MB
  -> APIService.predict multipart/form-data
  -> FastAPI /predict
  -> Pillow verify + convert JPEG sạch
  -> requests.post(HF_API_URL)
  -> parse plant/disease/confidence
  -> nếu quá thấp: unrecognized
  -> nếu thành công: upload Supabase Storage
  -> trả PredictResponse
```

### Giải thích code quan trọng

- `ScannerViewModel.analyzeImage`: chặn khi chưa chọn cây, ảnh không tạo JPEG được, ảnh > 5MB.
- `APIService.predict`: tự build multipart form gồm `selected_plant` và file ảnh.
- `_validate_upload_metadata`: kiểm tra content-type và filename trước khi đọc ảnh.
- `_sanitize_image`: đọc ảnh thật, chặn decompression bomb, convert RGB/JPEG.
- `parse_confidence_value`: nhận cả `0.92`, `92`, hoặc `92%` và chuẩn hóa về phần trăm.
- `PREDICT_UNRECOGNIZED_THRESHOLD`: dưới ngưỡng này app báo không nhận diện được.
- `PREDICT_LOW_CONFIDENCE_THRESHOLD`: dưới ngưỡng này nhưng vẫn trên unrecognized thì ghi vào `ai_feedback_cases` nếu user đã đăng nhập.

### Ví dụ đời thường

Ảnh upload giống một gói hàng. Nhìn nhãn ngoài “image/jpeg” chưa đủ; backend phải mở gói ra kiểm tra bên trong có đúng là ảnh, có quá lớn, có độc hại không, rồi mới đưa cho chuyên viên AI xem.

### Lỗi thường gặp và cách debug

- 415: file không phải jpg/png/webp hoặc tên file sai.
- 400: file không đọc được như ảnh.
- 413: ảnh quá lớn.
- 502: Hugging Face bận, trả JSON sai hoặc thiếu disease/confidence.
- 504: Hugging Face timeout.
- `image_url` null: confidence dưới ngưỡng unrecognized.

### 5 câu hỏi kiểm tra nhanh

Xem `docs/learning/quiz.md`, Module 3.

---

## Module 4. Supabase Auth, RLS và Storage bucket

### Mục tiêu cần hiểu

- Hiểu Auth, anon key, service role key.
- Hiểu RLS/policy theo user và expert.
- Hiểu bucket ảnh hoạt động ra sao.

### File liên quan

- `supabase/README.md`
- `supabase/sql/003_profiles_roles.sql`
- `supabase/sql/005_storage_avatars.sql`
- `supabase/sql/009_harden_profile_roles.sql`
- `supabase/sql/017_storage_plant_images.sql`
- `supabase/sql/027_profiles_account_metadata.sql`
- `PlantDiseaseDetector(IOS)/PlantDiseaseDetector/Services/SupabaseAuthService.swift`
- `PlantDiseaseDetector(IOS)/PlantDiseaseDetector/Services/SupabaseProfileService.swift`
- `deploy/auth.py`
- `deploy/config.py`

### Tính năng giải quyết vấn đề gì

Hệ thống cần biết ai đang dùng app, ai là expert, dữ liệu nào là của ai, và ảnh nào được phép upload/đọc.

### Luồng hoạt động

```text
iOS đăng nhập Supabase Auth
  -> nhận access_token/refresh_token
  -> AuthStore lưu Keychain
  -> ensure profile trong profiles
  -> RootView đọc role
  -> backend endpoint cần auth kiểm bearer token qua /auth/v1/user
```

### Giải thích code quan trọng

- `SupabaseAuthService.signIn/signUp/refresh`: gọi Supabase Auth REST API.
- `AuthStore.validAccessToken`: nếu token hết hạn thì refresh.
- `profiles.id references auth.users(id)`: mỗi user auth có một profile mở rộng.
- `public.is_expert(uid)`: helper SQL cho RLS kiểm expert.
- `storage.buckets plant-images`: bucket public để đọc ảnh qua URL, client thường không có quyền ghi tùy policy; backend dùng service role để upload.

### Ví dụ đời thường

RLS giống bảo vệ tầng chung cư. Mỗi người có chìa khóa căn của mình. Expert giống nhân viên kỹ thuật được cấp thẻ để vào phòng cần sửa. Service role là chìa khóa tổng của ban quản lý, chỉ đặt ở backend.

### Lỗi thường gặp và cách debug

- 401: thiếu bearer token hoặc token hết hạn.
- 403/RLS violation: policy không cho thao tác.
- Upload lỗi bucket: chưa chạy `017_storage_plant_images.sql` hoặc backend không dùng service role key.
- User không vào giao diện expert: `profiles.role` chưa là `expert` hoặc app chưa refresh role.

### 5 câu hỏi kiểm tra nhanh

Xem `docs/learning/quiz.md`, Module 4.

---

## Module 5. iOS SwiftUI app và luồng điều hướng

### Mục tiêu cần hiểu

- Hiểu app khởi động từ đâu.
- Hiểu guest/user/expert khác nhau ra sao.
- Hiểu service layer phía iOS.

### File liên quan

- `PlantDiseaseDetectorApp.swift`
- `RootView.swift`
- `AuthStore.swift`
- `UserTabView.swift`
- `ExpertTabView.swift`
- `MainTabView.swift`
- `GuestMainTabView.swift`
- `SupabaseConfig.swift`
- `APIService.swift`
- `SupabaseDataService.swift`

### Tính năng giải quyết vấn đề gì

Người dùng có nhiều vai trò và trạng thái: chưa đăng nhập, khách, user, expert. App cần chọn đúng giao diện và đúng quyền.

### Luồng hoạt động

```text
App init
  -> AuthStore load session từ Keychain
  -> refresh nếu hết hạn
  -> fetch role từ profiles
  -> RootView chọn Login/Guest/User/Expert
  -> TabView điều hướng từng chức năng
```

### Giải thích code quan trọng

- `@StateObject authStore`: giữ trạng thái đăng nhập toàn app.
- `@EnvironmentObject`: truyền store xuống các View con.
- `RootView.shouldShowChatFab`: chỉ hiện nút chat ở tab phù hợp.
- `APIService`: gọi backend FastAPI.
- `SupabaseDataService`: gọi thẳng Supabase REST cho bảng có RLS.

### Lỗi thường gặp và cách debug

- Không thấy tab expert: kiểm tra `profiles.role`.
- Session bị mất: kiểm tra Keychain/refresh token.
- Fatal error config: thiếu key trong `Info.plist`.
- JSON decode lỗi: response backend/Supabase khác struct Swift.

### 5 câu hỏi kiểm tra nhanh

Xem `docs/learning/quiz.md`, Module 5.

---

## Module 6. History, Cây của tôi, Discover và Bookmark

### Mục tiêu cần hiểu

- Hiểu lịch sử chẩn đoán và điều kiện tạo vùng dịch.
- Hiểu `user_plants`, `care_plans`, `care_tasks`.
- Hiểu Discover/Bookmark đọc tài nguyên công khai và lưu bookmark riêng.

### File liên quan

- `deploy/routers/history.py`
- `deploy/database.py`
- `supabase/sql/001_history_public_read_auth_write.sql`
- `supabase/sql/014_care_plants_tasks.sql`
- `supabase/sql/015_plant_knowledge_resources.sql`
- `supabase/sql/018_discover_resource_media.sql`
- `supabase/sql/019_discover_in_app_content.sql`
- `supabase/sql/020_history_owner_delete.sql`
- `supabase/sql/024_user_plants_current_diagnosis.sql`
- `HistoryView.swift`
- `PlantCollectionView.swift`
- `CareTasksView.swift`
- `DiscoverView.swift`
- `SupabaseDataService.swift`

### Tính năng giải quyết vấn đề gì

Sau khi chẩn đoán, user cần xem lại lịch sử, theo dõi cây của mình, lưu tài liệu hay, và biến kết quả chẩn đoán thành checklist chăm sóc.

### Luồng hoạt động

```text
Kết quả predict thành công
  -> user bấm lưu lịch sử
  -> APIService.saveHistory gửi /history/save
  -> backend insert history
  -> nếu đủ điều kiện: insert outbreak_cases
  -> iOS có thể thêm cây vào user_plants
  -> tạo care_plan/care_tasks từ AI hoặc thao tác thủ công
```

### Giải thích code quan trọng

- `_should_create_outbreak`: chỉ tạo vùng dịch khi có vị trí, confidence >= 60, disease không healthy.
- `user_plants.current_*`: lưu chẩn đoán hiện tại gắn với cây trong bộ sưu tập.
- `plant_knowledge`: kiến thức cây công khai.
- `plant_resources`: tài liệu/video/report công khai.
- `bookmarks`: bảng riêng theo user, unique `(created_by, resource_id)`.

### Lỗi thường gặp và cách debug

- Lưu history lỗi 401: chưa có token.
- Không tạo outbreak: thiếu lat/lng, confidence thấp, disease healthy.
- Bookmark lỗi RLS: `created_by` không đúng `auth.uid()`.
- Discover rỗng: chưa chạy seed SQL `015`, `018`, `019`.

### 5 câu hỏi kiểm tra nhanh

Xem `docs/learning/quiz.md`, Module 6.

---

## Module 7. Weather và outbreak map

### Mục tiêu cần hiểu

- Hiểu cách lấy thời tiết theo vị trí.
- Hiểu cách outbreak map lấy điểm và vùng tỉnh.
- Hiểu cache, validate và decorate dữ liệu.

### File liên quan

- `deploy/routers/weather.py`
- `deploy/routers/outbreaks.py`
- `deploy/geo.py`
- `deploy/validation.py`
- `supabase/sql/007_outbreak_cases.sql`
- `supabase/sql/021_outbreak_cases_diagnosis_links.sql`
- `supabase/sql/022_outbreak_cases_location_label.sql`
- `WeatherView.swift`
- `WeatherService.swift`
- `OutbreakMapView.swift`
- `OutbreakProvinceOverlayView.swift`
- `OutbreakService.swift`

### Tính năng giải quyết vấn đề gì

Bệnh cây liên quan nhiều đến độ ẩm, mưa, nhiệt độ và vùng lây lan. Weather và map giúp user không chỉ biết “cây đang bệnh gì” mà còn biết “khu vực mình có rủi ro gì”.

### Luồng hoạt động

```text
iOS lấy vị trí
  -> /weather?lat&lng
  -> validate lat/lng
  -> OpenWeather One Call
  -> backend tạo alerts
  -> iOS hiển thị

iOS mở map
  -> /outbreaks hoặc /outbreaks/areas
  -> backend query outbreak_cases
  -> lấy boundary tỉnh
  -> tính count/level
  -> iOS vẽ overlay
```

### Giải thích code quan trọng

- `validate_coordinates`: chặn lat ngoài -90..90 và lng ngoài -180..180.
- `cache_get/cache_set`: giảm số lần gọi OpenWeather/GeoJSON/Supabase.
- `compute_level`: tính mức vùng dịch dựa trên số ca.
- `_profiles_for_items`: lấy profile để hiển thị nguồn user/expert.

### Lỗi thường gặp và cách debug

- 400: tọa độ sai.
- 500: thiếu `OPENWEATHER_API_KEY`.
- 502: OpenWeather hoặc boundary URL lỗi.
- Map không có dữ liệu: bảng `outbreak_cases` rỗng hoặc query filter quá hẹp.

### 5 câu hỏi kiểm tra nhanh

Xem `docs/learning/quiz.md`, Module 7.

---

## Module 8. Chatbot, LLM advice, care plan

### Mục tiêu cần hiểu

- Hiểu backend gọi Gemini như thế nào.
- Hiểu cache LLM bằng input hash.
- Hiểu chat lưu có consent.
- Hiểu care plan/task tạo từ chẩn đoán.

### File liên quan

- `deploy/routers/llm.py`
- `deploy/gemini.py`
- `deploy/prompts.py`
- `deploy/cache.py`
- `deploy/database.py`
- `supabase/sql/008_llm_advice_cache.sql`
- `supabase/sql/013_chat_consent_sessions.sql`
- `supabase/sql/025_chat_message_attachments.sql`
- `LLMAdviceService.swift`
- `ChatService.swift`
- `ChatSessionStore.swift`
- `AssistantHubView.swift`
- `ChatSupportView.swift`
- `CarePlanPreviewView.swift`

### Tính năng giải quyết vấn đề gì

AI predict chỉ trả nhãn bệnh. Người dùng cần lời khuyên bằng tiếng Việt, checklist chăm sóc, và có thể hỏi thêm qua chatbot.

### Luồng hoạt động

```text
Kết quả diagnosis
  -> LLMAdviceService.adviceForDiagnosis
  -> /llm/advice/diagnosis
  -> hash payload
  -> đọc llm_advice_cache
  -> nếu miss: gọi Gemini
  -> validate JSON
  -> upsert cache
  -> trả advice về iOS
```

### Giải thích code quan trọng

- `canonical_json + sha256`: tạo khóa cache ổn định.
- `validate_advice_json`: ép Gemini output về cấu trúc app cần.
- `diagnosis_fallback_advice`: fallback khi Gemini 503.
- `chat_sessions.storage_consent`: chỉ lưu chat khi user đồng ý.
- `care_plan.tasks`: tạo task có `due_in_days`, `category`, `repeat_rule`, `reminder_hour`.

### Ví dụ đời thường

Gemini giống một chuyên gia tư vấn bận rộn. Cache giống cuốn sổ ghi lại câu trả lời cũ. Nếu hỏi cùng một ca bệnh, backend mở sổ trả lại thay vì gọi chuyên gia lần nữa.

### Lỗi thường gặp và cách debug

- 413: chat prompt quá dài.
- 429: gọi LLM quá nhiều.
- JSON Gemini sai schema: xem `deploy/gemini.py` validator.
- Chat không lưu: user chưa đồng ý storage hoặc chưa đăng nhập.
- Cache không hit: payload khác nhau do thêm field hoặc thay thời tiết.

### 5 câu hỏi kiểm tra nhanh

Xem `docs/learning/quiz.md`, Module 8.

---

## Module 9. Low-confidence feedback, report, consultation và expert workflow

### Mục tiêu cần hiểu

- Hiểu cách thu thập dữ liệu retrain AI.
- Hiểu user report và expert review.
- Hiểu consultation có ảnh và metadata.

### File liên quan

- `deploy/routers/predict.py`
- `deploy/routers/consultations.py`
- `supabase/sql/004_reports_and_consultations.sql`
- `supabase/sql/010_harden_workflow_updates.sql`
- `supabase/sql/011_ai_feedback_cases.sql`
- `supabase/sql/012_app_feedback.sql`
- `supabase/sql/016_consultation_workflow.sql`
- `supabase/sql/026_consultation_photos_and_feedback_admin.sql`
- `ReportIssueView.swift`
- `AppFeedbackView.swift`
- `ExpertFeedbackListView.swift`
- `ExpertQAManagementView.swift`
- `ValidationView.swift`
- `ExpertSupportHubView.swift`

### Tính năng giải quyết vấn đề gì

AI không thể đúng tuyệt đối. Hệ thống cần vòng phản hồi để user báo sai, expert duyệt, và dữ liệu sai/khó được lưu làm nguồn cải thiện mô hình.

### Luồng hoạt động

```text
Predict confidence thấp
  -> backend tự log ai_feedback_cases nếu user đã đăng nhập
hoặc user đồng ý gửi ảnh
  -> /ai-feedback/low-confidence
  -> upload image
  -> insert ai_feedback_cases
  -> expert mở danh sách pending
  -> cập nhật expert_label/expert_note/review_status
```

### Giải thích code quan trọng

- `ai_feedback_cases`: dataset retrain/kiểm tra AI.
- `report_cases`: user nói “AI đoán sai”, expert xử lý workflow.
- `consultation_requests`: user gửi câu hỏi/ảnh để expert tư vấn.
- RLS update expert: chỉ `public.is_expert(auth.uid())` mới patch workflow.
- `consultations/expert-request`: backend nhận multipart nhiều ảnh, upload Storage, insert consultation.

### Ví dụ đời thường

Một sinh viên làm bài trắc nghiệm AI chấm. Nếu điểm thấp hoặc sinh viên khiếu nại, bài được đưa cho giảng viên chấm lại. Kết quả chấm lại vừa giúp sinh viên, vừa làm dữ liệu cải thiện đề/đáp án.

### Lỗi thường gặp và cách debug

- User thường không update được status: đúng do RLS.
- Expert không thấy case: profile chưa có role expert.
- Consultation upload lỗi: thiếu bucket `plant-images` hoặc service role key.
- Feedback không insert: `created_by` khác `auth.uid()` hoặc chưa đăng nhập.

### 5 câu hỏi kiểm tra nhanh

Xem `docs/learning/quiz.md`, Module 9.

---

## Module 10. Security, validation, deploy Render và unit test

### Mục tiêu cần hiểu

- Hiểu hệ thống tự bảo vệ bằng security middleware, validation, rate limit.
- Hiểu Render deploy cần biến môi trường gì.
- Hiểu test đang kiểm những gì và cách chạy.

### File liên quan

- `deploy/security.py`
- `deploy/validation.py`
- `deploy/rate_limit.py`
- `render.yaml`
- `.env.example`
- `requirements.txt`
- `tests/test_predict_endpoint.py`
- `tests/test_predict_helpers.py`
- `tests/test_security.py`
- `tests/test_validation.py`
- `tests/test_history_router.py`
- `tests/test_llm_care_plan.py`
- `tests/test_health.py`
- `scripts/generate_unit_test_report.py`
- `unit_test_report.html`

### Tính năng giải quyết vấn đề gì

Khi app chạy public, hệ thống phải tránh spam, payload lạ, request sai, lỗi service ngoài, và cần có test chứng minh chức năng ổn định.

### Luồng kiểm tra hệ thống

```text
Local:
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  python3 -m unittest discover -s tests

Manual:
  uvicorn main:app --reload --port 8000
  GET /health
  GET /health/ready
  POST /predict bằng Postman

Report:
  python3 scripts/generate_unit_test_report.py
  open unit_test_report.html
```

### Giải thích code quan trọng

- `tests` dùng `unittest`, `TestClient`, `patch/mock`.
- Mock giúp test không gọi thật Hugging Face, Supabase, Gemini, OpenWeather.
- `generate_unit_test_report.py` chạy test thật và sinh HTML passed/failed để chụp đưa vào báo cáo.
- Render dùng `uvicorn main:app --host 0.0.0.0 --port $PORT`.

### Lỗi thường gặp và cách debug

- `python` không tồn tại trên macOS: dùng `python3`.
- Test fail vì rate limit state: nhiều test gọi `reset_rate_limits`.
- Deploy Render fail: xem build log, env vars, `/health/ready`.
- Postman `/predict` fail 415: chọn Body form-data, field file type File.
- Supabase key sai: `/health/ready` có warning về publishable/anon key.

### 5 câu hỏi kiểm tra nhanh

Xem `docs/learning/quiz.md`, Module 10.

---

# Cách học tương tác với giảng viên

1. Đọc Module 1 trong file này và sơ đồ ở `system_flow.md`.
2. Trả lời 5 câu hỏi Module 1 trong `quiz.md`.
3. Gửi câu trả lời cho giảng viên.
4. Giảng viên sẽ chấm, giải thích đáp án, rồi mới chuyển sang Module 2.
5. Lặp lại đến Module 10.
