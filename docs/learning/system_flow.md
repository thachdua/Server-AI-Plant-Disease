# Bản đồ tổng quan kiến trúc hệ thống Plant Disease Detector

Tài liệu này mô tả hệ thống ở mức kiến trúc và luồng dữ liệu. Mục tiêu là giúp nhìn được toàn bộ đường đi của dữ liệu từ giao diện iOS đến backend, Supabase, AI model và các dịch vụ ngoài.

## 1. Kiến trúc tổng quan

```mermaid
flowchart LR
    User["Người dùng / Khách / Chuyên gia"]
    IOS["iOS SwiftUI App"]
    Backend["FastAPI Backend trên Render"]
    HF["Hugging Face Space / AI Predict"]
    Supabase["Supabase Auth + Postgres + Storage"]
    Gemini["Gemini API"]
    Weather["OpenWeather API"]
    GIS["Boundary GeoJSON"]

    User --> IOS
    IOS -->|"multipart/form-data /predict"| Backend
    IOS -->|"JSON /history/save, /weather, /outbreaks, /llm/*"| Backend
    IOS -->|"Supabase REST/Auth"| Supabase

    Backend -->|"image"| HF
    HF -->|"plant, disease, confidence"| Backend
    Backend -->|"upload image, insert/select"| Supabase
    Backend -->|"prompt JSON/text"| Gemini
    Backend -->|"lat/lng"| Weather
    Backend -->|"province boundary"| GIS

    Backend -->|"JSON response"| IOS
    Supabase -->|"Auth/session/table rows/public image URL"| IOS
```

## 2. Ba lớp chính của đồ án

### 2.1. Lớp giao diện iOS

Các file chính:

- `PlantDiseaseDetector(IOS)/PlantDiseaseDetector/PlantDiseaseDetectorApp.swift`
- `PlantDiseaseDetector(IOS)/PlantDiseaseDetector/Features/Auth/RootView.swift`
- `PlantDiseaseDetector(IOS)/PlantDiseaseDetector/Features/Scanner/ScannerView.swift`
- `PlantDiseaseDetector(IOS)/PlantDiseaseDetector/Features/Scanner/ScannerViewModel.swift`
- `PlantDiseaseDetector(IOS)/PlantDiseaseDetector/Services/APIService.swift`
- `PlantDiseaseDetector(IOS)/PlantDiseaseDetector/Services/SupabaseDataService.swift`
- `PlantDiseaseDetector(IOS)/PlantDiseaseDetector/Services/LLMAdviceService.swift`
- `PlantDiseaseDetector(IOS)/PlantDiseaseDetector/Services/WeatherService.swift`
- `PlantDiseaseDetector(IOS)/PlantDiseaseDetector/Services/OutbreakService.swift`

Vai trò:

- Hiển thị giao diện.
- Lấy ảnh từ camera/thư viện.
- Gửi ảnh lên backend.
- Gọi Supabase Auth/REST để đăng nhập, lưu cây, bookmark, chat, phản hồi.
- Hiển thị lịch sử, bản đồ vùng dịch, thời tiết, tư vấn AI.

### 2.2. Lớp backend FastAPI

Các file chính:

- `main.py`
- `deploy/main.py`
- `deploy/config.py`
- `deploy/security.py`
- `deploy/auth.py`
- `deploy/routers/predict.py`
- `deploy/routers/history.py`
- `deploy/routers/llm.py`
- `deploy/routers/weather.py`
- `deploy/routers/outbreaks.py`
- `deploy/routers/consultations.py`
- `deploy/database.py`
- `deploy/gemini.py`
- `deploy/validation.py`
- `deploy/rate_limit.py`

Vai trò:

- Nhận request từ iOS.
- Chặn request sai định dạng, quá lớn, vượt rate limit.
- Xác thực bearer token Supabase khi endpoint cần đăng nhập.
- Xử lý ảnh upload và gọi Hugging Face.
- Ghi lịch sử, cache LLM, vùng dịch vào Supabase/Postgres.
- Gọi Gemini, OpenWeather, dữ liệu boundary.

### 2.3. Lớp dữ liệu Supabase

Các file chính:

- `supabase/README.md`
- `supabase/sql/003_profiles_roles.sql`
- `supabase/sql/011_ai_feedback_cases.sql`
- `supabase/sql/013_chat_consent_sessions.sql`
- `supabase/sql/014_care_plants_tasks.sql`
- `supabase/sql/015_plant_knowledge_resources.sql`
- `supabase/sql/017_storage_plant_images.sql`
- `supabase/sql/021_outbreak_cases_diagnosis_links.sql`
- `supabase/sql/024_user_plants_current_diagnosis.sql`
- `supabase/sql/025_chat_message_attachments.sql`
- `supabase/sql/026_consultation_photos_and_feedback_admin.sql`
- `supabase/sql/027_profiles_account_metadata.sql`

Vai trò:

- Auth: quản lý tài khoản, token, Google OAuth.
- Postgres: lưu hồ sơ, lịch sử, feedback, chat, cây của tôi, bookmark.
- Storage: lưu ảnh chẩn đoán, ảnh feedback/retrain, ảnh tư vấn.
- RLS: đảm bảo user chỉ thao tác dữ liệu của mình, expert mới được duyệt/cập nhật workflow.

## 3. Luồng chẩn đoán bệnh cây

```mermaid
sequenceDiagram
    participant U as Người dùng
    participant VM as ScannerViewModel
    participant API as APIService
    participant BE as FastAPI /predict
    participant HF as Hugging Face Model
    participant ST as Supabase Storage
    participant DB as Supabase/Postgres

    U->>VM: Chọn cây, chụp/chọn ảnh
    VM->>VM: Nén ảnh JPEG, kiểm tra < 5MB
    VM->>API: predict(selectedPlant, jpegData, accessToken?)
    API->>BE: POST /predict multipart/form-data
    BE->>BE: rate limit, auth optional, validate metadata
    BE->>BE: sanitize image, convert JPEG
    BE->>HF: POST image + selected_plant
    HF-->>BE: plant, disease, confidence
    alt confidence < unrecognized threshold
        BE-->>API: status=unrecognized, no image_url
    else success
        BE->>ST: upload predictions/*.jpg
        BE->>DB: insert ai_feedback_cases nếu confidence thấp và user đã đăng nhập
        BE-->>API: status=success, plant, disease, confidence, image_url
    end
    API-->>VM: PredictResponse
    VM-->>U: Hiển thị kết quả / cảnh báo
```

## 4. Luồng lưu lịch sử và tạo vùng dịch

```mermaid
sequenceDiagram
    participant IOS as iOS App
    participant BE as FastAPI /history/save
    participant Auth as Supabase Auth
    participant DB as Postgres

    IOS->>BE: POST /history/save + Bearer token
    BE->>Auth: GET /auth/v1/user để kiểm token
    Auth-->>BE: user id
    BE->>BE: validate disease, image_url, lat/lng
    BE->>DB: insert history
    alt Có lat/lng, confidence >= 60, disease không healthy
        BE->>DB: insert outbreak_cases
        BE-->>IOS: outbreak_saved=true
    else Không đủ điều kiện
        BE-->>IOS: outbreak_saved=false
    end
```

## 5. Luồng Supabase RLS dễ hiểu

Ví dụ đời thường: Supabase giống một tòa nhà chung cư.

- `auth.users`: danh sách cư dân.
- `profiles`: thẻ cư dân ghi vai trò user/expert.
- RLS là bảo vệ ở cửa từng phòng.
- User thường chỉ mở được phòng của mình.
- Expert có thẻ đặc biệt nên xem/cập nhật được các phòng cần thẩm định.
- Service role key là chìa khóa tổng của ban quản lý, chỉ backend giữ, tuyệt đối không đưa vào app.

## 6. Luồng chatbot và LLM

```mermaid
flowchart TD
    IOS["ChatSupportView / LLMAdviceService"]
    BE["/llm/chat, /llm/advice/*, /llm/care-plan/*"]
    Cache["llm_advice_cache / memory cache"]
    Gemini["Gemini API"]
    Supa["chat_sessions / chat_messages nếu user đồng ý lưu"]

    IOS --> BE
    BE -->|"hash input"| Cache
    Cache -->|"hit"| BE
    Cache -->|"miss"| Gemini
    Gemini --> BE
    BE --> IOS
    IOS -->|"opt-in save"| Supa
```

## 7. Luồng Discover, Bookmark, Cây của tôi

```mermaid
flowchart LR
    IOS["iOS SupabaseDataService"]
    PK["plant_knowledge"]
    PR["plant_resources"]
    BM["bookmarks"]
    UP["user_plants"]
    CP["care_plans"]
    CT["care_tasks"]

    IOS -->|"public select"| PK
    IOS -->|"public select"| PR
    IOS -->|"user own all"| BM
    IOS -->|"user own all"| UP
    IOS -->|"user own all"| CP
    IOS -->|"user own all"| CT

    PK --> PR
    PR --> BM
    UP --> CP
    UP --> CT
    CP --> CT
```

## 8. Git hiện tại khi lập tài liệu

Khi đọc repo, trạng thái git có:

- `PlantDiseaseDetector(IOS)` có trạng thái modified theo git parent.
- `supabase/sql/025_chat_message_attachments.sql` và `supabase/sql/026_consultation_photos_and_feedback_admin.sql` đang bị xóa vài dòng comment.
- `supabase/sql/027_profiles_account_metadata.sql` là file SQL mới chưa commit.

Không có thay đổi code ứng dụng nào được thực hiện trong quá trình tạo tài liệu học này.

