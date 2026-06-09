# Supabase SQL

Các file trong `supabase/sql/` là các script để cấu hình **RLS/Policies** cho dữ liệu public.

## `history` (public-read)
Bạn chọn **1** trong 2 hướng:

### A) Public-read + authenticated-write (khuyến nghị)
- File: `sql/001_history_public_read_auth_write.sql`
- Ai cũng xem được lịch sử.
- Chỉ user đã đăng nhập mới thêm bản ghi (giảm spam).

### B) Public-read + backend-write-only
- File: `sql/002_history_public_read_backend_write.sql`
- iOS không được insert trực tiếp; chỉ backend Render (service role) ghi DB.

> Lưu ý: Service role key phải nằm ở backend, không bao giờ đưa vào app iOS.

## `profiles` + role (RBAC)
- File: `sql/003_profiles_roles.sql`
- Tạo bảng `profiles` với cột `role` (`user` | `expert`) để định tuyến giao diện theo quyền.
- Policy đã chặn user tự đổi `role` của mình thành `expert`.
- Nếu bạn từng chạy bản cũ của `003_profiles_roles.sql`, hãy chạy thêm `sql/009_harden_profile_roles.sql` để vá policy hiện có.

## Backend key
- Backend Render nên dùng **service role key** trong biến `SUPABASE_SERVICE_ROLE_KEY`, vì backend cần upload ảnh chẩn đoán, ghi history và ghi cache.
- iOS chỉ dùng `SUPABASE_ANON_KEY`/publishable key. Không đưa service role key vào app.

## Reports + consultations
- File: `sql/004_reports_and_consultations.sql`
- User thường được tạo và xem bản ghi của chính mình.
- Expert được xem và cập nhật workflow/review/reply.
- Nếu bạn từng chạy bản cũ của `004_reports_and_consultations.sql`, hãy chạy thêm `sql/010_harden_workflow_updates.sql` để chặn user thường PATCH các trường workflow/expert từ custom client.

## Feature migrations 011-021
Chạy theo đúng thứ tự sau sau khi đã có `profiles`, `report_cases`, `consultation_requests` và helper `public.is_expert`:

1. `sql/011_ai_feedback_cases.sql`: dữ liệu retrain cho dự đoán sai/độ tin cậy thấp.
2. `sql/012_app_feedback.sql`: góp ý/lỗi app/chatbot/danh mục.
3. `sql/013_chat_consent_sessions.sql`: lưu chat khi user đồng ý.
4. `sql/014_care_plants_tasks.sql`: bộ sưu tập cây, care plan, task/checklist.
5. `sql/015_plant_knowledge_resources.sql`: knowledge/resources/bookmark.
6. `sql/016_consultation_workflow.sql`: consent, questionnaire, SLA và preference thông báo cho tư vấn chuyên gia.
7. `sql/017_storage_plant_images.sql`: Storage bucket `plant-images` để lưu ảnh chẩn đoán/history/retrain.
8. `sql/018_discover_resource_media.sql`: thêm ảnh/thumbnail, category và seed nội dung cho tab Khám phá.
9. `sql/019_discover_in_app_content.sql`: thêm loại tài liệu mới, nội dung đọc trong app và thay link nguồn bị lỗi.
10. `sql/020_history_owner_delete.sql`: cho phép user xoá lịch sử chẩn đoán của chính mình.
11. `sql/021_outbreak_cases_diagnosis_links.sql`: liên kết ca chẩn đoán/history/ảnh vào dữ liệu vùng dịch.
12. `sql/022_outbreak_cases_location_label.sql`: thêm địa danh dễ đọc cho chi tiết ca vùng dịch.
13. `sql/023_chat_sessions_owner_delete.sql`: cho phép user xoá chat đã lưu của chính mình.
