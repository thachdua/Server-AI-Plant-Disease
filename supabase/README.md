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
