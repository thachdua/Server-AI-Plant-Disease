# Project Workflow: Plant Disease Detector

## 1. Client (iOS - iPhone 8)
- Chụp ảnh lá cây.
- Tiền xử lý: Nén ảnh xuống ~28KB (JPEG).
- Gửi POST Request kèm ảnh tới Render (FastAPI).
- Nhận JSON kết quả và hiển thị.

## 2. Server (Render - FastAPI)
- Nhận ảnh từ Client.
- Chạy Model AI (71 classes) để phân loại.
- Upload ảnh vật lý lên Supabase -> Lấy Public URL.
- Lưu (Tên bệnh, Độ chính xác, URL ảnh) vào SQL Database Supabase.
- Trả kết quả cuối cùng về cho iOS.

## 3. Tech Stack
- Backend: FastAPI (Deploy trên Render).
- Storage: Supabase.
- Database: Supabase.
- Mobile: SwiftUI (iOS 16.0), MVVM.