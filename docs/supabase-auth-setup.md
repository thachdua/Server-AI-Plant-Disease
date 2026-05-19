# Supabase Auth setup (iOS + Render)

## 1) Lấy thông tin Supabase
- **Project URL**: `https://<project-ref>.supabase.co`
- **Anon key**: Settings → API → `anon public`

## 2) Auth → URL Configuration (quan trọng cho email + Google)

| Trường | Giá trị |
|--------|---------|
| **Site URL** | `plantdiseasedetector://auth-callback` |
| **Redirect URLs** (thêm từng dòng) | `plantdiseasedetector://auth-callback` |
| | `https://server-ai-plant-disease.onrender.com/auth/confirmed` |

Nếu thiếu bước này, link xác nhận email trong Gmail sẽ mở `*.supabase.co` và báo `{"error":"requested path is invalid"}`.

## 3) Google OAuth
### Google Cloud Console
- OAuth client type: **Web application**
- Authorized redirect URI: `plantdiseasedetector://auth-callback`

### Supabase
- Auth → Providers → Google: bật, dán Client ID + Secret

## 4) iOS
- URL scheme `plantdiseasedetector` trong `Info.plist`
- `SUPABASE_URL` + `SUPABASE_ANON_KEY` trong `Info.plist` (anon key được phép trong app client)

## 5) Render (backend bridge) — bắt buộc sau khi bỏ hardcode secret

Vào **Render → Environment** và thêm:

```
SUPABASE_URL=https://<project-ref>.supabase.co
SUPABASE_KEY=<anon_key>
DB_USER=postgres.<project-ref>
DB_PASSWORD=<database_password>
DB_HOST=aws-1-ap-southeast-1.pooler.supabase.com
DB_PORT=6543
DB_NAME=postgres
OPENWEATHER_API_KEY=...
GEMINI_API_KEY=...
```

Sau khi lưu → **Manual Deploy**. Kiểm tra:

```bash
curl "https://server-ai-plant-disease.onrender.com/outbreaks"
```

Phải trả `{"status":"success",...}` — không phải `Internal Server Error`.
