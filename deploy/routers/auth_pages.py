from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()


@router.get("/auth/confirmed", response_class=HTMLResponse)
def email_confirmed_page():
    return """<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Đăng ký thành công</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; padding: 32px 20px;
           background: #f4f7f4; color: #1a1a1a; text-align: center; }
    .card { max-width: 420px; margin: 40px auto; background: #fff; border-radius: 16px;
            padding: 28px 22px; box-shadow: 0 8px 24px rgba(0,0,0,.08); }
    h1 { font-size: 22px; margin: 0 0 12px; color: #2e7d32; }
    p { font-size: 16px; line-height: 1.5; margin: 0 0 10px; }
    a { color: #2e7d32; font-weight: 600; text-decoration: none; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Đăng ký thành công!</h1>
    <p id="status">Email đã được xác nhận. Đang mở ứng dụng…</p>
    <p>Nếu ứng dụng không tự mở, <a id="openApp" href="#">bấm vào đây</a>.</p>
  </div>
  <script>
    (function () {
      var hash = window.location.hash || "";
      var appUrl = "plantdiseasedetector://auth-callback" + hash;
      var link = document.getElementById("openApp");
      link.href = appUrl;
      if (hash.indexOf("access_token=") >= 0) {
        window.location.replace(appUrl);
      } else {
        document.getElementById("status").textContent =
          "Đăng ký thành công! Vui lòng mở ứng dụng Plant Disease Detector.";
      }
    })();
  </script>
</body>
</html>"""


@router.get("/auth/recovery", response_class=HTMLResponse)
def password_recovery_page():
    return """<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Đặt lại mật khẩu</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; padding: 32px 20px;
           background: #f4f7f4; color: #1a1a1a; text-align: center; }
    .card { max-width: 420px; margin: 40px auto; background: #fff; border-radius: 16px;
            padding: 28px 22px; box-shadow: 0 8px 24px rgba(0,0,0,.08); }
    h1 { font-size: 22px; margin: 0 0 12px; color: #2e7d32; }
    p { font-size: 16px; line-height: 1.5; margin: 0 0 10px; }
    a { color: #2e7d32; font-weight: 600; text-decoration: none; }
    .button { display: inline-block; margin-top: 12px; padding: 12px 18px; border-radius: 12px;
              background: #2e7d32; color: #fff; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Đặt lại mật khẩu</h1>
    <p id="status">Đang mở ứng dụng để đặt mật khẩu mới…</p>
    <p>Nếu ứng dụng không tự mở, hãy bấm nút bên dưới.</p>
    <a id="openApp" class="button" href="#">Mở ứng dụng</a>
  </div>
  <script>
    (function () {
      var hash = window.location.hash || "";
      var query = window.location.search || "";
      var payload = hash || query;
      var appUrl = "plantdiseasedetector://auth-callback" + payload;
      var link = document.getElementById("openApp");
      var status = document.getElementById("status");
      var isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent);
      link.href = appUrl;
      if (isIOS && (payload.indexOf("access_token=") >= 0 || payload.indexOf("error=") >= 0)) {
        setTimeout(function () { window.location.href = appUrl; }, 250);
      } else if (!isIOS) {
        status.textContent = "Vui lòng mở email đặt lại mật khẩu trên iPhone đã cài ứng dụng, hoặc bấm nút mở ứng dụng nếu đang dùng iPhone.";
      } else {
        status.textContent = "Liên kết đặt lại mật khẩu chưa có phiên hợp lệ. Hãy mở lại email reset mới nhất hoặc yêu cầu gửi lại email.";
      }
    })();
  </script>
</body>
</html>"""
