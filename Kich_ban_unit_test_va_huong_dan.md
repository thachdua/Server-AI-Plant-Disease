# KỊCH BẢN UNIT TEST VÀ HƯỚNG DẪN CÔNG CỤ KIỂM THỬ

## 1. Mục đích kiểm thử

Unit test được sử dụng để kiểm tra từng thành phần nhỏ của hệ thống Plant Disease Detector nhằm đảm bảo các hàm xử lý, API backend và một số logic phía ứng dụng iOS hoạt động đúng theo yêu cầu. Việc kiểm thử giúp phát hiện lỗi sớm, hạn chế lỗi phát sinh khi thay đổi mã nguồn và tăng độ tin cậy của hệ thống trước khi triển khai thực tế.

Trong đồ án này, unit test tập trung vào các nhóm chức năng chính:

- Kiểm thử API chẩn đoán bệnh cây.
- Kiểm thử xử lý ảnh đầu vào.
- Kiểm thử chuẩn hóa kết quả dự đoán từ mô hình AI.
- Kiểm thử lưu lịch sử chẩn đoán.
- Kiểm thử kiểm tra dữ liệu đầu vào của API.
- Kiểm thử bảo mật, giới hạn request và content-type.
- Kiểm thử API thời tiết, ổ dịch và tư vấn AI.
- Kiểm thử một số logic phía ứng dụng iOS.

## 2. Công cụ sử dụng để unit test

### 2.1. Python unittest

`unittest` là thư viện kiểm thử có sẵn trong Python, được sử dụng để viết và chạy các ca kiểm thử cho backend FastAPI. Ưu điểm của `unittest` là không cần cài thêm thư viện ngoài, cú pháp rõ ràng và phù hợp với các bài kiểm thử hàm, service hoặc API.

Trong đồ án, các file test backend được đặt trong thư mục `tests/`, ví dụ:

- `test_predict_endpoint.py`: kiểm thử API dự đoán bệnh cây.
- `test_predict_helpers.py`: kiểm thử các hàm hỗ trợ xử lý kết quả dự đoán và ảnh.
- `test_validation.py`: kiểm thử dữ liệu đầu vào.
- `test_security.py`: kiểm thử bảo mật API.
- `test_history_router.py`: kiểm thử lưu lịch sử chẩn đoán.
- `test_health.py`: kiểm thử endpoint kiểm tra trạng thái server.
- `test_rate_limit.py`: kiểm thử giới hạn request.
- `test_llm_care_plan.py`: kiểm thử chức năng tư vấn AI/lịch chăm sóc.
- `test_supabase_sql.py`: kiểm thử nội dung migration và policy Supabase.

Lệnh chạy toàn bộ test backend:

```bash
python -m unittest discover -s tests
```

### 2.2. FastAPI TestClient

`TestClient` là công cụ dùng để kiểm thử API FastAPI mà không cần chạy server thật bằng Uvicorn. TestClient cho phép gửi request trực tiếp đến ứng dụng FastAPI trong môi trường test, sau đó kiểm tra status code và nội dung response.

Ví dụ, hệ thống có thể gửi request test đến endpoint `/predict`, `/weather`, `/health`, `/outbreaks` hoặc `/llm/advice/diagnosis` để kiểm tra kết quả trả về.

Ưu điểm:

- Không cần deploy backend khi test.
- Không cần mở port server.
- Có thể kiểm tra response JSON, status code và lỗi trả về.
- Phù hợp kiểm thử API trong quá trình phát triển.

### 2.3. unittest.mock

`unittest.mock` được sử dụng để giả lập các dịch vụ bên ngoài như Hugging Face, Supabase, Gemini API hoặc OpenWeather API. Khi viết unit test, không nên gọi trực tiếp các dịch vụ thật vì có thể làm test chậm, tốn quota API và phụ thuộc vào mạng.

Trong đồ án, `mock` được dùng để:

- Giả lập kết quả trả về từ mô hình AI.
- Giả lập thao tác upload ảnh lên Supabase Storage.
- Giả lập thao tác ghi dữ liệu vào Supabase.
- Kiểm tra rằng API bên ngoài không bị gọi khi dữ liệu đầu vào không hợp lệ.

Ví dụ:

```python
with patch("deploy.routers.predict.requests.post", return_value=hf_response):
    response = client.post("/predict", files={...})
```

### 2.4. Pillow

`Pillow` là thư viện xử lý ảnh trong Python. Trong unit test, Pillow được sử dụng để tạo ảnh JPEG giả lập nhằm kiểm tra chức năng upload và xử lý ảnh mà không cần dùng ảnh thật từ máy.

Ví dụ tạo ảnh test:

```python
import io
from PIL import Image

def jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (12, 12), color=(40, 160, 80)).save(buf, format="JPEG")
    return buf.getvalue()
```

### 2.5. Xcode Testing / XCTest cho iOS

Đối với ứng dụng iOS, Xcode cung cấp công cụ kiểm thử tích hợp gồm Unit Test và UI Test. Unit Test dùng để kiểm tra các hàm xử lý logic, còn UI Test dùng để kiểm tra thao tác giao diện như mở app, chuyển màn hình, nhấn nút hoặc nhập dữ liệu.

Trong đồ án, thư mục iOS đã có sẵn:

- `PlantDiseaseDetectorTests`: dùng cho unit test logic iOS.
- `PlantDiseaseDetectorUITests`: dùng cho kiểm thử giao diện.

Các phần phù hợp để viết unit test iOS gồm:

- Kiểm tra hàm xử lý tên bệnh/localize bệnh.
- Kiểm tra logic chọn cây.
- Kiểm tra format độ tin cậy.
- Kiểm tra xử lý URL backend/Supabase config.
- Kiểm tra các helper tạo nội dung hiển thị kết quả.

### 2.6. Postman

Postman không phải công cụ unit test chính, nhưng được sử dụng để kiểm thử thủ công API trong quá trình phát triển. Postman giúp gửi request đến backend đã chạy local hoặc đã deploy trên Render để kiểm tra nhanh các endpoint.

Các endpoint có thể kiểm thử bằng Postman:

- `GET /health`
- `GET /health/ready`
- `POST /predict`
- `GET /weather`
- `POST /llm/advice/diagnosis`
- `GET /outbreaks`
- `GET /outbreaks/areas`

## 3. Kịch bản unit test backend

### 3.1. Nhóm kiểm thử API chẩn đoán bệnh cây

| Mã test | Mục tiêu kiểm thử | Dữ liệu đầu vào | Kết quả mong đợi |
|---|---|---|---|
| UT-PD-01 | Kiểm tra API `/predict` trả kết quả thành công | Ảnh JPEG hợp lệ, loại cây Tomato, mô hình trả confidence 0.9234 | Status code 200, trả về plant, disease, confidence dạng phần trăm và image_url |
| UT-PD-02 | Kiểm tra hệ thống từ chối kết quả thiếu dữ liệu từ mô hình | Mô hình trả thiếu trường disease | Status code 502, thông báo Prediction service returned incomplete result |
| UT-PD-03 | Kiểm tra ảnh có độ tin cậy quá thấp | Mô hình trả confidence thấp hơn ngưỡng nhận diện | Response có status `unrecognized`, không tự động lưu vào bảng phản hồi |
| UT-PD-04 | Kiểm tra ghi nhận ca độ tin cậy thấp khi người dùng đã đăng nhập | Mô hình trả confidence thấp nhưng vẫn trên ngưỡng unrecognized | Lưu dữ liệu vào bảng `ai_feedback_cases` với reason `low_confidence` |
| UT-PD-05 | Kiểm tra endpoint gửi phản hồi ảnh độ tin cậy thấp | Người dùng gửi ảnh, cây dự đoán, bệnh dự đoán, ghi chú | Status code 200, lưu phản hồi vào Supabase |

### 3.2. Nhóm kiểm thử xử lý ảnh

| Mã test | Mục tiêu kiểm thử | Dữ liệu đầu vào | Kết quả mong đợi |
|---|---|---|---|
| UT-IMG-01 | Kiểm tra hệ thống chấp nhận ảnh JPEG hợp lệ | File JPEG được tạo bằng Pillow | Hàm validate ảnh trả về thành công |
| UT-IMG-02 | Kiểm tra hệ thống từ chối file không phải ảnh | File text hoặc bytes không hợp lệ | Trả lỗi ảnh không được hỗ trợ |
| UT-IMG-03 | Kiểm tra content-type upload không hợp lệ | File có content-type không phải image/jpeg, image/png, image/webp | Status code 415 |
| UT-IMG-04 | Kiểm tra ảnh vượt dung lượng cho phép | File ảnh lớn hơn giới hạn cấu hình | Status code 413 |

### 3.3. Nhóm kiểm thử hàm hỗ trợ dự đoán

| Mã test | Mục tiêu kiểm thử | Dữ liệu đầu vào | Kết quả mong đợi |
|---|---|---|---|
| UT-HELP-01 | Tách tên cây từ nhãn mô hình | `Tomato___Late_blight` | Trả về `Tomato` |
| UT-HELP-02 | Chuyển confidence từ số thập phân sang phần trăm | `0.9234` | Trả về `92.34%` |
| UT-HELP-03 | Chuyển confidence dạng chuỗi phần trăm | `85%` | Trả về giá trị 85.0 |
| UT-HELP-04 | Xử lý confidence không hợp lệ | Chuỗi không phải số | Trả về `None` |

### 3.4. Nhóm kiểm thử validation dữ liệu đầu vào

| Mã test | Mục tiêu kiểm thử | Dữ liệu đầu vào | Kết quả mong đợi |
|---|---|---|---|
| UT-VAL-01 | Kiểm tra API thời tiết từ chối latitude sai | `lat = 120` | Status code 400, không gọi OpenWeather |
| UT-VAL-02 | Kiểm tra API tư vấn thời tiết từ chối longitude sai | `lng = 999` | Status code 400, không gọi OpenWeather |
| UT-VAL-03 | Kiểm tra API ổ dịch từ chối severity ngoài khoảng | `severity = 9` | Status code 400 |
| UT-VAL-04 | Kiểm tra API vùng dịch từ chối since_days quá lớn | `since_days = 3651` | Status code 400 |
| UT-VAL-05 | Kiểm tra API tư vấn bệnh yêu cầu có tên bệnh | disease rỗng | Status code 422 |

### 3.5. Nhóm kiểm thử lịch sử chẩn đoán

| Mã test | Mục tiêu kiểm thử | Dữ liệu đầu vào | Kết quả mong đợi |
|---|---|---|---|
| UT-HIS-01 | Kiểm tra lưu lịch sử theo user đã xác thực | Token user hợp lệ, dữ liệu chẩn đoán hợp lệ | Lưu lịch sử với đúng user_id |
| UT-HIS-02 | Kiểm tra tự tạo ca ổ dịch khi bệnh không khỏe và đủ tin cậy | Bệnh cây confidence cao, có vị trí | Tạo thêm dữ liệu outbreak |
| UT-HIS-03 | Kiểm tra không tạo ổ dịch khi confidence thấp | Dữ liệu bệnh nhưng confidence dưới ngưỡng | Không tạo outbreak |
| UT-HIS-04 | Kiểm tra không tạo ổ dịch khi cây khỏe mạnh | Kết quả healthy | Không tạo outbreak |
| UT-HIS-05 | Kiểm tra lỗi database khi lưu lịch sử thất bại | Supabase/database lỗi | Trả lỗi phù hợp, không làm ứng dụng crash |

### 3.6. Nhóm kiểm thử bảo mật API

| Mã test | Mục tiêu kiểm thử | Dữ liệu đầu vào | Kết quả mong đợi |
|---|---|---|---|
| UT-SEC-01 | Kiểm tra security headers | Gửi request đến backend | Response có security headers |
| UT-SEC-02 | Kiểm tra endpoint JSON từ chối content-type sai | Request JSON nhưng content-type không hợp lệ | Status code 415 |
| UT-SEC-03 | Kiểm tra request khai báo body quá lớn | Content-Length vượt giới hạn | Status code 413 |
| UT-SEC-04 | Kiểm tra global rate limit | Gửi nhiều request vượt giới hạn | Status code 429 |
| UT-SEC-05 | Kiểm tra endpoint chat từ chối field lạ | JSON có field không nằm trong schema | Status code lỗi validation |
| UT-SEC-06 | Kiểm tra `/predict` từ chối metadata upload không phải ảnh | File `.txt` hoặc content-type text/plain | Status code 415 |

### 3.7. Nhóm kiểm thử health check và cấu hình

| Mã test | Mục tiêu kiểm thử | Dữ liệu đầu vào | Kết quả mong đợi |
|---|---|---|---|
| UT-HEALTH-01 | Kiểm tra `/health` hoạt động nhẹ | GET `/health` | Status code 200, không phụ thuộc dịch vụ ngoài |
| UT-HEALTH-02 | Kiểm tra `/health/ready` báo thiếu cấu hình | Thiếu biến môi trường bắt buộc | Trả thông tin thiếu cấu hình nhưng không lộ secret |
| UT-CONFIG-01 | Kiểm tra backend ưu tiên service role key | Cấu hình Supabase service key | Dùng đúng key phía server |
| UT-CONFIG-02 | Kiểm tra cảnh báo khi dùng nhầm publishable key | Cấu hình key không phù hợp | Có cảnh báo bảo mật |

### 3.8. Nhóm kiểm thử tư vấn AI và lịch chăm sóc

| Mã test | Mục tiêu kiểm thử | Dữ liệu đầu vào | Kết quả mong đợi |
|---|---|---|---|
| UT-LLM-01 | Kiểm tra request tư vấn có chứa ngữ cảnh cây | Thông tin cây, bệnh, thời tiết | Payload gửi AI có đủ ngữ cảnh |
| UT-LLM-02 | Kiểm tra endpoint tạo lịch chăm sóc chuẩn hóa task | Dữ liệu task từ AI | Trả về danh sách task hợp lệ |
| UT-LLM-03 | Kiểm tra giới hạn độ dài prompt | Prompt quá dài | Từ chối trước khi gọi Gemini |
| UT-LLM-04 | Kiểm tra rate limit tư vấn AI | Gửi nhiều request liên tiếp | Trả lỗi 429 khi vượt giới hạn |

## 4. Kịch bản unit test iOS

### 4.1. Nhóm kiểm thử logic hiển thị kết quả

| Mã test | Mục tiêu kiểm thử | Dữ liệu đầu vào | Kết quả mong đợi |
|---|---|---|---|
| UT-IOS-01 | Kiểm tra format độ tin cậy | Confidence `92.34%` | Hiển thị đúng định dạng phần trăm |
| UT-IOS-02 | Kiểm tra thông báo khi kết quả unrecognized | Response có status `unrecognized` | Hiển thị thông báo không nhận diện được |
| UT-IOS-03 | Kiểm tra hiển thị kết quả thành công | Response gồm plant, disease, image_url | Hiển thị đầy đủ cây, bệnh, ảnh và độ tin cậy |

### 4.2. Nhóm kiểm thử cấu hình dịch vụ

| Mã test | Mục tiêu kiểm thử | Dữ liệu đầu vào | Kết quả mong đợi |
|---|---|---|---|
| UT-IOS-04 | Kiểm tra đọc `BACKEND_BASE_URL` | Info.plist có URL backend | Service lấy đúng URL backend |
| UT-IOS-05 | Kiểm tra đọc `SUPABASE_URL` | Info.plist có Supabase URL | App lấy đúng cấu hình Supabase |
| UT-IOS-06 | Kiểm tra thiếu cấu hình | Thiếu URL hoặc key | App trả lỗi dễ hiểu |

### 4.3. Nhóm kiểm thử xử lý chọn cây và ảnh

| Mã test | Mục tiêu kiểm thử | Dữ liệu đầu vào | Kết quả mong đợi |
|---|---|---|---|
| UT-IOS-07 | Kiểm tra chưa chọn cây | Người dùng chưa chọn cây trước khi quét | Hiển thị yêu cầu chọn cây |
| UT-IOS-08 | Kiểm tra chọn cây hợp lệ | Chọn Tomato hoặc Apple | ViewModel lưu đúng cây đã chọn |
| UT-IOS-09 | Kiểm tra ảnh rỗng hoặc ảnh lỗi | Không có ảnh được chọn | Không gửi request, hiển thị lỗi phù hợp |

## 5. Hướng dẫn chạy unit test backend

### 5.1. Cài đặt môi trường

Tạo môi trường ảo Python:

```bash
python -m venv .venv
source .venv/bin/activate
```

Cài đặt thư viện:

```bash
pip install -r requirements.txt
```

### 5.2. Chạy toàn bộ unit test

```bash
python -m unittest discover -s tests
```

### 5.3. Chạy một file test cụ thể

Ví dụ chạy test API dự đoán:

```bash
python -m unittest tests/test_predict_endpoint.py
```

Ví dụ chạy test validation:

```bash
python -m unittest tests/test_validation.py
```

### 5.4. Chạy một test case cụ thể

```bash
python -m unittest tests.test_predict_endpoint.PredictEndpointTests.test_predict_success_normalizes_response
```

## 6. Hướng dẫn chạy unit test iOS

### 6.1. Chạy bằng Xcode

Các bước thực hiện:

1. Mở project `PlantDiseaseDetector.xcodeproj` bằng Xcode.
2. Chọn scheme `PlantDiseaseDetector`.
3. Chọn thiết bị iOS Simulator phù hợp.
4. Nhấn `Command + U` để chạy toàn bộ test.
5. Xem kết quả trong Test Navigator của Xcode.

### 6.2. Chạy bằng Terminal

Có thể chạy test iOS bằng `xcodebuild`:

```bash
xcodebuild test \
  -project "PlantDiseaseDetector(IOS)/PlantDiseaseDetector.xcodeproj" \
  -scheme "PlantDiseaseDetector" \
  -destination 'platform=iOS Simulator,name=iPhone 15'
```

Tên simulator có thể thay đổi tùy máy. Có thể xem danh sách simulator bằng lệnh:

```bash
xcrun simctl list devices
```

## 7. Mẫu trình bày kết quả unit test trong báo cáo

| STT | Nhóm kiểm thử | Số test case | Kết quả mong đợi | Trạng thái |
|---|---|---:|---|---|
| 1 | API chẩn đoán bệnh cây | 5 | API trả kết quả đúng, xử lý được lỗi mô hình và độ tin cậy thấp | Đạt |
| 2 | Xử lý ảnh | 4 | Chấp nhận ảnh hợp lệ, từ chối ảnh sai định dạng hoặc quá lớn | Đạt |
| 3 | Validation dữ liệu đầu vào | 5 | Từ chối dữ liệu sai trước khi gọi dịch vụ ngoài | Đạt |
| 4 | Lịch sử chẩn đoán | 5 | Lưu đúng lịch sử, tạo ổ dịch đúng điều kiện | Đạt |
| 5 | Bảo mật API | 6 | Chặn content-type sai, body quá lớn và request vượt giới hạn | Đạt |
| 6 | Health check và cấu hình | 4 | Kiểm tra trạng thái backend và cấu hình không lộ secret | Đạt |
| 7 | Tư vấn AI và lịch chăm sóc | 4 | Tạo tư vấn/lịch chăm sóc hợp lệ, chặn prompt quá dài | Đạt |
| 8 | Logic iOS | 9 | Hiển thị đúng kết quả, đọc đúng cấu hình, xử lý chọn cây/ảnh | Đề xuất kiểm thử |

## 8. Kết luận

Thông qua unit test, hệ thống Plant Disease Detector được kiểm tra ở nhiều thành phần quan trọng như API dự đoán bệnh cây, xử lý ảnh, kiểm tra dữ liệu đầu vào, bảo mật API, lưu lịch sử chẩn đoán và tích hợp tư vấn AI. Các test backend giúp đảm bảo server xử lý đúng trong cả trường hợp thành công và lỗi. Đối với ứng dụng iOS, unit test và UI test giúp kiểm tra logic hiển thị, cấu hình dịch vụ và luồng thao tác người dùng.

Việc xây dựng kịch bản unit test giúp hệ thống ổn định hơn, dễ bảo trì hơn và giảm rủi ro phát sinh lỗi khi nâng cấp mô hình AI, thay đổi API hoặc bổ sung chức năng mới trong tương lai.

