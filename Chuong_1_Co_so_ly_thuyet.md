# CHƯƠNG 1. CƠ SỞ LÝ THUYẾT

## 1.1. Tổng quan về trí tuệ nhân tạo trong nhận diện bệnh cây

Trí tuệ nhân tạo là lĩnh vực nghiên cứu và phát triển các hệ thống có khả năng mô phỏng một số hoạt động thông minh của con người như học tập, nhận dạng, suy luận và đưa ra quyết định. Trong những năm gần đây, trí tuệ nhân tạo được ứng dụng rộng rãi trong nhiều lĩnh vực như y tế, giáo dục, giao thông, thương mại điện tử và nông nghiệp. Đối với nông nghiệp, AI có thể hỗ trợ phân tích dữ liệu thời tiết, dự đoán năng suất, phát hiện sâu bệnh, nhận diện tình trạng cây trồng và đưa ra gợi ý chăm sóc phù hợp.

Trong bài toán nhận diện bệnh cây, trí tuệ nhân tạo thường được sử dụng để phân tích hình ảnh lá, thân hoặc quả nhằm xác định cây đang khỏe mạnh hay mắc một loại bệnh cụ thể. Người dùng chỉ cần chụp ảnh cây trồng, hệ thống sẽ xử lý hình ảnh và đưa ra kết quả dự đoán. Cách tiếp cận này giúp rút ngắn thời gian kiểm tra, hỗ trợ người dùng chưa có nhiều kinh nghiệm và góp phần phát hiện bệnh sớm hơn so với phương pháp quan sát thủ công.

Các lợi ích chính của việc ứng dụng AI trong nhận diện bệnh cây bao gồm:

- Hỗ trợ phát hiện bệnh cây nhanh chóng thông qua hình ảnh.
- Giảm sự phụ thuộc vào kinh nghiệm cá nhân của người trồng.
- Hỗ trợ người dùng đưa ra hướng xử lý và chăm sóc kịp thời.
- Có thể kết hợp với dữ liệu thời tiết, vị trí và lịch sử chẩn đoán để tăng hiệu quả theo dõi.
- Tạo nền tảng cho các hệ thống nông nghiệp thông minh trong tương lai.

Trong đồ án này, trí tuệ nhân tạo được ứng dụng vào hệ thống **Plant Disease Detector** nhằm hỗ trợ nhận diện bệnh cây từ ảnh chụp trên ứng dụng iOS. Kết quả dự đoán bao gồm loại cây, loại bệnh và độ tin cậy, từ đó hệ thống có thể cung cấp thêm gợi ý chăm sóc bằng tiếng Việt cho người dùng.

## 1.2. Tổng quan về xử lý ảnh

Xử lý ảnh là lĩnh vực nghiên cứu các phương pháp thu nhận, biến đổi, phân tích và trích xuất thông tin từ hình ảnh. Trong các hệ thống nhận diện đối tượng, xử lý ảnh đóng vai trò quan trọng vì chất lượng ảnh đầu vào ảnh hưởng trực tiếp đến kết quả dự đoán của mô hình. Một ảnh rõ nét, đủ sáng, đúng đối tượng và ít nhiễu thường giúp mô hình nhận diện chính xác hơn.

Đối với bài toán nhận diện bệnh cây, ảnh đầu vào thường là ảnh lá cây hoặc bộ phận cây có biểu hiện bệnh. Hệ thống cần kiểm tra định dạng ảnh, kích thước ảnh, dung lượng ảnh và khả năng đọc ảnh trước khi gửi đến mô hình AI. Việc kiểm tra này giúp tránh các lỗi do ảnh không hợp lệ, ảnh quá lớn hoặc file không phải ảnh.

Các bước xử lý ảnh thường gặp trong hệ thống nhận diện bệnh cây bao gồm:

- Kiểm tra định dạng ảnh như JPEG, PNG hoặc WebP.
- Kiểm tra dung lượng ảnh để tránh ảnh quá lớn.
- Đọc và xác thực ảnh trước khi xử lý.
- Chuyển đổi ảnh về định dạng phù hợp với mô hình.
- Làm sạch ảnh trước khi lưu trữ hoặc gửi đến dịch vụ dự đoán.
- Trả về kết quả gồm nhãn dự đoán và độ tin cậy.

Trong đồ án, backend có nhiệm vụ kiểm tra ảnh tải lên, giới hạn kích thước ảnh, xử lý ảnh an toàn và chuyển ảnh sang định dạng JPEG trước khi gửi đến mô hình nhận diện cũng như lưu lên Supabase Storage.

## 1.3. Tổng quan về mô hình học sâu CNN

Học sâu là một nhánh của học máy, sử dụng các mạng nơ-ron nhiều tầng để học đặc trưng từ dữ liệu. Trong lĩnh vực thị giác máy tính, mạng nơ-ron tích chập, còn gọi là Convolutional Neural Network (CNN), là một trong những mô hình phổ biến nhất để xử lý và phân loại hình ảnh.

CNN có khả năng tự động trích xuất đặc trưng từ ảnh thông qua các lớp tích chập. Thay vì phải thiết kế đặc trưng thủ công, mô hình có thể học được các đặc điểm như cạnh, màu sắc, hình dạng, hoa văn, vùng tổn thương hoặc dấu hiệu bệnh trên lá cây. Nhờ đó, CNN phù hợp với các bài toán như nhận diện khuôn mặt, phân loại ảnh, phát hiện vật thể và nhận diện bệnh cây.

Cấu trúc cơ bản của CNN thường gồm:

- Lớp tích chập: dùng để trích xuất đặc trưng từ ảnh đầu vào.
- Lớp kích hoạt: thường sử dụng hàm ReLU để tăng khả năng học phi tuyến.
- Lớp pooling: giảm kích thước dữ liệu đặc trưng và giữ lại thông tin quan trọng.
- Lớp fully connected: tổng hợp đặc trưng để đưa ra kết quả phân loại.
- Lớp đầu ra: trả về xác suất hoặc nhãn dự đoán tương ứng với từng lớp bệnh.

Trong hệ thống Plant Disease Detector, mô hình nhận diện bệnh cây được sử dụng để phân tích ảnh cây trồng và trả về kết quả dự đoán. Kết quả này được backend xử lý tiếp để hiển thị trên ứng dụng iOS cho người dùng.

```text
Ảnh cây trồng
      |
      v
Tiền xử lý ảnh
      |
      v
Mô hình CNN
      |
      v
Kết quả dự đoán: cây, bệnh, độ tin cậy
```

Hình 1.1. Sơ đồ tổng quát quá trình nhận diện bệnh cây bằng CNN

## 1.4. Tổng quan về Swift

Swift là ngôn ngữ lập trình do Apple phát triển, được sử dụng để xây dựng ứng dụng trên các nền tảng như iOS, macOS, watchOS và tvOS. Swift có cú pháp hiện đại, dễ đọc, hiệu năng cao và hỗ trợ nhiều tính năng giúp lập trình viên phát triển ứng dụng an toàn hơn.

Một số đặc điểm nổi bật của Swift gồm:

- Cú pháp ngắn gọn, dễ hiểu và dễ bảo trì.
- Hiệu năng tốt, phù hợp với phát triển ứng dụng di động.
- Hỗ trợ kiểm soát kiểu dữ liệu chặt chẽ, giúp giảm lỗi trong quá trình lập trình.
- Hỗ trợ lập trình hướng đối tượng và lập trình hàm.
- Tích hợp tốt với hệ sinh thái Apple và công cụ Xcode.
- Có thể kết hợp với SwiftUI để xây dựng giao diện hiện đại.

Trong đồ án này, Swift được sử dụng để xây dựng ứng dụng iOS Plant Disease Detector. Ứng dụng cho phép người dùng đăng nhập, quét ảnh cây, xem kết quả chẩn đoán, xem lịch sử, sử dụng trợ lý AI, xem thời tiết và theo dõi bản đồ ổ dịch.

## 1.5. Tổng quan về SwiftUI

SwiftUI là framework xây dựng giao diện người dùng do Apple phát triển. SwiftUI cho phép lập trình viên mô tả giao diện theo phong cách khai báo, nghĩa là giao diện được xây dựng dựa trên trạng thái dữ liệu của ứng dụng. Khi dữ liệu thay đổi, giao diện sẽ tự động cập nhật tương ứng.

So với cách xây dựng giao diện truyền thống, SwiftUI giúp mã nguồn ngắn gọn, dễ đọc và dễ tái sử dụng hơn. Các thành phần giao diện được tổ chức thành các View, mỗi View có thể chứa các thành phần nhỏ hơn và được kết hợp để tạo thành màn hình hoàn chỉnh.

Các tính năng chính của SwiftUI gồm:

- Xây dựng giao diện theo phong cách khai báo.
- Tự động cập nhật giao diện khi trạng thái thay đổi.
- Hỗ trợ tái sử dụng thành phần giao diện.
- Tích hợp tốt với hệ sinh thái Apple.
- Hỗ trợ xem trước giao diện trong Xcode.
- Dễ dàng kết hợp với các thành phần như NavigationView, TabView, Sheet và Form.

Trong đồ án, SwiftUI được sử dụng để xây dựng các màn hình như trang chủ, màn hình quét bệnh cây, màn hình đăng nhập, màn hình lịch sử, màn hình thời tiết, màn hình bản đồ ổ dịch và màn hình trợ lý thông minh.

## 1.6. Tổng quan về FastAPI

FastAPI là một framework Python hiện đại dùng để xây dựng API với hiệu năng cao. FastAPI hỗ trợ khai báo kiểu dữ liệu, kiểm tra dữ liệu đầu vào, sinh tài liệu API tự động và xử lý bất đồng bộ. Nhờ đó, FastAPI phù hợp để xây dựng backend cho các ứng dụng web, ứng dụng di động và hệ thống tích hợp nhiều dịch vụ.

Một số đặc điểm nổi bật của FastAPI gồm:

- Hiệu năng cao nhờ sử dụng Starlette và Pydantic.
- Hỗ trợ kiểm tra dữ liệu đầu vào dựa trên kiểu dữ liệu Python.
- Tự động tạo tài liệu API thông qua Swagger UI và OpenAPI.
- Hỗ trợ xử lý bất đồng bộ.
- Dễ dàng tổ chức router theo từng nhóm chức năng.
- Phù hợp với các hệ thống cần giao tiếp qua RESTful API.

Trong đồ án, FastAPI được sử dụng để xây dựng backend cho hệ thống Plant Disease Detector. Backend có nhiệm vụ tiếp nhận ảnh từ ứng dụng iOS, kiểm tra dữ liệu đầu vào, gọi mô hình nhận diện bệnh cây, lưu ảnh lên Supabase Storage, lưu lịch sử chẩn đoán và cung cấp API cho các chức năng thời tiết, tư vấn AI, bản đồ ổ dịch và xác thực.

## 1.7. Tổng quan về RESTful API

REST (Representational State Transfer) là một kiểu kiến trúc được sử dụng phổ biến trong thiết kế API. RESTful API cho phép các hệ thống khác nhau giao tiếp với nhau thông qua giao thức HTTP. Mỗi tài nguyên trong hệ thống thường được biểu diễn bằng một đường dẫn URL, còn các thao tác với tài nguyên được thực hiện thông qua các phương thức HTTP.

Các phương thức HTTP thường dùng trong RESTful API gồm:

- GET: dùng để lấy dữ liệu từ server.
- POST: dùng để tạo mới dữ liệu hoặc gửi dữ liệu lên server.
- PUT/PATCH: dùng để cập nhật dữ liệu.
- DELETE: dùng để xóa dữ liệu.

RESTful API có ưu điểm là dễ hiểu, dễ triển khai và không phụ thuộc vào ngôn ngữ lập trình. Ứng dụng client có thể được xây dựng bằng Swift, JavaScript, Java hoặc bất kỳ ngôn ngữ nào miễn là có thể gửi request HTTP đến server.

Trong đồ án, ứng dụng iOS giao tiếp với backend FastAPI thông qua các API như API dự đoán bệnh cây, API lưu lịch sử, API tư vấn AI, API thời tiết và API ổ dịch. Ví dụ, khi người dùng chọn ảnh và nhấn phân tích, ứng dụng sẽ gửi request đến endpoint `/predict`; backend xử lý ảnh và trả kết quả về cho ứng dụng.

```text
Client gửi request HTTP
        |
        v
Backend xử lý yêu cầu
        |
        v
Trả response dạng JSON
```

Hình 1.2. Sơ đồ hoạt động cơ bản của RESTful API

## 1.8. Tổng quan về mô hình Client-Server

Mô hình Client-Server là mô hình kiến trúc phổ biến trong các hệ thống phần mềm hiện nay. Trong mô hình này, client là phía gửi yêu cầu, còn server là phía tiếp nhận yêu cầu, xử lý và trả kết quả. Client có thể là ứng dụng di động, trình duyệt web hoặc một phần mềm khác. Server thường là hệ thống backend có nhiệm vụ xử lý nghiệp vụ, quản lý dữ liệu và kết nối với các dịch vụ liên quan.

Nguyên tắc hoạt động của mô hình Client-Server như sau:

- Client gửi yêu cầu đến server thông qua mạng Internet.
- Server tiếp nhận và kiểm tra yêu cầu.
- Server xử lý nghiệp vụ hoặc truy vấn dữ liệu cần thiết.
- Server trả kết quả về client.
- Client hiển thị kết quả cho người dùng.

Mô hình Client-Server giúp tách biệt giao diện người dùng và phần xử lý nghiệp vụ. Nhờ đó, hệ thống dễ bảo trì, dễ mở rộng và có thể thay đổi backend hoặc client độc lập hơn.

Trong đồ án Plant Disease Detector, ứng dụng iOS đóng vai trò client, còn FastAPI backend đóng vai trò server. Ứng dụng iOS gửi ảnh cây trồng và thông tin loại cây đến server; server gọi mô hình AI, lưu dữ liệu và trả kết quả dự đoán về ứng dụng.

```text
Ứng dụng iOS
    |
    | Gửi ảnh và thông tin cây
    v
FastAPI Backend
    |
    | Gọi AI, Supabase, thời tiết, Gemini
    v
Kết quả trả về ứng dụng
```

Hình 1.3. Mô hình Client-Server trong hệ thống Plant Disease Detector

## 1.9. Tổng quan về Supabase

Supabase là nền tảng Backend-as-a-Service mã nguồn mở, cung cấp nhiều dịch vụ hỗ trợ phát triển ứng dụng như cơ sở dữ liệu PostgreSQL, xác thực người dùng, lưu trữ file và API thời gian thực. Supabase giúp lập trình viên xây dựng backend nhanh hơn mà không cần tự triển khai toàn bộ hệ thống cơ sở dữ liệu và xác thực từ đầu.

Các thành phần chính của Supabase gồm:

- Supabase Auth: hỗ trợ đăng ký, đăng nhập, xác thực người dùng và quản lý phiên đăng nhập.
- Supabase Database: sử dụng PostgreSQL để lưu trữ dữ liệu.
- Supabase Storage: lưu trữ file như ảnh, tài liệu hoặc avatar.
- Row Level Security: cơ chế bảo mật giúp kiểm soát quyền truy cập dữ liệu theo từng người dùng.
- API tự động: hỗ trợ truy cập dữ liệu thông qua API.

Trong đồ án, Supabase được sử dụng để quản lý người dùng, vai trò người dùng, lịch sử chẩn đoán, dữ liệu ổ dịch, phản hồi AI và lưu trữ ảnh cây trồng. Người dùng đăng nhập có thể xem lại lịch sử chẩn đoán của mình, trong khi chuyên gia có thể tham gia xử lý các trường hợp cần thẩm định hoặc tư vấn.

## 1.10. Tổng quan về PostgreSQL

PostgreSQL là hệ quản trị cơ sở dữ liệu quan hệ mã nguồn mở, được sử dụng rộng rãi trong các hệ thống phần mềm nhờ tính ổn định, bảo mật và khả năng mở rộng. PostgreSQL hỗ trợ các kiểu dữ liệu phong phú, truy vấn SQL mạnh mẽ, khóa ngoại, ràng buộc dữ liệu, transaction và nhiều tính năng nâng cao.

Một số đặc điểm của PostgreSQL gồm:

- Lưu trữ dữ liệu theo mô hình bảng, hàng và cột.
- Hỗ trợ truy vấn SQL tiêu chuẩn.
- Hỗ trợ ràng buộc dữ liệu và quan hệ giữa các bảng.
- Có tính ổn định và độ tin cậy cao.
- Hỗ trợ bảo mật và phân quyền truy cập.
- Có thể mở rộng cho nhiều loại ứng dụng khác nhau.

Trong hệ thống Plant Disease Detector, PostgreSQL được sử dụng thông qua Supabase để lưu các bảng như hồ sơ người dùng, lịch sử chẩn đoán, báo cáo, ca phản hồi AI, dữ liệu ổ dịch, tài nguyên kiến thức và thông tin liên quan đến quy trình tư vấn.

## 1.11. Tổng quan về Hugging Face

Hugging Face là một nền tảng phổ biến trong lĩnh vực trí tuệ nhân tạo, cung cấp kho mô hình, tập dữ liệu và công cụ triển khai mô hình. Nền tảng này hỗ trợ nhiều bài toán như xử lý ngôn ngữ tự nhiên, thị giác máy tính, nhận diện giọng nói và phân loại ảnh.

Hugging Face Spaces là một dịch vụ cho phép triển khai các ứng dụng AI hoặc mô hình học máy dưới dạng web app/API. Nhờ đó, lập trình viên có thể đưa mô hình đã huấn luyện lên nền tảng này và gọi mô hình từ các ứng dụng khác thông qua endpoint.

Các lợi ích của Hugging Face gồm:

- Cung cấp kho mô hình AI phong phú.
- Hỗ trợ triển khai mô hình nhanh chóng.
- Có thể tích hợp với backend thông qua API.
- Phù hợp cho các bài toán thử nghiệm, demo và triển khai mô hình AI.

Trong đồ án, backend FastAPI gọi đến Hugging Face Space hoặc endpoint dự đoán để chạy mô hình nhận diện bệnh cây. Kết quả từ mô hình được backend xử lý và trả về ứng dụng iOS dưới dạng dữ liệu JSON.

## 1.12. Tổng quan về Gemini API

Gemini API là dịch vụ trí tuệ nhân tạo tạo sinh do Google phát triển, có khả năng xử lý ngôn ngữ tự nhiên, sinh nội dung, trả lời câu hỏi và hỗ trợ hội thoại. Với khả năng hiểu ngữ cảnh và tạo phản hồi bằng ngôn ngữ tự nhiên, Gemini API có thể được tích hợp vào ứng dụng để xây dựng trợ lý thông minh hoặc chức năng tư vấn tự động.

Các chức năng thường gặp của Gemini API gồm:

- Sinh nội dung văn bản dựa trên yêu cầu đầu vào.
- Trả lời câu hỏi bằng ngôn ngữ tự nhiên.
- Tóm tắt thông tin.
- Đưa ra gợi ý hoặc hướng dẫn theo ngữ cảnh.
- Hỗ trợ hội thoại nhiều lượt.

Trong đồ án, Gemini API được sử dụng để cung cấp tư vấn chăm sóc cây bằng tiếng Việt. Sau khi hệ thống nhận diện được loại cây và bệnh, backend có thể gửi thông tin này đến Gemini API để tạo gợi ý xử lý, biện pháp chăm sóc và khuyến nghị khi cần liên hệ chuyên gia.

## 1.13. Tổng quan về OpenWeather API

OpenWeather API là dịch vụ cung cấp dữ liệu thời tiết thông qua API. Dữ liệu thời tiết có thể bao gồm nhiệt độ, độ ẩm, mô tả thời tiết, lượng mưa, gió và dự báo trong nhiều ngày. Dịch vụ này thường được tích hợp vào các ứng dụng cần hiển thị thông tin thời tiết theo vị trí địa lý.

Trong nông nghiệp, thời tiết là yếu tố quan trọng ảnh hưởng đến sinh trưởng của cây trồng và khả năng phát sinh bệnh. Nhiệt độ cao, độ ẩm lớn, mưa nhiều hoặc thiếu ánh sáng có thể tạo điều kiện cho một số loại bệnh phát triển. Vì vậy, việc kết hợp dữ liệu thời tiết với hệ thống chăm sóc cây giúp người dùng có thêm thông tin tham khảo khi theo dõi tình trạng cây.

Trong đồ án, OpenWeather API được tích hợp để hiển thị thời tiết hiện tại và dự báo trong những ngày tiếp theo. Ứng dụng có thể sử dụng dữ liệu này để cung cấp cảnh báo hoặc gợi ý chăm sóc cây phù hợp hơn với điều kiện môi trường.

## 1.14. Tổng quan về Render

Render là nền tảng triển khai ứng dụng trên cloud, hỗ trợ triển khai web service, API server, static site, cron job và cơ sở dữ liệu. Render cho phép lập trình viên đưa ứng dụng backend lên môi trường trực tuyến để client có thể truy cập thông qua Internet.

Một số ưu điểm của Render gồm:

- Hỗ trợ triển khai ứng dụng nhanh chóng.
- Có thể kết nối với GitHub để tự động deploy khi mã nguồn thay đổi.
- Hỗ trợ cấu hình biến môi trường.
- Phù hợp để triển khai API backend cho ứng dụng web và di động.
- Có thể sử dụng health check để theo dõi trạng thái hoạt động của service.

Trong đồ án, FastAPI backend có thể được triển khai trên Render. Ứng dụng iOS sẽ gọi đến URL backend đã triển khai để thực hiện các chức năng như nhận diện bệnh cây, lưu lịch sử, lấy dữ liệu thời tiết, lấy dữ liệu ổ dịch và tư vấn AI.

## 1.15. Tổng quan về bảo mật API

Bảo mật API là yếu tố quan trọng khi xây dựng hệ thống có giao tiếp giữa client và server. Nếu API không được bảo vệ đúng cách, hệ thống có thể gặp các rủi ro như gửi dữ liệu sai định dạng, upload file không hợp lệ, lạm dụng request, truy cập trái phép hoặc rò rỉ dữ liệu người dùng.

Một số biện pháp bảo mật API thường dùng gồm:

- Xác thực người dùng trước khi cho phép truy cập dữ liệu cá nhân.
- Kiểm tra định dạng dữ liệu đầu vào.
- Giới hạn kích thước request và file upload.
- Giới hạn số lượng request trong một khoảng thời gian.
- Kiểm tra Content-Type của request.
- Không đưa khóa bí mật vào ứng dụng client.
- Sử dụng cơ chế phân quyền để giới hạn thao tác của từng vai trò người dùng.

Trong đồ án, backend có các cơ chế như kiểm tra định dạng ảnh, giới hạn dung lượng ảnh, giới hạn request theo IP, kiểm tra content-type, xác thực người dùng Supabase và sử dụng service role key ở phía server. Các biện pháp này giúp hệ thống ổn định hơn và hạn chế việc sử dụng sai API.

## 1.16. Tổng quan kiến trúc hệ thống Plant Disease Detector

Plant Disease Detector là hệ thống hỗ trợ nhận diện bệnh cây từ hình ảnh, gồm ứng dụng iOS, backend FastAPI, mô hình AI và các dịch vụ hỗ trợ. Ứng dụng iOS là nơi người dùng thao tác trực tiếp, backend là nơi xử lý nghiệp vụ và kết nối với các dịch vụ như Supabase, Hugging Face, OpenWeather và Gemini.

Luồng xử lý chính của hệ thống như sau:

- Người dùng mở ứng dụng iOS và chọn chức năng quét bệnh cây.
- Người dùng chụp ảnh hoặc chọn ảnh từ thư viện.
- Ứng dụng gửi ảnh và loại cây đã chọn đến backend.
- Backend kiểm tra ảnh, xử lý ảnh và gọi mô hình AI.
- Mô hình trả về kết quả dự đoán bệnh cây.
- Backend lưu ảnh lên Supabase Storage và lưu lịch sử nếu người dùng đã đăng nhập.
- Ứng dụng hiển thị kết quả gồm loại cây, loại bệnh, độ tin cậy và gợi ý xử lý.

```text
Người dùng
    |
    v
Ứng dụng iOS SwiftUI
    |
    v
Backend FastAPI
    |
    +--> Hugging Face / Mô hình AI
    |
    +--> Supabase Auth, Database, Storage
    |
    +--> OpenWeather API
    |
    +--> Gemini API
```

Hình 1.4. Kiến trúc tổng quan hệ thống Plant Disease Detector

Kiến trúc này giúp hệ thống tách biệt rõ ràng giữa giao diện, xử lý nghiệp vụ, lưu trữ dữ liệu và các dịch vụ AI. Nhờ đó, hệ thống dễ bảo trì, dễ mở rộng và có thể bổ sung thêm chức năng trong tương lai như cải thiện mô hình nhận diện, thêm loại cây mới, mở rộng bản đồ ổ dịch hoặc nâng cấp chức năng tư vấn chuyên gia.

---

# PROMPT GỢI Ý ĐỂ NHỜ CHATGPT TẠO FILE WORD

```text
Hãy chuyển nội dung Markdown sau thành Chương 1 của file Word báo cáo đồ án.

Yêu cầu định dạng:
- Khổ giấy A4.
- Font Times New Roman.
- Cỡ chữ nội dung 13.
- Giãn dòng 1.5.
- Căn đều hai bên.
- Lề trái 3 cm, lề phải 2 cm, lề trên 2 cm, lề dưới 2 cm.
- Tiêu đề "CHƯƠNG 1. CƠ SỞ LÝ THUYẾT" in hoa, in đậm, căn giữa.
- Các mục 1.1, 1.2, 1.3... in đậm.
- Giữ nguyên các hình minh họa dạng sơ đồ text và chú thích Hình 1.x.
- Có thể chỉnh nhẹ câu chữ cho mạch lạc, nhưng không thay đổi nội dung chính.
```

