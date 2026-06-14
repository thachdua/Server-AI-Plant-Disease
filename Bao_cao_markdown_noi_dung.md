# NỘI DUNG MARKDOWN DÙNG ĐỂ TẠO FILE WORD BÁO CÁO

> Bạn có thể gửi toàn bộ nội dung Markdown này cho ChatGPT và yêu cầu:  
> “Hãy chuyển nội dung sau thành file Word báo cáo đồ án, định dạng trang A4, font Times New Roman cỡ 13, giãn dòng 1.5, căn đều hai bên, tiêu đề chương in hoa và căn giữa.”

---

# LỜI CẢM ƠN

Trước hết, em xin gửi lời cảm ơn chân thành đến quý thầy cô đã tận tình giảng dạy, hướng dẫn và truyền đạt cho em những kiến thức quý báu trong suốt quá trình học tập. Những kiến thức về lập trình, cơ sở dữ liệu, trí tuệ nhân tạo, phát triển ứng dụng và triển khai hệ thống là nền tảng quan trọng giúp em có thể thực hiện đồ án này.

Em xin đặc biệt cảm ơn thầy/cô hướng dẫn đã luôn quan tâm, góp ý và định hướng cho em trong quá trình nghiên cứu, xây dựng và hoàn thiện đề tài. Những nhận xét và chỉ dẫn của thầy/cô đã giúp em hiểu rõ hơn về yêu cầu của một hệ thống thực tế, từ đó cải thiện chất lượng sản phẩm và báo cáo.

Em cũng xin cảm ơn gia đình, bạn bè đã luôn động viên, hỗ trợ và tạo điều kiện thuận lợi để em hoàn thành đồ án. Trong quá trình thực hiện, mặc dù em đã cố gắng tìm hiểu và hoàn thiện hệ thống, nhưng do thời gian và kinh nghiệm còn hạn chế nên đồ án khó tránh khỏi những thiếu sót. Em rất mong nhận được sự góp ý của quý thầy cô để đề tài được hoàn thiện hơn.

Em xin chân thành cảm ơn!

---

# LỜI MỞ ĐẦU

Trong những năm gần đây, công nghệ thông tin và trí tuệ nhân tạo ngày càng được ứng dụng rộng rãi trong nhiều lĩnh vực của đời sống, đặc biệt là trong nông nghiệp. Việc phát hiện sớm bệnh trên cây trồng có vai trò quan trọng trong quá trình chăm sóc và bảo vệ cây, giúp người trồng kịp thời đưa ra biện pháp xử lý, hạn chế thiệt hại về năng suất và chất lượng nông sản. Tuy nhiên, trong thực tế, việc nhận biết bệnh cây bằng mắt thường còn phụ thuộc nhiều vào kinh nghiệm, dễ nhầm lẫn giữa các loại bệnh có biểu hiện tương tự và mất nhiều thời gian khi cần kiểm tra trên diện rộng.

Xuất phát từ nhu cầu đó, đồ án **“Plant Disease Detector”** được xây dựng với mục tiêu phát triển một ứng dụng hỗ trợ nhận diện bệnh cây thông qua hình ảnh. Ứng dụng cho phép người dùng chụp hoặc chọn ảnh lá cây, gửi lên hệ thống để mô hình trí tuệ nhân tạo phân tích và trả về kết quả dự đoán gồm loại cây, loại bệnh và độ tin cậy. Bên cạnh chức năng nhận diện bệnh, hệ thống còn hỗ trợ lưu lịch sử chẩn đoán, cung cấp tư vấn chăm sóc bằng tiếng Việt, hiển thị thông tin thời tiết nông nghiệp, bản đồ ổ dịch và các nội dung tham khảo liên quan đến bệnh cây.

Về mặt kỹ thuật, đồ án được xây dựng gồm ứng dụng iOS sử dụng SwiftUI, backend FastAPI, mô hình nhận diện bệnh cây, cơ sở dữ liệu và lưu trữ ảnh bằng Supabase. Ngoài ra, hệ thống còn tích hợp các dịch vụ như OpenWeather API để lấy dữ liệu thời tiết và Gemini API để hỗ trợ tư vấn thông minh cho người dùng. Sự kết hợp giữa các công nghệ này giúp ứng dụng không chỉ thực hiện việc nhận diện bệnh cây mà còn hướng đến một công cụ hỗ trợ chăm sóc cây trồng toàn diện hơn.

Trong quá trình thực hiện đồ án, em đã có cơ hội tìm hiểu và vận dụng các kiến thức về lập trình ứng dụng di động, xây dựng API, xử lý ảnh, tích hợp mô hình AI, quản lý cơ sở dữ liệu, xác thực người dùng và triển khai hệ thống thực tế. Đây là cơ hội giúp em củng cố kiến thức đã học, đồng thời rèn luyện khả năng phân tích, thiết kế và phát triển một sản phẩm phần mềm hoàn chỉnh.

---

# TÊN ĐỀ TÀI

**Xây dựng ứng dụng nhận diện bệnh cây bằng trí tuệ nhân tạo trên nền tảng iOS**

Tên tiếng Anh của hệ thống: **Plant Disease Detector**

---

# LÝ DO CHỌN ĐỀ TÀI

Nông nghiệp là lĩnh vực có vai trò quan trọng trong đời sống và kinh tế. Trong quá trình canh tác, cây trồng thường gặp nhiều loại bệnh khác nhau do nấm, vi khuẩn, virus, sâu hại hoặc điều kiện môi trường không phù hợp. Nếu không phát hiện và xử lý kịp thời, bệnh cây có thể lây lan nhanh, làm giảm năng suất, ảnh hưởng đến chất lượng sản phẩm và gây thiệt hại kinh tế cho người trồng.

Hiện nay, việc nhận diện bệnh cây thường dựa vào kinh nghiệm cá nhân hoặc cần sự hỗ trợ từ chuyên gia nông nghiệp. Cách làm này có thể mất nhiều thời gian, khó thực hiện với người chưa có kinh nghiệm và không phải lúc nào cũng thuận tiện. Trong khi đó, điện thoại thông minh ngày càng phổ biến, giúp người dùng dễ dàng chụp ảnh cây trồng và gửi dữ liệu đến hệ thống phân tích.

Vì vậy, việc xây dựng một ứng dụng có khả năng nhận diện bệnh cây từ hình ảnh là cần thiết và có ý nghĩa thực tiễn. Ứng dụng giúp người dùng nhanh chóng có được thông tin tham khảo về tình trạng cây, từ đó đưa ra hướng xử lý phù hợp. Đồng thời, việc kết hợp thêm các chức năng như lịch sử chẩn đoán, tư vấn AI, thời tiết nông nghiệp và bản đồ ổ dịch giúp hệ thống trở nên hữu ích hơn trong quá trình chăm sóc cây trồng.

---

# MỤC TIÊU CỦA ĐỀ TÀI

## Mục tiêu tổng quát

Xây dựng một hệ thống hỗ trợ nhận diện bệnh cây thông qua hình ảnh, hoạt động trên nền tảng iOS, có khả năng phân tích ảnh cây trồng, trả về kết quả chẩn đoán và cung cấp các thông tin hỗ trợ chăm sóc cây.

## Mục tiêu cụ thể

- Xây dựng ứng dụng iOS bằng SwiftUI với giao diện thân thiện, dễ sử dụng.
- Cho phép người dùng chụp ảnh hoặc chọn ảnh lá/cây để gửi lên hệ thống.
- Tích hợp backend FastAPI để tiếp nhận ảnh, xử lý yêu cầu và trả kết quả dự đoán.
- Kết nối mô hình AI nhận diện bệnh cây để phân tích ảnh.
- Trả về kết quả gồm loại cây, loại bệnh, độ tin cậy và hình ảnh đã xử lý/lưu trữ.
- Lưu lịch sử chẩn đoán cho người dùng đã đăng nhập.
- Tích hợp Supabase để quản lý xác thực, cơ sở dữ liệu và lưu trữ ảnh.
- Cung cấp chức năng tư vấn chăm sóc bằng tiếng Việt thông qua Gemini API.
- Hiển thị thông tin thời tiết phục vụ chăm sóc cây trồng.
- Xây dựng bản đồ ổ dịch để người dùng có thể theo dõi các khu vực có ghi nhận bệnh cây.
- Phân quyền người dùng thường và chuyên gia để hỗ trợ quy trình tư vấn, phản hồi và quản lý.

---

# PHẠM VI THỰC HIỆN

Đồ án tập trung xây dựng hệ thống nhận diện bệnh cây từ hình ảnh và các chức năng hỗ trợ trên ứng dụng iOS. Phạm vi thực hiện bao gồm:

- Ứng dụng di động iOS sử dụng SwiftUI.
- Backend API sử dụng FastAPI.
- Chức năng nhận diện bệnh cây thông qua ảnh.
- Chức năng đăng nhập, đăng ký, đăng nhập Google và chế độ khách.
- Chức năng lưu và xem lịch sử chẩn đoán.
- Chức năng tư vấn AI bằng tiếng Việt.
- Chức năng xem thời tiết nông nghiệp.
- Chức năng bản đồ ổ dịch.
- Chức năng phản hồi, báo cáo và hỗ trợ người dùng.
- Cơ sở dữ liệu, phân quyền và lưu trữ ảnh bằng Supabase.

Đề tài không đặt mục tiêu thay thế hoàn toàn chuyên gia nông nghiệp. Kết quả nhận diện và tư vấn từ hệ thống mang tính chất hỗ trợ tham khảo, người dùng vẫn cần kết hợp với quan sát thực tế hoặc ý kiến chuyên gia trong các trường hợp nghiêm trọng.

---

# ĐỐI TƯỢNG SỬ DỤNG

Hệ thống hướng đến các nhóm người dùng chính sau:

- Người trồng cây, nông dân, người làm vườn hoặc người chăm sóc cây tại nhà.
- Người dùng cần công cụ hỗ trợ nhận diện nhanh bệnh cây từ hình ảnh.
- Chuyên gia hoặc người có chuyên môn nông nghiệp tham gia hỗ trợ tư vấn, phản hồi và quản lý ca bệnh.
- Sinh viên, giảng viên hoặc người nghiên cứu quan tâm đến ứng dụng AI trong nông nghiệp.

---

# CÔNG NGHỆ SỬ DỤNG

## Ứng dụng di động

- Ngôn ngữ lập trình: Swift.
- Giao diện: SwiftUI.
- Nền tảng: iOS.
- Chức năng chính: quét ảnh, hiển thị kết quả, quản lý lịch sử, đăng nhập, tư vấn, thời tiết, bản đồ ổ dịch.

## Backend

- Ngôn ngữ lập trình: Python.
- Framework: FastAPI.
- Chức năng chính: nhận ảnh từ ứng dụng, xác thực request, gọi mô hình nhận diện, lưu ảnh, lưu dữ liệu lịch sử, cung cấp API cho các chức năng phụ.
- Triển khai: Render hoặc môi trường server tương thích.

## Trí tuệ nhân tạo

- Mô hình nhận diện bệnh cây từ ảnh.
- Tích hợp Hugging Face Space hoặc endpoint dự đoán để xử lý ảnh.
- Tích hợp Gemini API để sinh tư vấn chăm sóc cây bằng tiếng Việt.

## Cơ sở dữ liệu và lưu trữ

- Supabase Auth: quản lý đăng nhập, đăng ký và xác thực người dùng.
- Supabase Postgres: lưu lịch sử chẩn đoán, hồ sơ người dùng, vai trò, báo cáo, ca phản hồi AI và dữ liệu ổ dịch.
- Supabase Storage: lưu ảnh cây trồng, ảnh chẩn đoán và avatar người dùng.

## Dịch vụ bên ngoài

- OpenWeather API: cung cấp dữ liệu thời tiết.
- Gemini API: hỗ trợ tư vấn AI.
- Hugging Face: chạy mô hình nhận diện bệnh cây.

---

# CẤU TRÚC HỆ THỐNG

Hệ thống được thiết kế theo mô hình gồm ba phần chính:

1. **Ứng dụng iOS**  
   Người dùng thao tác trực tiếp trên ứng dụng, chụp hoặc chọn ảnh cây, đăng nhập, xem kết quả, xem lịch sử, thời tiết, bản đồ ổ dịch và nhận tư vấn.

2. **Backend FastAPI**  
   Backend tiếp nhận yêu cầu từ ứng dụng iOS, kiểm tra dữ liệu đầu vào, xử lý ảnh, gọi mô hình AI, kết nối Supabase và trả kết quả về cho ứng dụng.

3. **Dịch vụ dữ liệu và AI**  
   Bao gồm Supabase, Hugging Face, OpenWeather API và Gemini API. Các dịch vụ này hỗ trợ xác thực, lưu trữ, dự đoán bệnh, lấy thời tiết và sinh nội dung tư vấn.

Sơ đồ tổng quát:

```text
Ứng dụng iOS SwiftUI
        |
        v
Backend FastAPI
        |
        +--> Mô hình AI nhận diện bệnh cây
        |
        +--> Supabase Auth, Database, Storage
        |
        +--> OpenWeather API
        |
        +--> Gemini API
```

---

# CHỨC NĂNG CHÍNH CỦA HỆ THỐNG

## Chức năng quét và nhận diện bệnh cây

Người dùng có thể chụp ảnh trực tiếp hoặc chọn ảnh từ thiết bị. Sau đó, ứng dụng gửi ảnh cùng loại cây đã chọn đến backend. Backend kiểm tra định dạng và kích thước ảnh, xử lý ảnh an toàn, gọi mô hình nhận diện bệnh cây và trả về kết quả dự đoán gồm tên cây, tên bệnh, độ tin cậy và URL ảnh đã lưu.

## Chức năng quản lý tài khoản

Hệ thống hỗ trợ đăng ký, đăng nhập bằng email/password, đăng nhập Google OAuth và chế độ khách. Người dùng đã đăng nhập có thể lưu lịch sử chẩn đoán và sử dụng các chức năng cá nhân hóa. Hệ thống cũng có cơ chế phân quyền giữa người dùng thường và chuyên gia.

## Chức năng lưu lịch sử chẩn đoán

Sau mỗi lần nhận diện bệnh, hệ thống có thể lưu lại kết quả chẩn đoán gồm ảnh, loại cây, loại bệnh, độ tin cậy, thời gian và thông tin người dùng. Chức năng này giúp người dùng theo dõi quá trình chăm sóc cây và xem lại các lần chẩn đoán trước đó.

## Chức năng tư vấn AI

Hệ thống tích hợp Gemini API để cung cấp gợi ý chăm sóc cây bằng tiếng Việt. Dựa trên kết quả chẩn đoán, loại cây, bệnh và một số thông tin liên quan, AI có thể đưa ra hướng xử lý, lưu ý chăm sóc và biện pháp phòng ngừa phù hợp.

## Chức năng thời tiết nông nghiệp

Ứng dụng cung cấp thông tin thời tiết nhằm hỗ trợ người dùng trong quá trình chăm sóc cây. Các yếu tố như nhiệt độ, độ ẩm và điều kiện thời tiết có thể ảnh hưởng đến sự phát triển của cây cũng như khả năng phát sinh bệnh.

## Chức năng bản đồ ổ dịch

Hệ thống có chức năng hiển thị bản đồ ổ dịch, giúp theo dõi các khu vực có ghi nhận ca bệnh cây. Chức năng này hỗ trợ người dùng quan sát tình hình bệnh cây theo vị trí địa lý và nâng cao khả năng cảnh báo sớm.

## Chức năng phản hồi và hỗ trợ chuyên gia

Khi kết quả dự đoán có độ tin cậy thấp hoặc người dùng cần hỗ trợ thêm, hệ thống có thể ghi nhận phản hồi để chuyên gia xem xét. Chuyên gia có thể quản lý, phản hồi và hỗ trợ xử lý các trường hợp cần tư vấn chuyên sâu hơn.

---

# Ý NGHĨA CỦA ĐỀ TÀI

Đề tài có ý nghĩa thực tiễn trong việc ứng dụng trí tuệ nhân tạo vào nông nghiệp, giúp người dùng có thêm công cụ hỗ trợ phát hiện bệnh cây nhanh chóng và thuận tiện. Thay vì chỉ dựa vào kinh nghiệm cá nhân, người dùng có thể sử dụng điện thoại để chụp ảnh và nhận kết quả tham khảo từ hệ thống.

Bên cạnh đó, đồ án còn giúp em củng cố và vận dụng nhiều kiến thức đã học, bao gồm phát triển ứng dụng di động, xây dựng backend API, xử lý ảnh, tích hợp mô hình AI, thiết kế cơ sở dữ liệu, xác thực người dùng, bảo mật API và triển khai hệ thống. Đây là nền tảng quan trọng để em tiếp tục phát triển các sản phẩm phần mềm có tính ứng dụng thực tế trong tương lai.

---

# BỐ CỤC BÁO CÁO

Báo cáo đồ án được trình bày gồm các chương chính như sau:

## Chương 1: Tổng quan đề tài

Chương này trình bày lý do chọn đề tài, mục tiêu thực hiện, phạm vi nghiên cứu, đối tượng sử dụng và ý nghĩa của hệ thống nhận diện bệnh cây bằng trí tuệ nhân tạo.

## Chương 2: Cơ sở lý thuyết và công nghệ sử dụng

Chương này trình bày các kiến thức nền tảng liên quan đến nhận diện ảnh, trí tuệ nhân tạo, mô hình học sâu, phát triển ứng dụng iOS bằng SwiftUI, backend FastAPI, Supabase, API thời tiết và Gemini API.

## Chương 3: Phân tích và thiết kế hệ thống

Chương này trình bày yêu cầu chức năng, yêu cầu phi chức năng, kiến trúc tổng thể, thiết kế cơ sở dữ liệu, luồng xử lý nhận diện bệnh cây, phân quyền người dùng và các thành phần chính của hệ thống.

## Chương 4: Xây dựng và triển khai hệ thống

Chương này trình bày quá trình xây dựng ứng dụng iOS, phát triển backend, tích hợp mô hình AI, kết nối Supabase, triển khai API, xử lý ảnh, lưu lịch sử, hiển thị thời tiết, bản đồ ổ dịch và tư vấn AI.

## Chương 5: Kiểm thử và đánh giá

Chương này trình bày quá trình kiểm thử các chức năng chính của hệ thống như đăng nhập, quét ảnh, nhận diện bệnh, lưu lịch sử, tư vấn AI, thời tiết, bản đồ ổ dịch và kiểm tra API backend.

## Chương 6: Kết luận và hướng phát triển

Chương này tổng kết kết quả đạt được, nêu những hạn chế còn tồn tại và đề xuất các hướng phát triển tiếp theo cho hệ thống.

---

# KẾT LUẬN MỞ ĐẦU

Tóm lại, đồ án **Plant Disease Detector** hướng đến việc xây dựng một ứng dụng hỗ trợ nhận diện bệnh cây bằng trí tuệ nhân tạo, kết hợp giữa ứng dụng di động, backend API, cơ sở dữ liệu và các dịch vụ AI. Hệ thống không chỉ giúp người dùng nhận diện bệnh cây từ hình ảnh mà còn cung cấp các chức năng hỗ trợ như lưu lịch sử, tư vấn chăm sóc, xem thời tiết và theo dõi ổ dịch. Đây là một đề tài có tính thực tiễn, phù hợp với xu hướng ứng dụng công nghệ hiện đại vào nông nghiệp.

---

# PROMPT GỢI Ý ĐỂ NHỜ CHATGPT TẠO FILE WORD

Bạn có thể copy đoạn dưới đây gửi cho ChatGPT:

```text
Hãy chuyển nội dung Markdown sau thành một file Word báo cáo đồ án.

Yêu cầu định dạng:
- Khổ giấy A4.
- Font chữ Times New Roman.
- Cỡ chữ nội dung 13.
- Giãn dòng 1.5.
- Căn đều hai bên.
- Lề trái 3 cm, lề phải 2 cm, lề trên 2 cm, lề dưới 2 cm.
- Các tiêu đề lớn in hoa, căn giữa, in đậm.
- Các mục nhỏ đánh số hoặc định dạng rõ ràng.
- Giữ văn phong trang trọng, phù hợp báo cáo đồ án.
- Có thể chỉnh sửa câu chữ cho mạch lạc hơn nhưng không thay đổi nội dung chính.
```

