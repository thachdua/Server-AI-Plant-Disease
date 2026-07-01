# Đáp án gợi ý cho 200 câu hỏi bảo vệ đồ án Plant Disease Detector

Gợi ý sử dụng: mỗi đáp án nên được xem như “ý chính để trả lời miệng”, không cần học thuộc từng chữ. Khi bảo vệ, hãy trả lời ngắn, đúng trọng tâm, sau đó bổ sung ví dụ trong hệ thống của mình.

---

## A. Tổng quan đề tài và nghiệp vụ

1. Đề tài xây dựng ứng dụng iOS nhận diện bệnh cây từ ảnh, kết hợp backend FastAPI, mô hình AI, Supabase và các dịch vụ hỗ trợ như thời tiết, bản đồ vùng dịch, chatbot tư vấn.
2. Vì bệnh cây là bài toán thực tế, có nhu cầu cao trong nông nghiệp; ảnh lá cây có thể xử lý bằng AI và phù hợp với ứng dụng di động.
3. Người trồng cây, người làm vườn, nông dân, người chăm sóc cây tại nhà và chuyên gia nông nghiệp.
4. Hệ thống có khách, người dùng đã đăng nhập và chuyên gia.
5. Khách dùng chức năng giới hạn; người dùng đăng nhập lưu lịch sử/cây/chat/bookmark; chuyên gia thẩm định phản hồi, tư vấn và xử lý workflow.
6. Điểm nổi bật là kết hợp nhận diện bệnh cây, lưu lịch sử, bản đồ vùng dịch, thời tiết, tư vấn AI và feedback để cải thiện mô hình.
7. Không. AI chỉ hỗ trợ tham khảo; trường hợp nghiêm trọng vẫn cần chuyên gia vì ảnh và model có thể sai.
8. Là thông tin hỗ trợ tham khảo kèm độ tin cậy, không phải kết luận tuyệt đối.
9. Nếu ảnh không hợp lệ thì backend trả lỗi; nếu ảnh hợp lệ nhưng model không chắc thì trả unrecognized hoặc ghi nhận low-confidence.
10. Chọn loại cây giúp model/backend có ngữ cảnh, giảm nhầm lẫn giữa bệnh của các cây khác nhau.
11. Model có thể dự đoán sai hoặc confidence thấp vì dữ liệu đầu vào không khớp loại cây.
12. Đăng nhập, lịch sử, cây của tôi, care tasks, Discover, bookmark, weather, outbreak map, chatbot, feedback, tư vấn chuyên gia.
13. Giúp người dùng xem lại các lần chẩn đoán, theo dõi tình trạng cây theo thời gian và tạo dữ liệu vùng dịch.
14. Lưu bộ sưu tập cây cá nhân, thông tin chăm sóc, bệnh hiện tại và các việc cần làm.
15. Cung cấp tài nguyên kiến thức và cho phép người dùng lưu lại tài liệu hữu ích.
16. Giúp theo dõi các khu vực có ghi nhận bệnh, hỗ trợ cảnh báo sớm và quan sát xu hướng lây lan.
17. Thời tiết như độ ẩm, mưa, nhiệt độ ảnh hưởng đến nấm bệnh và stress cây, nên hỗ trợ gợi ý chăm sóc.
18. Vì AI có thể sai; feedback giúp chuyên gia thẩm định và tạo dữ liệu cải thiện model.
19. Vì một số thao tác như duyệt phản hồi, cập nhật workflow cần người có chuyên môn và quyền cao hơn.
20. Nên ưu tiên cải thiện độ chính xác model, UX chụp ảnh, feedback chuyên gia và dữ liệu cây/bệnh phù hợp địa phương.

## B. Kiến trúc hệ thống

21. Kiến trúc gồm iOS SwiftUI app, FastAPI backend trên Render, Supabase Auth/Postgres/Storage, Hugging Face model, Gemini API và OpenWeather API.
22. Tách lớp giúp dễ bảo trì, bảo mật secret, mở rộng và thay đổi model/backend mà không phụ thuộc hoàn toàn vào app.
23. Vì iOS không nên giữ API key/service key, model nặng, khó kiểm soát upload và khó cập nhật model.
24. Backend nhận request, validate, xử lý ảnh, gọi model/API ngoài, ghi DB và trả JSON cho app.
25. Supabase quản lý đăng nhập, database, storage ảnh và RLS phân quyền dữ liệu.
26. Hugging Face được gọi trong endpoint `/predict` để chạy mô hình nhận diện bệnh cây.
27. Gemini dùng cho chatbot, tư vấn chẩn đoán, tư vấn thời tiết, care plan và care metrics.
28. OpenWeather cung cấp dữ liệu thời tiết theo vị trí.
29. Render deploy backend FastAPI để iOS gọi qua Internet.
30. iOS chụp ảnh → gửi `/predict` → backend validate ảnh → gọi Hugging Face → upload Storage → trả plant/disease/confidence/image_url.
31. iOS gửi `/history/save` kèm token → backend xác thực → ghi `history` → nếu đủ điều kiện ghi `outbreak_cases`.
32. iOS gửi tin nhắn tới `/llm/chat` → backend kiểm rate limit/prompt → gọi Gemini → trả reply → nếu user đồng ý thì lưu chat vào Supabase.
33. iOS gọi `/outbreaks` hoặc `/outbreaks/areas` → backend query `outbreak_cases`, lấy boundary, tính level → trả dữ liệu để vẽ map.
34. Backend dùng cho nghiệp vụ cần secret/xử lý phức tạp; Supabase REST dùng cho bảng đã có RLS và thao tác CRUD trực tiếp.
35. Predict, LLM, weather cần API key/xử lý server; cây/bookmark/chat có thể dùng Supabase REST vì RLS bảo vệ.
36. Predict, history save, weather, outbreak, LLM và consultation qua backend bị ảnh hưởng.
37. Đăng nhập, lưu ảnh, lưu lịch sử, bảng dữ liệu, chat, bookmark, cây của tôi đều bị ảnh hưởng.
38. Backend trả lỗi timeout 504 hoặc 502 tùy lỗi, app hiển thị thông báo thân thiện và có thể retry.
39. Có cache, stale cache hoặc fallback advice trong một số endpoint LLM.
40. Có, vì các module tách router/service, database tách bảng, model endpoint có thể thay bằng URL khác.

## C. AI và mô hình nhận diện bệnh cây

41. Đây là bài toán phân loại ảnh, cụ thể là phân loại bệnh cây từ hình ảnh.
42. Đầu vào là ảnh cây/lá và loại cây chọn; đầu ra là plant, disease và confidence.
43. CNN phù hợp vì có khả năng trích xuất đặc trưng hình ảnh như màu sắc, đốm bệnh, viền lá, hoa văn bệnh.
44. CNN là mạng nơ-ron tích chập dùng nhiều trong thị giác máy tính.
45. Lớp convolution trích xuất đặc trưng cục bộ từ ảnh bằng các kernel/filter.
46. Pooling giảm kích thước feature map, giữ thông tin quan trọng và giảm tính toán.
47. Fully connected tổng hợp đặc trưng để đưa ra xác suất phân loại.
48. Confidence là mức độ mô hình tin vào dự đoán, thường hiển thị dạng phần trăm.
49. Không luôn đúng; model có thể tự tin sai nếu dữ liệu lệch, ảnh mơ hồ hoặc ngoài tập huấn luyện.
50. Model dễ nhầm class, confidence có thể thấp hoặc dự đoán sai bệnh tương tự.
51. Ảnh mờ, thiếu sáng, sai góc, nền rối làm model khó nhận ra triệu chứng.
52. Ảnh xấu, chọn sai cây, bệnh ngoài tập train, class imbalance, triệu chứng giống nhau, môi trường thực khác dữ liệu train.
53. Cần đủ ảnh mỗi class, đa dạng điều kiện, nhãn đúng, cân bằng dữ liệu, chia train/validation/test hợp lý.
54. Để model học đặc trưng bệnh thật thay vì chỉ học điều kiện chụp cố định.
55. Class imbalance là số ảnh giữa các lớp lệch nhau; model có xu hướng thiên về lớp nhiều dữ liệu.
56. Overfitting là model học quá kỹ dữ liệu train nhưng kém trên ảnh mới.
57. Dùng data augmentation, regularization, dropout, early stopping, thêm dữ liệu và validation tốt.
58. Thu thập ảnh cây mới, gán nhãn bệnh, cập nhật class, train/fine-tune model, deploy endpoint mới và cập nhật app nếu cần.
59. Thu thập ảnh bệnh mới, chuẩn hóa nhãn, train lại model với class mới, test và cập nhật localizer/dictionary.
60. Vì các ảnh model không chắc thường là dữ liệu khó, hữu ích để expert gán nhãn và retrain.

## D. Luồng predict và xử lý ảnh

61. Endpoint `/predict`.
62. Dạng `multipart/form-data`.
63. Để đảm bảo request đúng loại dữ liệu trước khi xử lý, tránh gửi JSON/text vào endpoint ảnh.
64. Để chặn nhanh file có metadata/tên không hợp lệ như `.js`, `.txt`.
65. Vì Content-Type có thể giả; Pillow xác minh nội dung file thật sự là ảnh.
66. Là ảnh được thiết kế để giải nén ra kích thước cực lớn, gây tốn RAM/CPU.
67. Để tránh quá tải server, giảm thời gian xử lý và hạn chế tấn công upload.
68. Để chuẩn hóa định dạng, loại metadata lạ và phù hợp với model/storage.
69. Để giảm rủi ro bảo mật và đảm bảo ảnh gửi đi sạch.
70. Middleware/content-type và `_validate_upload_metadata` chặn trước; nếu qua được thì Pillow vẫn kiểm tra nội dung.
71. Kiểm tra bytes ảnh, giới hạn dung lượng/kích thước, verify ảnh, convert RGB/JPEG sạch.
72. Kiểm tra content-type và filename extension có thuộc JPEG/PNG/WebP không.
73. Xử lý số 0-1, số 0-100 và chuỗi có dấu `%`.
74. Tách tên cây từ nhãn bệnh như `Tomato___Late_blight`.
75. Trả `status="unrecognized"`, message xin lỗi, confidence, disease dự đoán và `image_url=None`.
76. Upload ảnh, trả success, và nếu có user_id thì log vào `ai_feedback_cases`.
77. Để tránh tự động lưu dữ liệu quá không chắc chắn khi user chưa đồng ý.
78. Để app có URL hiển thị, lưu history, feedback, outbreak và phục vụ truy vết.
79. Trả HTTP 502 với lỗi prediction service returned incomplete result.
80. Trả HTTP 502 với lỗi Hugging Face returned invalid JSON.

## E. Backend FastAPI và API design

81. FastAPI là framework Python xây API nhanh, hỗ trợ Pydantic validation, docs tự động và async.
82. Tạo app, gắn middleware, exception handler và include các router.
83. Router chia endpoint theo nhóm chức năng, giúp code dễ quản lý.
84. Predict, history, consultations, outbreaks, llm, weather, health, auth_pages.
85. Để gom biến môi trường, key, URL, threshold, rate limit và cấu hình DB.
86. Để bảo mật secret, dễ đổi môi trường local/Render mà không sửa code.
87. `main.py` root là entrypoint import app từ `deploy.main`, tiện cho Render/local command.
88. Vì các thư viện như requests, psycopg2, Supabase client là blocking; threadpool tránh nghẽn event loop.
89. Kiểm tra backend còn sống, dùng cho uptime/health check.
90. `/health/ready` kiểm tra cấu hình bắt buộc như Supabase, DB, HF URL.
91. Vì health endpoint public; lộ secret sẽ nguy hiểm.
92. Định nghĩa schema request/response, validate kiểu dữ liệu và giới hạn field.
93. Từ chối field lạ, giảm rủi ro gửi payload ngoài dự kiến.
94. Giúp client hiểu lỗi do input, auth, payload, rate limit hay service ngoài.
95. 400 thường là lỗi logic/giá trị do code tự raise; 422 là lỗi validation schema Pydantic.
96. Thiếu bearer token hoặc token không hợp lệ.
97. Body JSON/request hoặc ảnh quá lớn.
98. Content-Type hoặc loại file upload không được hỗ trợ.
99. Gửi quá nhiều request vượt giới hạn.
100. Lỗi khi gọi dịch vụ phụ thuộc như Hugging Face/OpenWeather hoặc kết quả service không hợp lệ.

## F. Supabase database, RLS và Storage

101. Supabase Auth, Postgres database, Storage bucket và Row Level Security.
102. Lưu hồ sơ user, role user/expert, display name, avatar và metadata tài khoản.
103. Để mỗi profile gắn trực tiếp với user đăng nhập trong Supabase Auth.
104. User thao tác dữ liệu cá nhân; expert có quyền xem/cập nhật các ca thẩm định, phản hồi, tư vấn.
105. RLS là cơ chế bảo mật theo từng dòng trong database PostgreSQL.
106. Để user chỉ truy cập dữ liệu của mình và expert chỉ làm đúng quyền.
107. User có thể đọc/sửa dữ liệu người khác nếu gọi REST API trực tiếp.
108. Chỉ được select dòng có `created_by = auth.uid()` hoặc điều kiện tương tự.
109. Chỉ được insert dòng có `created_by = auth.uid()`.
110. Để tránh tự cấp quyền chuyên gia và truy cập dữ liệu quản trị.
111. Kiểm tra user hiện tại có role expert trong `profiles` không.
112. Anon key dùng client, bị RLS giới hạn; service role key quyền rất cao, có thể bypass RLS.
113. Vì nếu lộ trong app, người khác có thể thao tác toàn bộ dữ liệu.
114. Ảnh chẩn đoán, ảnh feedback/retrain, ảnh consultation và ảnh liên quan cây.
115. Để app hiển thị ảnh đã upload và liên kết ảnh với history/feedback/outbreak.
116. Cây, bệnh, confidence, image_url, thời gian và user tạo.
117. Tọa độ, cây, bệnh, confidence, ảnh, history_id, source, province, severity, thời gian.
118. Lưu ca AI confidence thấp hoặc user xác nhận sai để expert duyệt và làm dữ liệu retrain.
119. `report_cases` là workflow báo sai/chuyên gia xử lý; `ai_feedback_cases` thiên về dữ liệu cải thiện model.
120. Lưu yêu cầu tư vấn chuyên gia, câu hỏi, ảnh, trạng thái, phản hồi chuyên gia và metadata.

## G. History, Cây của tôi, Discover và Bookmark

121. Khi predict thành công và user đăng nhập gửi request lưu lịch sử.
122. Để gắn dữ liệu với chủ sở hữu và cho phép xem/xóa theo quyền.
123. Có lat/lng, confidence >= 60 và disease không phải healthy.
124. Vì cây khỏe không đại diện ca bệnh cần cảnh báo vùng dịch.
125. Vì dữ liệu không đủ tin cậy, đưa lên bản đồ có thể gây nhiễu.
126. Lưu cây cá nhân, thông tin chăm sóc, kích thước, nước, ánh sáng, bệnh hiện tại.
127. Gắn chẩn đoán mới nhất với cây cá nhân để theo dõi tình trạng hiện tại.
128. Lưu kế hoạch chăm sóc được tạo từ chẩn đoán hoặc AI.
129. Lưu các việc cần làm như tưới, kiểm tra, cắt lá bệnh, nhắc nhở.
130. Để tạo công việc lặp lại hằng ngày/tuần/tháng/năm.
131. Để biết task có đặt thông báo cục bộ hay không.
132. Lưu kiến thức nền về cây: tên, tóm tắt chăm sóc, nhiệt độ, ánh sáng, tưới.
133. Lưu bài viết, report, video, nội dung Discover.
134. Lưu tài nguyên mà user đánh dấu để xem lại.
135. Để một user không bookmark trùng cùng một tài nguyên nhiều lần.
136. Vì đây là tài liệu công khai, không chứa dữ liệu riêng tư.
137. `care_tasks` nên bị xóa cascade hoặc tách tùy thiết kế; trong SQL user_plant_id on delete cascade.
138. Discover là tài nguyên/bài viết; Disease Dictionary là tra cứu bệnh và triệu chứng.
139. Kiến thức tĩnh ổn định, dễ kiểm duyệt; tư vấn AI linh hoạt theo ngữ cảnh.
140. Cập nhật/insert vào `plant_resources`, có thể qua SQL seed như `015`, `018`, `019`.

## H. Weather, outbreak map và dữ liệu địa lý

141. Từ OpenWeather API.
142. Current weather, hourly, daily, alerts, lat/lng và status.
143. Độ ẩm, mưa, nhiệt độ cao, gió mạnh.
144. Độ ẩm cao tạo điều kiện cho nấm bệnh phát triển.
145. Mưa làm tăng ẩm, bắn nước lan mầm bệnh và khiến lá ướt lâu.
146. Để tránh gọi API ngoài với dữ liệu sai và trả lỗi rõ ràng.
147. Giảm độ trễ, giảm quota API và tránh gọi lặp cùng vị trí.
148. Bảng `outbreak_cases`.
149. `/outbreaks` trả danh sách điểm; `/outbreaks/areas` tổng hợp theo vùng/tỉnh để vẽ overlay.
150. Xác định ranh giới tỉnh/khu vực để kiểm tra điểm bệnh thuộc vùng nào.
151. Chủ yếu theo số ca trong khu vực, có xét max severity.
152. Confidence là độ tin cậy model, không phải mức nghiêm trọng thực tế của ổ dịch.
153. Backend lấy `created_by`, đọc `profiles` và format thành Người dùng/Chuyên gia + tên.
154. Để tăng độ tin cậy và minh bạch nguồn dữ liệu.
155. Không hiển thị trên bản đồ điểm được vì thiếu tọa độ.
156. Weather theo vị trí, lưu history có lat/lng và tự tạo outbreak bị ảnh hưởng.
157. Backend thường trả 502 từ OpenWeather hoặc 500 nếu thiếu key.
158. Trả 502 boundary fetch failed.
159. Bổ sung boundary toàn bộ tỉnh, tối ưu cache, query theo bbox và dữ liệu địa phương.
160. Dùng ngưỡng confidence, expert review, chỉ hiển thị ca đã xác nhận hoặc phân biệt auto/expert.

## I. Chatbot, Gemini, LLM và tư vấn

161. Chatbot, tư vấn chẩn đoán, tư vấn thời tiết, care plan, care metrics.
162. Nhận hội thoại và trả lời trợ lý AI.
163. Tạo lời khuyên xử lý dựa trên plant/disease/confidence.
164. Tạo lời khuyên chăm sóc dựa trên thời tiết, cây, bệnh và care context.
165. Tạo lịch chăm sóc/checklist sau chẩn đoán.
166. Vì người dùng mục tiêu dùng tiếng Việt, dễ hiểu hơn.
167. Để tránh tư vấn nguy hiểm; ưu tiên IPM và khuyên hỏi chuyên gia khi cần.
168. Vì app cần cấu trúc ổn định để decode và hiển thị.
169. Dùng validator/fallback, hoặc trả lỗi thân thiện.
170. Lưu kết quả tư vấn để tái sử dụng, giảm chi phí và tăng tốc.
171. Để định danh input giống nhau một cách ngắn gọn và ổn định.
172. Vì thời tiết thay đổi theo thời gian, cache cũ có thể không còn đúng.
173. Khi Gemini lỗi/tạm unavailable, đặc biệt lỗi 503 hoặc không có response hợp lệ.
174. Là việc chỉ lưu chat lên cloud khi user đồng ý.
175. Vì chat có thể chứa thông tin riêng tư; cần tôn trọng quyền riêng tư.
176. `chat_sessions` là cuộc trò chuyện; `chat_messages` là từng tin nhắn thuộc session.
177. Để phân biệt tin nhắn của user và assistant khi hiển thị/gửi ngữ cảnh.
178. Trả 413 prompt too long trước khi gọi Gemini.
179. Vì LLM tốn chi phí/quota và dễ bị spam.
180. Dùng review người dùng, app_feedback, expert đánh giá, test case thực tế và so sánh với nguồn chuyên môn.

## J. iOS SwiftUI app

181. Điểm khởi động app, tạo store dùng chung và mở `RootView`.
182. Dựa trên `isLoading`, `session`, `role`, `isGuest`.
183. Session, trạng thái loading, guest mode, role, recovery session và auth notice.
184. Keychain an toàn hơn UserDefaults cho token đăng nhập.
185. Dùng ASWebAuthenticationSession với PKCE, nhận callback URL, đổi code lấy session.
186. Từ `Info.plist`.
187. `APIService` gọi backend FastAPI; `SupabaseDataService` gọi Supabase REST.
188. Quản lý camera, chọn cây, xử lý ảnh, gọi predict, lưu trạng thái kết quả/lỗi.
189. Để UI phản ứng theo trạng thái: đang phân tích, có lỗi, có kết quả.
190. Để giảm dung lượng, thống nhất định dạng và phù hợp backend/model.
191. Để app biết khi nào hiển thị trạng thái không nhận diện được thay vì kết quả bình thường.
192. Để người dùng không thấy lỗi kỹ thuật khó hiểu và biết cách thử lại.
193. Bắt URLError, hiển thị thông báo mất mạng/timeout và cho retry ảnh vừa chọn.
194. Để không bắt user phải chụp/chọn lại ảnh khi lỗi mạng hoặc server chậm.
195. Chuyển nhãn cây/bệnh từ dạng model/API sang tên tiếng Việt dễ hiểu.
196. Vì người dùng mục tiêu là người Việt, giúp hiểu kết quả và hướng xử lý nhanh hơn.
197. `WeatherService` gọi `/weather` và `/weather/overview`.
198. `OutbreakService` gọi `/outbreaks` và `/outbreaks/areas`.
199. Gọi `/llm/advice/weather`, `/llm/advice/diagnosis`, `/llm/care-plan/diagnosis`, `/llm/care-metrics`.
200. Có thể tái sử dụng backend, database, API, model, Supabase; phải viết lại UI Android, camera, storage token và service client native Android.

