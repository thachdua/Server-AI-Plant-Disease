# 200 câu hỏi có thể gặp khi bảo vệ đồ án Plant Disease Detector

Vai trò giả định: giảng viên phản biện/giảng viên hội đồng trong buổi bảo vệ đồ án tốt nghiệp.  
Phạm vi: AI, server, database, Supabase, iOS, nghiệp vụ, bảo mật, deploy, kiểm thử và hướng phát triển.

---

## A. Tổng quan đề tài và nghiệp vụ

1. Em hãy giới thiệu ngắn gọn đề tài của mình giải quyết vấn đề gì?
2. Vì sao em chọn bài toán nhận diện bệnh cây thay vì một bài toán AI khác?
3. Đối tượng người dùng chính của hệ thống là ai?
4. Hệ thống của em hỗ trợ những vai trò người dùng nào?
5. Khách, người dùng đã đăng nhập và chuyên gia khác nhau ở điểm nào?
6. Điểm mới hoặc điểm nổi bật nhất của đồ án là gì?
7. Hệ thống có thay thế hoàn toàn chuyên gia nông nghiệp không? Vì sao?
8. Kết quả chẩn đoán của AI nên được hiểu là kết luận chắc chắn hay thông tin tham khảo?
9. Nếu người dùng chụp ảnh không rõ thì hệ thống xử lý thế nào?
10. Vì sao trước khi quét bệnh cần chọn loại cây?
11. Nếu người dùng chọn sai loại cây thì kết quả có thể bị ảnh hưởng thế nào?
12. Hệ thống có những chức năng chính nào ngoài chẩn đoán bệnh cây?
13. Chức năng lịch sử chẩn đoán mang lại lợi ích gì?
14. Chức năng Cây của tôi giải quyết nhu cầu gì?
15. Chức năng Discover/Bookmark phục vụ mục đích gì?
16. Chức năng bản đồ vùng dịch có ý nghĩa thực tế như thế nào?
17. Chức năng thời tiết hỗ trợ quá trình chăm sóc cây ra sao?
18. Vì sao cần có chức năng phản hồi kết quả AI?
19. Vì sao hệ thống cần phân quyền chuyên gia?
20. Nếu triển khai thực tế cho nông dân, em sẽ ưu tiên cải thiện chức năng nào trước?

## B. Kiến trúc hệ thống

21. Em hãy mô tả kiến trúc tổng thể của hệ thống.
22. Vì sao hệ thống tách thành iOS app, backend FastAPI và Supabase?
23. Vì sao không gọi trực tiếp mô hình AI từ ứng dụng iOS?
24. Backend FastAPI đóng vai trò gì trong hệ thống?
25. Supabase đóng vai trò gì trong hệ thống?
26. Hugging Face được sử dụng ở bước nào?
27. Gemini API được sử dụng cho chức năng nào?
28. OpenWeather API được sử dụng để làm gì?
29. Render được sử dụng trong hệ thống như thế nào?
30. Luồng dữ liệu từ lúc người dùng chụp ảnh đến lúc nhận kết quả là gì?
31. Luồng dữ liệu khi người dùng lưu lịch sử chẩn đoán là gì?
32. Luồng dữ liệu khi người dùng hỏi chatbot là gì?
33. Luồng dữ liệu khi người dùng xem bản đồ vùng dịch là gì?
34. Điểm khác nhau giữa gọi backend FastAPI và gọi trực tiếp Supabase REST là gì?
35. Vì sao một số chức năng đi qua backend, còn một số chức năng iOS gọi thẳng Supabase?
36. Nếu backend bị lỗi, những chức năng nào của app bị ảnh hưởng nhiều nhất?
37. Nếu Supabase bị lỗi, hệ thống bị ảnh hưởng như thế nào?
38. Nếu Hugging Face bị timeout, hệ thống trả gì cho người dùng?
39. Nếu Gemini API lỗi, hệ thống có phương án dự phòng nào?
40. Em đánh giá kiến trúc hiện tại có dễ mở rộng không? Vì sao?

## C. AI và mô hình nhận diện bệnh cây

41. Bài toán AI trong đồ án thuộc loại bài toán gì?
42. Đầu vào và đầu ra của mô hình nhận diện bệnh cây là gì?
43. Vì sao bài toán này phù hợp với mô hình CNN?
44. CNN là gì?
45. Lớp convolution trong CNN có vai trò gì?
46. Lớp pooling trong CNN có vai trò gì?
47. Fully connected layer trong CNN dùng để làm gì?
48. Confidence trong kết quả dự đoán có ý nghĩa gì?
49. Confidence cao có luôn đồng nghĩa kết quả đúng không? Vì sao?
50. Nếu hai bệnh có triệu chứng giống nhau thì mô hình có thể gặp khó khăn gì?
51. Chất lượng ảnh đầu vào ảnh hưởng đến mô hình như thế nào?
52. Những yếu tố nào có thể làm mô hình dự đoán sai?
53. Dataset huấn luyện ảnh bệnh cây cần đảm bảo những yếu tố nào?
54. Vì sao cần dữ liệu ảnh ở nhiều điều kiện ánh sáng và góc chụp khác nhau?
55. Class imbalance là gì và ảnh hưởng thế nào đến mô hình?
56. Overfitting là gì trong bài toán nhận diện bệnh cây?
57. Làm sao để giảm overfitting?
58. Nếu muốn thêm một loại cây mới vào model thì cần làm những bước nào?
59. Nếu muốn thêm một loại bệnh mới vào model thì cần làm những bước nào?
60. Vì sao dữ liệu low-confidence có thể giúp cải thiện model?

## D. Luồng predict và xử lý ảnh

61. Endpoint nào của backend dùng để chẩn đoán bệnh cây?
62. iOS gửi ảnh lên backend bằng định dạng request nào?
63. Vì sao backend phải kiểm tra Content-Type của request?
64. Vì sao backend phải kiểm tra phần mở rộng file ảnh?
65. Vì sao backend vẫn phải mở ảnh bằng Pillow dù Content-Type đã là image?
66. Decompression bomb là gì?
67. Vì sao cần giới hạn kích thước ảnh upload?
68. Vì sao backend chuyển ảnh về JPEG trước khi gửi sang AI?
69. Vì sao cần loại bỏ metadata hoặc payload lạ trong ảnh?
70. Nếu người dùng upload file JavaScript đổi tên thành `.jpg` thì hệ thống chặn ở đâu?
71. Hàm `_sanitize_image` có nhiệm vụ gì?
72. Hàm `_validate_upload_metadata` có nhiệm vụ gì?
73. Hàm `parse_confidence_value` xử lý những dạng confidence nào?
74. Hàm `infer_plant_from_disease_label` dùng để làm gì?
75. Khi confidence thấp hơn ngưỡng unrecognized, backend trả kết quả gì?
76. Khi confidence thấp nhưng vẫn trên ngưỡng unrecognized, hệ thống làm gì?
77. Vì sao trường hợp unrecognized không upload ảnh tự động?
78. Vì sao ảnh thành công được upload lên Supabase Storage?
79. Nếu Hugging Face trả thiếu trường disease thì backend xử lý thế nào?
80. Nếu Hugging Face trả JSON không hợp lệ thì backend xử lý thế nào?

## E. Backend FastAPI và API design

81. FastAPI là gì và vì sao phù hợp với đồ án này?
82. `deploy/main.py` có nhiệm vụ gì?
83. Router trong FastAPI giúp tổ chức code như thế nào?
84. Các router chính trong backend gồm những router nào?
85. Vì sao backend cần file `config.py`?
86. Vì sao nên cấu hình qua biến môi trường thay vì hard-code trong code?
87. `main.py` ở root khác gì với `deploy/main.py`?
88. Vì sao backend dùng `run_in_threadpool` ở một số đoạn?
89. Endpoint `/health` dùng để làm gì?
90. Endpoint `/health/ready` khác `/health` ở điểm nào?
91. Vì sao `/health/ready` không được trả secret ra ngoài?
92. Pydantic model trong `deploy/models.py` dùng để làm gì?
93. `extra="forbid"` trong Pydantic có ý nghĩa gì?
94. Vì sao API cần trả status code phù hợp như 400, 401, 413, 415, 429, 502?
95. HTTP 400 khác 422 như thế nào trong hệ thống này?
96. HTTP 401 thường xảy ra khi nào?
97. HTTP 413 thường xảy ra khi nào?
98. HTTP 415 thường xảy ra khi nào?
99. HTTP 429 thường xảy ra khi nào?
100. HTTP 502 thường xảy ra khi nào?

## F. Supabase database, RLS và Storage

101. Supabase gồm những thành phần nào được dùng trong đồ án?
102. Bảng `profiles` dùng để làm gì?
103. Vì sao `profiles.id` tham chiếu đến `auth.users(id)`?
104. Role `user` và `expert` khác nhau thế nào?
105. RLS là gì?
106. Vì sao cần bật RLS cho các bảng dữ liệu người dùng?
107. Nếu không bật RLS thì rủi ro gì có thể xảy ra?
108. Policy `select own` thường có ý nghĩa gì?
109. Policy `insert own` thường có ý nghĩa gì?
110. Vì sao user không được tự cập nhật role thành expert?
111. Function `is_expert(uid)` dùng để làm gì?
112. Service role key khác anon key ở điểm nào?
113. Vì sao service role key chỉ được đặt ở backend?
114. Bucket `plant-images` dùng để lưu dữ liệu gì?
115. Vì sao ảnh chẩn đoán cần public URL?
116. Bảng `history` lưu những thông tin gì?
117. Bảng `outbreak_cases` lưu những thông tin gì?
118. Bảng `ai_feedback_cases` phục vụ mục đích gì?
119. Bảng `report_cases` khác gì với `ai_feedback_cases`?
120. Bảng `consultation_requests` dùng trong nghiệp vụ nào?

## G. History, Cây của tôi, Discover và Bookmark

121. Khi nào hệ thống lưu lịch sử chẩn đoán?
122. Vì sao lưu lịch sử cần user đăng nhập?
123. Hàm `_should_create_outbreak` kiểm tra những điều kiện nào?
124. Vì sao cây khỏe mạnh không tạo ca vùng dịch?
125. Vì sao confidence thấp không tạo ca vùng dịch?
126. `user_plants` lưu dữ liệu gì?
127. Các trường `current_disease`, `current_confidence`, `current_history_id` trong `user_plants` có ý nghĩa gì?
128. `care_plans` dùng để làm gì?
129. `care_tasks` dùng để làm gì?
130. Vì sao `care_tasks` cần trường `repeat_rule`?
131. Vì sao cần trường `notification_enabled`?
132. `plant_knowledge` lưu dữ liệu gì?
133. `plant_resources` lưu dữ liệu gì?
134. `bookmarks` dùng để làm gì?
135. Vì sao bảng `bookmarks` cần unique `(created_by, resource_id)`?
136. Tại sao `plant_resources` có thể public select?
137. Nếu user xóa một cây trong `user_plants`, các `care_tasks` liên quan nên xử lý thế nào?
138. Discover khác Disease Dictionary như thế nào?
139. Vì sao hệ thống cần cả kiến thức tĩnh và tư vấn AI động?
140. Nếu muốn thêm tài liệu mới vào Discover thì cần cập nhật ở đâu?

## H. Weather, outbreak map và dữ liệu địa lý

141. Chức năng thời tiết lấy dữ liệu từ đâu?
142. Endpoint `/weather` trả những thông tin chính nào?
143. Hệ thống tạo cảnh báo thời tiết dựa trên những yếu tố nào?
144. Vì sao độ ẩm cao có liên quan đến bệnh cây?
145. Vì sao mưa có thể làm tăng nguy cơ lan bệnh?
146. Vì sao cần validate lat/lng trước khi gọi OpenWeather?
147. Cache thời tiết giúp ích gì?
148. Chức năng outbreak map dùng bảng nào?
149. `/outbreaks` và `/outbreaks/areas` khác nhau thế nào?
150. Boundary GeoJSON được dùng để làm gì?
151. Hệ thống tính vùng dịch theo số ca hay theo severity?
152. Vì sao không nên lấy confidence làm severity trực tiếp?
153. `source_display` trong outbreak case được tạo như thế nào?
154. Vì sao cần hiển thị nguồn là người dùng hay chuyên gia?
155. Nếu một ca bệnh không có lat/lng thì có hiển thị trên bản đồ được không?
156. Nếu người dùng không cấp quyền vị trí thì chức năng nào bị ảnh hưởng?
157. Nếu OpenWeather hết quota thì hệ thống trả lỗi gì?
158. Nếu dữ liệu boundary tải thất bại thì backend xử lý thế nào?
159. Làm sao để mở rộng bản đồ từ vài tỉnh sang toàn quốc?
160. Làm sao để giảm sai lệch dữ liệu vùng dịch do AI dự đoán nhầm?

## I. Chatbot, Gemini, LLM và tư vấn

161. Gemini API được dùng cho những chức năng nào?
162. Endpoint `/llm/chat` dùng để làm gì?
163. Endpoint `/llm/advice/diagnosis` dùng để làm gì?
164. Endpoint `/llm/advice/weather` dùng để làm gì?
165. Endpoint `/llm/care-plan/diagnosis` dùng để làm gì?
166. Vì sao prompt cần yêu cầu trả lời tiếng Việt?
167. Vì sao prompt cần hạn chế đưa liều lượng hóa chất nguy hiểm?
168. Vì sao backend cần validate JSON trả về từ Gemini?
169. Nếu Gemini trả lời sai format thì hệ thống nên xử lý thế nào?
170. `llm_advice_cache` dùng để làm gì?
171. Vì sao cache LLM cần `input_hash`?
172. Vì sao weather advice cache có thể hết hạn?
173. Fallback advice dùng trong trường hợp nào?
174. Chat storage consent là gì?
175. Vì sao không nên tự động lưu mọi cuộc chat nếu người dùng chưa đồng ý?
176. `chat_sessions` và `chat_messages` khác nhau thế nào?
177. Vì sao chat message cần trường `role`?
178. Nếu prompt quá dài thì backend xử lý ra sao?
179. Vì sao LLM cần rate limit riêng?
180. Làm sao để đánh giá chất lượng câu trả lời của chatbot?

## J. iOS SwiftUI app

181. `PlantDiseaseDetectorApp` có vai trò gì?
182. `RootView` quyết định màn hình dựa trên những trạng thái nào?
183. `AuthStore` quản lý những thông tin gì?
184. Vì sao session được lưu trong Keychain?
185. Google OAuth trong iOS hoạt động theo cơ chế nào?
186. `SupabaseConfig` lấy cấu hình từ đâu?
187. `APIService` khác `SupabaseDataService` thế nào?
188. `ScannerViewModel` có nhiệm vụ gì?
189. Vì sao ViewModel nên giữ trạng thái `isAnalyzing`, `errorMessage`, `predictResponse`?
190. Vì sao ảnh được nén/chuyển thành JPEG trước khi upload?
191. `PredictResponse.isUnrecognized` dùng để làm gì?
192. Vì sao app cần xử lý lỗi thân thiện với người dùng?
193. Nếu mất mạng khi đang gửi ảnh thì app xử lý thế nào?
194. Vì sao cần cho phép retry ảnh vừa chọn?
195. `DiseaseLocalizer` dùng để làm gì?
196. Vì sao cần hiển thị tên bệnh bằng tiếng Việt?
197. `WeatherService` gọi endpoint nào?
198. `OutbreakService` gọi endpoint nào?
199. `LLMAdviceService` gọi các endpoint nào?
200. Nếu muốn port app sang Android, phần nào có thể tái sử dụng và phần nào phải viết lại?

