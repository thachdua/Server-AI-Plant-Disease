# Đáp án bảo vệ: Cache của chatbot, LLM, thời tiết và địa lý

File này dùng để ôn trả lời khi giảng viên hỏi xoáy về cache trong đồ án Plant Disease Detector.

---

## 1. Cache trong hệ thống của em gồm những loại nào?

Trong đồ án hiện tại có 3 kiểu lưu/cache chính:

1. **Cache RAM ngắn hạn ở backend**
   - File: `deploy/cache.py`
   - Dùng dictionary `_cache` lưu dữ liệu theo key và thời gian hết hạn.
   - Mất khi server restart hoặc Render spin down.

2. **Cache LLM trong database Supabase/PostgreSQL**
   - Bảng: `llm_advice_cache`
   - File SQL: `supabase/sql/008_llm_advice_cache.sql`
   - Hàm đọc/ghi: `llm_cache_get`, `llm_cache_upsert` trong `deploy/database.py`
   - Dùng cho các câu trả lời Gemini có cấu trúc như tư vấn bệnh, tư vấn thời tiết, care plan, care metrics.

3. **Cache cục bộ phiên chat trên iOS**
   - File: `PlantDiseaseDetector(IOS)/PlantDiseaseDetector/Services/ChatSessionStore.swift`
   - Dùng `UserDefaults` để giữ tạm nội dung chat hiện tại trên thiết bị.
   - Nếu người dùng đồng ý lưu cloud, chat mới được ghi vào Supabase qua `chat_sessions` và `chat_messages`.

---

## 2. Cache RAM ngắn hạn hoạt động thế nào?

File `deploy/cache.py`:

```python
_cache: dict[str, tuple[float, object]] = {}

def cache_get(key: str):
    item = _cache.get(key)
    if not item:
        return None
    exp, val = item
    if time.time() > exp:
        _cache.pop(key, None)
        return None
    return val

def cache_set(key: str, val, ttl_seconds: int):
    _cache[key] = (time.time() + ttl_seconds, val)
```

Ý nghĩa:

- Khi lưu cache, hệ thống lưu `key`, `value`, và thời điểm hết hạn.
- Khi đọc cache, nếu chưa hết hạn thì trả dữ liệu.
- Nếu hết hạn thì xóa key và trả `None`.

Câu trả lời ngắn:

**Dạ, cache RAM của em là cache đơn giản theo key-value có TTL. Mỗi item lưu kèm thời điểm hết hạn. Khi request sau có cùng cache key, backend trả dữ liệu cache thay vì gọi lại API ngoài hoặc query lại DB. Cache này nhanh nhưng chỉ nằm trong RAM, mất khi server restart hoặc Render sleep.**

---

## 3. Cache chatbot hoạt động thế nào?

Phải phân biệt 2 phần:

### 3.1. Chatbot backend `/llm/chat` có cache câu trả lời Gemini không?

Hiện tại **không cache câu trả lời `/llm/chat` ở backend**.

File: `deploy/routers/llm.py`

Endpoint `/llm/chat`:

- Nhận tối đa 12 tin nhắn gần nhất.
- Kiểm tra tổng số ký tự không vượt `LLM_CHAT_MAX_CHARS`.
- Tạo `system_prompt` theo mode `agriculture` hoặc `general`.
- Gọi Gemini bằng `call_gemini_text`.
- Trả reply về app.

Lý do không cache trực tiếp chat:

- Chat là hội thoại nhiều lượt, ngữ cảnh thay đổi liên tục.
- Cùng một câu hỏi nhưng lịch sử trước đó khác nhau thì ý nghĩa khác nhau.
- Cache nhầm có thể trả câu trả lời không đúng mạch hội thoại.

Câu trả lời ngắn:

**Dạ, riêng endpoint chatbot `/llm/chat` hiện tại em không cache câu trả lời Gemini ở backend, vì chat là hội thoại nhiều lượt, phụ thuộc vào lịch sử tin nhắn. Nếu cache không cẩn thận, cùng một câu hỏi nhưng context khác nhau có thể trả sai.**

### 3.2. Vậy chat có được lưu không?

Có, nhưng đó là **lưu phiên chat**, không phải cache câu trả lời AI để tái sử dụng.

Phía iOS:

- `ChatSessionStore.swift` lưu messages hiện tại vào `UserDefaults`.
- Nếu user thoát app rồi mở lại, app có thể restore phiên chat cục bộ.

Phía Supabase:

- `chat_sessions`: lưu thông tin cuộc chat.
- `chat_messages`: lưu từng tin nhắn.
- Chỉ lưu khi `storage_consent = true`.

Câu trả lời ngắn:

**Dạ, app có cache cục bộ phiên chat bằng UserDefaults để người dùng không mất đoạn chat đang dùng. Nếu người dùng đồng ý lưu lên cloud thì app ghi vào Supabase `chat_sessions` và `chat_messages`. Đây là lưu lịch sử chat, không phải cache để thay thế Gemini.**

---

## 4. Cache LLM advice hoạt động thế nào?

Các endpoint LLM có cấu trúc dùng cache database:

- `/llm/advice/diagnosis`
- `/llm/advice/weather`
- `/llm/care-plan/diagnosis`
- `/llm/care-metrics`

Luồng chung:

```text
Tạo payload context
    |
canonical_json(payload)
    |
sha256(...) => input_hash
    |
Đọc bảng llm_advice_cache theo kind + input_hash + lang
    |
Nếu có cache hợp lệ => trả cache
Nếu không => gọi Gemini
    |
Validate JSON
    |
Upsert vào llm_advice_cache
    |
Trả kết quả về app
```

File liên quan:

- `deploy/routers/llm.py`
- `deploy/database.py`
- `deploy/utils.py`
- `supabase/sql/008_llm_advice_cache.sql`

Câu trả lời ngắn:

**Dạ, với các khuyến nghị có cấu trúc như tư vấn bệnh, tư vấn thời tiết, care plan, care metrics, backend tạo payload context, chuẩn hóa JSON bằng `canonical_json`, băm SHA-256 thành `input_hash`, rồi tra bảng `llm_advice_cache`. Nếu đã có kết quả hợp lệ thì trả cache, nếu chưa có thì gọi Gemini, validate JSON rồi mới lưu vào cache.**

---

## 5. Vì sao cần `canonical_json` và `sha256`?

File `deploy/utils.py`:

```python
def canonical_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
```

Giải thích:

- `canonical_json` giúp JSON luôn có thứ tự key ổn định.
- Nếu payload giống nhau nhưng thứ tự field khác nhau, hash vẫn giống nhau.
- `sha256` biến context dài thành một chuỗi hash ngắn, dùng làm cache key.

Ví dụ:

```json
{"plant":"Tomato","disease":"Late blight"}
```

và:

```json
{"disease":"Late blight","plant":"Tomato"}
```

Sau `canonical_json`, hai payload có cùng thứ tự key, nên tạo cùng hash.

Câu trả lời ngắn:

**Dạ, em dùng `canonical_json` để chuẩn hóa payload trước khi hash, tránh trường hợp cùng dữ liệu nhưng thứ tự key khác nhau lại sinh cache key khác. Sau đó dùng SHA-256 để tạo `input_hash` ngắn, ổn định và không phải lưu toàn bộ context làm khóa.**

---

## 6. Cache tư vấn theo bệnh cây đã chẩn đoán

Endpoint:

```text
POST /llm/advice/diagnosis
```

Payload cache gồm:

```python
payload = {
    "plant": req.plant,
    "disease": disease,
    "confidence": req.confidence,
    "user_note": req.user_note,
    "weather_snapshot": req.weather_snapshot,
    "lang": "vi",
}
```

Ý nghĩa:

- Cùng cây, cùng bệnh, cùng confidence, cùng ghi chú và weather snapshot thì dùng lại kết quả.
- Nếu bệnh khác hoặc thời tiết khác thì `input_hash` khác, nên Gemini được gọi lại.

Câu trả lời ngắn:

**Dạ, tư vấn theo bệnh được cache theo toàn bộ context gồm cây, bệnh, confidence, ghi chú người dùng, weather snapshot và ngôn ngữ. Vì vậy nếu context thay đổi thì cache key thay đổi, hệ thống không dùng nhầm câu trả lời cũ.**

---

## 7. Cache tư vấn theo thời tiết

Endpoint:

```text
POST /llm/advice/weather
```

Payload cache gồm:

```python
payload = {
    "lat": round(req.lat, 3),
    "lng": round(req.lng, 3),
    "plant": req.plant,
    "disease": req.disease,
    "care_context": req.care_context,
    "snapshot": snapshot,
    "lang": "vi",
}
```

Điểm quan trọng:

- Lat/lng được làm tròn 3 chữ số thập phân.
- 3 chữ số thập phân tương đương khoảng hơn 100m, giúp giảm cache key quá chi tiết.
- Weather cache có kiểm tra hết hạn bằng `is_cache_expired("weather", updated_at)`.
- Weather LLM cache hết hạn sau khoảng 6 giờ.

Câu trả lời ngắn:

**Dạ, tư vấn thời tiết được cache theo vị trí đã làm tròn, cây, bệnh, care context và snapshot thời tiết. Vì thời tiết thay đổi theo thời gian, cache weather được kiểm tra hết hạn, nếu quá khoảng 6 giờ thì gọi Gemini lại thay vì dùng cache cũ.**

---

## 8. Cache tạo lịch chăm sóc

Endpoint:

```text
POST /llm/care-plan/diagnosis
```

Payload cache gồm:

```python
payload = {
    "plant": req.plant,
    "disease": disease,
    "confidence": req.confidence,
    "availability_mode": req.availability_mode or "normal",
    "user_note": req.user_note,
    "weather_snapshot": req.weather_snapshot,
    "lang": "vi",
}
```

Ý nghĩa:

- Lịch chăm sóc phụ thuộc cây, bệnh, confidence, mức độ rảnh/bận của người dùng, ghi chú và thời tiết.
- Nếu người dùng đổi `availability_mode` từ `busy` sang `flexible`, cache key đổi và lịch có thể khác.

Câu trả lời ngắn:

**Dạ, care plan được cache theo cây, bệnh, confidence, chế độ rảnh/bận, ghi chú và weather snapshot. Nhờ vậy lịch tạo ra bám theo đúng ngữ cảnh. Nếu người dùng thay đổi chế độ chăm sóc hoặc thông tin cây thì cache key thay đổi và hệ thống tạo lại lịch.**

Lưu ý kỹ thuật để trả lời nếu bị hỏi sâu:

**Trong code hiện tại có dùng kind `care_plan`, vì vậy schema SQL của `llm_advice_cache` cần cho phép thêm kind này. Nếu migration cũ chỉ check `weather, diagnosis` thì cần migration mở rộng CHECK constraint để chứa `care_plan` và `care_metrics`.**

---

## 9. Cache care metrics

Endpoint:

```text
POST /llm/care-metrics
```

Payload cache gồm:

```python
payload = {
    "plant": req.plant,
    "disease": req.disease,
    "confidence": confidence,
    "pot_diameter_cm": pot,
    "plant_height_cm": height,
    "measured_lux": lux,
    "lang": "vi",
}
```

Điểm hay:

- `pot_diameter_cm` làm tròn 1 chữ số.
- `plant_height_cm` làm tròn 1 chữ số.
- `measured_lux` làm tròn theo bội số 10.
- Làm tròn giúp tránh tạo cache key quá nhiều vì sai số đo nhỏ.

Câu trả lời ngắn:

**Dạ, care metrics được cache theo cây, bệnh, confidence, đường kính chậu, chiều cao cây và lux đo được. Các số đo được làm tròn trước khi hash để tránh cùng một tình huống nhưng sai số rất nhỏ lại tạo cache key mới.**

---

## 10. Cache thời tiết `/weather` hoạt động thế nào?

Endpoint:

```text
GET /weather?lat=...&lng=...
```

Cache key:

```python
lat_key = round(lat, 3)
lng_key = round(lng, 3)
cache_key = f"weather|{lat_key}|{lng_key}"
```

TTL:

```python
cache_set(cache_key, out, ttl_seconds=300)
```

Tức là cache 5 phút.

Lý do:

- Thời tiết hiện tại không cần gọi lại mỗi giây.
- Cache 5 phút giúp giảm quota OpenWeather và tăng tốc phản hồi.
- Vẫn đủ mới cho mục tiêu khuyến nghị chăm sóc cây.

Câu trả lời ngắn:

**Dạ, endpoint `/weather` cache theo tọa độ làm tròn 3 chữ số và TTL 300 giây. Trong 5 phút, nếu nhiều người hoặc cùng người gọi cùng khu vực, backend trả cache thay vì gọi lại OpenWeather.**

---

## 11. Cache weather overview

Endpoint:

```text
GET /weather/overview
```

Cache key:

```python
cache_key = f"weather_overview|{lat_key}|{lng_key}"
```

TTL:

```python
ttl_seconds=900
```

Tức là 15 phút.

Câu trả lời ngắn:

**Dạ, weather overview ít cần cập nhật liên tục hơn dữ liệu current weather, nên em cache 15 phút. Nếu OpenWeather trả overview không phải tiếng Việt, backend có fallback tiếng Việt cơ bản.**

---

## 12. Cache địa lý và bản đồ vùng dịch

Các endpoint outbreak dùng cache RAM ngắn hạn:

| Endpoint | Cache key theo | TTL |
|---|---|---:|
| `/outbreaks` | filter bệnh, cây, tỉnh, xã, source, confidence, vị trí gần, limit | 20 giây |
| `/outbreaks/areas` | level, parent, since_days, filter, geometry, vị trí gần | 300 giây |
| `/outbreaks/admin/provinces` | danh sách tỉnh | 3600 giây |
| `/outbreaks/admin/wards` | province_id | 3600 giây |
| `/outbreaks/filter-options` | since/province/ward | 300 giây |
| `/outbreaks/nearby-advice` | lat/lng/radius/plant/disease/min severity | 60 giây |
| `/outbreaks/summary` | filter + since + vị trí gần | 60 giây |

Ý nghĩa:

- Danh sách ca bệnh thay đổi tương đối nhanh, nên `/outbreaks` chỉ cache 20 giây.
- Vùng/tổng hợp/summary nặng hơn, cache 1-5 phút.
- Danh sách tỉnh/xã ít thay đổi, cache 1 giờ.

Câu trả lời ngắn:

**Dạ, dữ liệu địa lý và bản đồ vùng dịch dùng cache RAM theo từng bộ lọc. Endpoint danh sách ca bệnh cache ngắn 20 giây để dữ liệu vẫn mới. Các endpoint tổng hợp vùng, filter, nearby advice cache 60-300 giây. Danh sách tỉnh/xã ít thay đổi nên cache 1 giờ.**

---

## 13. Vì sao cache địa lý không để quá lâu?

Vì dữ liệu `outbreak_cases` có thể thay đổi khi người dùng lưu chẩn đoán mới hoặc expert cập nhật trạng thái.

Nếu cache quá lâu:

- Bản đồ không hiện ca mới.
- Risk level không cập nhật.
- Người dùng thấy dữ liệu cũ.

Câu trả lời ngắn:

**Dạ, bản đồ vùng dịch cần tương đối mới vì ca bệnh có thể được tạo từ lịch sử chẩn đoán. Do đó các endpoint liên quan outbreak chỉ cache ngắn, ví dụ 20 giây cho danh sách ca và 60-300 giây cho tổng hợp.**

---

## 14. Nếu cache sai thì có ảnh hưởng gì?

Các rủi ro:

- Trả khuyến nghị cũ không đúng thời tiết hiện tại.
- Bản đồ vùng dịch chậm cập nhật.
- Care plan không phản ánh thay đổi context mới.
- Chat nếu cache sai có thể trả lệch hội thoại, nên hiện tại không cache `/llm/chat`.

Cách giảm rủi ro:

- Cache key phải chứa đầy đủ context.
- Weather cache có TTL và kiểm tra hết hạn.
- Chỉ cache Gemini sau khi validate output.
- Không cache trực tiếp chatbot hội thoại.
- Outbreak cache TTL ngắn.

Câu trả lời ngắn:

**Dạ, để tránh cache sai, em đưa các yếu tố quan trọng vào cache key như cây, bệnh, confidence, vị trí, thời tiết, mode chăm sóc. Với dữ liệu thay đổi nhanh như weather/outbreak thì TTL ngắn. Với LLM, chỉ kết quả đã validate mới được lưu.**

---

## 15. Cache có bảo mật dữ liệu người dùng không?

Cần nói rõ:

- Cache RAM có thể chứa response tạm thời.
- LLM DB cache có thể chứa content_json/content_text tư vấn.
- `input_hash` không lưu raw payload làm key, nhưng content_json vẫn là nội dung tư vấn.
- Không nên đưa dữ liệu quá nhạy cảm vào prompt/cache.
- Chat cloud chỉ lưu khi user đồng ý.

Câu trả lời ngắn:

**Dạ, cache giúp tối ưu hiệu năng nhưng cũng cần chú ý quyền riêng tư. Em không dùng raw context làm khóa mà dùng `input_hash`. Với chat, app chỉ lưu lên cloud khi người dùng đồng ý. Nếu triển khai thực tế, em sẽ hạn chế dữ liệu nhạy cảm trong prompt/cache và có chính sách xóa dữ liệu theo người dùng.**

---

## 16. Cache có bị mất khi Render sleep/restart không?

Có với cache RAM.

- `_cache` nằm trong memory của process Python.
- Render Free có thể spin down khi không có request.
- Khi server khởi động lại, cache RAM mất.

Nhưng:

- LLM advice cache trong database vẫn còn.
- Chat cloud trong Supabase vẫn còn nếu user đã lưu.

Câu trả lời ngắn:

**Dạ, cache RAM như weather/outbreak sẽ mất khi Render restart hoặc spin down vì nó nằm trong memory. Đây là chấp nhận được vì đó là cache ngắn hạn. Riêng cache LLM trong bảng `llm_advice_cache` và chat đã lưu Supabase thì vẫn tồn tại sau restart.**

---

## 17. Vì sao không dùng Redis?

Câu trả lời tốt:

**Dạ, Redis là lựa chọn tốt hơn cho production vì cache chia sẻ được giữa nhiều instance, có TTL chuẩn và hiệu năng cao. Tuy nhiên trong phạm vi đồ án và Render free, em dùng cache RAM đơn giản để giảm phụ thuộc hạ tầng. Nếu triển khai thực tế, em sẽ thay cache RAM bằng Redis hoặc managed cache.**

---

## 18. Nếu nhiều instance backend thì cache RAM có vấn đề gì?

Vấn đề:

- Mỗi instance có cache riêng.
- Request vào instance A có cache nhưng instance B không có.
- Dữ liệu cache không đồng bộ.

Câu trả lời ngắn:

**Dạ, cache RAM chỉ phù hợp một instance. Nếu scale nhiều instance, cache sẽ không đồng bộ giữa các instance. Khi đó nên dùng Redis hoặc database cache chung.**

---

## 19. Vì sao cache LLM dùng database thay vì RAM?

Vì LLM:

- Tốn chi phí/quota.
- Kết quả tư vấn có thể dùng lại lâu hơn.
- Cần tồn tại sau restart.
- Cần chia sẻ nếu backend scale.

Câu trả lời ngắn:

**Dạ, LLM tốn chi phí và độ trễ cao hơn weather/outbreak query, nên em lưu cache LLM vào database để tồn tại sau restart và dùng lại ổn định. RAM cache chỉ dùng cho dữ liệu ngắn hạn như thời tiết, bản đồ hoặc fallback tạm.**

---

## 20. Cache fallback là gì?

Trong `/llm/advice/diagnosis`, nếu Gemini lỗi 503:

```python
fallback_key = f"llm_fallback|diagnosis|{input_hash}"
cached_fb = cache_get(fallback_key)
...
cache_set(fallback_key, cached_fb, ttl_seconds=300)
```

Ý nghĩa:

- Nếu Gemini tạm lỗi, backend tạo fallback advice.
- Fallback được cache RAM 5 phút để không tạo lại liên tục.

Câu trả lời ngắn:

**Dạ, fallback cache là cache tạm trong RAM cho lời khuyên dự phòng khi Gemini lỗi 503. Nó giúp các request giống nhau trong vài phút nhận cùng fallback ổn định mà không phải xử lý lại.**

---

## 21. Giảng viên hỏi: “Cache có làm kết quả Gemini bị cũ không?”

Trả lời:

**Dạ, có thể nếu TTL hoặc cache key thiết kế không tốt. Vì vậy em đưa context vào cache key. Với weather advice còn kiểm tra `updated_at`, quá 6 giờ thì cache hết hạn. Với dữ liệu thời tiết thô, cache chỉ 5 phút. Với outbreak, cache ngắn 20-300 giây. Những dữ liệu ít thay đổi như tỉnh/xã mới cache 1 giờ.**

---

## 22. Giảng viên hỏi: “Nếu cùng cây và bệnh nhưng ở vị trí khác thì có dùng chung cache không?”

Trả lời:

**Dạ, tùy endpoint. Với tư vấn diagnosis nếu payload không có weather/vị trí thì có thể dùng chung vì context giống nhau. Nhưng với weather advice, payload có lat/lng đã làm tròn và weather snapshot, nên vị trí khác sẽ sinh `input_hash` khác và không dùng nhầm cache.**

---

## 23. Giảng viên hỏi: “Nếu thời tiết thay đổi mà cache vẫn còn thì sao?”

Trả lời:

**Dạ, dữ liệu `/weather` chỉ cache 5 phút nên thay đổi nhỏ trong vài phút được chấp nhận. Với LLM weather advice, cache DB có kiểm tra hết hạn khoảng 6 giờ; nếu quá hạn thì gọi Gemini lại. Nếu cần chính xác hơn trong production, em có thể giảm TTL hoặc đưa thời điểm/weather snapshot chi tiết hơn vào cache key.**

---

## 24. Giảng viên hỏi: “Vì sao lat/lng lại round 3 chữ số?”

Trả lời:

**Dạ, round 3 chữ số giúp gom các vị trí rất gần nhau vào cùng cache key, khoảng hơn 100m. Điều này giảm số lần gọi OpenWeather/Gemini vì thời tiết trong khu vực rất gần thường không khác nhiều. Nếu cần chính xác hơn, có thể tăng lên 4 chữ số; nếu muốn tiết kiệm hơn, giảm xuống 2 chữ số.**

---

## 25. Giảng viên hỏi: “Cache của em có xóa tự động không?”

Trả lời:

**Dạ, cache RAM được xóa lazy khi có request đọc key đã hết hạn. Còn cache DB hiện tại được upsert và dùng `updated_at` để kiểm tra hết hạn ở một số kind như weather. Nếu production, em sẽ thêm job định kỳ dọn cache DB cũ để tránh bảng phình to.**

---

## 26. Giảng viên hỏi: “Có cache ảnh predict không?”

Trả lời:

**Dạ, hiện tại em không cache kết quả predict theo ảnh. Mỗi ảnh gửi lên sẽ được xử lý và gọi model. Lý do là ảnh có thể khác nhau rất nhỏ nhưng bệnh khác, và hash ảnh/cache predict cần thiết kế kỹ để tránh trả sai. Nếu mở rộng, em có thể hash ảnh sau khi normalize để phát hiện ảnh trùng, nhưng vẫn phải cân nhắc quyền riêng tư và độ chính xác.**

---

## 27. Giảng viên hỏi: “Tại sao không cache chatbot mà lại cache advice?”

Trả lời:

**Dạ, chatbot là hội thoại tự do, phụ thuộc lịch sử tin nhắn nên cache dễ sai context. Còn advice/care plan/weather là request có cấu trúc, context rõ ràng như cây, bệnh, confidence, thời tiết, vị trí, nên có thể hash context để cache an toàn hơn.**

---

## 28. Giảng viên hỏi: “Cache có giảm chi phí không?”

Trả lời:

**Dạ có. Cache giúp giảm số lần gọi Gemini và OpenWeather, hai dịch vụ có quota/chi phí. Nó cũng giảm thời gian phản hồi và giảm tải cho Supabase/query địa lý.**

---

## 29. Giảng viên hỏi: “Nếu cache DB bị RLS chặn ghi thì sao?”

Trả lời:

**Dạ, bảng `llm_advice_cache` chặn client ghi trực tiếp, nhưng backend dùng kết nối DB/service role để ghi. Nếu bị lỗi quyền hoặc thiếu migration, hàm `llm_cache_upsert` sẽ log lỗi và trả False; hệ thống vẫn có thể trả kết quả vừa gọi Gemini cho user, chỉ là không cache được.**

---

## 30. Giảng viên hỏi: “Điểm cần cải thiện trong cache hiện tại là gì?”

Trả lời:

**Dạ, có vài điểm: dùng Redis thay cho RAM cache khi scale nhiều instance; thêm job dọn cache DB cũ; chuẩn hóa migration để `llm_advice_cache.kind` hỗ trợ đủ `diagnosis`, `weather`, `care_plan`, `care_metrics`; thêm invalidation khi dữ liệu outbreak thay đổi; và cân nhắc cache predict theo hash ảnh nếu đảm bảo quyền riêng tư.**

---

## 31. Câu trả lời tổng hợp ngắn khi bị hỏi về cache

**Dạ, hệ thống của em có hai lớp cache chính. Lớp thứ nhất là cache RAM ngắn hạn trong `deploy/cache.py`, dùng cho dữ liệu thay đổi nhanh như thời tiết, bản đồ vùng dịch, danh sách tỉnh/xã và fallback khi Gemini lỗi. Cache này dùng key-value kèm TTL, rất nhanh nhưng mất khi Render restart. Lớp thứ hai là cache LLM trong bảng `llm_advice_cache`, dùng cho các câu trả lời có cấu trúc từ Gemini như tư vấn bệnh, tư vấn thời tiết, care plan và care metrics. Backend tạo `input_hash` từ context gồm cây, bệnh, confidence, thời tiết, vị trí, mode chăm sóc... rồi dùng hash đó để đọc/ghi cache. Riêng chatbot `/llm/chat` hiện không cache câu trả lời backend vì hội thoại phụ thuộc lịch sử tin nhắn, nhưng app có lưu phiên chat cục bộ bằng UserDefaults và chỉ lưu cloud vào Supabase khi user đồng ý.**

