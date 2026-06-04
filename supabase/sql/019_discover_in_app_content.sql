-- Expand Discover with more document types and in-app readable content.

alter table public.plant_resources
  drop constraint if exists plant_resources_resource_type_check;

alter table public.plant_resources
  add constraint plant_resources_resource_type_check
  check (resource_type in ('article','report','youtube','guide','checklist','infographic'));

alter table public.plant_resources
  add column if not exists content_markdown text,
  add column if not exists key_points text[] not null default '{}',
  add column if not exists read_time_minutes integer;

delete from public.plant_resources a
using public.plant_resources b
where a.ctid < b.ctid
  and a.url = b.url;

create unique index if not exists plant_resources_url_unique_idx
on public.plant_resources (url);

update public.plant_resources
set
  url = 'https://www.mdpi.com/2073-4395/11/12/2590',
  source_name = 'MDPI Agronomy',
  title = 'Coffee Leaf Rust: tổng quan và quản lý',
  summary = 'Bài tổng quan về bệnh gỉ sắt cà phê, triệu chứng và các hướng quản lý như giống kháng, vệ sinh vườn và phun phòng theo khuyến cáo địa phương.',
  image_url = 'https://images.unsplash.com/photo-1447933601403-0c6688de566e?auto=format&fit=crop&w=1200&q=80',
  category = 'disease',
  duration_label = 'Báo cáo',
  resource_type = 'report'
where url like '%cabi.org%'
   or url like '%cabidigitallibrary%'
   or lower(title) like '%coffee leaf rust%';

insert into public.plant_resources
  (
    plant_key,
    title,
    url,
    resource_type,
    source_name,
    summary,
    language,
    image_url,
    category,
    duration_label,
    is_featured,
    content_markdown,
    key_points,
    read_time_minutes
  )
values
  (
    null,
    'Checklist 7 ngày sau khi AI chẩn đoán bệnh',
    'app://discover/post-diagnosis-checklist',
    'checklist',
    'Plant Disease Detector',
    'Checklist nội bộ giúp người dùng xử lý an toàn sau khi nhận kết quả chẩn đoán.',
    'vi',
    'https://images.unsplash.com/photo-1416879595882-3373a0480b5b?auto=format&fit=crop&w=1200&q=80',
    'diagnosis',
    'Checklist',
    true,
    $body$
## Ngày 1: cách ly và quan sát
- Đặt cây bệnh cách xa cây khỏe nếu có thể.
- Chụp lại lá, thân, quả và mặt dưới lá để so sánh sau vài ngày.
- Không tưới phun lên lá vào buổi tối.

## Ngày 2-3: vệ sinh nguồn bệnh
- Cắt bỏ phần lá hoặc cành đã hỏng nặng bằng dụng cụ sạch.
- Thu gom lá rụng quanh gốc, không để mầm bệnh tồn tại trong tàn dư.
- Rửa tay hoặc khử trùng kéo sau khi thao tác.

## Ngày 4-7: theo dõi và điều chỉnh chăm sóc
- Kiểm tra vết bệnh có lan nhanh không.
- Điều chỉnh tưới gốc, tăng thông thoáng và giảm ẩm kéo dài.
- Nếu bệnh lan nhanh hoặc cây có giá trị cao, hỏi chuyên gia/kỹ thuật viên địa phương.
$body$,
    array[
      'Không lưu kết quả độ tin cậy thấp vào lịch sử chăm sóc.',
      'Ưu tiên vệ sinh, cách ly và giảm ẩm trước khi dùng thuốc.',
      'Nếu dùng thuốc bảo vệ thực vật, luôn theo nhãn sản phẩm và khuyến cáo địa phương.'
    ],
    4
  ),
  (
    null,
    'Cách tưới nước để giảm nấm bệnh trên lá',
    'app://discover/watering-against-leaf-disease',
    'guide',
    'Plant Disease Detector',
    'Hướng dẫn tưới gốc, chọn thời điểm tưới và theo dõi độ ẩm để giảm nguy cơ bệnh lá.',
    'vi',
    'https://images.unsplash.com/photo-1466692476868-aef1dfb1e735?auto=format&fit=crop&w=1200&q=80',
    'watering',
    '5 phút đọc',
    true,
    $body$
## Nguyên tắc chính
Tưới nước không chỉ là bổ sung nước cho cây. Với cây đang có dấu hiệu bệnh, cách tưới quyết định lá có khô nhanh hay tiếp tục ẩm lâu, tạo điều kiện cho nấm và vi khuẩn.

## Nên làm
- Tưới vào gốc, tránh làm ướt mặt lá khi không cần thiết.
- Tưới vào buổi sáng để cây có thời gian khô trước đêm.
- Kiểm tra mặt đất hoặc giá thể trước khi tưới lại.
- Với cây chậu, đảm bảo nước thoát ra được và không đọng dưới đáy.

## Không nên làm
- Không tưới phun mưa vào chiều tối khi cây đang bệnh.
- Không tưới theo lịch cứng nếu đất vẫn còn ướt.
- Không để nhiều cây quá sát nhau làm không khí khó lưu thông.
$body$,
    array[
      'Tưới sáng tốt hơn tưới tối khi cây đang bị bệnh lá.',
      'Tưới gốc giúp hạn chế nước đọng trên lá.',
      'Đất úng làm rễ yếu và cây dễ nhiễm bệnh hơn.'
    ],
    5
  ),
  (
    null,
    'Infographic: dấu hiệu cây thiếu nước, dư nước và thiếu sáng',
    'app://discover/stress-signs-infographic',
    'infographic',
    'Plant Disease Detector',
    'Bảng nhận biết nhanh các dấu hiệu chăm sóc sai thường bị nhầm với bệnh.',
    'vi',
    'https://images.unsplash.com/photo-1485955900006-10f4d324d411?auto=format&fit=crop&w=1200&q=80',
    'care',
    'Hình ảnh',
    false,
    $body$
## Thiếu nước
- Lá rũ mềm, mép lá khô, đất khô sâu.
- Cây hồi lại sau khi tưới đúng lượng.

## Dư nước
- Lá vàng từ dưới lên, đất ẩm lâu, rễ có mùi hoặc thâm.
- Cây vẫn rũ dù đất đang ướt.

## Thiếu sáng
- Thân vươn dài, lá nhạt màu, cây nghiêng về phía cửa sáng.
- Cây ít ra hoa hoặc chậm lớn.

## Gợi ý xử lý
So sánh dấu hiệu chăm sóc với kết quả AI. Nếu độ tin cậy thấp hoặc triệu chứng không khớp, hãy chụp lại ảnh rõ hơn và không vội dùng thuốc.
$body$,
    array[
      'Không phải mọi đốm vàng đều là bệnh.',
      'Thiếu sáng và dư nước rất dễ bị nhầm với nấm bệnh.',
      'Ảnh chụp rõ mặt trên/mặt dưới lá giúp chẩn đoán tốt hơn.'
    ],
    3
  ),
  (
    'tomato',
    'Bệnh cháy lá sớm trên cà chua và khoai tây',
    'https://extension.umn.edu/node/2681',
    'article',
    'University of Minnesota Extension',
    'Nguồn tham khảo về bệnh cháy lá sớm, triệu chứng vòng đồng tâm và cách giảm lây lan.',
    'vi',
    'https://images.unsplash.com/photo-1592841200221-a6898f307baa?auto=format&fit=crop&w=1200&q=80',
    'disease',
    '6 phút đọc',
    true,
    $body$
## Dấu hiệu cần để ý
Bệnh cháy lá sớm thường bắt đầu từ lá già gần gốc. Vết bệnh có màu nâu, đôi khi thấy các vòng đồng tâm giống bia bắn. Khi nặng, lá vàng quanh vết và rụng dần.

## Vì sao dễ lan
Bệnh phát triển mạnh khi lá ẩm lâu, cây quá rậm hoặc tàn dư cây bệnh còn lại trong vườn.

## Gợi ý chăm sóc
- Tỉa bỏ lá bệnh nặng và thu gom khỏi vườn.
- Tưới gốc, tránh tưới lên lá.
- Giữ khoảng cách cây thông thoáng.
- Luân canh, hạn chế trồng cà chua/khoai tây liên tục cùng vị trí.
$body$,
    array[
      'Vết bệnh vòng đồng tâm là dấu hiệu đáng chú ý.',
      'Lá ẩm lâu và vườn rậm làm bệnh dễ lan.',
      'Luân canh giúp giảm nguồn bệnh trong đất/tàn dư.'
    ],
    6
  ),
  (
    'tomato',
    'Quản lý bệnh hại trong vườn rau gia đình',
    'https://extension.umn.edu/planting-and-growing-guides/managing-plant-diseases-home-garden',
    'guide',
    'University of Minnesota Extension',
    'Tổng hợp nguyên tắc phòng bệnh: giống sạch bệnh, luân canh, tưới hợp lý, vệ sinh tàn dư.',
    'vi',
    'https://images.unsplash.com/photo-1591857177580-dc82b9ac4e1e?auto=format&fit=crop&w=1200&q=80',
    'disease',
    '7 phút đọc',
    false,
    $body$
## Tư duy phòng bệnh
Mục tiêu không phải là làm vườn hoàn toàn không có bệnh, mà là giữ bệnh ở mức cây vẫn sinh trưởng và cho thu hoạch tốt.

## Các bước quan trọng
- Mua giống/cây con khỏe, không có dấu hiệu bệnh.
- Tránh trồng cùng họ cây ở một vị trí trong nhiều vụ liên tiếp.
- Tưới và bố trí cây để lá khô nhanh sau mưa hoặc tưới.
- Kiểm tra cây định kỳ, xử lý sớm khi bệnh còn ít.
- Loại bỏ phần cây bệnh nặng và tàn dư cuối vụ.
$body$,
    array[
      'Phòng bệnh thường hiệu quả hơn chữa khi bệnh đã nặng.',
      'Độ ẩm kéo dài là yếu tố lớn với nhiều bệnh nấm/vi khuẩn.',
      'Vệ sinh tàn dư giúp giảm nguồn bệnh cho vụ sau.'
    ],
    7
  ),
  (
    'rice',
    'Danh mục bệnh phổ biến trên lúa',
    'https://www.knowledgebank.irri.org/training/fact-sheets/pest-management/diseases',
    'report',
    'IRRI Rice Knowledge Bank',
    'Tài liệu tham khảo về nhiều bệnh lúa như đạo ôn, bạc lá, khô vằn, lem lép hạt.',
    'vi',
    'https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1200&q=80',
    'disease',
    'Tài liệu',
    false,
    $body$
## Khi xem bệnh lúa
Bệnh lúa cần được đánh giá theo giai đoạn sinh trưởng, thời tiết gần đây và vị trí triệu chứng trên ruộng. Một vết bệnh trên lá chưa đủ để kết luận nếu thiếu bối cảnh.

## Cần ghi lại
- Giai đoạn lúa: mạ, đẻ nhánh, làm đòng, trổ hay chín.
- Vị trí bệnh: rìa ruộng, vùng trũng, vùng bón đạm nhiều.
- Thời tiết: mưa, ẩm cao, gió mạnh hoặc rét/nắng kéo dài.
- Tốc độ lan và tỷ lệ cây bị ảnh hưởng.
$body$,
    array[
      'Bệnh lúa phụ thuộc mạnh vào giai đoạn sinh trưởng.',
      'Mưa ẩm và bón đạm cao có thể làm một số bệnh nặng hơn.',
      'Nên ghi bối cảnh ruộng trước khi hỏi chuyên gia.'
    ],
    5
  ),
  (
    'rice',
    'Bạc lá lúa: nhận biết nhanh',
    'https://www.knowledgebank.irri.org/decision-tools/rice-doctor/rice-doctor-fact-sheets/item/bacterial-blight',
    'article',
    'IRRI Rice Doctor',
    'Nguồn tham khảo về bạc lá lúa, điều kiện phát sinh và dấu hiệu lá vàng khô.',
    'vi',
    'https://images.unsplash.com/photo-1537355251707-c65ffa1a83d4?auto=format&fit=crop&w=1200&q=80',
    'disease',
    '5 phút đọc',
    false,
    $body$
## Dấu hiệu thường gặp
Bạc lá lúa thường tạo vệt vàng hoặc khô từ mép lá, có thể lan dài theo phiến lá. Trên ruộng ẩm, gió mạnh và mưa kéo dài có thể làm bệnh lan nhanh hơn.

## Cần phân biệt
Thiếu dinh dưỡng, cháy nắng, ngộ độc hoặc tổn thương cơ học cũng có thể làm lá vàng/khô. Hãy quan sát nhiều cây và nhiều vị trí ruộng.

## Gợi ý an toàn
- Không bón thừa đạm khi ruộng đang có dấu hiệu bệnh.
- Giữ vệ sinh ruộng và quản lý tàn dư sau vụ.
- Hỏi cán bộ kỹ thuật địa phương nếu bệnh lan nhanh.
$body$,
    array[
      'Vệt vàng khô từ mép lá là dấu hiệu cần chú ý.',
      'Mưa gió có thể giúp vi khuẩn lan rộng.',
      'Không tăng đạm khi ruộng đang nghi bệnh.'
    ],
    5
  ),
  (
    'rose',
    'Cẩm nang chăm sóc hoa hồng',
    'https://www.rhs.org.uk/plants/roses/growing-guide',
    'guide',
    'RHS',
    'Hướng dẫn nền về ánh sáng, đất, tưới, cắt tỉa và các vấn đề thường gặp trên hoa hồng.',
    'vi',
    'https://images.unsplash.com/photo-1496062031456-07b8f162a322?auto=format&fit=crop&w=1200&q=80',
    'care',
    '6 phút đọc',
    false,
    $body$
## Điều kiện ưa thích
Hoa hồng cần nhiều nắng, đất giữ ẩm nhưng thoát nước tốt, và không gian thoáng để giảm nấm lá.

## Chăm sóc cơ bản
- Ưu tiên nắng sáng hoặc vị trí có nhiều ánh sáng.
- Tưới gốc, tránh làm ướt lá vào buổi tối.
- Cắt bỏ hoa tàn và lá bệnh.
- Bón phân vừa đủ theo giai đoạn sinh trưởng.

## Khi cây có bệnh
Không cắt quá nhiều lá cùng lúc. Nếu bệnh nặng, xử lý từng phần và theo dõi cây phục hồi.
$body$,
    array[
      'Hoa hồng cần nắng và không khí lưu thông.',
      'Tưới gốc buổi sáng giúp giảm nấm lá.',
      'Cắt tỉa hợp lý giúp cây khỏe và ra hoa tốt hơn.'
    ],
    6
  ),
  (
    'coffee',
    'Coffee Leaf Rust: tổng quan và quản lý',
    'https://www.mdpi.com/2073-4395/11/12/2590',
    'report',
    'MDPI Agronomy',
    'Bài tổng quan về bệnh gỉ sắt cà phê và các nhóm biện pháp quản lý.',
    'vi',
    'https://images.unsplash.com/photo-1447933601403-0c6688de566e?auto=format&fit=crop&w=1200&q=80',
    'disease',
    'Báo cáo',
    false,
    $body$
## Dấu hiệu chính
Gỉ sắt cà phê thường thấy ở mặt dưới lá dạng bột màu vàng cam. Lá bệnh có thể vàng, rụng sớm và làm cây suy yếu.

## Hướng quản lý
- Theo dõi tán lá, đặc biệt mùa ẩm.
- Cắt tỉa để tán thông thoáng.
- Quản lý dinh dưỡng để cây không bị suy.
- Dùng giống kháng và biện pháp phòng trừ theo khuyến cáo địa phương.

## Lưu ý
Không tự pha liều thuốc theo cảm tính. Với cây sản xuất, cần theo hướng dẫn của kỹ thuật viên hoặc khuyến nông.
$body$,
    array[
      'Bột vàng cam mặt dưới lá là dấu hiệu điển hình.',
      'Tán rậm và ẩm kéo dài làm bệnh dễ phát triển.',
      'Giống kháng và quản lý tán là hướng bền vững hơn chỉ phụ thuộc thuốc.'
    ],
    6
  ),
  (
    null,
    'Bón phân cho cây chậu: tránh quá tay',
    'https://extension.umn.edu/manage-soil-nutrients/how-manage-soil-and-nutrients-home-gardens',
    'guide',
    'University of Minnesota Extension',
    'Hướng dẫn đọc tình trạng cây, bón phân vừa đủ và tránh làm cây bệnh nặng hơn.',
    'vi',
    'https://images.unsplash.com/photo-1585314062604-1a357de8b000?auto=format&fit=crop&w=1200&q=80',
    'fertilizing',
    '5 phút đọc',
    false,
    $body$
## Vì sao cần cẩn thận
Cây đang bệnh hoặc đang stress không phải lúc nào cũng cần thêm phân. Bón quá nhiều có thể làm rễ yếu, cháy mép lá hoặc làm một số bệnh phát triển mạnh hơn.

## Cách làm an toàn
- Chỉ bón khi cây đang có điều kiện ánh sáng và nước ổn định.
- Dùng liều thấp trước, quan sát 7-10 ngày.
- Không bón phân đậm đặc lên đất đang khô.
- Với cây bệnh, ưu tiên ổn định tưới, ánh sáng và thông thoáng trước.
$body$,
    array[
      'Cây yếu không nhất thiết cần thêm phân ngay.',
      'Bón quá tay có thể làm rễ stress.',
      'Ổn định nước và ánh sáng trước khi tăng dinh dưỡng.'
    ],
    5
  ),
  (
    null,
    'Video gợi ý: phòng bệnh cây trong vườn nhà',
    'https://www.youtube.com/results?search_query=plant+disease+management+extension+home+garden',
    'youtube',
    'YouTube',
    'Danh sách video từ các nguồn extension/khuyến nông về phòng bệnh cây trong vườn nhà.',
    'vi',
    'https://images.unsplash.com/photo-1523348837708-15d4a09cfac2?auto=format&fit=crop&w=1200&q=80',
    'video',
    'Video',
    false,
    $body$
## Cách dùng video hiệu quả
Video hữu ích để quan sát thao tác thực tế như cắt tỉa, vệ sinh vườn, tưới gốc và kiểm tra mặt dưới lá.

## Khi xem video
- Ưu tiên nguồn từ trường đại học, extension hoặc khuyến nông.
- Không áp dụng thuốc/hóa chất nếu video không nói rõ cây trồng, liều và nhãn sản phẩm.
- So sánh với tình trạng cây thật của bạn trước khi làm theo.
$body$,
    array[
      'Video tốt cho thao tác thực hành.',
      'Không làm theo liều thuốc mơ hồ.',
      'Ưu tiên nguồn extension/khuyến nông.'
    ],
    3
  )
on conflict (url) do update set
  plant_key = excluded.plant_key,
  title = excluded.title,
  resource_type = excluded.resource_type,
  source_name = excluded.source_name,
  summary = excluded.summary,
  language = excluded.language,
  image_url = excluded.image_url,
  category = excluded.category,
  duration_label = excluded.duration_label,
  is_featured = excluded.is_featured,
  content_markdown = excluded.content_markdown,
  key_points = excluded.key_points,
  read_time_minutes = excluded.read_time_minutes;
