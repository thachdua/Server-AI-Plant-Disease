-- Add media metadata for Discover resources and seed image-backed demo content.

alter table public.plant_resources
  add column if not exists image_url text,
  add column if not exists category text not null default 'article',
  add column if not exists duration_label text,
  add column if not exists is_featured boolean not null default false;

create unique index if not exists plant_resources_url_unique_idx
on public.plant_resources (url);

update public.plant_resources
set
  category = case
    when resource_type = 'youtube' then 'youtube'
    when resource_type = 'report' then 'report'
    else 'guide'
  end,
  image_url = coalesce(
    image_url,
    case
      when plant_key = 'tomato' then 'https://images.unsplash.com/photo-1591857177580-dc82b9ac4e1e?auto=format&fit=crop&w=1200&q=80'
      when plant_key = 'rice' then 'https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1200&q=80'
      when plant_key = 'rose' then 'https://images.unsplash.com/photo-1496062031456-07b8f162a322?auto=format&fit=crop&w=1200&q=80'
      when plant_key = 'coffee' then 'https://images.unsplash.com/photo-1447933601403-0c6688de566e?auto=format&fit=crop&w=1200&q=80'
      else 'https://images.unsplash.com/photo-1416879595882-3373a0480b5b?auto=format&fit=crop&w=1200&q=80'
    end
  ),
  duration_label = coalesce(duration_label, case when resource_type = 'youtube' then 'Video' else '5 phút đọc' end),
  is_featured = coalesce(is_featured, false);

insert into public.plant_resources
  (plant_key, title, url, resource_type, source_name, summary, language, image_url, category, duration_label, is_featured)
values
  (
    'tomato',
    'Hướng dẫn chăm sóc cà chua khi thời tiết ẩm',
    'https://extension.umn.edu/vegetables/growing-tomatoes-home-gardens',
    'article',
    'University of Minnesota Extension',
    'Các nguyên tắc tưới, ánh sáng, đất và phòng bệnh thường gặp cho cà chua.',
    'vi',
    'https://images.unsplash.com/photo-1591857177580-dc82b9ac4e1e?auto=format&fit=crop&w=1200&q=80',
    'guide',
    '6 phút đọc',
    true
  ),
  (
    'rice',
    'Rice Knowledge Bank: kỹ thuật và sâu bệnh lúa',
    'https://www.knowledgebank.irri.org/',
    'report',
    'IRRI',
    'Kho tài liệu kỹ thuật về canh tác, sâu bệnh và quản lý ruộng lúa.',
    'vi',
    'https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1200&q=80',
    'report',
    'Tài liệu',
    false
  ),
  (
    'rose',
    'Cách chăm sóc hoa hồng khỏe và ít nấm bệnh',
    'https://www.rhs.org.uk/plants/roses/growing-guide',
    'article',
    'RHS',
    'Gợi ý ánh sáng, tưới nước, cắt tỉa và theo dõi bệnh phổ biến trên hoa hồng.',
    'vi',
    'https://images.unsplash.com/photo-1496062031456-07b8f162a322?auto=format&fit=crop&w=1200&q=80',
    'guide',
    '5 phút đọc',
    false
  ),
  (
    'coffee',
    'Báo cáo tham khảo về bệnh gỉ sắt cà phê',
    'https://www.cabi.org/isc/datasheet/26182',
    'report',
    'CABI',
    'Thông tin nền về tác nhân, triệu chứng và quản lý bệnh gỉ sắt trên cà phê.',
    'vi',
    'https://images.unsplash.com/photo-1447933601403-0c6688de566e?auto=format&fit=crop&w=1200&q=80',
    'report',
    'Báo cáo',
    false
  ),
  (
    'tomato',
    'Video: quản lý bệnh cà chua và chăm sóc sau chẩn đoán',
    'https://www.youtube.com/results?search_query=tomato+disease+management+extension',
    'youtube',
    'YouTube',
    'Danh sách video hướng dẫn chăm sóc và quản lý bệnh cà chua từ các nguồn khuyến nông.',
    'vi',
    'https://images.unsplash.com/photo-1416879595882-3373a0480b5b?auto=format&fit=crop&w=1200&q=80',
    'youtube',
    'Video',
    true
  ),
  (
    null,
    'Hướng dẫn bón phân an toàn cho cây trong chậu',
    'https://extension.umn.edu/manage-soil-nutrients/how-manage-soil-and-nutrients-home-gardens',
    'article',
    'University of Minnesota Extension',
    'Các nguyên tắc cơ bản về dinh dưỡng, đất và bón phân cho vườn nhà.',
    'vi',
    'https://images.unsplash.com/photo-1466692476868-aef1dfb1e735?auto=format&fit=crop&w=1200&q=80',
    'guide',
    '7 phút đọc',
    false
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
  is_featured = excluded.is_featured;
