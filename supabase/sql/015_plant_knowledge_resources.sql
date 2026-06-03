-- Curated plant knowledge, related plants, resources, and user bookmarks.

create table if not exists public.plant_knowledge (
  plant_key text primary key,
  display_name text not null,
  care_summary text,
  temperature_min_c double precision,
  temperature_max_c double precision,
  sunlight text,
  light_guide text,
  watering_guide text,
  related_plant_keys text[] not null default '{}',
  updated_at timestamptz not null default now()
);

create table if not exists public.plant_resources (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  plant_key text null references public.plant_knowledge(plant_key) on delete set null,
  title text not null,
  url text not null,
  resource_type text not null check (resource_type in ('article','report','youtube')),
  source_name text,
  summary text,
  language text not null default 'vi'
);

create table if not exists public.bookmarks (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  created_by uuid not null references auth.users(id) on delete cascade,
  resource_id uuid not null references public.plant_resources(id) on delete cascade,
  unique (created_by, resource_id)
);

create index if not exists plant_resources_plant_key_idx on public.plant_resources (plant_key, created_at desc);
create index if not exists bookmarks_created_by_idx on public.bookmarks (created_by, created_at desc);

alter table public.plant_knowledge enable row level security;
alter table public.plant_resources enable row level security;
alter table public.bookmarks enable row level security;

drop policy if exists "plant_knowledge_public_select" on public.plant_knowledge;
create policy "plant_knowledge_public_select" on public.plant_knowledge
for select to public using (true);

drop policy if exists "plant_resources_public_select" on public.plant_resources;
create policy "plant_resources_public_select" on public.plant_resources
for select to public using (true);

drop policy if exists "bookmarks_all_own" on public.bookmarks;
create policy "bookmarks_all_own" on public.bookmarks
for all to authenticated
using (created_by = auth.uid())
with check (created_by = auth.uid());

revoke insert, update, delete on table public.plant_knowledge from anon, authenticated;
revoke insert, update, delete on table public.plant_resources from anon, authenticated;

insert into public.plant_knowledge
  (plant_key, display_name, care_summary, temperature_min_c, temperature_max_c, sunlight, light_guide, watering_guide, related_plant_keys)
values
  ('tomato','Cà chua','Ưa nắng, cần đất thoát nước và theo dõi nấm bệnh khi ẩm cao.',18,30,'Nắng trực tiếp 6-8 giờ/ngày','Đặt nơi có nắng sáng; nếu cây vươn dài, thiếu sáng.','Tưới gốc khi mặt đất se khô, tránh ướt lá chiều tối.', array['bell_pepper','potato']),
  ('rice','Lúa','Cần nước ổn định, theo dõi đạo ôn và bạc lá sau mưa ẩm.',20,34,'Nắng đầy đủ','Ruộng cần đủ sáng và thông thoáng, hạn chế che bóng kéo dài.','Giữ mực nước theo giai đoạn sinh trưởng, tránh ngập sâu kéo dài.', array['corn','soybean']),
  ('rose','Hoa hồng','Cần nắng, thông thoáng và cắt tỉa lá bệnh thường xuyên.',16,30,'Nắng sáng 4-6 giờ/ngày','Thiếu sáng làm ít hoa và dễ nấm; ưu tiên nắng sáng.','Tưới gốc buổi sáng, hạn chế nước đọng trên lá.', array['strawberry','grape']),
  ('coffee','Cà phê','Ưa khí hậu ấm, cần che bóng vừa phải và thoát nước tốt.',18,28,'Nắng tán xạ/che bóng nhẹ','Ánh sáng quá gắt dễ stress; che bóng nhẹ ở giai đoạn non.','Tưới sâu theo chu kỳ khô hạn, tránh úng rễ.', array['orange','cassava'])
on conflict (plant_key) do update set
  display_name = excluded.display_name,
  care_summary = excluded.care_summary,
  temperature_min_c = excluded.temperature_min_c,
  temperature_max_c = excluded.temperature_max_c,
  sunlight = excluded.sunlight,
  light_guide = excluded.light_guide,
  watering_guide = excluded.watering_guide,
  related_plant_keys = excluded.related_plant_keys,
  updated_at = now();

insert into public.plant_resources (plant_key, title, url, resource_type, source_name, summary)
values
  ('tomato','Quản lý bệnh hại cà chua an toàn','https://extension.umn.edu/vegetables/growing-tomatoes-home-gardens','article','University of Minnesota Extension','Tổng quan chăm sóc và phòng bệnh cà chua.'),
  ('rice','Rice Knowledge Bank','https://www.knowledgebank.irri.org/','report','IRRI','Tài liệu kỹ thuật về canh tác và sâu bệnh lúa.'),
  ('rose','Rose care basics','https://www.rhs.org.uk/plants/roses/growing-guide','article','RHS','Hướng dẫn chăm sóc hoa hồng.'),
  ('coffee','Coffee leaf rust overview','https://www.cabi.org/isc/datasheet/26182','report','CABI','Thông tin tham khảo về bệnh gỉ sắt cà phê.'),
  ('tomato','Tomato care video search','https://www.youtube.com/results?search_query=tomato+disease+management+extension','youtube','YouTube','Danh sách video hướng dẫn chăm sóc và quản lý bệnh cà chua.')
on conflict do nothing;
