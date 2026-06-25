-- Demo outbreak cases for map QA and app demos.
-- Safe to re-run: it replaces only rows tagged with source = 'demo_seed'.

delete from public.outbreak_cases
where source = 'demo_seed';

with disease_catalog(idx, plant, disease, base_severity) as (
  values
    (1, 'Tomato', 'Cháy lá muộn (Late Blight)', 5),
    (2, 'Tomato', 'Đốm lá vi khuẩn (Bacterial Spot)', 4),
    (3, 'Pepper', 'Cháy lá vi khuẩn (Bacterial Blight)', 4),
    (4, 'Apple', 'Ghẻ táo (Apple Scab)', 3),
    (5, 'Grape', 'Sương mai (Downy Mildew)', 4),
    (6, 'Corn', 'Rỉ sắt (Rust)', 3),
    (7, 'Potato', 'Cháy lá sớm (Early Blight)', 4),
    (8, 'Rice', 'Đạo ôn (Rice Blast)', 5),
    (9, 'Cucumber', 'Phấn trắng (Powdery Mildew)', 3),
    (10, 'Mango', 'Thán thư (Anthracnose)', 4)
),
ranked_wards as (
  select
    w.code as ward_code,
    w.province_code,
    gw.geom,
    row_number() over (
      partition by w.province_code
      order by md5(w.code || ':' || w.province_code)
    ) as ward_rank,
    dense_rank() over (order by w.province_code) as province_rank
  from public.wards w
  join public.gis_wards gw on gw.ward_code = w.code
),
selected_wards as (
  select *
  from ranked_wards
  where ward_rank <= 18
),
demo_rows as (
  select
    st_pointonsurface(sw.geom) as pt,
    dc.plant,
    dc.disease,
    greatest(1, least(5, dc.base_severity + ((sw.ward_rank + sw.province_rank) % 3) - 1)) as severity,
    68 + ((sw.ward_rank * 7 + sw.province_rank * 3) % 29) as confidence,
    now() - (((sw.ward_rank * 5 + sw.province_rank * 11) % 90)::text || ' days')::interval as reported_at,
    sw.province_code,
    sw.ward_code
  from selected_wards sw
  join disease_catalog dc
    on dc.idx = ((sw.ward_rank + sw.province_rank) % 10) + 1
)
insert into public.outbreak_cases (
  lat,
  lng,
  plant,
  disease,
  confidence,
  severity,
  reported_at,
  note,
  source,
  review_status
)
select
  st_y(pt),
  st_x(pt),
  plant,
  disease,
  confidence,
  severity,
  reported_at,
  'Ca demo seed cho bản đồ vùng dịch V2; có thể xoá bằng source=demo_seed.',
  'demo_seed',
  'auto_accepted'
from demo_rows;

select count(*) as demo_seed_count
from public.outbreak_cases
where source = 'demo_seed';
