-- Vietnamese administrative GIS support for "Bản đồ vùng dịch".
--
-- Data source:
--   thanglequoc/vietnamese-provinces-database
--   MIT License, GIS dataset v4.0.0, WGS84/SRID 4326.
--
-- Import order after this migration:
--   1. postgresql/postgres_ImportData_vn_units.sql
--   2. postgresql/gis/postgresql_ImportData_gis_2026-06-20__12_32_01.sql
--
-- The large import files are intentionally not committed into this app repo.

create extension if not exists postgis;

create table if not exists public.administrative_regions (
  id integer primary key,
  name varchar(255) not null,
  name_en varchar(255) not null,
  code_name varchar(255),
  code_name_en varchar(255)
);

create table if not exists public.administrative_units (
  id integer primary key,
  full_name varchar(255),
  full_name_en varchar(255),
  short_name varchar(255),
  short_name_en varchar(255),
  code_name varchar(255),
  code_name_en varchar(255)
);

create table if not exists public.provinces (
  code varchar(20) primary key,
  name varchar(255) not null,
  name_en varchar(255),
  full_name varchar(255) not null,
  full_name_en varchar(255),
  code_name varchar(255),
  administrative_unit_id integer
);

do $$
begin
  if not exists (
    select 1 from pg_constraint where conname = 'provinces_administrative_unit_id_fkey'
  ) then
    alter table public.provinces
      add constraint provinces_administrative_unit_id_fkey
      foreign key (administrative_unit_id) references public.administrative_units(id);
  end if;
end $$;

create index if not exists idx_provinces_unit on public.provinces(administrative_unit_id);

create table if not exists public.wards (
  code varchar(20) primary key,
  name varchar(255) not null,
  name_en varchar(255),
  full_name varchar(255),
  full_name_en varchar(255),
  code_name varchar(255),
  province_code varchar(20),
  administrative_unit_id integer
);

do $$
begin
  if not exists (
    select 1 from pg_constraint where conname = 'wards_administrative_unit_id_fkey'
  ) then
    alter table public.wards
      add constraint wards_administrative_unit_id_fkey
      foreign key (administrative_unit_id) references public.administrative_units(id);
  end if;

  if not exists (
    select 1 from pg_constraint where conname = 'wards_province_code_fkey'
  ) then
    alter table public.wards
      add constraint wards_province_code_fkey
      foreign key (province_code) references public.provinces(code);
  end if;
end $$;

create index if not exists idx_wards_province on public.wards(province_code);
create index if not exists idx_wards_unit on public.wards(administrative_unit_id);

create table if not exists public.gis_provinces (
  id integer primary key generated always as identity,
  province_code varchar(20) not null,
  gis_server_id varchar(50),
  area_km2 numeric(12,5),
  bbox geometry(Polygon, 4326),
  geom geometry(MultiPolygon, 4326)
);

do $$
begin
  if not exists (
    select 1 from pg_constraint where conname = 'gis_provinces_province_code_fkey'
  ) then
    alter table public.gis_provinces
      add constraint gis_provinces_province_code_fkey
      foreign key (province_code) references public.provinces(code);
  end if;
end $$;

create index if not exists idx_gis_provinces_province_code on public.gis_provinces(province_code);
create index if not exists idx_gis_provinces_bbox on public.gis_provinces using gist (bbox);
create index if not exists idx_gis_provinces_geom on public.gis_provinces using gist (geom);

create table if not exists public.gis_wards (
  id integer primary key generated always as identity,
  ward_code varchar(20) not null,
  gis_server_id varchar(50),
  area_km2 numeric(12,5),
  bbox geometry(Polygon, 4326),
  geom geometry(MultiPolygon, 4326)
);

do $$
begin
  if not exists (
    select 1 from pg_constraint where conname = 'gis_wards_ward_code_fkey'
  ) then
    alter table public.gis_wards
      add constraint gis_wards_ward_code_fkey
      foreign key (ward_code) references public.wards(code);
  end if;
end $$;

create index if not exists idx_gis_wards_ward_code on public.gis_wards(ward_code);
create index if not exists idx_gis_wards_bbox on public.gis_wards using gist (bbox);
create index if not exists idx_gis_wards_geom on public.gis_wards using gist (geom);

alter table public.outbreak_cases
  add column if not exists geom geometry(Point, 4326),
  add column if not exists ward_id text,
  add column if not exists ward_name text;

create index if not exists outbreak_cases_geom_gist_idx
on public.outbreak_cases using gist (geom);

create index if not exists outbreak_cases_admin_reported_idx
on public.outbreak_cases (province_id, ward_id, reported_at desc);

create or replace function public.resolve_admin_for_point(
  p_lng double precision,
  p_lat double precision
)
returns table (
  province_id text,
  province_name text,
  ward_id text,
  ward_name text,
  location_label text
)
language sql
stable
as $$
  with pt as (
    select st_setsrid(st_makepoint(p_lng, p_lat), 4326) as geom
  ),
  ward_match as (
    select
      p.code::text as province_id,
      p.name::text as province_name,
      w.code::text as ward_id,
      coalesce(w.full_name, w.name)::text as ward_name
    from pt
    join public.gis_wards gw
      on gw.bbox && pt.geom
     and st_covers(gw.geom, pt.geom)
    join public.wards w on w.code = gw.ward_code
    join public.provinces p on p.code = w.province_code
    order by gw.area_km2 nulls last, w.code
    limit 1
  ),
  province_match as (
    select
      p.code::text as province_id,
      p.name::text as province_name
    from pt
    join public.gis_provinces gp
      on gp.bbox && pt.geom
     and st_covers(gp.geom, pt.geom)
    join public.provinces p on p.code = gp.province_code
    order by gp.area_km2 nulls last, p.code
    limit 1
  )
  select
    coalesce(wm.province_id, pm.province_id) as province_id,
    coalesce(wm.province_name, pm.province_name) as province_name,
    wm.ward_id,
    wm.ward_name,
    case
      when wm.ward_name is not null and coalesce(wm.province_name, pm.province_name) is not null
        then wm.ward_name || ', ' || coalesce(wm.province_name, pm.province_name) || ', Vietnam'
      when coalesce(wm.province_name, pm.province_name) is not null
        then coalesce(wm.province_name, pm.province_name) || ', Vietnam'
      else null
    end as location_label
  from province_match pm
  full join ward_match wm on true;
$$;

create or replace function public.outbreak_cases_set_admin_fields()
returns trigger
language plpgsql
as $$
declare
  admin record;
begin
  if new.lng is not null and new.lat is not null then
    new.geom := st_setsrid(st_makepoint(new.lng, new.lat), 4326);

    select * into admin
    from public.resolve_admin_for_point(new.lng, new.lat)
    limit 1;

    if admin.province_id is not null then
      new.province_id := admin.province_id;
      new.province_name := admin.province_name;
    end if;

    if admin.ward_id is not null then
      new.ward_id := admin.ward_id;
      new.ward_name := admin.ward_name;
    end if;

    if coalesce(btrim(new.location_label), '') = '' and admin.location_label is not null then
      new.location_label := admin.location_label;
    end if;
  end if;

  return new;
end;
$$;

drop trigger if exists outbreak_cases_set_admin_fields_trigger on public.outbreak_cases;
create trigger outbreak_cases_set_admin_fields_trigger
before insert or update of lat, lng, location_label
on public.outbreak_cases
for each row
execute function public.outbreak_cases_set_admin_fields();

update public.outbreak_cases
set geom = st_setsrid(st_makepoint(lng, lat), 4326)
where geom is null;

with resolved as (
  select
    oc.id,
    admin.province_id,
    admin.province_name,
    admin.ward_id,
    admin.ward_name,
    admin.location_label
  from public.outbreak_cases oc
  cross join lateral public.resolve_admin_for_point(oc.lng, oc.lat) admin
  where oc.lat is not null
    and oc.lng is not null
)
update public.outbreak_cases oc
set
  province_id = coalesce(resolved.province_id, oc.province_id),
  province_name = coalesce(resolved.province_name, oc.province_name),
  ward_id = coalesce(resolved.ward_id, oc.ward_id),
  ward_name = coalesce(resolved.ward_name, oc.ward_name),
  location_label = case
    when coalesce(btrim(oc.location_label), '') = '' then coalesce(resolved.location_label, oc.location_label)
    else oc.location_label
  end
from resolved
where resolved.id = oc.id;
