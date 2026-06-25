-- Extra indexes for the V2 outbreak map filters and nearby warnings.
-- Run after 037_vn_admin_gis_outbreak_map.sql.

create index if not exists outbreak_cases_plant_reported_idx
on public.outbreak_cases (plant, reported_at desc);

create index if not exists outbreak_cases_source_reported_idx
on public.outbreak_cases (source, reported_at desc);

create index if not exists outbreak_cases_review_status_reported_idx
on public.outbreak_cases (review_status, reported_at desc);

create index if not exists outbreak_cases_confidence_idx
on public.outbreak_cases (confidence);

create index if not exists outbreak_cases_reported_severity_idx
on public.outbreak_cases (reported_at desc, severity);

create index if not exists outbreak_cases_geom_geography_gist_idx
on public.outbreak_cases using gist ((geom::geography))
where geom is not null;

comment on index public.outbreak_cases_geom_geography_gist_idx is
  'Supports radius-based nearby outbreak queries in meters.';
