-- Link diagnosis history with outbreak cases and store safe display fields.
-- Confidence is only used as data-quality metadata, not as outbreak severity.

alter table public.outbreak_cases
  add column if not exists plant text,
  add column if not exists confidence double precision,
  add column if not exists image_url text,
  add column if not exists history_id text,
  add column if not exists created_by uuid null references auth.users(id) on delete set null,
  add column if not exists review_status text not null default 'auto_accepted',
  add column if not exists province_id text,
  add column if not exists province_name text;

create index if not exists outbreak_cases_province_reported_idx
on public.outbreak_cases (province_id, reported_at desc);

create index if not exists outbreak_cases_disease_reported_idx
on public.outbreak_cases (disease, reported_at desc);

create index if not exists outbreak_cases_history_id_idx
on public.outbreak_cases (history_id);
