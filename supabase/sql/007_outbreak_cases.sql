-- Outbreak cases (curated) for "Bản đồ vùng dịch"
-- Run in Supabase SQL editor.

create table if not exists public.outbreak_cases (
  id uuid primary key default gen_random_uuid(),
  lat double precision not null,
  lng double precision not null,
  disease text not null,
  severity int not null default 3 check (severity >= 1 and severity <= 5),
  reported_at timestamptz not null default now(),
  note text,
  source text,
  created_at timestamptz not null default now()
);

create index if not exists outbreak_cases_reported_at_idx on public.outbreak_cases (reported_at desc);
create index if not exists outbreak_cases_disease_idx on public.outbreak_cases (disease);

alter table public.outbreak_cases enable row level security;

-- Public can read (Guest allowed)
drop policy if exists "outbreak_cases_public_select" on public.outbreak_cases;
create policy "outbreak_cases_public_select"
on public.outbreak_cases
for select
to public
using (true);

-- Block writes from anon/authenticated (curated by admin/service role)
revoke insert, update, delete on table public.outbreak_cases from anon, authenticated;

