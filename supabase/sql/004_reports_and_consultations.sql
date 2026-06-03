-- Reports (user feedback) + Consultations (Q&A / booking) with RBAC via profiles.role
-- Run in Supabase SQL editor.

-- Helper: check expert role
create or replace function public.is_expert(uid uuid)
returns boolean
language sql
stable
security definer
set search_path = public
as $$
  select exists (
    select 1 from public.profiles p
    where p.id = uid and p.role = 'expert'
  );
$$;

-- =========================================================
-- 1) Report cases: user reports incorrect AI result
-- =========================================================

create table if not exists public.report_cases (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),

  created_by uuid null references auth.users(id) on delete set null,
  plant text,
  predicted_disease text,
  confidence double precision,
  image_url text,

  -- user feedback
  user_correct_disease text,
  user_note text,

  -- workflow
  status text not null default 'pending' check (status in ('pending','reviewing','resolved','rejected')),

  -- expert review
  reviewed_by uuid null references auth.users(id) on delete set null,
  expert_final_disease text,
  expert_note text
);

-- updated_at trigger (reuse if exists)
create or replace function public.touch_updated_at()
returns trigger as $$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

drop trigger if exists report_cases_touch_updated_at on public.report_cases;
create trigger report_cases_touch_updated_at
before update on public.report_cases
for each row execute function public.touch_updated_at();

alter table public.report_cases enable row level security;

-- SELECT:
-- - user can read their own reports
-- - experts can read all
drop policy if exists "report_cases_select_own_or_expert" on public.report_cases;
create policy "report_cases_select_own_or_expert"
on public.report_cases
for select
to authenticated
using (
  created_by = auth.uid()
  or public.is_expert(auth.uid())
);

-- INSERT: any authenticated user
drop policy if exists "report_cases_insert_authenticated" on public.report_cases;
create policy "report_cases_insert_authenticated"
on public.report_cases
for insert
to authenticated
with check (created_by = auth.uid());

-- UPDATE:
-- - experts can update any (for review)
-- - regular users only INSERT reports; they cannot PATCH workflow/expert fields from a custom client
drop policy if exists "report_cases_update_owner_or_expert" on public.report_cases;
drop policy if exists "report_cases_update_expert" on public.report_cases;
create policy "report_cases_update_expert"
on public.report_cases
for update
to authenticated
using (public.is_expert(auth.uid()))
with check (public.is_expert(auth.uid()));

-- =========================================================
-- 2) Consultations: user request + expert reply / schedule
-- =========================================================

create table if not exists public.consultation_requests (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),

  created_by uuid null references auth.users(id) on delete set null,
  title text,
  question text,
  image_url text,

  -- scheduling
  requested_time timestamptz,
  contact_phone text,

  -- workflow
  status text not null default 'pending' check (status in ('pending','assigned','answered','closed')),

  assigned_expert uuid null references auth.users(id) on delete set null,
  expert_reply text,
  expert_reply_at timestamptz
);

drop trigger if exists consultation_requests_touch_updated_at on public.consultation_requests;
create trigger consultation_requests_touch_updated_at
before update on public.consultation_requests
for each row execute function public.touch_updated_at();

alter table public.consultation_requests enable row level security;

-- SELECT:
-- - user reads own
-- - expert reads all (or assigned)
drop policy if exists "consult_select_own_or_expert" on public.consultation_requests;
create policy "consult_select_own_or_expert"
on public.consultation_requests
for select
to authenticated
using (
  created_by = auth.uid()
  or public.is_expert(auth.uid())
);

-- INSERT: authenticated user
drop policy if exists "consult_insert_authenticated" on public.consultation_requests;
create policy "consult_insert_authenticated"
on public.consultation_requests
for insert
to authenticated
with check (created_by = auth.uid());

-- UPDATE:
-- - experts can update any (reply/assign/close)
-- - regular users only INSERT requests; they cannot PATCH expert reply/status fields from a custom client
drop policy if exists "consult_update_owner_or_expert" on public.consultation_requests;
drop policy if exists "consult_update_expert" on public.consultation_requests;
create policy "consult_update_expert"
on public.consultation_requests
for update
to authenticated
using (public.is_expert(auth.uid()))
with check (public.is_expert(auth.uid()));
