-- AI feedback dataset for low-confidence predictions and user-confirmed wrong diagnoses.

create table if not exists public.ai_feedback_cases (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  created_by uuid null references auth.users(id) on delete set null,
  plant text,
  predicted_disease text,
  confidence double precision,
  image_url text,
  source text not null default 'manual' check (source in ('predict','report','manual')),
  reason text not null default 'low_confidence' check (reason in ('low_confidence','wrong_diagnosis','unclear_image','other')),
  user_correct_disease text,
  user_note text,
  review_status text not null default 'pending' check (review_status in ('pending','reviewing','accepted','rejected')),
  reviewed_by uuid null references auth.users(id) on delete set null,
  expert_label text,
  expert_note text
);

create index if not exists ai_feedback_cases_created_at_idx on public.ai_feedback_cases (created_at desc);
create index if not exists ai_feedback_cases_created_by_idx on public.ai_feedback_cases (created_by);
create index if not exists ai_feedback_cases_review_status_idx on public.ai_feedback_cases (review_status);

drop trigger if exists ai_feedback_cases_touch_updated_at on public.ai_feedback_cases;
create trigger ai_feedback_cases_touch_updated_at
before update on public.ai_feedback_cases
for each row execute function public.touch_updated_at();

alter table public.ai_feedback_cases enable row level security;

drop policy if exists "ai_feedback_select_own_or_expert" on public.ai_feedback_cases;
create policy "ai_feedback_select_own_or_expert"
on public.ai_feedback_cases
for select
to authenticated
using (created_by = auth.uid() or public.is_expert(auth.uid()));

drop policy if exists "ai_feedback_insert_own" on public.ai_feedback_cases;
create policy "ai_feedback_insert_own"
on public.ai_feedback_cases
for insert
to authenticated
with check (created_by = auth.uid());

drop policy if exists "ai_feedback_update_expert" on public.ai_feedback_cases;
create policy "ai_feedback_update_expert"
on public.ai_feedback_cases
for update
to authenticated
using (public.is_expert(auth.uid()))
with check (public.is_expert(auth.uid()));

revoke delete on table public.ai_feedback_cases from anon, authenticated;
