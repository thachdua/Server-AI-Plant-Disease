-- Durable asynchronous prediction jobs for iOS scanner.

create table if not exists public.prediction_jobs (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  created_by uuid null references auth.users(id) on delete set null,
  selected_plant text not null,
  image_url text null,
  status text not null default 'pending'
    check (status in ('pending', 'processing', 'done', 'failed')),
  predicted_plant text null,
  predicted_disease text null,
  confidence double precision null,
  confidence_text text null,
  result_image_url text null,
  error_message text null,
  source text not null default 'ios'
);

drop trigger if exists prediction_jobs_touch_updated_at on public.prediction_jobs;
create trigger prediction_jobs_touch_updated_at
before update on public.prediction_jobs
for each row execute function public.touch_updated_at();

create index if not exists prediction_jobs_created_by_created_idx
on public.prediction_jobs (created_by, created_at desc);

create index if not exists prediction_jobs_status_created_idx
on public.prediction_jobs (status, created_at);

alter table public.prediction_jobs enable row level security;

revoke insert, update, delete on table public.prediction_jobs from anon, authenticated;
grant select on table public.prediction_jobs to authenticated;

drop policy if exists "prediction_jobs_select_own" on public.prediction_jobs;
create policy "prediction_jobs_select_own"
on public.prediction_jobs
for select
to authenticated
using (created_by = auth.uid());
