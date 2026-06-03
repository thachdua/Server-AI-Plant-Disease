-- Extend expert consultation workflow with consent, questionnaire, SLA, and notification preferences.

alter table public.consultation_requests
  add column if not exists storage_consent boolean not null default true,
  add column if not exists profile_snapshot jsonb not null default '{}'::jsonb,
  add column if not exists questionnaire_json jsonb not null default '{}'::jsonb,
  add column if not exists expected_reply_at timestamptz,
  add column if not exists notify_email boolean not null default false,
  add column if not exists notify_local boolean not null default true,
  add column if not exists user_email text,
  add column if not exists diagnosis_context jsonb not null default '{}'::jsonb;

alter table public.consultation_requests
  drop constraint if exists consultation_requests_status_check;

alter table public.consultation_requests
  add constraint consultation_requests_status_check
  check (status in ('pending','triage','assigned','answered','closed','cancelled'));

create index if not exists consultation_requests_expected_reply_idx
on public.consultation_requests (expected_reply_at);
