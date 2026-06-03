-- General app feedback: chatbot quality, catalog mistakes, bugs, and UX issues.

create table if not exists public.app_feedback (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  created_by uuid not null references auth.users(id) on delete cascade,
  category text not null check (category in ('chatbot','catalog','bug','advice','other')),
  message text not null,
  context_json jsonb not null default '{}'::jsonb,
  status text not null default 'pending' check (status in ('pending','reviewing','resolved','rejected')),
  reviewed_by uuid null references auth.users(id) on delete set null,
  admin_note text
);

create index if not exists app_feedback_created_at_idx on public.app_feedback (created_at desc);
create index if not exists app_feedback_created_by_idx on public.app_feedback (created_by);
create index if not exists app_feedback_status_idx on public.app_feedback (status);

drop trigger if exists app_feedback_touch_updated_at on public.app_feedback;
create trigger app_feedback_touch_updated_at
before update on public.app_feedback
for each row execute function public.touch_updated_at();

alter table public.app_feedback enable row level security;

drop policy if exists "app_feedback_select_own_or_expert" on public.app_feedback;
create policy "app_feedback_select_own_or_expert"
on public.app_feedback
for select
to authenticated
using (created_by = auth.uid() or public.is_expert(auth.uid()));

drop policy if exists "app_feedback_insert_own" on public.app_feedback;
create policy "app_feedback_insert_own"
on public.app_feedback
for insert
to authenticated
with check (created_by = auth.uid());

drop policy if exists "app_feedback_update_expert" on public.app_feedback;
create policy "app_feedback_update_expert"
on public.app_feedback
for update
to authenticated
using (public.is_expert(auth.uid()))
with check (public.is_expert(auth.uid()));

revoke delete on table public.app_feedback from anon, authenticated;
