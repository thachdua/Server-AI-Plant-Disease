-- Personal history (owner-only) for `public.history`
-- Run in Supabase SQL editor.

alter table public.history enable row level security;

-- Remove overly-broad public read policies (if they exist)
drop policy if exists "history_public_select" on public.history;

-- Authenticated users can only read their own rows
drop policy if exists "history_select_own" on public.history;
create policy "history_select_own"
on public.history
for select
to authenticated
using (created_by = auth.uid());

