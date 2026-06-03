-- Public-read + authenticated-write policy for `history`
-- Apply in Supabase SQL editor.

-- 1) Optional: add created_by for attribution
alter table if exists public.history
add column if not exists created_by uuid null references auth.users(id);

-- 2) Enable RLS
alter table public.history enable row level security;

-- 3) Public SELECT (everyone can read)
drop policy if exists "history_public_select" on public.history;
create policy "history_public_select"
on public.history
for select
to public
using (true);

-- 4) INSERT only for authenticated users (helps prevent spam)
drop policy if exists "history_authenticated_insert" on public.history;
create policy "history_authenticated_insert"
on public.history
for insert
to authenticated
with check (true);

-- 5) Optional: allow authenticated users to update/delete only their own rows
-- (requires you to set created_by on insert)
-- drop policy if exists "history_owner_update" on public.history;
-- create policy "history_owner_update"
-- on public.history
-- for update
-- to authenticated
-- using (created_by = auth.uid())
-- with check (created_by = auth.uid());
--
-- drop policy if exists "history_owner_delete" on public.history;
-- create policy "history_owner_delete"
-- on public.history
-- for delete
-- to authenticated
-- using (created_by = auth.uid());

