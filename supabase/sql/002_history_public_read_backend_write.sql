-- Public-read + backend-write-only policy for `history`
-- Use this if you want ONLY your Render backend (service role) to insert,
-- and block direct client inserts completely.
--
-- Important:
-- - Service role bypasses RLS.
-- - Keep service role key on backend only. Never ship it to iOS.

alter table public.history enable row level security;

-- Public SELECT (everyone can read)
drop policy if exists "history_public_select" on public.history;
create policy "history_public_select"
on public.history
for select
to public
using (true);

-- Block inserts/updates/deletes from anon/authenticated by default
revoke insert, update, delete on table public.history from anon, authenticated;

