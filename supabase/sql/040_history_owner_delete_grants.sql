-- Restore table privileges needed by authenticated clients for owner-only history.
-- RLS still limits reads/deletes to rows where created_by = auth.uid().

alter table public.history enable row level security;

revoke select, insert, update, delete on table public.history from anon;
revoke insert, update on table public.history from authenticated;

grant select, delete on table public.history to authenticated;

drop policy if exists "history_select_own" on public.history;
create policy "history_select_own"
on public.history
for select
to authenticated
using (created_by = auth.uid());

drop policy if exists "history_owner_delete" on public.history;
create policy "history_owner_delete"
on public.history
for delete
to authenticated
using (created_by = auth.uid());
