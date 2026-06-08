-- Allow authenticated users to delete only their own saved diagnosis history.
-- Run after history has a created_by column and owner-select policy.

alter table public.history enable row level security;

drop policy if exists "history_owner_delete" on public.history;
create policy "history_owner_delete"
on public.history
for delete
to authenticated
using (created_by = auth.uid());
