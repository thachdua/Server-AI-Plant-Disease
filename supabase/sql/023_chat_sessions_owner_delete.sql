-- Allow users to delete their own saved chat sessions.
-- chat_messages are removed automatically through the existing ON DELETE CASCADE FK.

drop policy if exists "chat_sessions_delete_own" on public.chat_sessions;
create policy "chat_sessions_delete_own"
on public.chat_sessions
for delete
to authenticated
using (created_by = auth.uid());
