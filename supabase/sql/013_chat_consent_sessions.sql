-- Chat storage is opt-in. The app only writes these tables when the user consents.

create table if not exists public.chat_sessions (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  created_by uuid not null references auth.users(id) on delete cascade,
  title text,
  mode text not null default 'agriculture',
  storage_consent boolean not null default true,
  source text not null default 'chat'
);

create table if not exists public.chat_messages (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  session_id uuid not null references public.chat_sessions(id) on delete cascade,
  created_by uuid not null references auth.users(id) on delete cascade,
  role text not null check (role in ('user','assistant')),
  text text not null
);

create index if not exists chat_sessions_created_by_idx on public.chat_sessions (created_by, updated_at desc);
create index if not exists chat_messages_session_idx on public.chat_messages (session_id, created_at);

drop trigger if exists chat_sessions_touch_updated_at on public.chat_sessions;
create trigger chat_sessions_touch_updated_at
before update on public.chat_sessions
for each row execute function public.touch_updated_at();

alter table public.chat_sessions enable row level security;
alter table public.chat_messages enable row level security;

drop policy if exists "chat_sessions_select_own" on public.chat_sessions;
create policy "chat_sessions_select_own"
on public.chat_sessions
for select
to authenticated
using (created_by = auth.uid());

drop policy if exists "chat_sessions_insert_own" on public.chat_sessions;
create policy "chat_sessions_insert_own"
on public.chat_sessions
for insert
to authenticated
with check (created_by = auth.uid() and storage_consent = true);

drop policy if exists "chat_sessions_update_own" on public.chat_sessions;
create policy "chat_sessions_update_own"
on public.chat_sessions
for update
to authenticated
using (created_by = auth.uid())
with check (created_by = auth.uid());

drop policy if exists "chat_messages_select_own" on public.chat_messages;
create policy "chat_messages_select_own"
on public.chat_messages
for select
to authenticated
using (
  created_by = auth.uid()
  and exists (
    select 1 from public.chat_sessions s
    where s.id = chat_messages.session_id and s.created_by = auth.uid()
  )
);

drop policy if exists "chat_messages_insert_own" on public.chat_messages;
create policy "chat_messages_insert_own"
on public.chat_messages
for insert
to authenticated
with check (
  created_by = auth.uid()
  and exists (
    select 1 from public.chat_sessions s
    where s.id = chat_messages.session_id and s.created_by = auth.uid() and s.storage_consent = true
  )
);

revoke update, delete on table public.chat_messages from anon, authenticated;
