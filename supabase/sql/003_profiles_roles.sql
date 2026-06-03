-- Profiles + roles for RBAC (User vs Expert)
-- Run in Supabase SQL editor.

-- 1) Table
create table if not exists public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  role text not null default 'user' check (role in ('user', 'expert')),
  display_name text,
  avatar_url text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- Helper used by RLS policies to compare the current stored role.
-- SECURITY DEFINER prevents policy recursion when reading profiles inside policies.
create or replace function public.profile_role(uid uuid)
returns text
language sql
stable
security definer
set search_path = public
as $$
  select p.role from public.profiles p where p.id = uid
$$;

-- 2) Keep updated_at fresh
create or replace function public.set_updated_at()
returns trigger as $$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

drop trigger if exists profiles_set_updated_at on public.profiles;
create trigger profiles_set_updated_at
before update on public.profiles
for each row execute function public.set_updated_at();

-- 3) RLS: user can read/write their own profile
alter table public.profiles enable row level security;

drop policy if exists "profiles_select_own" on public.profiles;
create policy "profiles_select_own"
on public.profiles
for select
to authenticated
using (id = auth.uid());

drop policy if exists "profiles_insert_own" on public.profiles;
create policy "profiles_insert_own"
on public.profiles
for insert
to authenticated
with check (
  id = auth.uid()
  and role = 'user'
);

drop policy if exists "profiles_update_own" on public.profiles;
create policy "profiles_update_own"
on public.profiles
for update
to authenticated
using (id = auth.uid())
with check (
  id = auth.uid()
  and role = public.profile_role(auth.uid())
);

-- 4) Optional: auto-create profile row on signup
-- Requires Supabase "Auth Hook" / trigger on auth.users (not supported in all setups).
-- Many projects create profile lazily from client instead.
