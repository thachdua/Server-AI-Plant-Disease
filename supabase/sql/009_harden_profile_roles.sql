-- Harden profile role policies for existing Supabase projects.
-- Run this if you already applied 003_profiles_roles.sql before this hardening change.
--
-- Problem fixed:
-- The original owner-update policy allowed a malicious client to PATCH
-- public.profiles.role from "user" to "expert" for their own profile.

create or replace function public.profile_role(uid uuid)
returns text
language sql
stable
security definer
set search_path = public
as $$
  select p.role from public.profiles p where p.id = uid
$$;

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
