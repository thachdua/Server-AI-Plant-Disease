-- Harden profile password storage.
-- Passwords must only be handled by Supabase Auth, which hashes them one-way.
-- The public.profiles table may keep safe login metadata, but must not store
-- raw, reversible, or duplicate password/hash fields.

do $$
declare
  unsafe_column text;
begin
  foreach unsafe_column in array array[
    'password',
    'current_password',
    'raw_password',
    'plain_password',
    'password_hash'
  ]
  loop
    if exists (
      select 1
      from information_schema.columns
      where table_schema = 'public'
        and table_name = 'profiles'
        and column_name = unsafe_column
    ) then
      execute format('alter table public.profiles drop column %I', unsafe_column);
    end if;
  end loop;
end $$;

alter table public.profiles
  add column if not exists has_password_login boolean not null default false,
  add column if not exists password_enabled_at timestamptz;
