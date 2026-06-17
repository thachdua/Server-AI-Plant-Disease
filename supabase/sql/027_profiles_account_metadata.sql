alter table public.profiles
  add column if not exists email text,
  add column if not exists auth_provider text not null default 'email',
  add column if not exists last_login_at timestamptz;

create index if not exists profiles_email_idx
on public.profiles (email)
where email is not null;

create index if not exists profiles_auth_provider_idx
on public.profiles (auth_provider);
