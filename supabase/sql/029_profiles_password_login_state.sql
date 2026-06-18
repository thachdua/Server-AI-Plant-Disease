alter table public.profiles
  add column if not exists has_password_login boolean not null default false,
  add column if not exists password_enabled_at timestamptz;

update public.profiles
set
  has_password_login = true,
  password_enabled_at = coalesce(password_enabled_at, last_login_at, updated_at, created_at, now())
where auth_provider = 'email'
  and has_password_login = false;

create index if not exists profiles_has_password_login_idx
on public.profiles (has_password_login);
