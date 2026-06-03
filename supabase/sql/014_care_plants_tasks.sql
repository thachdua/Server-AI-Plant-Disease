-- Plant collection, care plans, and reminder tasks.

create table if not exists public.user_plants (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  created_by uuid not null references auth.users(id) on delete cascade,
  plant_key text,
  display_name text not null,
  note text,
  watering_method text,
  pot_diameter_cm double precision,
  plant_height_cm double precision,
  water_ml double precision,
  cup_count double precision,
  temperature_min_c double precision,
  temperature_max_c double precision,
  sunlight text
);

create table if not exists public.care_plans (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  created_by uuid not null references auth.users(id) on delete cascade,
  user_plant_id uuid null references public.user_plants(id) on delete set null,
  plant text,
  disease text,
  confidence double precision,
  summary_vi text,
  safety_note text,
  source text not null default 'diagnosis'
);

create table if not exists public.care_tasks (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  created_by uuid not null references auth.users(id) on delete cascade,
  user_plant_id uuid null references public.user_plants(id) on delete cascade,
  care_plan_id uuid null references public.care_plans(id) on delete set null,
  title text not null,
  detail text,
  category text not null default 'inspection',
  due_at timestamptz,
  repeat_rule text not null default 'none' check (repeat_rule in ('none','daily','weekly','monthly','yearly')),
  notification_enabled boolean not null default false,
  completed_at timestamptz
);

create index if not exists user_plants_created_by_idx on public.user_plants (created_by, updated_at desc);
create index if not exists care_plans_created_by_idx on public.care_plans (created_by, created_at desc);
create index if not exists care_tasks_created_by_due_idx on public.care_tasks (created_by, due_at);

drop trigger if exists user_plants_touch_updated_at on public.user_plants;
create trigger user_plants_touch_updated_at before update on public.user_plants
for each row execute function public.touch_updated_at();

drop trigger if exists care_plans_touch_updated_at on public.care_plans;
create trigger care_plans_touch_updated_at before update on public.care_plans
for each row execute function public.touch_updated_at();

drop trigger if exists care_tasks_touch_updated_at on public.care_tasks;
create trigger care_tasks_touch_updated_at before update on public.care_tasks
for each row execute function public.touch_updated_at();

alter table public.user_plants enable row level security;
alter table public.care_plans enable row level security;
alter table public.care_tasks enable row level security;

drop policy if exists "user_plants_all_own" on public.user_plants;
create policy "user_plants_all_own" on public.user_plants
for all to authenticated
using (created_by = auth.uid())
with check (created_by = auth.uid());

drop policy if exists "care_plans_all_own" on public.care_plans;
create policy "care_plans_all_own" on public.care_plans
for all to authenticated
using (created_by = auth.uid())
with check (created_by = auth.uid());

drop policy if exists "care_tasks_all_own" on public.care_tasks;
create policy "care_tasks_all_own" on public.care_tasks
for all to authenticated
using (created_by = auth.uid())
with check (created_by = auth.uid());
