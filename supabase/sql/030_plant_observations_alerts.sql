-- Plant health observations and proactive alerts.

create table if not exists public.plant_observations (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  created_by uuid not null references auth.users(id) on delete cascade,
  user_plant_id uuid not null references public.user_plants(id) on delete cascade,
  type text not null check (type in ('manual','diagnosis','water','light','task','photo','recovery')),
  note text,
  image_url text,
  measured_lux double precision,
  water_ml double precision,
  health_score integer check (health_score is null or (health_score >= 0 and health_score <= 100)),
  disease_status text check (disease_status is null or disease_status in ('healthy','better','same','worse','unknown')),
  metadata_json jsonb not null default '{}'::jsonb
);

create index if not exists plant_observations_user_plant_created_idx
on public.plant_observations (user_plant_id, created_at desc);

create index if not exists plant_observations_created_by_created_idx
on public.plant_observations (created_by, created_at desc);

alter table public.plant_observations enable row level security;

drop policy if exists "plant_observations_all_own" on public.plant_observations;
create policy "plant_observations_all_own" on public.plant_observations
for all to authenticated
using (created_by = auth.uid())
with check (created_by = auth.uid());

create table if not exists public.plant_alerts (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  resolved_at timestamptz,
  snoozed_until timestamptz,
  created_by uuid not null references auth.users(id) on delete cascade,
  user_plant_id uuid null references public.user_plants(id) on delete cascade,
  kind text not null check (kind in ('weather','outbreak','care','water','light','disease','recovery')),
  severity text not null default 'info' check (severity in ('info','warning','critical')),
  title text not null,
  message text not null,
  action_title text,
  metadata_json jsonb not null default '{}'::jsonb
);

create index if not exists plant_alerts_created_by_created_idx
on public.plant_alerts (created_by, created_at desc);

create index if not exists plant_alerts_user_plant_created_idx
on public.plant_alerts (user_plant_id, created_at desc);

create index if not exists plant_alerts_open_idx
on public.plant_alerts (created_by, resolved_at, created_at desc);

alter table public.plant_alerts enable row level security;

drop policy if exists "plant_alerts_all_own" on public.plant_alerts;
create policy "plant_alerts_all_own" on public.plant_alerts
for all to authenticated
using (created_by = auth.uid())
with check (created_by = auth.uid());

drop policy if exists "plant_images_authenticated_insert_own_folder" on storage.objects;
create policy "plant_images_authenticated_insert_own_folder"
on storage.objects
for insert
to authenticated
with check (
  bucket_id = 'plant-images'
  and (storage.foldername(name))[1] = auth.uid()::text
);

drop policy if exists "plant_images_authenticated_update_own_folder" on storage.objects;
create policy "plant_images_authenticated_update_own_folder"
on storage.objects
for update
to authenticated
using (
  bucket_id = 'plant-images'
  and (storage.foldername(name))[1] = auth.uid()::text
)
with check (
  bucket_id = 'plant-images'
  and (storage.foldername(name))[1] = auth.uid()::text
);
