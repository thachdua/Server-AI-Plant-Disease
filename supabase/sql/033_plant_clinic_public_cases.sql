-- Plant Clinic: expert verdicts, care plans, and anonymous public case library.

create table if not exists public.clinic_public_cases (
    id uuid primary key default gen_random_uuid(),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    published_at timestamptz not null default now(),
    published_by uuid references auth.users(id) on delete set null,
    source text not null check (source in ('report', 'consultation')),
    source_report_id uuid references public.report_cases(id) on delete set null,
    source_consultation_id uuid references public.consultation_requests(id) on delete set null,
    plant text,
    disease text,
    severity text not null default 'medium' check (severity in ('low', 'medium', 'high')),
    summary text,
    care_steps_json jsonb not null default '[]'::jsonb,
    image_urls text[] not null default '{}',
    tags text[] not null default '{}',
    helpful_count integer not null default 0
);

drop trigger if exists clinic_public_cases_touch_updated_at on public.clinic_public_cases;
create trigger clinic_public_cases_touch_updated_at
before update on public.clinic_public_cases
for each row execute function public.touch_updated_at();

create index if not exists clinic_public_cases_published_idx
on public.clinic_public_cases (published_at desc);

create index if not exists clinic_public_cases_plant_disease_idx
on public.clinic_public_cases (plant, disease);

alter table public.clinic_public_cases enable row level security;

drop policy if exists "clinic_public_cases_read_published" on public.clinic_public_cases;
create policy "clinic_public_cases_read_published"
on public.clinic_public_cases
for select
using (published_at is not null);

drop policy if exists "clinic_public_cases_insert_expert" on public.clinic_public_cases;
create policy "clinic_public_cases_insert_expert"
on public.clinic_public_cases
for insert
to authenticated
with check (public.is_expert(auth.uid()));

drop policy if exists "clinic_public_cases_update_expert" on public.clinic_public_cases;
create policy "clinic_public_cases_update_expert"
on public.clinic_public_cases
for update
to authenticated
using (public.is_expert(auth.uid()))
with check (public.is_expert(auth.uid()));

drop policy if exists "clinic_public_cases_delete_expert" on public.clinic_public_cases;
create policy "clinic_public_cases_delete_expert"
on public.clinic_public_cases
for delete
to authenticated
using (public.is_expert(auth.uid()));

grant select on table public.clinic_public_cases to anon, authenticated;
grant insert, update, delete on table public.clinic_public_cases to authenticated;

alter table public.report_cases
    add column if not exists user_plant_id uuid references public.user_plants(id) on delete set null,
    add column if not exists photo_urls text[] not null default '{}',
    add column if not exists symptom_snapshot jsonb not null default '{}'::jsonb,
    add column if not exists expert_severity text check (expert_severity in ('low', 'medium', 'high')),
    add column if not exists expert_care_plan_json jsonb not null default '[]'::jsonb,
    add column if not exists expert_follow_up_days integer,
    add column if not exists expert_reviewed_at timestamptz,
    add column if not exists expert_edited_at timestamptz,
    add column if not exists publish_to_community boolean not null default false,
    add column if not exists public_case_id uuid references public.clinic_public_cases(id) on delete set null;

alter table public.consultation_requests
    add column if not exists user_plant_id uuid references public.user_plants(id) on delete set null,
    add column if not exists expert_final_disease text,
    add column if not exists expert_severity text check (expert_severity in ('low', 'medium', 'high')),
    add column if not exists expert_care_plan_json jsonb not null default '[]'::jsonb,
    add column if not exists expert_follow_up_days integer,
    add column if not exists publish_to_community boolean not null default false,
    add column if not exists public_case_id uuid references public.clinic_public_cases(id) on delete set null;

create index if not exists report_cases_user_plant_idx
on public.report_cases (user_plant_id, created_at desc);

create index if not exists consultation_requests_user_plant_idx
on public.consultation_requests (user_plant_id, created_at desc);

alter table public.care_tasks
    drop constraint if exists care_tasks_source_check;

alter table public.care_tasks
    add constraint care_tasks_source_check
    check (source in ('manual','diagnosis','diagnosis_plan','water','light','observation','weather','outbreak','recovery','template','expert'));
