-- Smart care task metadata and quick-action support.

alter table public.care_tasks
add column if not exists priority integer not null default 3
check (priority >= 1 and priority <= 5);

alter table public.care_tasks
add column if not exists source text not null default 'manual'
check (source in ('manual','diagnosis','diagnosis_plan','water','light','observation','weather','outbreak','recovery','template'));

alter table public.care_tasks
add column if not exists skipped_at timestamptz;

create index if not exists care_tasks_open_priority_due_idx
on public.care_tasks (created_by, completed_at, priority, due_at);

create index if not exists care_tasks_user_plant_open_idx
on public.care_tasks (user_plant_id, completed_at, due_at);
