-- Care task event log for agenda stats, streaks, and lightweight activity history.
create table if not exists public.care_task_events (
    id uuid primary key default gen_random_uuid(),
    created_at timestamptz not null default now(),
    created_by uuid not null references auth.users(id) on delete cascade,
    care_task_id uuid references public.care_tasks(id) on delete set null,
    user_plant_id uuid references public.user_plants(id) on delete set null,
    event_type text not null check (
        event_type in (
            'created',
            'completed',
            'reopened',
            'rescheduled',
            'snoozed',
            'skipped',
            'deleted',
            'repeat_created'
        )
    ),
    task_title text,
    category text,
    source text,
    priority integer,
    previous_due_at timestamptz,
    next_due_at timestamptz,
    metadata_json jsonb not null default '{}'::jsonb
);

create index if not exists care_task_events_created_by_created_at_idx
    on public.care_task_events (created_by, created_at desc);

create index if not exists care_task_events_care_task_id_idx
    on public.care_task_events (care_task_id);

create index if not exists care_task_events_user_plant_created_at_idx
    on public.care_task_events (user_plant_id, created_at desc);

alter table public.care_task_events enable row level security;

drop policy if exists "care_task_events_select_own" on public.care_task_events;
create policy "care_task_events_select_own"
    on public.care_task_events for select
    using (created_by = auth.uid());

drop policy if exists "care_task_events_insert_own" on public.care_task_events;
create policy "care_task_events_insert_own"
    on public.care_task_events for insert
    with check (created_by = auth.uid());

drop policy if exists "care_task_events_update_own" on public.care_task_events;
create policy "care_task_events_update_own"
    on public.care_task_events for update
    using (created_by = auth.uid())
    with check (created_by = auth.uid());

drop policy if exists "care_task_events_delete_own" on public.care_task_events;
create policy "care_task_events_delete_own"
    on public.care_task_events for delete
    using (created_by = auth.uid());
