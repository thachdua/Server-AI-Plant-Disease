-- Store the diagnosis context currently being tracked for each saved plant.
-- Notes remain user-authored notes; diagnosis metadata lives in structured fields.

alter table public.user_plants
    add column if not exists current_disease text,
    add column if not exists current_confidence double precision,
    add column if not exists current_history_id text,
    add column if not exists diagnosed_at timestamptz;

create index if not exists user_plants_current_history_id_idx
    on public.user_plants (current_history_id);

comment on column public.user_plants.current_disease is
    'Current disease being tracked for this plant, usually copied from a saved diagnosis history row.';

comment on column public.user_plants.current_confidence is
    'AI confidence percentage for current_disease, if the plant was added from diagnosis history.';

comment on column public.user_plants.current_history_id is
    'Source history row id when this plant was created from a saved diagnosis.';

comment on column public.user_plants.diagnosed_at is
    'Timestamp of the diagnosis that populated current_disease/current_confidence.';
