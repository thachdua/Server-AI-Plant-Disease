-- Smart Clinic follow-up support for expert verdicts and recovery tasks.

alter table public.plant_observations
  drop constraint if exists plant_observations_type_check;

alter table public.plant_observations
  add constraint plant_observations_type_check
  check (type in (
    'manual',
    'diagnosis',
    'water',
    'light',
    'task',
    'photo',
    'recovery',
    'expert_verdict'
  ));

alter table public.care_tasks
  drop constraint if exists care_tasks_source_check;

alter table public.care_tasks
  add constraint care_tasks_source_check
  check (source in (
    'manual',
    'diagnosis',
    'diagnosis_plan',
    'water',
    'light',
    'observation',
    'weather',
    'outbreak',
    'recovery',
    'template',
    'expert'
  ));
