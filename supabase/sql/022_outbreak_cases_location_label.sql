-- Store a readable location label for outbreak cases created from diagnosis history.
-- This keeps map details understandable even when users do not recognize coordinates.

alter table public.outbreak_cases
  add column if not exists location_label text;
