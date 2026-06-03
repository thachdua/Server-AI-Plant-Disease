-- LLM advice cache (structured JSON) for weather + diagnosis
-- Run in Supabase SQL Editor.

create table if not exists public.llm_advice_cache (
  id uuid primary key default gen_random_uuid(),
  kind text not null check (kind in ('weather', 'diagnosis')),
  input_hash text not null,
  lang text not null default 'vi',
  model text,
  content_json jsonb not null,
  content_text text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create unique index if not exists llm_advice_cache_kind_hash_lang_uidx
  on public.llm_advice_cache (kind, input_hash, lang);

create index if not exists llm_advice_cache_kind_idx
  on public.llm_advice_cache (kind);

create index if not exists llm_advice_cache_updated_at_idx
  on public.llm_advice_cache (updated_at desc);

-- Auto-update updated_at on update
create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists llm_advice_cache_set_updated_at on public.llm_advice_cache;
create trigger llm_advice_cache_set_updated_at
before update on public.llm_advice_cache
for each row execute function public.set_updated_at();

-- RLS: allow read (public) but block direct writes from clients.
alter table public.llm_advice_cache enable row level security;

drop policy if exists "llm_advice_cache_public_read" on public.llm_advice_cache;
create policy "llm_advice_cache_public_read"
on public.llm_advice_cache
for select
using (true);

drop policy if exists "llm_advice_cache_no_write" on public.llm_advice_cache;
create policy "llm_advice_cache_no_write"
on public.llm_advice_cache
for all
using (false)
with check (false);

