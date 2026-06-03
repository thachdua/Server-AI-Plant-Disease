-- Avatar upload setup (Supabase Storage)
-- 1) Create bucket "avatars" in Storage UI (recommended: public = true for easy display)
-- 2) Run this SQL to allow authenticated users to upload/update their own avatar objects.

-- Enable RLS on storage objects (usually enabled by default, but safe)
alter table storage.objects enable row level security;

-- Helper: only allow user to write within folder "<uid>/..."
-- We check object name starts with "{uid}/"

drop policy if exists "avatars_public_read" on storage.objects;
create policy "avatars_public_read"
on storage.objects
for select
to public
using (bucket_id = 'avatars');

drop policy if exists "avatars_user_insert_own" on storage.objects;
create policy "avatars_user_insert_own"
on storage.objects
for insert
to authenticated
with check (
  bucket_id = 'avatars'
  and (name like (auth.uid()::text || '/%'))
);

drop policy if exists "avatars_user_update_own" on storage.objects;
create policy "avatars_user_update_own"
on storage.objects
for update
to authenticated
using (
  bucket_id = 'avatars'
  and (name like (auth.uid()::text || '/%'))
)
with check (
  bucket_id = 'avatars'
  and (name like (auth.uid()::text || '/%'))
);

