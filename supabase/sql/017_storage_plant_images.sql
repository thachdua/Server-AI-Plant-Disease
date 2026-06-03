-- Storage bucket for prediction/history/retrain images.
-- Required by backend uploads to bucket `plant-images`.

insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values (
  'plant-images',
  'plant-images',
  true,
  5242880,
  array['image/jpeg', 'image/png', 'image/webp']
)
on conflict (id) do update
set
  public = excluded.public,
  file_size_limit = excluded.file_size_limit,
  allowed_mime_types = excluded.allowed_mime_types;

drop policy if exists "plant_images_public_read" on storage.objects;
create policy "plant_images_public_read"
on storage.objects
for select
to anon, authenticated
using (bucket_id = 'plant-images');
