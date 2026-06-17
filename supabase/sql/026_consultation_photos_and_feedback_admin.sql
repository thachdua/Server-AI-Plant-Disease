-- Expert consultation photos and admin feedback management.

alter table public.consultation_requests
  add column if not exists photo_urls text[] not null default '{}',
  add column if not exists primary_photo_url text,
  add column if not exists diagnostic_flow_version text not null default 'expert_wizard_v1';

grant select, update, delete on table public.app_feedback to authenticated;

drop policy if exists "app_feedback_delete_expert" on public.app_feedback;
create policy "app_feedback_delete_expert"
on public.app_feedback
for delete
to authenticated
using (public.is_expert(auth.uid()));
