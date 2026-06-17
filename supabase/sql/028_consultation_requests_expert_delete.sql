grant delete on table public.consultation_requests to authenticated;

drop policy if exists "consult_delete_expert" on public.consultation_requests;
create policy "consult_delete_expert"
on public.consultation_requests
for delete
to authenticated
using (public.is_expert(auth.uid()));
