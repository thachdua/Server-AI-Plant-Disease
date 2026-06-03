-- Harden report/consultation workflow updates for existing projects.
-- Run this if you already applied 004_reports_and_consultations.sql before this change.
--
-- Regular users should create their own report/consultation rows and read their own rows.
-- Only experts should update workflow/status/review/reply fields.

drop policy if exists "report_cases_update_owner_or_expert" on public.report_cases;
drop policy if exists "report_cases_update_expert" on public.report_cases;
create policy "report_cases_update_expert"
on public.report_cases
for update
to authenticated
using (public.is_expert(auth.uid()))
with check (public.is_expert(auth.uid()));

drop policy if exists "consult_update_owner_or_expert" on public.consultation_requests;
drop policy if exists "consult_update_expert" on public.consultation_requests;
create policy "consult_update_expert"
on public.consultation_requests
for update
to authenticated
using (public.is_expert(auth.uid()))
with check (public.is_expert(auth.uid()));
