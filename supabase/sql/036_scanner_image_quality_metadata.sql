-- Store lightweight client-side scanner image quality metadata for review and retraining.

alter table public.ai_feedback_cases
    add column if not exists quality_json jsonb not null default '{}'::jsonb,
    add column if not exists client_flow_version text;

alter table public.report_cases
    add column if not exists quality_json jsonb not null default '{}'::jsonb,
    add column if not exists client_flow_version text;

comment on column public.ai_feedback_cases.quality_json is
    'Client-side scanner quality report, such as brightness, sharpness, min dimension, warnings, and upload bytes.';

comment on column public.ai_feedback_cases.client_flow_version is
    'Client scanner flow version that produced this feedback case.';

comment on column public.report_cases.quality_json is
    'Client-side scanner quality report attached to a wrong-diagnosis report, when available.';

comment on column public.report_cases.client_flow_version is
    'Client scanner flow version that produced this report, when available.';
