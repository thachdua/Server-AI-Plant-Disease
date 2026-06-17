-- Optional image attachments for saved chatbot conversations.
-- Existing RLS on chat_messages stays unchanged because ownership still flows through session_id/created_by.

alter table public.chat_messages
  add column if not exists image_url text,
  add column if not exists attachment_type text,
  add column if not exists diagnosis_json jsonb;

create index if not exists chat_messages_attachment_type_idx
on public.chat_messages (attachment_type)
where attachment_type is not null;
