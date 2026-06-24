from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SupabaseSQLTests(unittest.TestCase):
    def test_profile_policies_prevent_self_role_escalation(self):
        sql = (ROOT / "supabase/sql/003_profiles_roles.sql").read_text()

        self.assertIn("and role = 'user'", sql)
        self.assertIn("and role = public.profile_role(auth.uid())", sql)
        self.assertIn("security definer", sql.lower())

    def test_profile_hardening_migration_exists(self):
        sql = (ROOT / "supabase/sql/009_harden_profile_roles.sql").read_text()

        self.assertIn("drop policy if exists \"profiles_insert_own\"", sql)
        self.assertIn("drop policy if exists \"profiles_update_own\"", sql)
        self.assertIn("and role = 'user'", sql)
        self.assertIn("and role = public.profile_role(auth.uid())", sql)

    def test_workflow_updates_are_expert_only(self):
        sql = (ROOT / "supabase/sql/004_reports_and_consultations.sql").read_text()

        self.assertIn('create policy "report_cases_update_expert"', sql)
        self.assertIn('create policy "consult_update_expert"', sql)
        self.assertIn("using (public.is_expert(auth.uid()))", sql)
        self.assertNotIn('create policy "report_cases_update_owner_or_expert"', sql)
        self.assertNotIn('create policy "consult_update_owner_or_expert"', sql)

    def test_workflow_hardening_migration_exists(self):
        sql = (ROOT / "supabase/sql/010_harden_workflow_updates.sql").read_text()

        self.assertIn('drop policy if exists "report_cases_update_owner_or_expert"', sql)
        self.assertIn('drop policy if exists "consult_update_owner_or_expert"', sql)
        self.assertIn('create policy "report_cases_update_expert"', sql)
        self.assertIn('create policy "consult_update_expert"', sql)

    def test_new_feature_migrations_exist_with_rls(self):
        expected = {
            "011_ai_feedback_cases.sql": "ai_feedback_cases",
            "012_app_feedback.sql": "app_feedback",
            "013_chat_consent_sessions.sql": "chat_sessions",
            "014_care_plants_tasks.sql": "care_tasks",
            "015_plant_knowledge_resources.sql": "plant_resources",
            "016_consultation_workflow.sql": "expected_reply_at",
            "032_care_task_events.sql": "care_task_events",
            "033_plant_clinic_public_cases.sql": "clinic_public_cases",
            "035_clinic_smart_care_followup.sql": "expert_verdict",
            "036_scanner_image_quality_metadata.sql": "quality_json",
        }
        for filename, marker in expected.items():
            sql = (ROOT / f"supabase/sql/{filename}").read_text()
            self.assertIn(marker, sql)
            if filename not in {"016_consultation_workflow.sql", "035_clinic_smart_care_followup.sql", "036_scanner_image_quality_metadata.sql"}:
                self.assertIn("enable row level security", sql.lower())

    def test_scanner_image_quality_metadata_columns(self):
        sql = (ROOT / "supabase/sql/036_scanner_image_quality_metadata.sql").read_text().lower()

        self.assertIn("public.ai_feedback_cases", sql)
        self.assertIn("public.report_cases", sql)
        self.assertIn("quality_json jsonb", sql)
        self.assertIn("client_flow_version text", sql)

    def test_clinic_smart_care_followup_constraints_include_expert_sources(self):
        sql = (ROOT / "supabase/sql/035_clinic_smart_care_followup.sql").read_text().lower()

        self.assertIn("plant_observations_type_check", sql)
        self.assertIn("'expert_verdict'", sql)
        self.assertIn("care_tasks_source_check", sql)
        self.assertIn("'expert'", sql)
        self.assertIn("'recovery'", sql)

    def test_clinic_public_cases_do_not_store_private_contact_fields(self):
        sql = (ROOT / "supabase/sql/033_plant_clinic_public_cases.sql").read_text().lower()
        table_sql = sql.split(");", 1)[0]

        self.assertNotIn("user_email", table_sql)
        self.assertNotIn("contact_phone", table_sql)
        self.assertNotIn("profile_snapshot", table_sql)
        self.assertIn("clinic_public_cases_read_published", sql)
        self.assertIn("public.is_expert(auth.uid())", sql)

    def test_profiles_password_storage_hardening_drops_unsafe_columns(self):
        sql = (ROOT / "supabase/sql/034_profiles_password_storage_hardening.sql").read_text().lower()

        for column in ["password", "current_password", "raw_password", "plain_password", "password_hash"]:
            self.assertIn(f"'{column}'", sql)
        self.assertIn("drop column", sql)
        self.assertIn("supabase auth", sql)
        self.assertIn("has_password_login", sql)
        self.assertIn("password_enabled_at", sql)


if __name__ == "__main__":
    unittest.main()
