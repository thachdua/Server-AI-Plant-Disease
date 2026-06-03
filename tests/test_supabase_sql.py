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
        }
        for filename, marker in expected.items():
            sql = (ROOT / f"supabase/sql/{filename}").read_text()
            self.assertIn(marker, sql)
            if filename != "016_consultation_workflow.sql":
                self.assertIn("enable row level security", sql.lower())


if __name__ == "__main__":
    unittest.main()
