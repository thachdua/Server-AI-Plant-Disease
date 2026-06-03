import importlib
import os
import unittest
from unittest.mock import patch

import deploy.config as config


class ConfigTests(unittest.TestCase):
    def reload_config(self, env):
        with patch.dict(os.environ, env, clear=True):
            return importlib.reload(config)

    def tearDown(self):
        importlib.reload(config)

    def test_prefers_explicit_service_role_key(self):
        cfg = self.reload_config(
            {
                "SUPABASE_URL": "https://example.supabase.co",
                "SUPABASE_SERVICE_ROLE_KEY": "service-role",
                "SUPABASE_KEY": "legacy",
                "DB_USER": "u",
                "DB_PASSWORD": "p",
            }
        )

        self.assertEqual(cfg.SUPABASE_KEY, "service-role")
        self.assertEqual(cfg.SUPABASE_KEY_SOURCE, "SUPABASE_SERVICE_ROLE_KEY")
        self.assertEqual(cfg.config_status()["warnings"], [])

    def test_warns_for_publishable_backend_key(self):
        cfg = self.reload_config(
            {
                "SUPABASE_URL": "https://example.supabase.co",
                "SUPABASE_KEY": "sb_publishable_abc",
                "DB_USER": "u",
                "DB_PASSWORD": "p",
            }
        )

        status = cfg.config_status()
        self.assertEqual(cfg.SUPABASE_KEY_SOURCE, "SUPABASE_KEY")
        self.assertIn("backend_is_using_supabase_publishable_key", status["warnings"])


if __name__ == "__main__":
    unittest.main()
