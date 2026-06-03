import unittest
from unittest.mock import patch

from psycopg2 import errors

from deploy import database


class FakeCursor:
    def __init__(self, rows=None, side_effects=None):
        self.rows = rows or []
        self.side_effects = list(side_effects or [])
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query, params=None):
        self.executed.append((query, params))
        if self.side_effects:
            effect = self.side_effects.pop(0)
            if effect is not None:
                raise effect

    def fetchone(self):
        return self.rows[0] if self.rows else None


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True


class DatabaseTests(unittest.TestCase):
    def test_save_to_db_commits_and_closes(self):
        cursor = FakeCursor()
        conn = FakeConnection(cursor)

        with patch("deploy.database._db_connect", return_value=conn):
            with patch("deploy.database.print"):
                database.save_to_db("Tomato", "Late blight", 92.0, "https://img", "user-1")

        self.assertEqual(conn.commits, 1)
        self.assertEqual(conn.rollbacks, 0)
        self.assertTrue(conn.closed)
        self.assertEqual(len(cursor.executed), 1)
        self.assertEqual(cursor.executed[0][1][-1], "user-1")

    def test_save_to_db_falls_back_when_created_by_missing(self):
        first_cursor = FakeCursor(side_effects=[errors.UndefinedColumn("created_by")])
        second_cursor = FakeCursor()
        first_conn = FakeConnection(first_cursor)
        second_conn = FakeConnection(second_cursor)

        with patch("deploy.database._db_connect", side_effect=[first_conn, second_conn]):
            with patch("deploy.database.print"):
                database.save_to_db("Tomato", "Late blight", 92.0, "https://img", "user-1")

        self.assertEqual(first_conn.rollbacks, 1)
        self.assertTrue(first_conn.closed)
        self.assertEqual(second_conn.commits, 1)
        self.assertTrue(second_conn.closed)
        self.assertEqual(len(second_cursor.executed[0][1]), 5)

    def test_llm_cache_get_closes_connection(self):
        row = ({"summary_vi": "ok"}, "ok", "model", "date")
        cursor = FakeCursor(rows=[row])
        conn = FakeConnection(cursor)

        with patch("deploy.database._db_connect", return_value=conn):
            result = database.llm_cache_get("diagnosis", "hash", "vi")

        self.assertTrue(conn.closed)
        self.assertEqual(result["model"], "model")
        self.assertEqual(result["content_text"], "ok")


if __name__ == "__main__":
    unittest.main()
