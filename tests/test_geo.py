import unittest

from deploy.geo import compute_level


class GeoLevelTests(unittest.TestCase):
    def test_compute_level_uses_case_count_not_severity(self):
        self.assertEqual(compute_level(0, 5), 0)
        self.assertEqual(compute_level(1, 5), 1)
        self.assertEqual(compute_level(2, 5), 1)
        self.assertEqual(compute_level(3, 1), 2)
        self.assertEqual(compute_level(5, 1), 2)
        self.assertEqual(compute_level(6, 1), 3)
        self.assertEqual(compute_level(10, 1), 3)
        self.assertEqual(compute_level(11, 1), 4)


if __name__ == "__main__":
    unittest.main()
