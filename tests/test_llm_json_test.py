import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from llm_json_test import expected_layout_count


class LlmJsonTestTests(unittest.TestCase):
    def test_two_sided_jpeg_order_requires_two_files(self):
        self.assertEqual(expected_layout_count(["100"], "4-4"), 2)
        self.assertEqual(expected_layout_count(["100", "101"], "4-4"), 4)

    def test_ten_two_sided_orders_require_twenty_files(self):
        order_ids = [str(value) for value in range(10)]
        self.assertEqual(expected_layout_count(order_ids, "4-4"), 20)

    def test_one_sided_orders_require_one_file_each(self):
        self.assertEqual(expected_layout_count(["100", "101"], "4-0"), 2)


if __name__ == "__main__":
    unittest.main()
