import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from website_parser import OrderDataError, fetch_suborders, parse_suborders_response


class WebsiteParserTests(unittest.TestCase):
    def test_duplicates_are_removed_without_changing_api_order(self):
        self.assertEqual(
            parse_suborders_response("30, 10, 30, 20, 10"),
            ["30", "10", "20"],
        )

    def test_invalid_item_is_rejected(self):
        with self.assertRaises(OrderDataError):
            parse_suborders_response("10, unexpected, 20")

    @patch("website_parser.time.sleep")
    @patch("website_parser.requests.post")
    def test_temporary_network_error_is_retried(self, post, sleep):
        response = Mock(text="20,10")
        response.raise_for_status.return_value = None
        post.side_effect = [requests.ConnectionError("temporary"), response]

        result = fetch_suborders("123", api_key="key", retries=2, backoff=0.1)

        self.assertEqual(result, ["20", "10"])
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once_with(0.1)


if __name__ == "__main__":
    unittest.main()
