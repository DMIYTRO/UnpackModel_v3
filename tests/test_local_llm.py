import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from local_llm import LocalLlmError, request_side_proposal


class LocalLlmTests(unittest.TestCase):
    @patch("local_llm.requests.post")
    def test_valid_proposal_is_accepted(self, post):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"status":"proposal","assignments":['
                            '{"order_id":"123","face":"a.tif","back":"b.tif"}],'
                            '"confidence":0.8,"reason":"явные признаки"}'
                        )
                    }
                }
            ]
        }
        post.return_value = response

        proposal = request_side_proposal("archive.zip", ["a.tif", "b.tif"], ["123"])

        self.assertEqual(proposal.assignments[0].face, "a.tif")
        self.assertEqual(proposal.confidence, 0.8)

    @patch("local_llm.requests.post")
    def test_invented_filename_is_rejected(self, post):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"status":"proposal","assignments":['
                            '{"order_id":"123","face":"a.tif","back":"invented.tif"}],'
                            '"confidence":0.9,"reason":"guess"}'
                        )
                    }
                }
            ]
        }
        post.return_value = response

        with self.assertRaises(LocalLlmError):
            request_side_proposal("archive.zip", ["a.tif", "b.tif"], ["123"])

    @patch("local_llm.requests.post")
    def test_markdown_wrapped_ambiguous_json_is_treated_as_refusal(self, post):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '```json\n{"status":"ambiguous","assignments":[],'
                            '"confidence":0,"reason":"недостаточно данных"}\n```'
                        )
                    }
                }
            ]
        }
        post.return_value = response

        with self.assertRaisesRegex(LocalLlmError, "недостаточно данных"):
            request_side_proposal("archive.zip", ["a.tif", "b.tif"], ["123"])


if __name__ == "__main__":
    unittest.main()
