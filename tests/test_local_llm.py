import base64
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image
from pypdf import PdfWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from local_llm import (
    LocalLlmError,
    _image_data_url,
    request_side_proposal,
    request_vision_side_proposal,
)


class LocalLlmTests(unittest.TestCase):
    def test_pdf_first_page_is_converted_to_bounded_jpeg(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "layout.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=2000, height=1000)
            with path.open("wb") as stream:
                writer.write(stream)

            data_url = _image_data_url(path)
            encoded = data_url.removeprefix("data:image/jpeg;base64,")
            with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
                self.assertEqual(image.format, "JPEG")
                self.assertLessEqual(max(image.size), 1600)

    @patch("local_llm._image_data_url", return_value="data:image/jpeg;base64,AA==")
    @patch("local_llm.requests.post")
    def test_valid_vision_proposal_is_accepted(self, post, image_data):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"status":"proposal","assignments":['
                            '{"order_id":"123","face":"a.tif","back":"b.tif"}],'
                            '"confidence":0.95,"reason":"логотип и контакты"}'
                        )
                    }
                }
            ]
        }
        post.return_value = response

        proposal = request_vision_side_proposal(
            "archive.zip",
            [Path("/tmp/a.tif"), Path("/tmp/b.tif")],
            ["123"],
        )

        self.assertEqual(proposal.assignments[0].face, "a.tif")
        self.assertEqual(proposal.confidence, 0.95)
        self.assertEqual(image_data.call_count, 2)
        user_content = post.call_args.kwargs["json"]["messages"][1]["content"]
        self.assertEqual(
            sum(item["type"] == "image_url" for item in user_content),
            2,
        )

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

    @patch("local_llm._image_data_url", return_value="data:image/jpeg;base64,AA==")
    @patch("local_llm.requests.post")
    def test_vision_uses_dedicated_model_setting(self, post, _image_data):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"status":"proposal","assignments":['
                            '{"order_id":"123","face":"a.tif","back":"b.tif"}],'
                            '"confidence":0.95,"reason":"визуальные признаки"}'
                        )
                    }
                }
            ]
        }
        post.return_value = response

        with patch.dict(
            "os.environ",
            {
                "LOCAL_LLM_MODEL": "gemma4:e2b",
                "LOCAL_LLM_VISION_MODEL": "moondream:latest",
            },
        ):
            request_vision_side_proposal(
                "archive.zip",
                [Path("/tmp/a.tif"), Path("/tmp/b.tif")],
                ["123"],
            )

        self.assertEqual(post.call_args.kwargs["json"]["model"], "moondream:latest")

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
