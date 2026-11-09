import base64
import io
import json
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


def _side_response(side, confidence, *, has_logo=False, has_contacts=False):
    """Ответ модели на вопрос про одну сторону макета."""
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "choices": [{"message": {"content": json.dumps({
            "has_logo_or_title": has_logo,
            "has_contacts": has_contacts,
            "side": side,
            "confidence": confidence,
        })}}]
    }
    return response

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
        # Лицо — второй файл: модель не должна зависеть от порядка показа.
        post.side_effect = [
            _side_response("back", 0.9, has_contacts=True),
            _side_response("face", 0.95, has_logo=True),
        ]

        proposal = request_vision_side_proposal(
            "archive.zip",
            [Path("/tmp/a.tif"), Path("/tmp/b.tif")],
            ["123"],
        )

        self.assertEqual(proposal.assignments[0].order_id, "123")
        self.assertEqual(proposal.assignments[0].face, "b.tif")
        self.assertEqual(proposal.assignments[0].back, "a.tif")
        self.assertEqual(proposal.confidence, 0.9)
        self.assertIn("a.tif: оборот", proposal.reason)
        self.assertEqual(image_data.call_count, 2)
        self.assertEqual(post.call_count, 2)
        for call in post.call_args_list:
            body = call.kwargs["json"]
            user_content = body["messages"][1]["content"]
            self.assertEqual(
                sum(item["type"] == "image_url" for item in user_content), 1
            )
            response_format = body["response_format"]
            self.assertEqual(response_format["type"], "json_schema")
            self.assertTrue(response_format["json_schema"]["strict"])
            side = response_format["json_schema"]["schema"]["properties"]["side"]
            self.assertEqual(side["enum"], ["face", "back", "unknown"])

    @patch("local_llm._image_data_url", return_value="data:image/jpeg;base64,AA==")
    @patch("local_llm.requests.post")
    def test_vision_rejects_pair_with_same_side(self, post, _image_data):
        post.side_effect = [
            _side_response("back", 0.95, has_contacts=True),
            _side_response("back", 0.9, has_contacts=True),
        ]

        with self.assertRaisesRegex(LocalLlmError, "не различаются"):
            request_vision_side_proposal(
                "archive.zip",
                [Path("/tmp/a.tif"), Path("/tmp/b.tif")],
                ["123"],
            )

    @patch("local_llm._image_data_url", return_value="data:image/jpeg;base64,AA==")
    @patch("local_llm.requests.post")
    def test_vision_rejects_unknown_side(self, post, _image_data):
        post.side_effect = [
            _side_response("face", 0.95, has_logo=True),
            _side_response("unknown", 0.3),
        ]

        with self.assertRaisesRegex(LocalLlmError, "неясно"):
            request_vision_side_proposal(
                "archive.zip",
                [Path("/tmp/a.tif"), Path("/tmp/b.tif")],
                ["123"],
            )

    @patch("local_llm._image_data_url", return_value="data:image/jpeg;base64,AA==")
    @patch("local_llm.requests.post")
    def test_vision_keeps_cyrillic_names_out_of_grammar(self, post, _image_data):
        post.side_effect = [
            _side_response("face", 0.9, has_logo=True),
            _side_response("back", 0.9, has_contacts=True),
        ]
        names = ["візитка_лице.pdf", "візитка_зворот.pdf"]

        proposal = request_vision_side_proposal(
            "archive.zip",
            [Path("/tmp/a.pdf"), Path("/tmp/b.pdf")],
            ["123"],
            filenames=names,
        )

        self.assertEqual(proposal.assignments[0].face, names[0])
        self.assertEqual(proposal.assignments[0].back, names[1])
        for call in post.call_args_list:
            schema = json.dumps(call.kwargs["json"]["response_format"])
            self.assertTrue(schema.isascii())

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
    def test_text_prompt_defines_numbered_side_convention(self, post):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{"message": {"content": (
                '{"status":"proposal","assignments":['
                '{"order_id":"123","face":"1_1.jpeg","back":"1_2.jpeg"}],'
                '"confidence":1,"reason":"правило имени"}'
            )}}]
        }
        post.return_value = response

        request_side_proposal(
            "archive.zip", ["1_1.jpeg", "1_2.jpeg"], ["123"]
        )

        user_prompt = post.call_args.kwargs["json"]["messages"][1]["content"]
        task = json.loads(user_prompt)["task"]
        self.assertIn("side 1 is face and side 2 is back", task)
        self.assertIn("1-based position", task)

    @patch("local_llm._image_data_url", return_value="data:image/jpeg;base64,AA==")
    @patch("local_llm.requests.post")
    def test_vision_uses_dedicated_model_setting(self, post, _image_data):
        post.side_effect = [
            _side_response("face", 0.95, has_logo=True),
            _side_response("back", 0.95, has_contacts=True),
        ]

        with patch.dict(
            "os.environ",
            {
                "LOCAL_LLM_MODEL": "gemma4:e2b",
                "LOCAL_LLM_VISION_MODEL": "qwen2.5vl:3b",
            },
        ):
            request_vision_side_proposal(
                "archive.zip",
                [Path("/tmp/a.tif"), Path("/tmp/b.tif")],
                ["123"],
            )

        for call in post.call_args_list:
            self.assertEqual(call.kwargs["json"]["model"], "qwen2.5vl:3b")

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
