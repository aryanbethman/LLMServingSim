import base64
import io
import json
import unittest

from inference_serving.controller import Controller


class _Process:
    def __init__(self):
        self.stdin = io.StringIO()


class ExecutionPayloadProtocolTest(unittest.TestCase):
    def test_controller_encodes_rank_payloads_on_one_framed_line(self):
        process = _Process()
        Controller(total_num=2).write_payloads(
            process, {0: b"rank-zero", 4: bytes([0, 1, 255])}
        )

        line = process.stdin.getvalue()
        self.assertTrue(line.startswith("ET_PAYLOADS "))
        self.assertTrue(line.endswith("\n"))
        encoded = json.loads(line[len("ET_PAYLOADS ") : -1])
        self.assertEqual(base64.b64decode(encoded["0"]), b"rank-zero")
        self.assertEqual(base64.b64decode(encoded["4"]), bytes([0, 1, 255]))

    def test_controller_encodes_template_bundle_and_tracks_cached_ids(self):
        process = _Process()
        controller = Controller(total_num=2)
        bundle = {
            "templates": {"template-a": ["bm9kZQ=="]},
            "bindings": {"0": {"template_id": "template-a"}},
        }
        controller.write_template_bundle(process, bundle)

        line = process.stdin.getvalue()
        self.assertTrue(line.startswith("ET_TEMPLATE_BUNDLE "))
        self.assertEqual(json.loads(line[len("ET_TEMPLATE_BUNDLE ") : -1]), bundle)
        self.assertEqual(controller.sent_template_ids, {"template-a"})


if __name__ == "__main__":
    unittest.main()
