import base64
import io
import json
import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from serving.core.controller import Controller


class _Process:
    def __init__(self):
        self.stdin = io.StringIO()


class ExecutionPayloadProtocolTest(unittest.TestCase):
    def test_malformed_ready_fails_instead_of_waiting_for_another_record(self):
        process = _Process()
        process.stdout = io.StringIO("READY [logger] diagnostic\n0 0 12 0\n")
        with self.assertRaisesRegex(RuntimeError, "Malformed ASTRA READY"):
            Controller(total_num=1).read_completion(process)

    def test_compact_reader_consumes_sideband_before_ready(self):
        process = _Process()
        process.stdout = io.StringIO("TEMPLATE_CACHE 1 42 3 84 2 5\nREADY 0 7 123 10\n")
        controller = Controller(total_num=1)
        self.assertEqual(controller.read_completion(process), {"sys": 0, "id": 7, "cycle": 123})
        self.assertEqual(controller.get_template_transport_stats()["astra_cache_entries"], 1)

    def test_legacy_reader_fails_on_eof(self):
        process = _Process()
        process.stdout = io.StringIO("diagnostic only\n")
        with self.assertRaisesRegex(RuntimeError, "closed stdout"):
            Controller(total_num=1).read_wait(process)

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
        self.assertEqual(
            controller.get_template_transport_stats(),
            {
                "bundles": 1,
                "wire_bytes": len(json.dumps(bundle, separators=(",", ":")).encode("utf-8")),
                "template_definitions": 1,
                "template_nodes": 1,
                "duplicate_template_definitions": 0,
                "duplicate_template_nodes": 0,
                "rank_bindings": 1,
                "template_releases": 0,
                "astra_cache_entries": 0,
                "astra_cache_nodes": 0,
                "astra_cache_high_water_entries": 0,
                "astra_cache_high_water_nodes": 0,
                "astra_cache_evictions": 0,
                "astra_cache_blocked_evictions": 0,
                "astra_template_decode_ns": 0,
                "astra_binding_parse_ns": 0,
                "astra_direct_feeder_init_ns": 0,
                "astra_direct_feeder_inits": 0,
                "cached_template_definitions": 1,
            },
        )

    def test_controller_forgets_template_only_after_astra_release(self):
        controller = Controller(total_num=1)
        template_id = "a" * 64
        controller.sent_template_ids.add(template_id)

        self.assertIsNone(controller.parse_output(f"TEMPLATE_RELEASE {template_id}\n"))
        self.assertEqual(controller.sent_template_ids, set())
        self.assertEqual(controller.get_template_transport_stats()["template_releases"], 1)

        self.assertIsNone(controller.parse_output("TEMPLATE_CACHE 1 42 3 84 2 5\n"))
        stats = controller.get_template_transport_stats()
        self.assertEqual(stats["astra_cache_entries"], 1)
        self.assertEqual(stats["astra_cache_nodes"], 42)
        self.assertEqual(stats["astra_cache_high_water_entries"], 3)
        self.assertEqual(stats["astra_cache_high_water_nodes"], 84)
        self.assertEqual(stats["astra_cache_evictions"], 2)
        self.assertEqual(stats["astra_cache_blocked_evictions"], 5)


if __name__ == "__main__":
    unittest.main()
