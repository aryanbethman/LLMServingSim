import argparse
import json
from pathlib import Path
import tempfile
import unittest

from workloads.generators.replicate_sessions import load_sessions, run
from serving.core.router import Router


class ReplicationTest(unittest.TestCase):
    def test_complete_sessions_preserve_prefixes_lengths_and_delays(self):
        source = {"session_id": "task", "arrival_time_ns": 100,
                  "sub_requests": [
                      {"input_toks": 2, "output_toks": 1, "tool_duration_ns": 17,
                       "input_tok_ids": [1, 2], "output_tok_ids": [3]},
                      {"input_toks": 4, "output_toks": 1, "tool_duration_ns": 0,
                       "input_tok_ids": [1, 2, 3, 4], "output_tok_ids": [5]}]}
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "source.jsonl", Path(tmp) / "out.jsonl"
            src.write_text(json.dumps(source) + "\n")
            args = argparse.Namespace(source=src, output=dst, num_sessions=3,
                                      sps=3.2, seed=42, first_arrival_sec=0,
                                      token_policy="isolate-copies")
            run(args)
            rows = load_sessions(dst)
            self.assertEqual(len({r["session_id"] for r in rows}), 3)
            self.assertEqual(len({r["arrival_time_ns"] for r in rows}), 3)
            for row in rows:
                a, b = row["sub_requests"]
                self.assertEqual(b["input_tok_ids"][:3], a["input_tok_ids"] + a["output_tok_ids"])
                self.assertEqual([a["input_toks"], b["input_toks"]], [2, 4])
                self.assertEqual(a["tool_duration_ns"], 17)
            self.assertNotEqual(rows[0]["sub_requests"][0]["input_tok_ids"],
                                rows[1]["sub_requests"][0]["input_tok_ids"])
            # Exercise the actual closed-loop release path, not just JSON shape.
            router = Router(0, [], 0)
            for row in rows:
                router._load_agentic_session(row, True)
            router._pending_requests.sort(key=lambda r: r["arrival_time_ns"])
            first = router._pending_requests[0]
            router._pending_idx = 1
            router.notify_request_completed(first["index"], 1000)
            released = next(r for r in router._pending_requests if r.get("sub_request_index") == 1)
            self.assertEqual(released["arrival_time_ns"], 1017)
            with self.assertRaises(FileExistsError):
                run(args)
            again = Path(tmp) / "again.jsonl"
            args.output = again
            run(args)
            self.assertEqual(dst.read_bytes(), again.read_bytes())


if __name__ == "__main__":
    unittest.main()
