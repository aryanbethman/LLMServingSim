import csv
from pathlib import Path
import tempfile
import unittest

from analysis import realized_arrivals as ra

S = 1_000_000_000
WORKLOAD = [
    {"session_id": "a", "arrival_time_ns": 0, "sub_requests": [
        {"input_toks": 10, "output_toks": 1, "tool_duration_ns": 1 * S},
        {"input_toks": 12, "output_toks": 1, "tool_duration_ns": 0}]},
    {"input_toks": 5, "output_toks": 1, "arrival_time_ns": 1 * S},
    {"session_id": "b", "arrival_time_ns": 2 * S, "sub_requests": [
        {"input_toks": 10, "output_toks": 1, "tool_duration_ns": 2 * S},
        {"input_toks": 12, "output_toks": 1, "tool_duration_ns": 0}]},
]
# id: (instance, arrival, end, queuing, ttft). Session a: call 0 ends at 3 s, call 1
# arrives at 3 + 1. Session b: call 0 ends at 5 s, call 1 arrives at 5 + 2.
ROWS = {0: (0, 0, 3, 1, 1), 1: (1, 4, 6, 0, 1), 2: (0, 1, 2, 0, 1), 3: (1, 2, 5, 0, 1), 4: (0, 7, 8, 0, 1)}


def _requests(rows=ROWS):
    return {rid: {"instance": inst, "arrival": a * S, "end": e * S, "latency": (e - a) * S,
                  "queuing": q * S, "ttft": t * S} for rid, (inst, a, e, q, t) in rows.items()}


class JoinTests(unittest.TestCase):
    def test_ids_follow_the_file(self):
        sessions = ra.join(_requests(), WORKLOAD)
        self.assertEqual([len(s) for s in sessions], [2, 1, 2])
        self.assertEqual([c["tool"] for c in sessions[2]], [2 * S, 0])

    def test_missing_or_extra_requests(self):
        rows = dict(ROWS)
        del rows[4]
        with self.assertRaises(ValueError):
            ra.join(_requests(rows), WORKLOAD)
        with self.assertRaises(ValueError):
            ra.join(_requests({**ROWS, 5: (0, 9, 10, 0, 1)}), WORKLOAD)

    def test_csv_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["instance id", "request id", "model", "input", "output", "arrival",
                                 "end_time", "latency", "queuing_delay", "TTFT", "TPOT", "ITL"])
                writer.writerow([2, 7, "m", 10, 1, 5, 9, 4, 1, 2, 0, "[]"])
            self.assertEqual(ra.load_requests(path), {7: {"instance": 2, "arrival": 5, "end": 9, "latency": 4,
                                                          "queuing": 1, "ttft": 2}})


class AnalyzeTests(unittest.TestCase):
    def test_whole_run(self):
        report = ra.analyze(ra.join(_requests(), WORKLOAD), spec_rate=0.5, replicas=2)
        self.assertEqual(report["window_sec"], [0.0, 8.0])
        self.assertEqual(report["closed_loop_mismatches"], 0)
        # Session gaps: a = 1 s queuing + 2 s service + 1 s tool; b = 3 s service + 2 s tool.
        self.assertEqual(report["call_gap_share"], {"queuing": 1 / 9, "service": 5 / 9, "tool": 3 / 9})
        # Latencies 3 + 2 + 1 + 3 + 1 = 10 s over 8 s; queuing removes 1 s.
        self.assertEqual(report["mean_calls_in_system"], 10 / 8)
        self.assertEqual(report["mean_calls_in_service"], 9 / 8)
        self.assertEqual(report["mean_calls_in_service_per_replica"], 9 / 16)
        # Lifetimes: a 0-6, flat 1-2, b 2-8 -> 13 s over 8 s.
        self.assertEqual(report["mean_live_sessions"], 13 / 8)
        self.assertEqual(report["per_replica_arrivals"], [3, 2])
        self.assertEqual(report["carried_calls_per_sec"], 5 / 8)
        self.assertEqual(report["offered_calls_per_sec"], 0.5 * 5 / 3)

    def test_window_clips(self):
        report = ra.analyze(ra.join(_requests(), WORKLOAD), window_sec=(2, 6))
        self.assertEqual(report["calls"], 5)
        self.assertEqual(report["per_replica_arrivals"], [0, 2])  # arrivals at 2 and 4 s
        # Overlap with [2, 6): 1 + 2 + 0 + 3 + 0 = 6 s over 4 s.
        self.assertEqual(report["mean_calls_in_system"], 1.5)
        with self.assertRaises(ValueError):
            ra.analyze(ra.join(_requests(), WORKLOAD), window_sec=(5, 5))

    def test_closed_loop_mismatch_is_counted(self):
        rows = dict(ROWS)
        rows[4] = (0, 6, 8, 0, 1)  # released 1 s early
        self.assertEqual(ra.analyze(ra.join(_requests(rows), WORKLOAD))["closed_loop_mismatches"], 1)


if __name__ == "__main__":
    unittest.main()
