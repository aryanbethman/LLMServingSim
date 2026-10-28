import unittest

import numpy as np

from workloads.generators import swebench as sb


def _event(i, ts, source, action=None, observation=None, response_id=None, prompt=0, completion=0, args=None):
    e = {"id": i, "timestamp": ts, "source": source}
    if action:
        e["action"] = action
    if observation:
        e["observation"] = observation
    if args is not None:
        e["args"] = args
    if response_id:
        e["tool_call_metadata"] = {"model_response": {
            "id": response_id, "usage": {"prompt_tokens": prompt, "completion_tokens": completion}}}
    return e


def _two_call_events():
    return [
        _event(0, "2025-07-31T19:00:00.000000", "agent", action="system",
               args={"content": "SYSTEM", "tools": [{"type": "function", "function": {"name": "bash"}}]}),
        _event(1, "2025-07-31T19:00:00.100000", "user", action="message", args={"content": "TASK"}),
        _event(2, "2025-07-31T19:00:03.000000", "agent", action="run", response_id="r1", prompt=100, completion=10),
        _event(3, "2025-07-31T19:00:03.750000", "agent", observation="run", response_id="r1", prompt=100, completion=10),
        _event(4, "2025-07-31T19:00:06.000000", "agent", action="finish", response_id="r2", prompt=140, completion=5),
    ]


def _calls(*shape):
    return [sb.LLMCall(prompt_toks=p, completion_toks=c, tool_duration_ns=g) for p, c, g in shape]


class ParseOpenHandsTrajectoryTests(unittest.TestCase):
    def test_one_call_per_distinct_response_in_order(self):
        traj = sb.parse_openhands_trajectory("t-1", _two_call_events())
        self.assertEqual([(c.prompt_toks, c.completion_toks) for c in traj.calls], [(100, 10), (140, 5)])

    def test_tool_duration_runs_from_response_to_last_event_before_next_response(self):
        traj = sb.parse_openhands_trajectory("t-1", _two_call_events())
        self.assertEqual(traj.calls[0].tool_duration_ns, 750_000_000)

    def test_last_call_has_zero_tool_duration(self):
        traj = sb.parse_openhands_trajectory("t-1", _two_call_events())
        self.assertEqual(traj.calls[-1].tool_duration_ns, 0)

    def test_keeps_system_prompt_tools_and_task(self):
        traj = sb.parse_openhands_trajectory("t-1", _two_call_events())
        self.assertEqual((traj.system_prompt, traj.task, traj.tools[0]["function"]["name"]), ("SYSTEM", "TASK", "bash"))

    def test_rejects_trajectory_without_llm_calls(self):
        events = _two_call_events()[:2]
        with self.assertRaisesRegex(ValueError, "t-1"):
            sb.parse_openhands_trajectory("t-1", events)


class TruncateCallsTests(unittest.TestCase):
    def test_max_calls_keeps_first_n(self):
        calls = _calls((10, 1, 5), (20, 1, 5), (30, 1, 0))
        self.assertEqual(len(sb.truncate_calls(calls, max_calls=2, max_kv_toks=0)), 2)

    def test_max_kv_toks_stops_before_first_call_over_the_limit(self):
        calls = _calls((10, 1, 5), (20, 1, 5), (10, 1, 0))
        self.assertEqual(sb.truncate_calls(calls, max_calls=0, max_kv_toks=15), calls[:1])

    def test_zero_limits_keep_everything(self):
        calls = _calls((10, 1, 5), (20, 1, 0))
        self.assertEqual(sb.truncate_calls(calls, max_calls=0, max_kv_toks=0), calls)


class BuildSessionTests(unittest.TestCase):
    def _session(self, first_prompt_ids, calls, seed=0):
        return sb.build_session("t-1-run0", 123, first_prompt_ids, calls, np.random.default_rng(seed), vocab_size=1000)

    def test_token_counts_come_from_the_recorded_calls(self):
        calls = _calls((8, 3, 50), (15, 2, 0))
        subs = self._session(list(range(8)), calls)["sub_requests"]
        self.assertEqual([(s["input_toks"], s["output_toks"]) for s in subs], [(8, 3), (15, 2)])

    def test_id_lengths_match_token_counts(self):
        calls = _calls((8, 3, 50), (15, 2, 0))
        subs = self._session(list(range(5)), calls)["sub_requests"]
        self.assertEqual([(len(s["input_tok_ids"]), len(s["output_tok_ids"])) for s in subs], [(8, 3), (15, 2)])

    def test_first_prompt_starts_with_the_rendered_ids(self):
        calls = _calls((8, 3, 0))
        subs = self._session([7, 7, 7, 7, 7], calls)["sub_requests"]
        self.assertEqual(subs[0]["input_tok_ids"][:5], [7, 7, 7, 7, 7])

    def test_rendered_ids_longer_than_the_prompt_are_cut(self):
        calls = _calls((4, 3, 0))
        subs = self._session(list(range(10)), calls)["sub_requests"]
        self.assertEqual(subs[0]["input_tok_ids"], [0, 1, 2, 3])

    def test_next_prompt_extends_previous_prompt_and_output(self):
        calls = _calls((8, 3, 50), (15, 2, 0))
        subs = self._session(list(range(8)), calls)["sub_requests"]
        previous = subs[0]["input_tok_ids"] + subs[0]["output_tok_ids"]
        self.assertEqual(subs[1]["input_tok_ids"][:len(previous)], previous)

    def test_shrinking_prompt_reuses_a_prefix_of_the_history(self):
        calls = _calls((8, 3, 50), (9, 2, 0))
        subs = self._session(list(range(8)), calls)["sub_requests"]
        previous = subs[0]["input_tok_ids"] + subs[0]["output_tok_ids"]
        self.assertEqual(subs[1]["input_tok_ids"], previous[:9])

    def test_tool_durations_are_kept_and_the_last_is_zero(self):
        calls = _calls((8, 3, 50), (15, 2, 70))
        subs = self._session(list(range(8)), calls)["sub_requests"]
        self.assertEqual([s["tool_duration_ns"] for s in subs], [50, 0])

    def test_reruns_of_a_task_share_the_first_prompt_but_not_the_output(self):
        calls = _calls((8, 3, 0))
        a = self._session(list(range(8)), calls, seed=1)["sub_requests"][0]
        b = self._session(list(range(8)), calls, seed=2)["sub_requests"][0]
        self.assertEqual(a["input_tok_ids"], b["input_tok_ids"])
        self.assertNotEqual(a["output_tok_ids"], b["output_tok_ids"])

    def test_session_carries_id_and_arrival(self):
        session = self._session(list(range(8)), _calls((8, 3, 0)))
        self.assertEqual((session["session_id"], session["arrival_time_ns"]), ("t-1-run0", 123))


class SampleSessionsTests(unittest.TestCase):
    def test_fewer_sessions_than_tasks_uses_each_task_at_most_once(self):
        picks = sb.sample_sessions(["a", "b", "c"], 2, np.random.default_rng(0))
        self.assertEqual(len({task for task, _ in picks}), 2)

    def test_more_sessions_than_tasks_cycles_with_increasing_run_index(self):
        picks = sb.sample_sessions(["a", "b"], 5, np.random.default_rng(0))
        self.assertEqual(sorted(run for _, run in picks), [0, 0, 1, 1, 2])

    def test_each_pass_uses_every_task_once(self):
        picks = sb.sample_sessions(["a", "b", "c"], 6, np.random.default_rng(0))
        self.assertEqual(sorted(task for task, run in picks if run == 1), ["a", "b", "c"])

    def test_same_seed_gives_same_order(self):
        a = sb.sample_sessions(["a", "b", "c", "d"], 4, np.random.default_rng(7))
        b = sb.sample_sessions(["d", "c", "b", "a"], 4, np.random.default_rng(7))
        self.assertEqual(a, b)

    def test_empty_pool_is_rejected(self):
        with self.assertRaises(ValueError):
            sb.sample_sessions([], 1, np.random.default_rng(0))


class PoissonArrivalsTests(unittest.TestCase):
    def test_arrivals_are_increasing_and_offset(self):
        arrivals = sb.poisson_arrivals(50, sps=2.0, first_arrival_ns=5_000, rng=np.random.default_rng(0))
        self.assertTrue(all(a < b for a, b in zip(arrivals, arrivals[1:])))
        self.assertGreater(arrivals[0], 5_000)

    def test_mean_gap_matches_rate(self):
        arrivals = sb.poisson_arrivals(20_000, sps=4.0, first_arrival_ns=0, rng=np.random.default_rng(0))
        self.assertAlmostEqual(arrivals[-1] / len(arrivals) / 1e9, 0.25, delta=0.01)


class ParseS3ListingTests(unittest.TestCase):
    LISTING = (b'<?xml version="1.0" encoding="UTF-8"?>'
               b'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
               b'<Contents><Key>run/trajs/a__a-1/trajectory.json</Key></Contents>'
               b'<Contents><Key>run/trajs/a__a-1/other.log</Key></Contents>'
               b'<NextContinuationToken>tok+/=</NextContinuationToken>'
               b'</ListBucketResult>')

    def test_keeps_only_trajectory_files(self):
        keys, _ = sb.parse_s3_listing(self.LISTING)
        self.assertEqual(keys, ["run/trajs/a__a-1/trajectory.json"])

    def test_returns_continuation_token(self):
        _, token = sb.parse_s3_listing(self.LISTING)
        self.assertEqual(token, "tok+/=")

    def test_last_page_has_no_token(self):
        listing = self.LISTING.replace(b"<NextContinuationToken>tok+/=</NextContinuationToken>", b"")
        _, token = sb.parse_s3_listing(listing)
        self.assertIsNone(token)


if __name__ == "__main__":
    unittest.main()
