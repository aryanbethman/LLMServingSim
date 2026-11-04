import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from analysis import arrival_fit as fit
from analysis import tool_delay_audit as audit
from workloads.generators.arrivals import ArrivalSpec, arrival_times_ns


def _times(process, num, **params):
    spec = ArrivalSpec(process=process, seed=11, unit="requests", params=params)
    return np.array(arrival_times_ns(spec, num)) / 1e9


class SpecialFunctionTests(unittest.TestCase):
    def test_digamma_and_trigamma(self):
        self.assertAlmostEqual(fit.digamma(1.0), -0.5772156649015329, places=10)
        self.assertAlmostEqual(fit.digamma(0.5), -1.9635100260214235, places=10)
        self.assertAlmostEqual(fit.trigamma(1.0), math.pi**2 / 6, places=10)


class DistributionFitTests(unittest.TestCase):
    def test_fits_recover_their_own_distribution(self):
        rng = np.random.default_rng(1)
        samples = {
            "gamma": rng.gamma(0.25, 4.0, 50_000),
            "weibull": rng.weibull(0.7, 50_000) * 2.0,
            "lognormal": rng.lognormal(0.0, 1.0, 50_000),
        }
        self.assertAlmostEqual(fit.fit_gamma(samples["gamma"])["shape"], 0.25, delta=0.01)
        self.assertAlmostEqual(fit.fit_weibull(samples["weibull"])["shape"], 0.7, delta=0.01)
        self.assertAlmostEqual(fit.fit_lognormal(samples["lognormal"])["sigma"], 1.0, delta=0.01)
        for name, gaps in samples.items():
            with self.subTest(name=name):
                fits = fit.fit_gaps(gaps, seed=0)
                self.assertEqual(min(fits, key=lambda k: fits[k]["aic"]), name)
                self.assertLess(fits[name]["ks_d"], 0.01)

    def test_ks_distance(self):
        a = np.arange(100.0)
        self.assertEqual(fit.ks_distance(a, a), 0.0)
        self.assertEqual(fit.ks_distance(a, a + 1000), 1.0)


class AnalyzeTests(unittest.TestCase):
    def test_poisson_looks_poisson(self):
        report = fit.analyze(_times("poisson", 30_000, rate=5.0))
        self.assertAlmostEqual(report["rate_per_sec"], 5.0, delta=0.15)
        self.assertAlmostEqual(report["gap_cv"], 1.0, delta=0.03)
        self.assertAlmostEqual(report["fits"]["gamma"]["cv"], 1.0, delta=0.03)
        for value in report["count_dispersion"].values():
            self.assertAlmostEqual(value, 1.0, delta=0.25)
        for value in report["gap_autocorrelation"].values():
            self.assertLess(abs(value), 0.03)
        self.assertIsNone(report["daily_profile"])
        self.assertTrue(report["flags"]["renewal"].startswith("--arrival-process "))

    def test_bursty_and_daily_traces(self):
        bursty = fit.analyze(_times("mmpp", 30_000, rate=2.0, high_sec=5.0, low_sec=30.0, low_frac=0.05))
        self.assertGreater(bursty["count_dispersion"]["60"], 5)
        daily = fit.analyze(_times("nhpp", 60_000, schedule=[(0, 0.05), (43_200, 0.25)], period_sec=86_400))
        shape = daily["daily_profile"]["relative_rate"]
        self.assertEqual(len(shape), 24)
        self.assertLess(shape[0], shape[12])
        self.assertIn("--rate-schedule-period-sec 86400", daily["daily_profile"]["flags"])
        # Inside an hour the process is Poisson again.
        self.assertAlmostEqual(daily["rate_profile"]["detrended_gap_cv"], 1.0, delta=0.1)


class ResolutionTests(unittest.TestCase):
    def test_quanta_and_bins(self):
        self.assertEqual(fit.resolution(np.array([3.0, 4.0, 9.0, 12.0])), 1.0)
        self.assertEqual(fit.resolution(np.array([0.001, 0.004, 0.0105])), 1e-4)
        self.assertEqual(fit.resolution(np.array([0.1234567, 0.3141593, 0.2718282])), 0.0)
        binned = np.repeat(np.arange(0.0, 300.0, 3.0), 5)
        self.assertEqual(fit.resolution(binned), 3.0)

    def test_dequantize_spreads_ties_inside_the_quantum(self):
        times = np.repeat([0.0, 3.0, 6.0], 4)
        spread = fit.dequantize(times, 3.0, seed=0)
        self.assertEqual(len(np.unique(spread)), 12)
        self.assertTrue(np.all((spread >= 0) & (spread < 9)))
        self.assertEqual(np.histogram(spread, bins=[0, 3, 6, 9])[0].tolist(), [4, 4, 4])


class LoadTests(unittest.TestCase):
    def test_presets(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "b.csv").write_text("Timestamp,Model,Log Type\n5,ChatGPT,API log\n2,GPT-4,API log\n"
                                       "9,ChatGPT,API log\n7,ChatGPT,API log\n")
            self.assertEqual(fit.load_times(tmp / "b.csv", "burstgpt", ["Model=ChatGPT"]).tolist(), [5, 7, 9])
            (tmp / "a.csv").write_text("TIMESTAMP,ContextTokens,GeneratedTokens\n"
                                       "2023-11-16 18:17:03.9799600,1,1\n2023-11-16 18:17:04.0319605,1,1\n"
                                       "2023-11-16 18:17:05.0000000,1,1\n")
            azure = fit.load_times(tmp / "a.csv", "azure")
            # Epoch seconds as floats keep ~2e-7 s, so the 7th digit is approximate.
            self.assertAlmostEqual(azure[1] - azure[0], 0.0520005, delta=1e-6)
            (tmp / "m.jsonl").write_text("".join(json.dumps({"timestamp": t}) + "\n" for t in (0, 1500, 3000)))
            self.assertEqual(fit.load_times(tmp / "m.jsonl", "mooncake").tolist(), [0, 1.5, 3])
            (tmp / "p.txt").write_text("# s\n2\n1\n\n3\n")
            self.assertEqual(fit.load_times(tmp / "p.txt", "plain").tolist(), [1, 2, 3])
            with self.assertRaises(ValueError):
                fit.load_times(tmp / "b.csv", "burstgpt", ["Model=GPT-4"])


def _event(i, ts, response=None, action=None, observation=None, command=None):
    event = {"id": i, "timestamp": f"2025-07-31T19:00:{ts:09.6f}", "source": "agent"}
    if action:
        event["action"] = action
        event["args"] = {"command": command} if command else {}
    if observation:
        event["observation"] = observation
    if response:
        event["tool_call_metadata"] = {"model_response": {
            "id": response, "usage": {"prompt_tokens": 10, "completion_tokens": 1}}}
    return event


class ToolDelayAuditTests(unittest.TestCase):
    EVENTS = [
        {"id": 0, "timestamp": "2025-07-31T19:00:00.000000", "source": "agent", "action": "system",
         "args": {"content": "S"}},
        {"id": 1, "timestamp": "2025-07-31T19:00:00.100000", "source": "user", "action": "message",
         "args": {"content": "T"}},
        _event(2, 3.0, "r1", action="run", command="cd /w && FOO=1 pytest -q"),
        _event(3, 3.62, "r1", observation="run"),
        _event(4, 6.0, "r2", action="read"),
        _event(5, 6.01, "r2", observation="read"),
        _event(6, 9.0, "r3", action="finish"),
    ]

    def test_steps(self):
        steps = audit.response_steps(self.EVENTS)
        self.assertEqual([s["tools"] for s in steps], ["run", "read"])
        self.assertEqual(steps[0]["commands"], ["pytest"])
        self.assertAlmostEqual(steps[0]["tool_sec"], 0.62, places=6)
        self.assertAlmostEqual(steps[0]["llm_gap_sec"], 2.38, places=6)
        self.assertAlmostEqual(steps[1]["tool_sec"], 0.01, places=6)

    def test_audit_matches_generator(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "task-1").mkdir()
            (Path(tmp) / "task-1" / "trajectory.json").write_text(json.dumps(self.EVENTS))
            report = audit.audit(Path(tmp))
        self.assertEqual((report["trajectories"], report["steps"]), (1, 2))
        self.assertEqual(report["by_tools"]["run"]["count"], 1)
        self.assertEqual(report["run_by_command"]["pytest"]["count"], 1)
        self.assertEqual(report["share_in_0.60_0.65_sec"], 0.5)
        self.assertAlmostEqual(report["tool_share_of_session_time"], 0.63 / (0.63 + 2.38 + 2.99), places=6)


if __name__ == "__main__":
    unittest.main()
