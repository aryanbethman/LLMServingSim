import argparse
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from analysis.offered_load import summarize
from workloads.generators import replicate_sessions, sharegpt, swebench
from workloads.generators.arrivals import (PROCESSES, ArrivalSpec, ToolDelaySpec, add_arrival_args,
                                           add_tool_delay_args, arrival_times_ns, spec_from_args,
                                           tool_delay_spec_from_args, weibull_shape)


def _spec(process, start_ns=0, **params):
    return ArrivalSpec(process=process, seed=42, unit="sessions", start_ns=start_ns, params=params)


def _poisson(**overrides):
    return _spec("poisson", **{"rate": 4.0, **overrides})


def _pulse(**overrides):
    return _spec("pulse", start_ns=1_000, **{"n": 3, "period_ns": 10_000_000_000, **overrides})


def _mmpp(**overrides):
    return _spec("mmpp", **{"rate": 1.0, "high_sec": 10.0, "low_sec": 50.0, "low_frac": 0.1, **overrides})


def _nhpp(**overrides):
    return _spec("nhpp", **{"schedule": [(0, 1.0), (100, 5.0)], "period_sec": 200.0, **overrides})


def _gaps_sec(spec, num):
    return np.diff(arrival_times_ns(spec, num)) / 1e9


def _window_counts(times_ns, width_sec):
    t = np.array(times_ns) / 1e9
    return np.bincount((t // width_sec).astype(int))[:-1]  # drop the partial last window


class TraceFile:
    def __init__(self, times):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "trace.txt"
        self.path.write_text("# seconds\n" + "\n".join(map(str, times)) + "\n")

    def spec(self, **extra):
        import hashlib
        return _spec("trace", start_ns=5, path=str(self.path),
                     sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(), **extra)


def _every_process():
    trace = TraceFile(np.cumsum(np.random.default_rng(0).exponential(1.0, 500)))
    return trace, [
        _poisson(), _spec("gamma", rate=2.0, cv=2.0), _spec("weibull", rate=2.0, cv=0.5),
        _spec("lognormal", rate=2.0, cv=1.5), _pulse(), _pulse(burst_rate=2.0), _mmpp(),
        _nhpp(), trace.spec(), trace.spec(rate=3.0),
    ]


class ConventionTests(unittest.TestCase):
    def test_every_process_meets_the_conventions(self):
        trace, specs = _every_process()
        with trace.dir:
            self.assertEqual({s.process for s in specs}, set(PROCESSES))
            for spec in specs:
                with self.subTest(process=spec.process, params=spec.params):
                    times = arrival_times_ns(spec, 300)
                    self.assertEqual(len(times), 300)
                    self.assertTrue(all(type(t) is int for t in times))
                    self.assertTrue(all(a <= b for a, b in zip(times, times[1:])))
                    if spec.process != "nhpp":
                        self.assertEqual(times[0], spec.start_ns)
                    self.assertEqual(arrival_times_ns(spec, 40), times[:40])
                    self.assertEqual(arrival_times_ns(spec, 0), [])
                    json.dumps(spec.to_dict())

    def test_seed_changes_random_processes(self):
        for spec in (_poisson(), _spec("gamma", rate=2.0, cv=2.0), _mmpp(), _nhpp()):
            other = ArrivalSpec(spec.process, 43, spec.unit, spec.start_ns, spec.params)
            self.assertNotEqual(arrival_times_ns(spec, 20), arrival_times_ns(other, 20))

    def test_stream_is_not_the_plain_seed_stream(self):
        # A generator that samples content from default_rng(seed) must not
        # replay the arrival gaps.
        plain = np.rint(np.cumsum(np.random.default_rng(42).exponential(0.25e9, size=9))).astype(np.int64)
        self.assertNotEqual(arrival_times_ns(_poisson(), 10)[1:], plain.tolist())


class PoissonAndRenewalTests(unittest.TestCase):
    def test_poisson_gaps(self):
        gaps = _gaps_sec(_poisson(), 20_001)
        self.assertAlmostEqual(gaps.mean(), 0.25, delta=0.01)
        self.assertAlmostEqual(gaps.std() / gaps.mean(), 1.0, delta=0.03)

    def test_renewal_gaps_hit_rate_and_cv(self):
        for process in ("gamma", "weibull", "lognormal"):
            for cv in (0.5, 2.0):
                with self.subTest(process=process, cv=cv):
                    gaps = _gaps_sec(_spec(process, rate=2.0, cv=cv), 200_001)
                    self.assertAlmostEqual(gaps.mean(), 0.5, delta=0.5 * 0.03)
                    self.assertAlmostEqual(gaps.std() / gaps.mean(), cv, delta=cv * 0.1)

    def test_weibull_shape(self):
        self.assertAlmostEqual(weibull_shape(1.0), 1.0, places=9)
        self.assertGreater(weibull_shape(0.5), 1.0)
        self.assertLess(weibull_shape(2.0), 1.0)
        with self.assertRaises(ValueError):
            weibull_shape(1e-6)


class PulseTests(unittest.TestCase):
    def test_bursts_start_on_the_period(self):
        period = 10_000_000_000
        self.assertEqual(arrival_times_ns(_pulse(), 7),
                         [1_000] * 3 + [1_000 + period] * 3 + [1_000 + 2 * period])

    def test_spread_bursts_start_on_the_period(self):
        spec = _pulse(n=4, burst_rate=2.0)
        times = arrival_times_ns(spec, 12)
        for burst in range(3):
            chunk = times[4 * burst: 4 * burst + 4]
            self.assertEqual(chunk[0], 1_000 + burst * spec.params["period_ns"])
            self.assertTrue(all(a < b for a, b in zip(chunk, chunk[1:])))

    def test_overlapping_bursts_come_back_sorted(self):
        times = arrival_times_ns(_pulse(n=50, period_ns=1_000, burst_rate=2.0), 200)
        self.assertEqual(times, sorted(times))

    def test_mean_rate(self):
        self.assertAlmostEqual(_pulse().mean_rate(), 0.3)


class MmppTests(unittest.TestCase):
    def test_state_rates_give_the_mean(self):
        high, low = _mmpp().to_dict()["params"]["high_rate"], _mmpp().to_dict()["params"]["low_rate"]
        self.assertAlmostEqual(high, 4.0)
        self.assertAlmostEqual(low, 0.4)

    def test_long_run_rate_and_overdispersion(self):
        times = arrival_times_ns(_mmpp(), 50_000)
        self.assertAlmostEqual((len(times) - 1) / (times[-1] / 1e9), 1.0, delta=0.1)
        counts = _window_counts(times, 10.0)
        self.assertGreater(counts.var() / counts.mean(), 3.0)  # Poisson would be ~1

    def test_on_off_is_silent_when_off(self):
        times = arrival_times_ns(_mmpp(low_frac=0.0, high_sec=1.0, low_sec=20.0), 5_000)
        gaps = np.diff(times) / 1e9
        self.assertGreater((gaps > 5).sum(), 50)


class NhppTests(unittest.TestCase):
    def test_mean_rate_is_the_schedule_average(self):
        self.assertAlmostEqual(_nhpp().mean_rate(), 3.0)
        self.assertEqual(_nhpp(period_sec=None).mean_rate(), 5.0)

    def test_counts_follow_the_schedule(self):
        t = np.array(arrival_times_ns(_nhpp(), 60_000)) / 1e9
        phase = t % 200
        rising, high = ((phase < 50).sum(), ((phase >= 100) & (phase < 150)).sum())
        # Rate 1->3 then 5->3 in those windows: 100 vs 200 arrivals per cycle.
        self.assertAlmostEqual(high / rising, 2.0, delta=0.1)
        self.assertAlmostEqual((len(t) - 1) / t[-1], 3.0, delta=0.1)

    def test_schedule_is_anchored_at_start(self):
        times = arrival_times_ns(_spec("nhpp", start_ns=7_000_000_000,
                                       schedule=[(0, 0.0), (10, 2.0)], period_sec=None), 5)
        self.assertGreater(times[0], 7_000_000_000)


class TraceTests(unittest.TestCase):
    def test_replay_and_rescale(self):
        trace = TraceFile([16, 10, 11, 13])
        with trace.dir:
            self.assertEqual(arrival_times_ns(trace.spec(), 4), [5, 1_000_000_005, 3_000_000_005, 6_000_000_005])
            self.assertEqual(arrival_times_ns(trace.spec(rate=1.0), 4),
                             [5, 500_000_005, 1_500_000_005, 3_000_000_005])
            self.assertAlmostEqual(trace.spec().mean_rate(), 0.5)
            with self.assertRaises(ValueError):
                arrival_times_ns(trace.spec(), 5)

    def test_changed_trace_is_rejected(self):
        trace = TraceFile([0, 1, 2])
        with trace.dir:
            spec = trace.spec()
            trace.path.write_text("0\n1\n3\n")
            with self.assertRaises(ValueError):
                ArrivalSpec(spec.process, spec.seed, spec.unit, spec.start_ns, spec.params)


class ValidationTests(unittest.TestCase):
    def test_rejects_bad_specs(self):
        bad = [
            ("poisson", {"rate": 0.0}), ("poisson", {"rate": float("inf")}), ("gamma", {"rate": 1.0}),
            ("gamma", {"rate": 1.0, "cv": 0.0}), ("pulse", {"n": 0, "period_ns": 1}),
            ("pulse", {"n": 1, "period_ns": 0}), ("mmpp", {"rate": 1.0, "high_sec": 1, "low_sec": 1}),
            ("mmpp", {"rate": 1.0, "high_sec": 1, "low_sec": 1, "low_frac": 1.0}),
            ("nhpp", {"schedule": [(1, 1.0)]}), ("nhpp", {"schedule": [(0, 1.0), (0, 2.0)]}),
            ("nhpp", {"schedule": [(0, 0.0)]}), ("nhpp", {"schedule": [(0, 1.0), (5, 0.0)]}),
            ("nhpp", {"schedule": [(0, 1.0), (5, 1.0)], "period_sec": 5.0}), ("brownian", {}),
        ]
        for process, params in bad:
            with self.subTest(process=process, params=params), self.assertRaises(ValueError):
                _spec(process, **params)
        with self.assertRaises(ValueError):
            _spec("poisson", start_ns=-1, rate=1.0)

    def test_manifest_dict(self):
        self.assertEqual(_poisson().to_dict(), {"process": "poisson", "seed": 42, "unit": "sessions",
                                                "start_ns": 0, "params": {"rate": 4.0},
                                                "mean_rate_per_sec": 4.0})


class GeneratorFlagTests(unittest.TestCase):
    ARGV = ["--sps", "0.5", "--seed", "7", "--first-arrival-sec", "1.5"]

    def _parse(self, extra=(), register=swebench.register_args,
               required=("--model", "m", "--num-sessions", "5", "--output", "o")):
        parser = argparse.ArgumentParser()
        register(parser)
        return parser.parse_args(self.ARGV + list(required) + list(extra))

    def test_every_generator_draws_the_same_times(self):
        specs = [
            spec_from_args(self._parse(register=sharegpt.register_args,
                                       required=["--model", "m", "--num-reqs", "5", "--output", "o"]),
                           "sessions"),
            spec_from_args(self._parse(), "sessions"),
            spec_from_args(self._parse(register=replicate_sessions.register_args,
                                       required=["--source", "s", "--output", "o", "--num-sessions", "5"]),
                           "sessions"),
        ]
        self.assertEqual(specs[0], specs[1])
        self.assertEqual(specs[1], specs[2])
        self.assertEqual(specs[0].start_ns, 1_500_000_000)
        self.assertEqual(arrival_times_ns(specs[0], 5), arrival_times_ns(specs[2], 5))

    def test_process_flags(self):
        cases = [
            (["--arrival-process", "gamma", "--arrival-cv", "3"], "gamma", {"rate": 0.5, "cv": 3.0}),
            (["--pulse", "--pulse-n", "2", "--pulse-delay-sec", "0.25"], "pulse",
             {"n": 2, "period_ns": 250_000_000}),
            (["--pulse", "--pulse-poisson"], "pulse", {"n": 10, "period_ns": 60_000_000_000, "burst_rate": 0.5}),
            (["--arrival-process", "mmpp", "--mmpp-low-frac", "0"], "mmpp",
             {"rate": 0.5, "high_sec": 10.0, "low_sec": 50.0, "low_frac": 0.0}),
            (["--arrival-process", "nhpp", "--rate-schedule", "0:0.2,30:1", "--rate-schedule-period-sec", "60"],
             "nhpp", {"schedule": [(0.0, 0.2), (30.0, 1.0)], "period_sec": 60.0}),
        ]
        for extra, process, params in cases:
            with self.subTest(extra=extra):
                spec = spec_from_args(self._parse(extra), "sessions")
                self.assertEqual((spec.process, spec.params), (process, params))

    def test_flag_errors(self):
        for extra in (["--pulse", "--arrival-process", "gamma"], ["--arrival-process", "nhpp"],
                      ["--arrival-process", "trace"]):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                spec_from_args(self._parse(extra), "sessions")
        parser = argparse.ArgumentParser()
        parser.add_argument("--seed", type=int, default=1)
        add_arrival_args(parser, "sessions")
        with self.assertRaises(ValueError):
            spec_from_args(parser.parse_args([]), "sessions")

    def test_programmatic_namespace_without_new_flags(self):
        args = argparse.Namespace(sps=3.2, seed=42, first_arrival_sec=0)
        self.assertEqual(spec_from_args(args, "sessions"), _poisson(rate=3.2))
        self.assertEqual(tool_delay_spec_from_args(args), ToolDelaySpec(seed=42))


class ToolDelayTests(unittest.TestCase):
    RECORDED = [1_000, 2_000, 0]

    def test_modes(self):
        self.assertEqual(ToolDelaySpec().sampler()(self.RECORDED), self.RECORDED)
        self.assertEqual(ToolDelaySpec("scale", 0, {"factor": 2.5}).sampler()(self.RECORDED), [2_500, 5_000, 0])
        self.assertEqual(ToolDelaySpec("constant", 0, {"sec": 0.5}).sampler()(self.RECORDED),
                         [500_000_000, 500_000_000, 0])

    def test_lognormal_median_and_last_call(self):
        draw = ToolDelaySpec("lognormal", 3, {"sec": 2.0, "sigma": 1.0}).sampler()
        delays = [d for _ in range(2_000) for d in draw([0] * 11)[:-1]]
        self.assertAlmostEqual(np.median(delays) / 1e9, 2.0, delta=0.1)
        self.assertEqual(draw([0, 0])[-1], 0)
        again = ToolDelaySpec("lognormal", 3, {"sec": 2.0, "sigma": 1.0}).sampler()
        self.assertEqual(again([0] * 11)[:-1], delays[:10])

    def test_flags_and_validation(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--seed", type=int, default=5)
        add_tool_delay_args(parser)
        spec = tool_delay_spec_from_args(parser.parse_args(["--tool-delay", "lognormal", "--tool-delay-sec", "3"]))
        self.assertEqual(spec, ToolDelaySpec("lognormal", 5, {"sec": 3.0, "sigma": 1.0}))
        with self.assertRaises(ValueError):
            tool_delay_spec_from_args(parser.parse_args(["--tool-delay", "constant"]))
        for mode, params in (("scale", {"factor": -1}), ("lognormal", {"sec": 0.0, "sigma": 1}), ("nope", {})):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                ToolDelaySpec(mode, 0, params)


class OfferedLoadTests(unittest.TestCase):
    def test_session_and_flat_rows(self):
        rows = [
            {"session_id": "a", "arrival_time_ns": 0, "sub_requests": [
                {"input_toks": 100, "output_toks": 10, "tool_duration_ns": 2_000_000_000},
                {"input_toks": 130, "output_toks": 20, "tool_duration_ns": 0}]},
            {"input_toks": 50, "output_toks": 5, "arrival_time_ns": 4_000_000_000},
        ]
        out = summarize(rows, mean_rate=0.5, replicas=2, service_sec=3.0)
        self.assertEqual(out["per_row"], {"calls": 1.5, "input_toks": 140.0, "uncached_input_toks": 85.0,
                                          "output_toks": 17.5, "tool_delay_sec": 1.0})
        self.assertEqual(out["empirical_rate_per_sec"], 0.25)
        self.assertEqual(out["offered"]["calls_per_sec"], 0.75)
        self.assertEqual(out["offered"]["min_concurrent_rows"], 0.5)
        self.assertEqual(out["offered_per_replica"]["output_toks_per_sec"], 4.375)
        self.assertEqual(out["utilization"], 0.75)


if __name__ == "__main__":
    unittest.main()
