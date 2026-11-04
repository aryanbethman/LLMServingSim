import argparse
import unittest

import numpy as np

from workloads.generators import replicate_sessions, sharegpt, swebench
from workloads.generators.arrivals import ArrivalSpec, arrival_times_ns, spec_from_args


def _poisson(**overrides):
    fields = dict(process="poisson", rate=4.0, seed=42, unit="sessions")
    fields.update(overrides)
    return ArrivalSpec(**fields)


def _pulse(**overrides):
    fields = dict(process="pulse", rate=2.0, seed=42, unit="requests", start_ns=1_000,
                  pulse_n=3, pulse_period_ns=10_000_000_000)
    fields.update(overrides)
    return ArrivalSpec(**fields)


class PoissonTests(unittest.TestCase):
    def test_first_arrival_is_the_start(self):
        times = arrival_times_ns(_poisson(start_ns=5_000), 50)
        self.assertEqual(times[0], 5_000)
        self.assertTrue(all(a <= b for a, b in zip(times, times[1:])))
        self.assertTrue(all(type(t) is int for t in times))

    def test_gaps_are_exponential_at_the_rate(self):
        gaps = np.diff(arrival_times_ns(_poisson(), 20_001)) / 1e9
        self.assertAlmostEqual(gaps.mean(), 0.25, delta=0.01)
        self.assertAlmostEqual(gaps.std() / gaps.mean(), 1.0, delta=0.03)

    def test_same_seed_same_times_and_count_keeps_prefix(self):
        self.assertEqual(arrival_times_ns(_poisson(), 100), arrival_times_ns(_poisson(), 100))
        self.assertEqual(arrival_times_ns(_poisson(), 100)[:10], arrival_times_ns(_poisson(), 10))
        self.assertNotEqual(arrival_times_ns(_poisson(), 10), arrival_times_ns(_poisson(seed=43), 10))

    def test_stream_is_not_the_plain_seed_stream(self):
        # A generator that samples content from default_rng(seed) must not
        # replay the arrival gaps.
        plain = np.rint(np.cumsum(np.random.default_rng(42).exponential(0.25e9, size=9))).astype(np.int64)
        self.assertNotEqual(arrival_times_ns(_poisson(), 10)[1:], plain.tolist())

    def test_degenerate_counts(self):
        self.assertEqual(arrival_times_ns(_poisson(), 0), [])
        self.assertEqual(arrival_times_ns(_poisson(start_ns=7), 1), [7])


class PulseTests(unittest.TestCase):
    def test_bursts_start_on_the_period(self):
        times = arrival_times_ns(_pulse(), 7)
        period = 10_000_000_000
        self.assertEqual(times, [1_000] * 3 + [1_000 + period] * 3 + [1_000 + 2 * period])

    def test_poisson_bursts_start_on_the_period_and_spread_inside(self):
        spec = _pulse(pulse_n=4, pulse_poisson=True)
        times = arrival_times_ns(spec, 12)
        for burst in range(3):
            begin = 1_000 + burst * spec.pulse_period_ns
            chunk = times[4 * burst: 4 * burst + 4]
            self.assertEqual(chunk[0], begin)
            self.assertTrue(all(a < b for a, b in zip(chunk, chunk[1:])))

    def test_overlapping_bursts_come_back_sorted(self):
        times = arrival_times_ns(_pulse(pulse_n=50, pulse_period_ns=1_000, pulse_poisson=True), 200)
        self.assertEqual(times, sorted(times))
        self.assertEqual(len(times), 200)

    def test_mean_rate(self):
        self.assertAlmostEqual(_pulse().mean_rate(), 0.3)
        self.assertEqual(_poisson().mean_rate(), 4.0)


class ValidationTests(unittest.TestCase):
    def test_rejects_bad_specs(self):
        for bad in (dict(rate=0.0), dict(rate=float("inf")), dict(start_ns=-1), dict(process="gamma")):
            with self.assertRaises(ValueError):
                _poisson(**bad)
        for bad in (dict(pulse_n=0), dict(pulse_period_ns=0)):
            with self.assertRaises(ValueError):
                _pulse(**bad)

    def test_manifest_dict(self):
        self.assertEqual(_poisson().to_dict(), {"process": "poisson", "rate": 4.0, "seed": 42,
                                                "unit": "sessions", "start_ns": 0,
                                                "mean_rate_per_sec": 4.0})
        self.assertEqual(_pulse().to_dict()["pulse_period_ns"], 10_000_000_000)


class GeneratorFlagTests(unittest.TestCase):
    ARGV = ["--sps", "0.5", "--seed", "7", "--first-arrival-sec", "1.5"]

    def _spec(self, register, extra, unit):
        parser = argparse.ArgumentParser()
        register(parser)
        return spec_from_args(parser.parse_args(self.ARGV + extra), unit)

    def test_every_generator_draws_the_same_times(self):
        specs = [
            self._spec(sharegpt.register_args, ["--model", "m", "--num-reqs", "5", "--output", "o"], "sessions"),
            self._spec(swebench.register_args, ["--model", "m", "--num-sessions", "5", "--output", "o"], "sessions"),
            self._spec(replicate_sessions.register_args,
                       ["--source", "s", "--output", "o", "--num-sessions", "5"], "sessions"),
        ]
        self.assertEqual(specs[0], specs[1])
        self.assertEqual(specs[1], specs[2])
        self.assertEqual(specs[0].start_ns, 1_500_000_000)
        self.assertEqual(arrival_times_ns(specs[0], 5), arrival_times_ns(specs[2], 5))

    def test_pulse_flags(self):
        spec = self._spec(swebench.register_args,
                          ["--model", "m", "--num-sessions", "5", "--output", "o",
                           "--pulse", "--pulse-n", "2", "--pulse-delay-sec", "0.25"], "sessions")
        self.assertEqual((spec.process, spec.pulse_n, spec.pulse_period_ns, spec.pulse_poisson),
                         ("pulse", 2, 250_000_000, False))

    def test_programmatic_namespace_without_pulse_flags(self):
        args = argparse.Namespace(sps=3.2, seed=42, first_arrival_sec=0)
        self.assertEqual(spec_from_args(args, "sessions"), _poisson(rate=3.2))


if __name__ == "__main__":
    unittest.main()
