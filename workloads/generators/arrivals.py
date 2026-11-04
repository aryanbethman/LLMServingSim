"""Arrival processes shared by every workload generator.

A generator draws the start times of its top-level rows (flat requests or
agent sessions) here and nowhere else, so the same spec gives the same times
in every generator. Arrivals of later calls inside an agent session are not
drawn: the simulator releases them when their predecessor completes plus the
recorded tool delay.

Conventions, for every process:

* The first arrival is at exactly ``start_ns``. Later arrivals follow by the
  process's gaps, so ``n`` arrivals span ``n - 1`` gaps.
* Times are the rounded cumulative sum of the gaps, in integer nanoseconds,
  and never decrease.
* Arrivals have their own random stream, ``default_rng([seed, ARRIVAL_STREAM])``.
  Changing how a generator samples its content never changes its arrivals.

Processes:

* ``poisson`` -- independent exponential gaps at ``rate`` per second.
* ``pulse`` -- bursts of ``pulse_n`` arrivals, one burst every
  ``pulse_period_ns`` measured from burst start to burst start. A burst's
  arrivals share its start instant, or with ``pulse_poisson`` follow it with
  exponential gaps at ``rate``. Bursts may then overlap; their union is
  returned in time order.
"""

from __future__ import annotations

import argparse
import dataclasses
import math

import numpy as np

# Distinguishes the arrival stream from any other stream seeded with the same
# seed ("ARRV").
ARRIVAL_STREAM = 0x41525256
PROCESSES = ("poisson", "pulse")


@dataclasses.dataclass(frozen=True)
class ArrivalSpec:
    process: str
    rate: float  # arrivals / sec; for pulse, the within-burst rate (pulse_poisson only)
    seed: int
    unit: str  # what one arrival starts: "requests" or "sessions"
    start_ns: int = 0
    pulse_n: int = 0
    pulse_period_ns: int = 0
    pulse_poisson: bool = False

    def __post_init__(self):
        if self.process not in PROCESSES:
            raise ValueError(f"unknown arrival process {self.process!r}; expected one of {PROCESSES}")
        if not math.isfinite(self.rate) or self.rate <= 0:
            raise ValueError(f"arrival rate must be finite and positive, got {self.rate}")
        if self.start_ns < 0:
            raise ValueError(f"first arrival must be nonnegative, got {self.start_ns} ns")
        if self.process == "pulse" and (self.pulse_n <= 0 or self.pulse_period_ns <= 0):
            raise ValueError("pulse arrivals need a positive burst size and period")

    def mean_rate(self) -> float:
        """Long-run arrivals per second."""
        if self.process == "pulse":
            return self.pulse_n * 1e9 / self.pulse_period_ns
        return self.rate

    def to_dict(self) -> dict:
        fields = dataclasses.asdict(self)
        if self.process != "pulse":
            for name in ("pulse_n", "pulse_period_ns", "pulse_poisson"):
                del fields[name]
        fields["mean_rate_per_sec"] = self.mean_rate()
        return fields


def arrival_times_ns(spec: ArrivalSpec, num: int) -> list[int]:
    """The first ``num`` arrival times of ``spec``, in nanoseconds."""
    if num < 0:
        raise ValueError(f"arrival count must be nonnegative, got {num}")
    rng = np.random.default_rng([spec.seed, ARRIVAL_STREAM])
    if spec.process == "poisson":
        return _from_gaps(spec.start_ns, rng.exponential(1e9 / spec.rate, size=max(num - 1, 0)), num)

    bursts = []
    for burst in range(math.ceil(num / spec.pulse_n)):
        size = min(spec.pulse_n, num - burst * spec.pulse_n)
        begin = spec.start_ns + burst * spec.pulse_period_ns
        gaps = (rng.exponential(1e9 / spec.rate, size=size - 1) if spec.pulse_poisson
                else np.zeros(size - 1))
        bursts.extend(_from_gaps(begin, gaps, size))
    return sorted(bursts)


def _from_gaps(start_ns: int, gaps_ns: np.ndarray, num: int) -> list[int]:
    if num == 0:
        return []
    offsets = np.rint(np.concatenate(([0.0], np.cumsum(gaps_ns)))).astype(np.int64)
    return (start_ns + offsets).tolist()


def add_arrival_args(p: argparse.ArgumentParser, unit: str) -> None:
    """Registers the arrival flags every generator shares."""
    p.add_argument("--sps", type=float, required=True,
                   help=f"Mean arrival rate ({unit} / sec) of the Poisson process; "
                        "with --pulse --pulse-poisson, the rate inside each burst.")
    p.add_argument("--first-arrival-sec", type=float, default=0,
                   dest="first_arrival_sec",
                   help=f"Time (seconds) of the first {unit[:-1]}'s arrival. Default 0.")
    p.add_argument("--pulse", action="store_true", default=False,
                   help=f"Burst arrivals: --pulse-n {unit} per burst, one burst "
                        "every --pulse-delay-sec seconds.")
    p.add_argument("--pulse-n", type=int, default=10,
                   dest="pulse_n",
                   help=f"(--pulse) {unit} per burst. Default 10.")
    p.add_argument("--pulse-delay-sec", type=float, default=60,
                   dest="pulse_delay_sec",
                   help="(--pulse) seconds from one burst's start to the next's. Default 60.")
    p.add_argument("--pulse-poisson", action="store_true", default=False,
                   dest="pulse_poisson",
                   help="(--pulse) spread each burst with exponential gaps at --sps "
                        "instead of starting it all at once.")


def spec_from_args(args: argparse.Namespace, unit: str) -> ArrivalSpec:
    """Builds the spec from add_arrival_args flags; absent pulse flags mean Poisson."""
    pulse = getattr(args, "pulse", False)
    if not math.isfinite(args.first_arrival_sec):
        raise ValueError("first-arrival-sec must be finite")
    return ArrivalSpec(
        process="pulse" if pulse else "poisson",
        rate=args.sps,
        seed=args.seed,
        unit=unit,
        start_ns=round(args.first_arrival_sec * 1e9),
        pulse_n=args.pulse_n if pulse else 0,
        pulse_period_ns=round(args.pulse_delay_sec * 1e9) if pulse else 0,
        pulse_poisson=bool(pulse and args.pulse_poisson),
    )
