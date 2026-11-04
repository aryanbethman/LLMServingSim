"""Arrival model shared by every workload generator.

The model has two layers (see docs/docs/workloads/arrival-processes.md):

1. Top-level arrivals -- the start times of flat requests or agent sessions --
   are drawn here by ``arrival_times_ns`` from an ``ArrivalSpec``.
2. Later calls inside an agent session are not drawn. The simulator releases
   call k+1 at (completion of call k) + (tool delay k). ``tool_delays_ns``
   only decides which tool delays a workload carries.

Conventions, for every process:

* The first arrival is at exactly ``start_ns``, except for ``nhpp``, whose
  rate schedule is anchored at ``start_ns`` and whose first arrival is the
  process's first event after it.
* Times are integer nanoseconds, rounded to nearest, and never decrease.
* The first ``n`` arrivals do not depend on how many are asked for.
* Arrivals have their own random stream, ``default_rng([seed, ARRIVAL_STREAM])``,
  and tool delays another, so neither moves when content sampling changes.

Processes and their ``params`` (rates are per second):

* ``poisson`` -- ``rate``. Independent exponential gaps.
* ``gamma`` / ``weibull`` / ``lognormal`` -- ``rate``, ``cv``. Renewal process
  whose i.i.d. gaps have mean 1/rate and coefficient of variation ``cv``.
  ``cv`` > 1 is burstier than Poisson, < 1 more regular.
* ``pulse`` -- ``n``, ``period_ns``, optional ``burst_rate``. Bursts of ``n``
  arrivals every ``period_ns`` (start to start). A burst shares one instant,
  or with ``burst_rate`` follows its start with exponential gaps at that rate.
* ``mmpp`` -- ``rate``, ``high_sec``, ``low_sec``, ``low_frac``. Two-state
  Markov-modulated Poisson process: exponential sojourns with the given means,
  Poisson arrivals at ``high_rate`` / ``low_rate = low_frac * high_rate``,
  scaled so the long-run mean is ``rate``. ``low_frac = 0`` is on/off traffic.
* ``nhpp`` -- ``schedule`` [(t_sec, rate), ...] from t=0, optional
  ``period_sec``. Non-homogeneous Poisson process whose rate is linear between
  schedule points and holds its last value, or with ``period_sec`` repeats the
  schedule, returning linearly to the first rate at t = ``period_sec``.
* ``trace`` -- ``path``, ``sha256``, optional ``rate``. Replays timestamps
  (seconds, one per line) shifted to ``start_ns`` and, with ``rate``, sped up
  or slowed down uniformly to that mean rate.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import math
from pathlib import Path

import numpy as np

# Distinguish these streams from any other stream seeded with the same seed
# ("ARRV", "TOOL").
ARRIVAL_STREAM = 0x41525256
TOOL_DELAY_STREAM = 0x544F4F4C
RENEWAL = ("gamma", "weibull", "lognormal")
PROCESSES = ("poisson", *RENEWAL, "pulse", "mmpp", "nhpp", "trace")
TOOL_DELAY_MODES = ("recorded", "scale", "constant", "lognormal")


@dataclasses.dataclass(frozen=True)
class ArrivalSpec:
    process: str
    seed: int
    unit: str  # what one arrival starts: "requests" or "sessions"
    start_ns: int = 0
    params: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.process not in PROCESSES:
            raise ValueError(f"unknown arrival process {self.process!r}; expected one of {PROCESSES}")
        if self.start_ns < 0:
            raise ValueError(f"first arrival must be nonnegative, got {self.start_ns} ns")
        p = self.params
        if self.process in ("poisson", "mmpp", *RENEWAL):
            _positive(p, "rate")
        if self.process in RENEWAL:
            _positive(p, "cv")
        if self.process == "pulse":
            _positive(p, "n")
            _positive(p, "period_ns")
            if "burst_rate" in p:
                _positive(p, "burst_rate")
        if self.process == "mmpp":
            _positive(p, "high_sec")
            _positive(p, "low_sec")
            if not 0 <= p.get("low_frac", -1) < 1:
                raise ValueError("mmpp low_frac must be in [0, 1)")
        if self.process == "nhpp":
            _check_schedule(p.get("schedule"), p.get("period_sec"))
        if self.process == "trace":
            if "rate" in p:
                _positive(p, "rate")
            if "path" not in p:
                raise ValueError("a trace spec needs a path")
            times = _trace_times(p["path"])
            if p.get("sha256") != _sha256(p["path"]):
                raise ValueError(f"trace {p['path']} does not match its recorded sha256")
            if len(times) < 2 or times[-1] <= times[0]:
                raise ValueError("a trace needs at least two distinct timestamps")

    def mean_rate(self) -> float:
        """Long-run arrivals per second."""
        p = self.params
        if self.process == "pulse":
            return p["n"] * 1e9 / p["period_ns"]
        if self.process == "nhpp":
            points = _schedule_points(p["schedule"], p.get("period_sec"))
            if p.get("period_sec") is None:
                return points[-1][1]
            area = sum((t1 - t0) * (r0 + r1) / 2 for (t0, r0), (t1, r1) in zip(points, points[1:]))
            return area / p["period_sec"]
        if self.process == "trace" and "rate" not in p:
            times = _trace_times(p["path"])
            return (len(times) - 1) / (times[-1] - times[0])
        return p["rate"]

    def to_dict(self) -> dict:
        fields = dataclasses.asdict(self)
        if self.process == "mmpp":
            fields["params"]["high_rate"], fields["params"]["low_rate"] = _mmpp_rates(self.params)
        fields["mean_rate_per_sec"] = self.mean_rate()
        return fields


def arrival_times_ns(spec: ArrivalSpec, num: int) -> list[int]:
    """The first ``num`` arrival times of ``spec``, in nanoseconds."""
    if num < 0:
        raise ValueError(f"arrival count must be nonnegative, got {num}")
    if num == 0:
        return []
    rng = np.random.default_rng([spec.seed, ARRIVAL_STREAM])
    p = spec.params
    if spec.process == "poisson":
        offsets = _cumulative(rng.exponential(1e9 / p["rate"], size=num - 1))
    elif spec.process in RENEWAL:
        offsets = _cumulative(_renewal_gaps(spec.process, 1e9 / p["rate"], p["cv"], num - 1, rng))
    elif spec.process == "pulse":
        return _pulse(spec, num, rng)
    elif spec.process == "mmpp":
        offsets = _mmpp(p, num, rng)
        offsets -= offsets[0]
    elif spec.process == "nhpp":
        offsets = _nhpp(p, num, rng)
    else:
        offsets = _trace(p, num)
    return (spec.start_ns + np.rint(offsets).astype(np.int64)).tolist()


def _cumulative(gaps_ns: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(gaps_ns)))


def _renewal_gaps(process: str, mean_ns: float, cv: float, size: int, rng) -> np.ndarray:
    if process == "gamma":
        shape = 1 / cv**2
        return rng.gamma(shape, mean_ns / shape, size=size)
    if process == "weibull":
        shape = weibull_shape(cv)
        return rng.weibull(shape, size=size) * mean_ns / math.gamma(1 + 1 / shape)
    sigma2 = math.log1p(cv**2)
    return rng.lognormal(math.log(mean_ns) - sigma2 / 2, math.sqrt(sigma2), size=size)


def weibull_shape(cv: float) -> float:
    """The Weibull shape whose coefficient of variation is ``cv``."""
    def cv_of(k):
        return math.sqrt(math.expm1(math.lgamma(1 + 2 / k) - 2 * math.lgamma(1 + 1 / k)))
    lo, hi = 0.05, 100.0  # cv from ~4e5 down to ~0.013
    if not cv_of(hi) <= cv <= cv_of(lo):
        raise ValueError(f"weibull cv {cv} is outside the supported range")
    for _ in range(200):  # cv decreases in k; bisect on log k
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if cv_of(mid) > cv else (lo, mid)
    return math.sqrt(lo * hi)


def _pulse(spec: ArrivalSpec, num: int, rng) -> list[int]:
    p = spec.params
    times = []
    for burst in range(math.ceil(num / p["n"])):
        size = min(p["n"], num - burst * p["n"])
        begin = spec.start_ns + burst * p["period_ns"]
        gaps = (rng.exponential(1e9 / p["burst_rate"], size=size - 1) if "burst_rate" in p
                else np.zeros(size - 1))
        times.extend((begin + np.rint(_cumulative(gaps)).astype(np.int64)).tolist())
    return sorted(times)


def _mmpp_rates(p: dict) -> tuple[float, float]:
    high_share = p["high_sec"] / (p["high_sec"] + p["low_sec"])
    high = p["rate"] / (high_share + p["low_frac"] * (1 - high_share))
    return high, p["low_frac"] * high


def _mmpp(p: dict, num: int, rng) -> np.ndarray:
    rates = _mmpp_rates(p)
    sojourn_ns = (p["high_sec"] * 1e9, p["low_sec"] * 1e9)
    state = 0 if rng.random() < sojourn_ns[0] / sum(sojourn_ns) else 1  # stationary start
    left = rng.exponential(sojourn_ns[state])
    t, out = 0.0, []
    while len(out) < num:
        gap = rng.exponential(1e9 / rates[state]) if rates[state] > 0 else math.inf
        if gap < left:
            t += gap
            left -= gap
            out.append(t)
        else:  # memoryless: the gap redrawn in the next state is exact
            t += left
            state ^= 1
            left = rng.exponential(sojourn_ns[state])
    return np.array(out)


def _check_schedule(schedule, period_sec):
    if not schedule:
        raise ValueError("nhpp needs a nonempty rate schedule")
    ts = [t for t, _ in schedule]
    if ts[0] != 0 or any(b <= a for a, b in zip(ts, ts[1:])):
        raise ValueError("nhpp schedule times must start at 0 and increase")
    if any(not math.isfinite(r) or r < 0 for _, r in schedule) or not any(r > 0 for _, r in schedule):
        raise ValueError("nhpp schedule rates must be finite, nonnegative and not all zero")
    if period_sec is None:
        if schedule[-1][1] <= 0:
            raise ValueError("a non-periodic nhpp schedule must end at a positive rate")
    elif period_sec <= ts[-1]:
        raise ValueError("nhpp period must be after the last schedule point")


def _schedule_points(schedule, period_sec):
    points = [(float(t), float(r)) for t, r in schedule]
    if period_sec is not None:
        points.append((float(period_sec), points[0][1]))
    return points


def _nhpp(p: dict, num: int, rng) -> np.ndarray:
    """Lewis-Shedler thinning of a Poisson process at the schedule's peak rate."""
    points = _schedule_points(p["schedule"], p.get("period_sec"))
    ts = np.array([t for t, _ in points])
    rs = np.array([r for _, r in points])
    peak = rs.max()
    period = p.get("period_sec")
    t, out = 0.0, []
    while len(out) < num:
        t += rng.exponential(1 / peak)
        phase = t % period if period is not None else t
        if rng.random() * peak < np.interp(phase, ts, rs):
            out.append(t * 1e9)
    return np.array(out)


def _trace_times(path) -> list[float]:
    values = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            values.append(float(line))
    return sorted(values)


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _trace(p: dict, num: int) -> np.ndarray:
    times = np.array(_trace_times(p["path"]))
    if num > len(times):
        raise ValueError(f"trace {p['path']} has {len(times)} arrivals, {num} requested")
    offsets = (times - times[0]) * 1e9
    if "rate" in p:
        native = (len(times) - 1) / (times[-1] - times[0])
        offsets *= native / p["rate"]
    return offsets[:num]


def _positive(p: dict, key: str):
    value = p.get(key)
    if value is None or not math.isfinite(value) or value <= 0:
        raise ValueError(f"arrival parameter {key} must be finite and positive, got {value}")


# ---------------------------------------------------------------------------
# Tool delays inside agent sessions
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ToolDelaySpec:
    """How a workload's tool delays relate to the recorded ones.

    ``recorded`` keeps them. ``scale`` multiplies them by ``factor``.
    ``constant`` sets every one to ``sec``. ``lognormal`` draws them i.i.d.
    with median ``sec`` and log-space standard deviation ``sigma``. The last
    call of a session always keeps delay 0: nothing follows it.
    """
    mode: str = "recorded"
    seed: int = 0
    params: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.mode not in TOOL_DELAY_MODES:
            raise ValueError(f"unknown tool-delay mode {self.mode!r}; expected one of {TOOL_DELAY_MODES}")
        if self.mode == "scale":
            if not math.isfinite(self.params.get("factor", -1)) or self.params.get("factor", -1) < 0:
                raise ValueError("tool-delay factor must be finite and nonnegative")
        if self.mode == "constant":
            if not math.isfinite(self.params.get("sec", -1)) or self.params.get("sec", -1) < 0:
                raise ValueError("tool-delay seconds must be finite and nonnegative")
        if self.mode == "lognormal":
            _positive(self.params, "sec")
            _positive(self.params, "sigma")

    def sampler(self):
        """Returns delays(recorded_ns_list) -> new_ns_list, drawing sequentially."""
        rng = np.random.default_rng([self.seed, TOOL_DELAY_STREAM])

        def delays(recorded: list[int]) -> list[int]:
            n = len(recorded)
            if self.mode == "recorded":
                new = np.array(recorded, dtype=float)
            elif self.mode == "scale":
                new = np.array(recorded, dtype=float) * self.params["factor"]
            elif self.mode == "constant":
                new = np.full(n, self.params["sec"] * 1e9)
            else:
                new = rng.lognormal(math.log(self.params["sec"] * 1e9),
                                    self.params["sigma"], size=n)
            new = np.rint(new).astype(np.int64)
            if n:
                new[-1] = 0
            return new.tolist()
        return delays

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------

def add_arrival_args(p: argparse.ArgumentParser, unit: str) -> None:
    """Registers the arrival flags every generator shares."""
    one = unit[:-1]
    p.add_argument("--arrival-process", choices=PROCESSES, default=None, dest="arrival_process",
                   help=f"Process for {one} arrivals. Default poisson (pulse with --pulse).")
    p.add_argument("--sps", type=float, default=None,
                   help=f"Mean arrival rate ({unit} / sec). Required for poisson, gamma, "
                        "weibull, lognormal and mmpp; for trace, rescales the trace to this "
                        "mean; with --pulse --pulse-poisson, the rate inside each burst.")
    p.add_argument("--first-arrival-sec", type=float, default=0,
                   dest="first_arrival_sec",
                   help=f"Time (seconds) of the first {one}'s arrival; for nhpp, where "
                        "its schedule starts. Default 0.")
    p.add_argument("--arrival-cv", type=float, default=1.0, dest="arrival_cv",
                   help="(gamma/weibull/lognormal) coefficient of variation of the gaps. "
                        "1 matches Poisson burstiness; larger is burstier. Default 1.")
    p.add_argument("--pulse", action="store_true", default=False,
                   help=f"Burst arrivals: --pulse-n {unit} per burst, one burst "
                        "every --pulse-delay-sec seconds. Same as --arrival-process pulse.")
    p.add_argument("--pulse-n", type=int, default=10,
                   dest="pulse_n",
                   help=f"(pulse) {unit} per burst. Default 10.")
    p.add_argument("--pulse-delay-sec", type=float, default=60,
                   dest="pulse_delay_sec",
                   help="(pulse) seconds from one burst's start to the next's. Default 60.")
    p.add_argument("--pulse-poisson", action="store_true", default=False,
                   dest="pulse_poisson",
                   help="(pulse) spread each burst with exponential gaps at --sps "
                        "instead of starting it all at once.")
    p.add_argument("--mmpp-high-sec", type=float, default=10, dest="mmpp_high_sec",
                   help="(mmpp) mean seconds in the high-rate state. Default 10.")
    p.add_argument("--mmpp-low-sec", type=float, default=50, dest="mmpp_low_sec",
                   help="(mmpp) mean seconds in the low-rate state. Default 50.")
    p.add_argument("--mmpp-low-frac", type=float, default=0.1, dest="mmpp_low_frac",
                   help="(mmpp) low-state rate as a fraction of the high-state rate; "
                        "0 is on/off. Default 0.1.")
    p.add_argument("--rate-schedule", default=None, dest="rate_schedule",
                   help="(nhpp) comma-separated t_sec:rate points starting at 0, "
                        "e.g. 0:0.2,300:1,600:0.2. Linear between points.")
    p.add_argument("--rate-schedule-period-sec", type=float, default=None,
                   dest="rate_schedule_period_sec",
                   help="(nhpp) repeat the schedule with this period (seconds).")
    p.add_argument("--arrival-trace", default=None, dest="arrival_trace",
                   help="(trace) file of arrival timestamps in seconds, one per line.")


def parse_schedule(text: str) -> list[tuple[float, float]]:
    points = []
    for item in text.split(","):
        t, _, rate = item.partition(":")
        points.append((float(t), float(rate)))
    return points


def spec_from_args(args: argparse.Namespace, unit: str) -> ArrivalSpec:
    """Builds the spec from add_arrival_args flags; flags a caller lacks take defaults."""
    def get(name, default=None):
        return getattr(args, name, default)

    process = get("arrival_process") or ("pulse" if get("pulse") else "poisson")
    if get("pulse") and process != "pulse":
        raise ValueError(f"--pulse conflicts with --arrival-process {process}")
    first = get("first_arrival_sec", 0)
    if not math.isfinite(first):
        raise ValueError("first-arrival-sec must be finite")
    rate = get("sps")
    if process in ("poisson", "mmpp", *RENEWAL) and rate is None:
        raise ValueError(f"--sps is required for --arrival-process {process}")

    if process == "poisson":
        params = {"rate": rate}
    elif process in RENEWAL:
        params = {"rate": rate, "cv": get("arrival_cv", 1.0)}
    elif process == "pulse":
        params = {"n": get("pulse_n", 10), "period_ns": round(get("pulse_delay_sec", 60) * 1e9)}
        if get("pulse_poisson"):
            if rate is None:
                raise ValueError("--pulse-poisson needs --sps")
            params["burst_rate"] = rate
    elif process == "mmpp":
        params = {"rate": rate, "high_sec": get("mmpp_high_sec", 10),
                  "low_sec": get("mmpp_low_sec", 50), "low_frac": get("mmpp_low_frac", 0.1)}
    elif process == "nhpp":
        if not get("rate_schedule"):
            raise ValueError("--arrival-process nhpp needs --rate-schedule")
        params = {"schedule": parse_schedule(get("rate_schedule")),
                  "period_sec": get("rate_schedule_period_sec")}
    else:
        if not get("arrival_trace"):
            raise ValueError("--arrival-process trace needs --arrival-trace")
        path = str(Path(get("arrival_trace")).resolve())
        params = {"path": path, "sha256": _sha256(path)}
        if rate is not None:
            params["rate"] = rate
    return ArrivalSpec(process=process, seed=args.seed, unit=unit,
                       start_ns=round(first * 1e9), params=params)


def add_tool_delay_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tool-delay", choices=TOOL_DELAY_MODES, default="recorded", dest="tool_delay",
                   help="Tool delays between a session's calls: keep the recorded ones, "
                        "scale them, set them constant or draw them lognormal. Default recorded.")
    p.add_argument("--tool-delay-factor", type=float, default=1.0, dest="tool_delay_factor",
                   help="(scale) multiplier on recorded delays. Default 1.")
    p.add_argument("--tool-delay-sec", type=float, default=None, dest="tool_delay_sec",
                   help="(constant) every delay; (lognormal) the median delay, in seconds.")
    p.add_argument("--tool-delay-sigma", type=float, default=1.0, dest="tool_delay_sigma",
                   help="(lognormal) standard deviation of log delay. Default 1.")


def tool_delay_spec_from_args(args: argparse.Namespace) -> ToolDelaySpec:
    mode = getattr(args, "tool_delay", "recorded")
    if mode == "recorded":
        return ToolDelaySpec(seed=args.seed)
    if mode == "scale":
        return ToolDelaySpec(mode, args.seed, {"factor": args.tool_delay_factor})
    if args.tool_delay_sec is None:
        raise ValueError(f"--tool-delay {mode} needs --tool-delay-sec")
    params = {"sec": args.tool_delay_sec}
    if mode == "lognormal":
        params["sigma"] = args.tool_delay_sigma
    return ToolDelaySpec(mode, args.seed, params)
