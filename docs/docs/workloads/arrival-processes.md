---
sidebar_position: 5
title: Arrival processes
---

# Arrival processes

Every workload generator draws arrival times from one module,
`workloads/generators/arrivals.py`. This page defines what those times mean,
which processes are available, and what they imply for the load a cluster
sees.

## The model

A workload has two layers of arrivals. Only the first is drawn by a generator.

**Top-level arrivals.** Rows are flat requests or agent sessions. Row `i`
starts at `S_i`, and `S_1 <= S_2 <= ...` are the points of an arrival
process with long-run rate `lambda` (`--sps`, per second).

**Arrivals inside a session.** Session `i` makes calls `k = 1..K_i`. Each
call has recorded input and output lengths, and a tool delay `T_ik` follows
it (`tool_duration_ns`; 0 after the last call). The simulator releases the
calls in a closed loop:

```text
A_i1     = S_i
A_i(k+1) = C_ik + T_ik
```

`C_ik` is the time call `k` completes in the simulation. Request arrivals are
therefore an output of the simulation, not an input. They depend on queuing,
batching and every other modeled cost.

Three consequences follow, and they hold for every process below.

1. **Call rate.** If the cluster keeps up, every arrived session eventually
   releases all of its calls. The long-run call rate is then
   `lambda * E[K]`, whatever the service times. The closed loop changes when
   calls arrive and how they correlate, not how many arrive per second.
2. **Concurrency.** By Little's law the mean number of live sessions is
   `lambda * E[D]`, where `D` is a session's lifetime. `D` is at least the
   sum of its tool delays, so `lambda * E[sum_k T_ik]` is a lower bound
   that needs no simulation.
3. **Utilization.** `rho = lambda * E[S] / R` for `R` replicas, where
   `E[S]` is the replica time one session costs. `E[S]` depends on batching
   and prefix-cache hits, so it is measured from a run, not assumed.

Even with Poisson session starts, calls are not Poisson. A session's calls
are spaced by its own service time plus tool delay, so they cluster in time.

## Conventions

- The first arrival is at exactly `--first-arrival-sec`, which accepts
  fractional seconds. The exception is `nhpp`, whose schedule starts at that
  time; its first arrival is the process's first event after it.
- Times are integer nanoseconds, rounded to nearest, and never decrease.
- The first `n` arrivals do not depend on how many are requested.
- Arrivals use their own random stream, derived from `--seed`, and tool
  delays use another. Changing filters, sampling or token ids never moves
  them.
- `--sps` is always the long-run mean rate, in requests per second for
  `sharegpt` and sessions per second for `swebench` and
  `replicate-sessions`. Sweeping `--sps` with any process changes the load,
  not the shape.

## Processes

Select a process with `--arrival-process` (default `poisson`). The flags
are the same in all three generators and in `analysis/session_scale.py
prepare`.

| Process | Flags | Definition | Use |
| --- | --- | --- | --- |
| `poisson` | `--sps` | i.i.d. exponential gaps, mean `1/lambda` | baseline; memoryless open traffic |
| `gamma` | `--sps`, `--arrival-cv` | i.i.d. Gamma gaps, mean `1/lambda`, coefficient of variation `cv` | one-knob burstiness at fixed rate |
| `weibull` | same | i.i.d. Weibull gaps with that mean and `cv` | heavier tail than Gamma for `cv > 1` |
| `lognormal` | same | i.i.d. lognormal gaps with that mean and `cv` | heaviest tail of the three |
| `pulse` | `--pulse-n`, `--pulse-delay-sec`, optional `--pulse-poisson` with `--sps` | `n` arrivals every period, start to start; at one instant, or spread at rate `--sps` | synchronized clients, cron-like spikes |
| `mmpp` | `--sps`, `--mmpp-high-sec`, `--mmpp-low-sec`, `--mmpp-low-frac` | two-state Markov-modulated Poisson process | bursts that last a while |
| `nhpp` | `--rate-schedule`, optional `--rate-schedule-period-sec` | Poisson with a time-varying rate | ramps, daily cycles |
| `trace` | `--arrival-trace`, optional `--sps` | replay of recorded timestamps | measured production arrivals |

`--pulse` is shorthand for `--arrival-process pulse`.

### Renewal processes

For `gamma`, `weibull` and `lognormal` the gaps are i.i.d. with mean
`m = 1/lambda` and coefficient of variation `cv` (standard deviation over
mean). `cv = 1` has the same burstiness as Poisson, `cv > 1` is burstier and
`cv < 1` more regular. Gamma with `cv = 1` is the exponential distribution;
so is Weibull.

| Distribution | Parameters from `m` and `cv` |
| --- | --- |
| Gamma | shape `k = 1/cv^2`, scale `m/k` |
| Weibull | shape `k` solving `Gamma(1+2/k)/Gamma(1+1/k)^2 = 1 + cv^2`, scale `m/Gamma(1+1/k)` |
| lognormal | `sigma^2 = ln(1 + cv^2)`, `mu = ln(m) - sigma^2/2` |

### MMPP

The process alternates between a high and a low state with exponential
sojourns, means `h = --mmpp-high-sec` and `l = --mmpp-low-sec`. It starts in
the high state with probability `p = h/(h+l)`, the long-run fraction of time
spent there. Arrivals are Poisson at `r_high` in the high state and
`r_low = f * r_high` in the low state, `f = --mmpp-low-frac`. The rates are
set so the mean equals `lambda`:

```text
r_high = lambda / (p + f * (1 - p))
```

`f = 0` gives on/off traffic. The manifest records `high_rate` and
`low_rate`. Counts over windows comparable to `h` and `l` are overdispersed:
their variance-to-mean ratio is well above Poisson's 1.

### NHPP

`--rate-schedule 0:0.2,300:1,600:0.2` gives `(t_sec, rate)` points starting
at `t = 0`. The rate is linear between points. Without a period it holds
the last rate forever, which must be positive. With
`--rate-schedule-period-sec P` the schedule repeats every `P` seconds and
returns linearly to the first rate at `t = P`. The mean rate is the
schedule's time average over one period (or its last rate). Times are drawn
by thinning a Poisson process at the schedule's peak rate.

### Trace replay

`--arrival-trace FILE` reads timestamps in seconds, one per line. Blank lines
and `#` comments are skipped, and timestamps are sorted. The trace is shifted
so its first timestamp lands at `--first-arrival-sec`. With `--sps` it is
also stretched or compressed uniformly so its mean rate is `--sps`. Gaps are
then scaled by `native_rate / sps`, and the relative pattern is unchanged.
Asking for more arrivals than the trace holds is an error. The manifest
records the trace's absolute path and SHA-256. A spec whose file has changed
is rejected.

## Tool delays

`replicate-sessions` (and `session_scale.py prepare`) can replace the
recorded `T_ik` with `--tool-delay`:

| Mode | Flags | Delays |
| --- | --- | --- |
| `recorded` | (default) | as recorded |
| `scale` | `--tool-delay-factor` | recorded times the factor |
| `constant` | `--tool-delay-sec` | that value |
| `lognormal` | `--tool-delay-sec`, `--tool-delay-sigma` | i.i.d. lognormal, median `--tool-delay-sec`, log-space s.d. `--tool-delay-sigma` |

The last call of every session keeps delay 0. Lengths, prefixes and call
order never change.

## Manifests

Each generator writes `<output>.manifest.json` with the arrival spec and, for
`replicate-sessions`, the tool-delay spec:

```json
{
  "arrival": {
    "process": "gamma",
    "seed": 42,
    "unit": "sessions",
    "start_ns": 0,
    "params": {"rate": 0.8, "cv": 3.0},
    "mean_rate_per_sec": 0.8
  },
  "tool_delay": {"mode": "scale", "seed": 42, "params": {"factor": 4.0}}
}
```

## Offered load

`analysis/offered_load.py` reports what a workload file asks of the cluster
before any simulation:

```bash
python analysis/offered_load.py outputs/scale256/sessions.jsonl --replicas 64
```

| Field | Meaning |
| --- | --- |
| `spec_mean_rate_per_sec` | `lambda` from the manifest |
| `empirical_rate_per_sec` | `(N-1)` over the span of the file's start times |
| `per_row.calls`, `input_toks`, `output_toks` | means per row |
| `per_row.uncached_input_toks` | input a call appends to its predecessor's prompt and output; the prefill left if each session always hits its own history |
| `offered.*_per_sec` | rate times the per-row mean, using the spec rate when known |
| `offered.min_concurrent_rows` | `lambda * E[sum of tool delays]`, the Little's-law lower bound |
| `utilization` | `lambda * E[S] / R`, only with `--service-sec E[S]` |

:::caution[Finite workloads and bursty processes]
A finite file realizes a rate that can differ a lot from `lambda`, more so
the burstier the process. For 200 replicated SWE-bench sessions at
`--sps 0.8`, Gamma with `cv = 3` realized 0.58 sessions/s, and on/off MMPP
1.18 sessions/s. Report both rates. Use enough rows or several seeds that the
measurement window sees the intended load.
:::
