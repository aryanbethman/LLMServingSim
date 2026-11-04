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

## Measured arrivals

`analysis/arrival_fit.py` fits these processes to a timestamp trace. It
reports maximum-likelihood gap fits with AIC and Kolmogorov-Smirnov distance
`D`, gap autocorrelation, count dispersion per window and the rate over time.
It also prints ready-to-use generator flags:

```bash
python analysis/arrival_fit.py BurstGPT_1.csv --preset burstgpt \
    --where Model=ChatGPT --where "Log Type=Conversation log"
```

Two numbers say the most. Count dispersion is the variance over the mean of
arrivals per window: 1 at every window for Poisson. For any renewal process,
gap autocorrelation is 0.

| Trace (subset) | Arrivals | Span | Rate/s | Gap CV | Dispersion at 10 s / 60 s / 1 h | Lag-1 gap autocorr. | Best gap fit (`D`) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Azure 2023 conversation | 19,366 | 1 h | 5.53 | 1.09 | 3.9 / 16 / - | 0.06 | Weibull, cv 1.09 (0.016) |
| Azure 2023 code | 8,819 | 1 h | 2.57 | 13.2 | 72 / 164 / - | 0.00 | none fits (Weibull 0.14) |
| Mooncake conversation (3 s bins) | 12,031 | 1 h | 3.40 | - | 1.5 / 2.9 / - | - | binned; see below |
| Mooncake tool-agent (3 s bins) | 23,608 | 1 h | 6.67 | - | 2.1 / 3.7 / - | - | binned; see below |
| BurstGPT ChatGPT conversation | 103,111 | 61 d | 0.020 | 4.9 | 1.2 / 2.1 / 55 | 0.18 | lognormal (0.048) |
| BurstGPT GPT-4 conversation | 51,108 | 61 d | 0.010 | 4.5 | 1.1 / 1.6 / 31 | 0.16 | lognormal (0.041) |
| BurstGPT ChatGPT API | 1,106,998 | 59 d | 0.22 | 98 | 29 / 165 / 6,721 | 0.10 | lognormal (0.048) |
| BurstGPT GPT-4 API | 168,520 | 59 d | 0.033 | 34 | 13 / 62 / 1,145 | 0.14 | Weibull (0.065) |

What the fits show:

1. **Interactive traffic is Poisson over seconds.** Over windows of 10 s or
   less, conversation traffic has dispersion of about 1 to 1.5:
   - BurstGPT conversation is 1.1-1.2 at 10 s.
   - Mooncake's 3 s bins have variance over mean 1.1 (conversation) and 1.6
     (tool-agent).
   - Divide Azure conversation's gaps by each minute's mean gap and their CV
     is 1.03.
2. **The rate itself moves.** Dispersion grows with the window in every
   trace. Long traces mix rates, which gives the positive gap autocorrelation
   and the heavy-tailed raw gaps. Lognormal "wins" BurstGPT because of this
   mixing, not because the gaps have a lognormal shape. Over the day,
   conversation traffic has a working-day profile with a midday dip and a
   peak-to-trough ratio of about 45. The script prints it as a periodic
   `nhpp` schedule with hours counted from the trace's start. API traffic
   follows a different profile.
3. **Programmatic traffic is a different process.** API logs and code
   completion are bursty even at 10 s: dispersion 13-72, gap CV 13-98. No
   single renewal distribution fits them. Bursts of calls, modeled as `mmpp`
   or `pulse`, fit them better than gap shape does.

The model for interactive traffic is therefore Poisson with a time-varying
rate. Use `poisson` within a measurement window that is short against the
rate's drift, and `nhpp` (from `daily_profile.flags`) for longer runs. Use
`mmpp` for API- or agent-driven bursts. High-CV renewal processes reproduce
the raw gap histogram of a long trace, but they get the reason wrong. They
put the burstiness in individual gaps, while in the traces it comes from the
rate changing.

:::caution[Binned and quantized timestamps]
Mooncake's conversation and tool-agent traces bucket arrivals into 3 s bins.
Its `arxiv` trace has the same per-bin counts as tool-agent, on 3.053 s bins.
The script spreads arrivals uniformly within a bin before taking gaps, so
those traces' gap fits come out Poisson by construction. Only their count
dispersion at windows of 6 s and over is evidence. BurstGPT timestamps are
whole seconds, and 46% of its arrivals share a second with another.
:::

Sources (downloaded 2026-11-04): BurstGPT v2.0 release `BurstGPT_1.csv`
(sha256 `4bb37836...`), Azure Public Dataset
`AzureLLMInferenceTrace_{conv,code}.csv` (`2f1e5b66...`, `54e9a6d2...`) and
Mooncake `FAST25-release/traces/{conversation,toolagent}_trace.jsonl`
(`b8cbb061...`, `48a2db1a...`).

### Recorded tool delays

`analysis/tool_delay_audit.py` splits a SWE-bench run's recorded tool delays
by tool and command. For each step it also checks that it reproduces the
swebench generator's delays exactly. For the default Qwen3-Coder OpenHands
run (500 trajectories, 30,287 steps):

| Tool | Share of steps | p50 | p90 | p99 |
| --- | --- | --- | --- | --- |
| `run` (bash) | 57% | 0.623 s | 1.13 s | 4.15 s |
| `read` | 22% | 6 ms | 16 ms | 19 ms |
| `edit` | 16% | 6 ms | 17 ms | 32 ms |
| `think` | 6% | 1 ms | 2 ms | 3 ms |

The delays are real harness latencies, not corrupted timestamps, but bash
delays measure OpenHands' polling more than the command. Every bash command
costs at least 0.62 s, `pwd` and `ls` included, and 97% of the longer ones
land at 0.62 + 0.5k s. Tool time is 7.7% of recorded session time. The other
92% is the gap from a tool result to the next LLM response, which the
simulator models itself. Use `--tool-delay` to test a different harness, for
example `--tool-delay scale --tool-delay-factor 0` for a zero-latency one.
