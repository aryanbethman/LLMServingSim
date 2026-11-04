#!/usr/bin/env python3
"""Fit arrival processes to a timestamp trace and report how well each fits.

Reads a request trace, takes its arrival timestamps, and reports:

* rate, span and timestamp resolution. Quantized timestamps (whole seconds,
  milliseconds, or a binned trace's bins) are spread uniformly inside their
  quantum before gaps are taken, so ties do not become zero gaps. Gaps
  shorter than a coarse quantum are then reconstructed, not measured: read
  a binned trace's count dispersion, not its gap fits.
* maximum-likelihood fits of the gaps to exponential, gamma, Weibull and
  lognormal, each with log-likelihood, AIC, the Kolmogorov-Smirnov distance D
  to the data and the gap CV it implies. With large traces every p-value is
  tiny, so compare D and AIC rather than test acceptance.
* renewal checks: gap autocorrelation (0 for any renewal process) and the
  index of dispersion of counts per window (1 for Poisson at every window).
* nonstationarity: the rate per --rate-window, written as an nhpp schedule,
  and the gap CV left after dividing each gap by its window's mean gap. For
  traces of two days or more, the mean hour-of-day profile as a periodic
  nhpp schedule.
* ready-to-use generator flags for the best renewal fit and the schedule.

Presets name the timestamp field of known public traces:
    burstgpt  BurstGPT CSV, "Timestamp" in seconds
    azure     Azure LLM inference 2023 CSV, "TIMESTAMP" datetime
    mooncake  Mooncake JSONL, "timestamp" in milliseconds
    plain     one timestamp in seconds per line (the --arrival-trace format)

Usage:
    python analysis/arrival_fit.py BurstGPT_1.csv --preset burstgpt \
        --where "Log Type=Conversation log" --output fit.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

PRESETS = {
    "burstgpt": ("csv", "Timestamp", "s"),
    "azure": ("csv", "TIMESTAMP", "datetime"),
    "mooncake": ("jsonl", "timestamp", "ms"),
    "plain": ("plain", None, "s"),
}
UNITS = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}
DISPERSION_WINDOWS = (1, 10, 60, 600, 3600)
KS_SIMULATED = 400_000


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_times(path: Path, preset: str, where=()) -> np.ndarray:
    """Arrival times in seconds, sorted, from a trace file."""
    kind, field, unit = PRESETS[preset]
    filters = [w.split("=", 1) for w in where]
    values = []
    if kind == "plain":
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                values.append(float(line))
    else:
        with path.open(newline="") as handle:
            rows = csv.DictReader(handle) if kind == "csv" else map(json.loads, handle)
            for row in rows:
                if all(str(row.get(k)) == v for k, v in filters):
                    values.append(_parse_time(row[field], unit))
    if len(values) < 3:
        raise ValueError(f"{path}: fewer than 3 arrivals after filtering")
    return np.sort(np.array(values, dtype=float))


def _parse_time(value, unit):
    if unit == "datetime":
        # Azure writes 7 fractional digits; datetime takes at most 6.
        head, _, frac = str(value).partition(".")
        return datetime.fromisoformat(f"{head}.{(frac + '000000')[:6]}").timestamp() + int(frac[6:7] or 0) * 1e-7
    return float(value) * UNITS[unit]


def resolution(times: np.ndarray) -> float:
    """The width of the bins timestamps were rounded into (0 if none).

    A binned trace -- most arrivals tied, distinct timestamps evenly spaced --
    has the spacing as its bin. Otherwise this is the largest power-of-ten
    quantum all timestamps are multiples of.
    """
    distinct = np.unique(times)
    if 2 < len(distinct) < len(times) / 2:
        spacing = np.diff(distinct)
        if spacing.std() < 0.05 * spacing.mean():
            return float(np.median(spacing))
    shifted = times - times[0]
    for exponent in range(3, -7, -1):
        quantum = 10.0 ** exponent
        if np.all(np.abs(shifted / quantum - np.rint(shifted / quantum)) < 1e-6):
            return quantum
    return 0.0


def dequantize(times: np.ndarray, quantum: float, seed: int) -> np.ndarray:
    if quantum <= 0:
        return times
    jitter = np.random.default_rng(seed).uniform(0, quantum, size=len(times))
    return np.sort(times + jitter)


# ---------------------------------------------------------------------------
# Distribution fits
# ---------------------------------------------------------------------------

# Recurrence up to x >= 10, then the asymptotic series (error < 1e-15).
def digamma(x: float) -> float:
    result = 0.0
    while x < 10:
        result -= 1 / x
        x += 1
    inv2 = 1 / (x * x)
    series = inv2 * (1 / 12 - inv2 * (1 / 120 - inv2 * (1 / 252 - inv2 * (1 / 240 - inv2 / 132))))
    return result + math.log(x) - 0.5 / x - series


def trigamma(x: float) -> float:
    result = 0.0
    while x < 10:
        result += 1 / (x * x)
        x += 1
    inv = 1 / x
    inv2 = inv * inv
    return result + inv + inv2 / 2 + inv * inv2 * (1 / 6 - inv2 * (1 / 30 - inv2 * (1 / 42 - inv2 / 30)))


def fit_gamma(x: np.ndarray) -> dict:
    mean, mean_log = x.mean(), np.log(x).mean()
    s = math.log(mean) - mean_log
    k = (3 - s + math.sqrt((s - 3) ** 2 + 24 * s)) / (12 * s)
    for _ in range(100):  # Newton on log(k) - digamma(k) = s
        step = (math.log(k) - digamma(k) - s) / (1 / k - trigamma(k))
        k = max(k - step, k / 10)
        if abs(step) < 1e-12 * k:
            break
    theta = mean / k
    n = len(x)
    ll = (k - 1) * np.log(x).sum() - x.sum() / theta - n * (k * math.log(theta) + math.lgamma(k))
    return {"shape": k, "scale": theta, "loglik": float(ll), "cv": 1 / math.sqrt(k)}


def fit_weibull(x: np.ndarray) -> dict:
    logs = np.log(x / x.mean())
    mean_log = logs.mean()

    def score(k):  # increasing in k; zero at the MLE
        w = k * logs
        w = np.exp(w - w.max())
        return float((w * logs).sum() / w.sum()) - 1 / k - mean_log

    lo, hi = 1e-3, 1e3
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        lo, hi = (lo, mid) if score(mid) > 0 else (mid, hi)
    k = math.sqrt(lo * hi)
    scale = x.mean() * float(np.mean(np.exp(k * logs))) ** (1 / k)
    n = len(x)
    ll = n * math.log(k) - n * k * math.log(scale) + (k - 1) * np.log(x).sum() - ((x / scale) ** k).sum()
    cv = math.sqrt(math.expm1(math.lgamma(1 + 2 / k) - 2 * math.lgamma(1 + 1 / k)))
    return {"shape": k, "scale": scale, "loglik": float(ll), "cv": cv}


def fit_lognormal(x: np.ndarray) -> dict:
    logs = np.log(x)
    mu, sigma = logs.mean(), logs.std()
    n = len(x)
    ll = -logs.sum() - n * math.log(sigma * math.sqrt(2 * math.pi)) - ((logs - mu) ** 2).sum() / (2 * sigma**2)
    return {"mu": float(mu), "sigma": float(sigma), "loglik": float(ll), "cv": math.sqrt(math.expm1(sigma**2))}


def fit_exponential(x: np.ndarray) -> dict:
    rate = 1 / x.mean()
    return {"rate": rate, "loglik": float(len(x) * math.log(rate) - rate * x.sum()), "cv": 1.0}


def _simulate(name: str, fit: dict, size: int, rng) -> np.ndarray:
    if name == "exponential":
        return rng.exponential(1 / fit["rate"], size)
    if name == "gamma":
        return rng.gamma(fit["shape"], fit["scale"], size)
    if name == "weibull":
        return rng.weibull(fit["shape"], size) * fit["scale"]
    return rng.lognormal(fit["mu"], fit["sigma"], size)


def ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic."""
    a, b = np.sort(a), np.sort(b)
    points = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, points, side="right") / len(a)
    cdf_b = np.searchsorted(b, points, side="right") / len(b)
    return float(np.abs(cdf_a - cdf_b).max())


def fit_gaps(gaps: np.ndarray, seed: int) -> dict:
    gaps = gaps[gaps > 0]
    rng = np.random.default_rng(seed)
    fits = {"exponential": fit_exponential(gaps), "gamma": fit_gamma(gaps),
            "weibull": fit_weibull(gaps), "lognormal": fit_lognormal(gaps)}
    for name, fit in fits.items():
        params = 1 if name == "exponential" else 2
        fit["aic"] = 2 * params - 2 * fit["loglik"]
        fit["ks_d"] = ks_distance(gaps, _simulate(name, fit, KS_SIMULATED, rng))
    return fits


# ---------------------------------------------------------------------------
# Renewal and stationarity checks
# ---------------------------------------------------------------------------

def autocorrelation(gaps: np.ndarray, lags=(1, 2, 3)) -> dict:
    centered = gaps - gaps.mean()
    denominator = float((centered**2).sum())
    return {str(lag): float((centered[:-lag] * centered[lag:]).sum() / denominator) for lag in lags}


def dispersion(times: np.ndarray, windows=DISPERSION_WINDOWS, quantum: float = 0.0) -> dict:
    """Variance-to-mean ratio of counts per window; windows finer than the
    timestamp quantum, or with under 20 windows in the trace, are skipped."""
    out = {}
    offsets = times - times[0]
    for width in windows:
        bins = int(offsets[-1] // width)
        if bins < 20 or width < 2 * quantum:
            continue
        counts = np.bincount((offsets // width).astype(np.int64), minlength=bins + 1)[:bins]
        out[str(width)] = float(counts.var() / counts.mean()) if counts.mean() > 0 else None
    return out


def rate_profile(times: np.ndarray, window: float) -> dict:
    offsets = times - times[0]
    bins = max(1, int(offsets[-1] // window))
    counts = np.bincount(np.minimum((offsets // window).astype(np.int64), bins - 1), minlength=bins)
    rates = counts / window
    index = np.minimum((offsets[1:] // window).astype(np.int64), bins - 1)
    local_mean_gap = np.where(rates[index] > 0, 1 / np.maximum(rates[index], 1e-300), np.nan)
    rescaled = np.diff(offsets) / local_mean_gap
    rescaled = rescaled[np.isfinite(rescaled) & (rescaled > 0)]
    schedule = ",".join(f"{i * window:g}:{r:.6g}" for i, r in enumerate(rates))
    return {"window_sec": window, "windows": bins, "rate_min": float(rates.min()),
            "rate_max": float(rates.max()), "rate_cv": float(rates.std() / rates.mean()),
            "detrended_gap_cv": float(rescaled.std() / rescaled.mean()), "schedule": schedule}


def daily_profile(times: np.ndarray, bins: int = 24) -> dict | None:
    """Mean rate per hour of day over whole days, as a periodic nhpp schedule.

    Hours count from the trace's first arrival, not a wall clock: traces such
    as BurstGPT store seconds since their start, so the phase is arbitrary.
    """
    offsets = times - times[0]
    days = int(offsets[-1] // 86_400)
    if days < 2:
        return None
    width = 86_400 / bins
    kept = offsets[offsets < days * 86_400]
    rates = np.bincount((kept % 86_400 // width).astype(np.int64), minlength=bins) / (days * width)
    schedule = ",".join(f"{i * width:g}:{r:.6g}" for i, r in enumerate(rates))
    return {"days": days, "relative_rate": (rates / rates.mean()).round(4).tolist(),
            "peak_to_trough": float(rates.max() / rates.min()) if rates.min() > 0 else None,
            "flags": f"--arrival-process nhpp --rate-schedule {schedule} --rate-schedule-period-sec 86400"}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def analyze(times: np.ndarray, rate_window: float | None = None, seed: int = 0) -> dict:
    quantum = resolution(times)
    spread = dequantize(times, quantum, seed)
    gaps = np.diff(spread)
    span = float(spread[-1] - spread[0])
    rate = (len(spread) - 1) / span
    fits = fit_gaps(gaps, seed)
    renewal = min(("gamma", "weibull", "lognormal"), key=lambda name: fits[name]["aic"])
    window = rate_window or (3600.0 if span >= 86_400 else max(60.0, span / 60))
    profile = rate_profile(spread, window)
    return {
        "arrivals": int(len(times)),
        "span_sec": span,
        "rate_per_sec": rate,
        "resolution_sec": quantum,
        "tied_timestamps": int((np.diff(times) == 0).sum()),
        "gap_cv": float(gaps.std() / gaps.mean()),
        "fits": fits,
        "best_by_aic": min(fits, key=lambda name: fits[name]["aic"]),
        "best_by_ks": min(fits, key=lambda name: fits[name]["ks_d"]),
        "gap_autocorrelation": autocorrelation(gaps),
        "count_dispersion": dispersion(spread, quantum=quantum),
        "rate_profile": profile,
        "daily_profile": daily_profile(spread),
        "flags": {
            "renewal": f"--arrival-process {renewal} --sps {rate:.6g} --arrival-cv {fits[renewal]['cv']:.4g}",
            "nhpp": f"--arrival-process nhpp --rate-schedule {profile['schedule']}",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--preset", choices=PRESETS, required=True)
    parser.add_argument("--where", action="append", default=[],
                        help="keep rows whose COLUMN equals VALUE (COLUMN=VALUE); repeatable")
    parser.add_argument("--rate-window", type=float, default=None,
                        help="seconds per rate-profile window; default 3600 for traces over a day, "
                             "else span/60 (at least 60)")
    parser.add_argument("--seed", type=int, default=0, help="seed for dequantization and KS draws")
    parser.add_argument("--output", type=Path, help="write the JSON report here instead of stdout")
    args = parser.parse_args()
    report = analyze(load_times(args.trace, args.preset, args.where), args.rate_window, args.seed)
    report = {"trace": str(args.trace), "preset": args.preset, "where": args.where, **report}
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
