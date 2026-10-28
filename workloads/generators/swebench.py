"""SWE-bench agent trajectories → LLMServingSim agentic-session JSONL.

Source: published OpenHands runs on SWE-bench, one ``trajectory.json`` per
task, from the public ``swe-bench-submissions`` S3 bucket (downloaded once
into ``--cache-dir``) or from a local directory with the same
``<instance_id>/trajectory.json`` layout. The default run is the Qwen team's
OpenHands + Qwen3-Coder-30B-A3B-Instruct run on SWE-bench Verified (500
tasks).

Each trajectory becomes one session; each LLM call becomes one
sub-request::

    {
      "session_id":      "<instance_id>-run<k>",
      "arrival_time_ns": <int>,
      "sub_requests": [
        {"input_toks": <int>, "output_toks": <int>, "tool_duration_ns": <int>,
         "input_tok_ids": [...], "output_tok_ids": [...]},
        ...
      ]
    }

* ``input_toks`` / ``output_toks`` are the serving engine's own counts
  (``usage.prompt_tokens`` / ``usage.completion_tokens``) for that call.
* ``tool_duration_ns`` is the recorded time from the model's response to
  the last event before the next LLM call (tool execution plus agent
  overhead).
* Token ids reproduce the prefix structure the block hashes see. The first
  prompt is the real system prompt, tool schemas and task rendered with
  ``--model``'s chat template, so every session shares the system prefix.
  Each later prompt is the previous prompt plus its output plus fresh ids
  for the appended tool result, so turn N+1 hits on turn N. Outputs and
  tool results use ids unique to the session, so reruns of one task share
  only the first prompt.

Sessions are sampled without replacement; asking for more sessions than
the run has tasks starts another pass with the next ``run<k>`` index.
Arrivals are Poisson at ``--sps`` sessions per second.

LLM calls that OpenHands logged without ``tool_call_metadata`` (a handful of
plain-text replies) carry no usage and are not emitted; their tokens show up
in the next call's prompt.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

from workloads.generators.sharegpt import _load_tokenizer

S3_BUCKET_URL = "https://swe-bench-submissions.s3.amazonaws.com"
DEFAULT_RUN = "verified/20250805-openhands-Qwen3-Coder-30B-A3B-Instruct"
_S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
_DOWNLOAD_ATTEMPTS = 5
_DOWNLOAD_THREADS = 16


# ---------------------------------------------------------------------------
# CLI plumbing — invoked from workloads.generators.__main__
# ---------------------------------------------------------------------------

# Registers the swebench subcommand's flags. Called by: workloads.generators.__main__.main.
def register_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True,
                   help="HF model id or local path whose tokenizer and chat "
                        "template render the first prompt. Use the source "
                        "run's model (default run: "
                        "Qwen/Qwen3-Coder-30B-A3B-Instruct).")
    p.add_argument("--source", default=DEFAULT_RUN,
                   help="Run prefix in the swe-bench-submissions S3 bucket, "
                        "or a local directory of <instance_id>/trajectory.json. "
                        f"Default: {DEFAULT_RUN}.")
    p.add_argument("--cache-dir", default="~/.cache/llmservingsim/swebench",
                   dest="cache_dir",
                   help="Where downloaded runs are kept. "
                        "Default ~/.cache/llmservingsim/swebench.")
    p.add_argument("--num-sessions", type=int, required=True,
                   dest="num_sessions",
                   help="Number of sessions to emit. More than the run's task "
                        "count repeats tasks as run1, run2, ...")
    p.add_argument("--sps", type=float, required=True,
                   help="Session arrival rate (sessions / sec). Poisson-distributed.")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for sampling, arrivals and token ids.")
    p.add_argument("--output", required=True,
                   help="Output JSONL path.")
    p.add_argument("--first-arrival-sec", type=int, default=0,
                   dest="first_arrival_sec",
                   help="Offset (seconds) added to the first session's arrival.")
    p.add_argument("--max-calls", type=int, default=0,
                   dest="max_calls",
                   help="Keep at most this many LLM calls per session "
                        "(0 = no cap). Default 0.")
    p.add_argument("--max-kv-toks", type=int, default=0,
                   dest="max_kv_toks",
                   help="End each session before its first call whose "
                        "input+output exceeds this (0 = no cap). Default 0.")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class LLMCall:
    prompt_toks: int
    completion_toks: int
    tool_duration_ns: int


@dataclasses.dataclass(frozen=True)
class Trajectory:
    instance_id: str
    system_prompt: str
    tools: list
    task: str
    calls: list


# Writes the session JSONL for the requested run and sample. Called by: workloads.generators.__main__.main. Calls: fetch_run, load_trajectories, build_session.
def run(args: argparse.Namespace) -> int:
    if args.num_sessions <= 0:
        raise ValueError(f"--num-sessions must be positive, got {args.num_sessions}")
    if args.sps <= 0:
        raise ValueError(f"--sps must be positive, got {args.sps}")

    source = Path(args.source).expanduser()
    if not source.is_dir():
        source = fetch_run(args.source, Path(args.cache_dir).expanduser())

    pool = {}
    for traj in load_trajectories(source):
        calls = truncate_calls(traj.calls, args.max_calls, args.max_kv_toks)
        if calls:
            pool[traj.instance_id] = dataclasses.replace(traj, calls=calls)
    if not pool:
        raise ValueError(f"no session in {source} fits --max-kv-toks {args.max_kv_toks}")

    tok = _load_tokenizer(args.model)
    rng = np.random.default_rng(args.seed)
    picks = sample_sessions(list(pool), args.num_sessions, rng)
    arrivals = poisson_arrivals(len(picks), args.sps, args.first_arrival_sec * 1_000_000_000, rng)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    first_prompts = {}
    num_calls = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for (instance_id, run_index), arrival_ns in zip(picks, arrivals):
            traj = pool[instance_id]
            if instance_id not in first_prompts:
                task_rng = np.random.default_rng([args.seed, zlib.crc32(instance_id.encode())])
                first_prompts[instance_id] = fit_ids(render_first_prompt(tok, traj),
                                                     traj.calls[0].prompt_toks, task_rng, tok.vocab_size)
            session_rng = np.random.default_rng([args.seed, zlib.crc32(instance_id.encode()), run_index + 1])
            session = build_session(f"{instance_id}-run{run_index}", arrival_ns, first_prompts[instance_id],
                                    traj.calls, session_rng, tok.vocab_size)
            fout.write(json.dumps(session) + "\n")
            num_calls += len(traj.calls)

    size_gb = out_path.stat().st_size / 1e9
    print(f"Wrote {len(picks)} sessions ({num_calls} LLM calls, {size_gb:.2f} GB) -> {out_path}")
    return 0


# ---------------------------------------------------------------------------
# Trajectory parsing
# ---------------------------------------------------------------------------

# Parses every <instance_id>/trajectory.json under a directory. Called by: run. Calls: parse_openhands_trajectory.
def load_trajectories(source: Path) -> list[Trajectory]:
    paths = sorted(source.glob("*/trajectory.json"))
    if not paths:
        raise ValueError(f"no <instance_id>/trajectory.json files under {source}")
    return [parse_openhands_trajectory(p.parent.name, json.loads(p.read_text())) for p in paths]


# Turns one OpenHands event list into its LLM calls. Called by: load_trajectories.
def parse_openhands_trajectory(instance_id: str, events: list[dict]) -> Trajectory:
    system = next((e for e in events if e.get("action") == "system"), None)
    task = next((e for e in events if e.get("source") == "user" and e.get("action") == "message"), None)
    if system is None or task is None:
        raise ValueError(f"{instance_id}: trajectory has no system prompt or task message")

    first_event = {}
    usage = {}
    for i, event in enumerate(events):
        meta = event.get("tool_call_metadata")
        if meta and meta["model_response"]["id"] not in first_event:
            first_event[meta["model_response"]["id"]] = i
            usage[meta["model_response"]["id"]] = meta["model_response"]["usage"]
    responses = list(first_event)
    if not responses:
        raise ValueError(f"{instance_id}: trajectory has no LLM calls with usage")

    calls = []
    for k, response_id in enumerate(responses):
        tool_duration_ns = 0
        if k + 1 < len(responses):
            start = _timestamp(events[first_event[response_id]])
            end = _timestamp(events[first_event[responses[k + 1]] - 1])
            tool_duration_ns = max(0, round((end - start) * 1e9))
        calls.append(LLMCall(usage[response_id]["prompt_tokens"], usage[response_id]["completion_tokens"],
                             tool_duration_ns))
    return Trajectory(instance_id, system["args"]["content"], system["args"].get("tools") or [],
                      task["args"]["content"], calls)


def _timestamp(event: dict) -> float:
    return datetime.fromisoformat(event["timestamp"]).timestamp()


# Drops the tail of a session past --max-calls or --max-kv-toks. Called by: run.
def truncate_calls(calls: list[LLMCall], max_calls: int, max_kv_toks: int) -> list[LLMCall]:
    kept = calls[:max_calls] if max_calls > 0 else list(calls)
    if max_kv_toks > 0:
        over = next((i for i, c in enumerate(kept) if c.prompt_toks + c.completion_toks > max_kv_toks), len(kept))
        kept = kept[:over]
    return kept


# ---------------------------------------------------------------------------
# Session construction
# ---------------------------------------------------------------------------

# Renders the system prompt, tool schemas and task as the first prompt's ids. Called by: run.
def render_first_prompt(tok, traj: Trajectory) -> list[int]:
    messages = [{"role": "system", "content": traj.system_prompt},
                {"role": "user", "content": traj.task}]
    text = tok.apply_chat_template(messages, tools=traj.tools or None, tokenize=False,
                                   add_generation_prompt=True)
    return tok(text, add_special_tokens=False)["input_ids"]


# Cuts ids to n, or pads them with fresh random ids up to n. Called by: run, build_session.
def fit_ids(ids: list[int], n: int, rng: np.random.Generator, vocab_size: int) -> list[int]:
    if len(ids) >= n:
        return list(ids[:n])
    return list(ids) + rng.integers(0, vocab_size, size=n - len(ids)).tolist()


# Builds one session's sub-requests with an append-only id history. Called by: run. Calls: fit_ids.
def build_session(session_id: str, arrival_ns: int, first_prompt_ids: list[int], calls: list[LLMCall],
                  rng: np.random.Generator, vocab_size: int) -> dict:
    sub_requests = []
    history = first_prompt_ids
    for k, call in enumerate(calls):
        prompt = fit_ids(history, call.prompt_toks, rng, vocab_size)
        output = rng.integers(0, vocab_size, size=call.completion_toks).tolist()
        sub_requests.append({
            "input_toks": call.prompt_toks,
            "output_toks": call.completion_toks,
            "tool_duration_ns": call.tool_duration_ns if k + 1 < len(calls) else 0,
            "input_tok_ids": prompt,
            "output_tok_ids": output,
        })
        history = prompt + output
    return {"session_id": session_id, "arrival_time_ns": arrival_ns, "sub_requests": sub_requests}


# Picks (instance_id, run_index) pairs, one shuffled pass over the tasks per run index. Called by: run.
def sample_sessions(instance_ids: list[str], num_sessions: int, rng: np.random.Generator) -> list[tuple[str, int]]:
    if not instance_ids:
        raise ValueError("no trajectories to sample sessions from")
    pool = sorted(instance_ids)
    picks = []
    run_index = 0
    while len(picks) < num_sessions:
        order = rng.permutation(len(pool))[: num_sessions - len(picks)]
        picks.extend((pool[i], run_index) for i in order)
        run_index += 1
    return picks


# Session arrival times (ns) of a Poisson process at sps. Called by: run.
def poisson_arrivals(num: int, sps: float, first_arrival_ns: int, rng: np.random.Generator) -> list[int]:
    gaps_ns = rng.exponential(scale=1e9 / sps, size=num)
    return (first_arrival_ns + np.cumsum(gaps_ns)).astype(np.int64).tolist()


# ---------------------------------------------------------------------------
# Download from the swe-bench-submissions bucket
# ---------------------------------------------------------------------------

# Downloads a run's trajectories into cache_dir/<run>, skipping files already there. Called by: run. Calls: S3 (HTTPS).
def fetch_run(run_prefix: str, cache_dir: Path) -> Path:
    run_prefix = run_prefix.strip("/")
    dest = cache_dir / run_prefix
    keys = _list_trajectory_keys(run_prefix)
    if not keys:
        raise ValueError(f"no trajectory.json files under s3://swe-bench-submissions/{run_prefix}/trajs/")
    trajs_prefix = f"{run_prefix}/trajs/"
    missing = [k for k in keys if not (dest / k[len(trajs_prefix):]).exists()]
    if missing:
        print(f"Downloading {len(missing)} of {len(keys)} trajectories -> {dest}")
        with ThreadPoolExecutor(_DOWNLOAD_THREADS) as pool:
            list(pool.map(lambda k: _download_json(f"{S3_BUCKET_URL}/{k}", dest / k[len(trajs_prefix):]), missing))
    return dest


# Lists every trajectory.json key under <run>/trajs/, following pagination. Called by: fetch_run. Calls: S3 (HTTPS).
def _list_trajectory_keys(run_prefix: str) -> list[str]:
    keys = []
    token = None
    while True:
        query = {"list-type": "2", "prefix": f"{run_prefix}/trajs/"}
        if token:
            query["continuation-token"] = token
        with urllib.request.urlopen(f"{S3_BUCKET_URL}/?{urllib.parse.urlencode(query)}", timeout=60) as resp:
            page_keys, token = parse_s3_listing(resp.read())
        keys.extend(page_keys)
        if token is None:
            return keys


# Extracts trajectory.json keys and the continuation token from one ListObjectsV2 page. Called by: _list_trajectory_keys.
def parse_s3_listing(xml_bytes: bytes) -> tuple[list[str], str | None]:
    root = ET.fromstring(xml_bytes)
    keys = [c.findtext("s3:Key", namespaces=_S3_NS) for c in root.findall("s3:Contents", _S3_NS)]
    token = root.findtext("s3:NextContinuationToken", namespaces=_S3_NS)
    return [k for k in keys if k.endswith("/trajectory.json")], token


# Fetches a JSON file with retries and writes it only once it parses. Called by: fetch_run. Calls: S3 (HTTPS).
def _download_json(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for _ in range(_DOWNLOAD_ATTEMPTS):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                body = resp.read()
            json.loads(body)
        except (OSError, ValueError) as e:
            last_error = e
            continue
        # Write-then-rename so an interrupted run never leaves a truncated file
        # that fetch_run would later treat as already downloaded.
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(body)
        tmp.replace(dest)
        return
    raise RuntimeError(f"failed to download {url} after {_DOWNLOAD_ATTEMPTS} attempts") from last_error
