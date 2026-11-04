#!/usr/bin/env python3
"""What the recorded tool delays of OpenHands SWE-bench trajectories measure.

The swebench generator sets a call's tool_duration_ns to the time from its
LLM response's first event (the action, logged when the response arrives) to
the last event before the next response (normally the observation, logged
when the tool returns). This audit splits those delays by the tools the
response invoked, and by the first word of bash commands. It also measures the
gap from that last event to the next response, which is the LLM call plus
agent overhead and which the simulator models itself rather than replays.

Every delay is cross-checked against workloads.generators.swebench, so the
audit describes exactly what the generator emits.

Usage:
    python analysis/tool_delay_audit.py \
        ~/.cache/llmservingsim/swebench/verified/20250805-openhands-Qwen3-Coder-30B-A3B-Instruct
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from workloads.generators.swebench import _timestamp, parse_openhands_trajectory

QUANTILES = (0.1, 0.5, 0.9, 0.99)


def response_steps(events: list[dict]) -> list[dict]:
    """One record per LLM response except the last: its tools, tool delay and the following LLM gap."""
    first, tools, commands = {}, defaultdict(list), defaultdict(list)
    for i, event in enumerate(events):
        meta = event.get("tool_call_metadata")
        if not meta:
            continue
        rid = meta["model_response"]["id"]
        first.setdefault(rid, i)
        if event.get("action"):
            tools[rid].append(event["action"])
            if event["action"] == "run":
                words = str((event.get("args") or {}).get("command", "")).split()
                commands[rid].append(_command_word(words))
    order = list(first)
    steps = []
    for rid, next_rid in zip(order, order[1:]):
        start = _timestamp(events[first[rid]])
        end = _timestamp(events[first[next_rid] - 1])
        steps.append({"tools": "+".join(sorted(set(tools[rid]))) or "none",
                      "commands": commands[rid],
                      "tool_sec": max(0.0, end - start),
                      "llm_gap_sec": _timestamp(events[first[next_rid]]) - end})
    return steps


def _command_word(words: list[str]) -> str:
    # Skip "cd <dir> &&" prefixes and env assignments to reach the program.
    while words:
        if words[0] == "cd" and len(words) >= 3 and words[2] == "&&":
            words = words[3:]
        elif "=" in words[0] and not words[0].startswith("="):
            words = words[1:]
        else:
            return Path(words[0]).name
    return ""


def summarize(values) -> dict:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"count": 0}
    out = {"count": int(len(values)), "mean": float(values.mean())}
    out.update({f"p{round(q * 100)}": float(np.quantile(values, q)) for q in QUANTILES})
    out["max"] = float(values.max())
    return out


def audit(source: Path) -> dict:
    by_tools, by_command, all_tool, all_gap = defaultdict(list), defaultdict(list), [], []
    near_floor = 0
    paths = sorted(source.glob("*/trajectory.json"))
    if not paths:
        raise ValueError(f"no <instance_id>/trajectory.json under {source}")
    for path in paths:
        events = json.loads(path.read_text())
        steps = response_steps(events)
        expected = [c.tool_duration_ns for c in parse_openhands_trajectory(path.parent.name, events).calls[:-1]]
        got = [max(0, round(s["tool_sec"] * 1e9)) for s in steps]
        if got != expected:
            raise RuntimeError(f"{path}: audit and generator disagree on tool delays")
        for step in steps:
            by_tools[step["tools"]].append(step["tool_sec"])
            if step["tools"] == "run" and len(step["commands"]) == 1:
                by_command[step["commands"][0]].append(step["tool_sec"])
            all_tool.append(step["tool_sec"])
            all_gap.append(step["llm_gap_sec"])
            near_floor += 0.6 <= step["tool_sec"] < 0.65
    total_tool, total_gap = sum(all_tool), sum(all_gap)
    commands = sorted(by_command.items(), key=lambda kv: -len(kv[1]))
    return {
        "trajectories": len(paths),
        "steps": len(all_tool),
        "tool_sec": summarize(all_tool),
        "llm_gap_sec": summarize(all_gap),
        "tool_share_of_session_time": total_tool / (total_tool + total_gap),
        "share_in_0.60_0.65_sec": near_floor / len(all_tool),
        "by_tools": {k: {"share": len(v) / len(all_tool), **summarize(v)}
                     for k, v in sorted(by_tools.items(), key=lambda kv: -len(kv[1]))},
        "run_by_command": {k: summarize(v) for k, v in commands[:15]},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="directory of <instance_id>/trajectory.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(audit(args.source.expanduser()), indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
