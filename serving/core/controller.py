import base64
import json
import re
from .logger import get_logger

# ASTRA-Sim's per-iteration report, the one line of its stdout the frontend
# has to parse. Compiled once: parse_output runs on every handshake, and at
# 8 NPUs a 10-request run makes 337,786 of them.
_ITERATION_RE = re.compile(
    r"sys\[(\d+)\] iteration (\d+) finished, (\d+) cycles, "
    r"exposed communication (\d+) cycles."
)

# The compact protocol's equivalent of the line above: same four numbers,
# no prose. Guarded by a startswith check at the call site so the common
# case costs a prefix compare rather than a regex.
_READY_RE = re.compile(r"READY (\d+) (\d+) (\d+) (\d+)\n?")

# Bookkeeping the template cache in ASTRA-Sim reports back.
_TEMPLATE_RELEASE_RE = re.compile(r"TEMPLATE_RELEASE ([0-9a-f]{64})\n?")
_TEMPLATE_CACHE_RE = re.compile(
    r"TEMPLATE_CACHE (\d+) (\d+) (\d+) (\d+) (\d+) (\d+)\n?"
)
_TEMPLATE_PROFILE_RE = re.compile(
    r"TEMPLATE_PROFILE (\d+) (\d+) (\d+) (\d+)\n?"
)


class Controller():
    def __init__(self, total_num):
        self.end_dict = {}
        self.total_num = total_num
        self.logger = get_logger(self.__class__)
        # Content ids ASTRA-Sim has acknowledged. A template already in its
        # cache never crosses the pipe again, however often a later batch
        # re-derives the same structure.
        self.sent_template_ids = set()
        self.template_transport_stats = {
            "bundles": 0,
            "wire_bytes": 0,
            "template_definitions": 0,
            "template_nodes": 0,
            "duplicate_template_definitions": 0,
            "duplicate_template_nodes": 0,
            "rank_bindings": 0,
            "template_releases": 0,
            "astra_cache_entries": 0,
            "astra_cache_nodes": 0,
            "astra_cache_high_water_entries": 0,
            "astra_cache_high_water_nodes": 0,
            "astra_cache_evictions": 0,
            "astra_cache_blocked_evictions": 0,
            "astra_template_decode_ns": 0,
            "astra_binding_parse_ns": 0,
            "astra_direct_feeder_init_ns": 0,
            "astra_direct_feeder_inits": 0,
        }
        for i in range(total_num):
            self.end_dict[i] = -1


    def read_wait(self, p):
        """Read ASTRA-Sim's stdout up to the "Waiting" prompt.

        Every line before the prompt is the iteration report; ASTRA-Sim used
        to interleave a per-tick "Checking ..." line per NPU, which made this
        loop 3.07M reads on a 10-request 8-NPU run against 675k now. See the
        ASTRA_SIM_TRACE_POLLING note in the analytical backend's main.cc.
        """
        out = [""]
        while "Waiting" not in out[-1] and out[-1] != "Checking Non-Exited Systems ...\n":
            line = p.stdout.readline()
            if line == "":
                raise RuntimeError("ASTRA closed stdout before a Waiting prompt")
            # For debugging
            # print(line, end='')
            out.append(line)
        return out

    def read_completion(self, p):
        """Return one compact completion record, discarding diagnostics.

        The compact protocol's counterpart to `read_wait`: instead of
        scanning for a prose prompt, read until a line parses as a
        record. Template bookkeeping lines parse to None and are consumed
        here rather than surfacing to the caller.
        """
        while True:
            line = p.stdout.readline()
            if line == "":
                raise RuntimeError("ASTRA closed stdout before a READY record")
            parsed = self.parse_output(line)
            if parsed is not None:
                return parsed
            if line.startswith("READY "):
                raise RuntimeError(f"Malformed ASTRA READY record: {line.rstrip()!r}")
            if line in {"COMPLETE\n", "INCOMPLETE\n"}:
                raise RuntimeError(f"ASTRA terminated early with {line.strip()}")

    def check_end(self, p, compact_protocol=False):
        if compact_protocol:
            while True:
                line = p.stdout.readline()
                if line == "":
                    raise RuntimeError("ASTRA closed stdout before completion")
                self.parse_output(line)
                if line == "COMPLETE\n":
                    print("ASTRA completed all requests")
                    return [line]
                if line == "INCOMPLETE\n":
                    raise RuntimeError("ASTRA reported incomplete requests")
        out = ["",""]
        while out[-2] != "All Request Has Been Exited\n" and out[-2] != "ERROR: Some Requests Remain\n":
            out.append(p.stdout.readline())
            p.stdout.flush()
        print(out[-4], end='')
        print(out[-2], end='')
        return out

    def write_flush(self, p, input):
        # For debugging
        # print(input)
        p.stdin.write(input+'\n')
        p.stdin.flush()
        return

    def write_payloads(self, p, payloads):
        """Send rank-indexed ET bytes over the existing line-oriented pipe.

        Base64 because the pipe is line-oriented and ET payloads are
        arbitrary bytes. Used when the graph is built in memory but not
        split into templates.
        """
        encoded = {
            str(rank): base64.b64encode(payload).decode("ascii")
            for rank, payload in payloads.items()
        }
        p.stdin.write("ET_PAYLOADS " + json.dumps(encoded, separators=(",", ":")) + "\n")
        p.stdin.flush()

    def write_template_bundle(self, p, bundle):
        """Send structural ET templates plus sparse rank overlays.

        Templates ASTRA-Sim has already acknowledged are stripped before
        the write, so a repeated structure costs only its rank overlays.
        Callers get the savings reported through
        `get_template_transport_stats`.
        """
        duplicate_template_ids = (
            self.sent_template_ids.intersection(bundle["templates"].keys())
        )
        duplicate_template_nodes = sum(
            len(bundle["templates"][template_id])
            for template_id in duplicate_template_ids
        )
        if duplicate_template_ids:
            bundle = {
                **bundle,
                "templates": {
                    template_id: nodes
                    for template_id, nodes in bundle["templates"].items()
                    if template_id not in duplicate_template_ids
                },
            }
        encoded = json.dumps(bundle, separators=(",", ":"))
        p.stdin.write("ET_TEMPLATE_BUNDLE " + encoded + "\n")
        p.stdin.flush()
        self.sent_template_ids.update(bundle["templates"].keys())
        stats = self.template_transport_stats
        stats["bundles"] += 1
        stats["wire_bytes"] += len(encoded.encode("utf-8"))
        stats["template_definitions"] += len(bundle["templates"])
        stats["template_nodes"] += sum(
            len(nodes) for nodes in bundle["templates"].values()
        )
        stats["duplicate_template_definitions"] += len(duplicate_template_ids)
        stats["duplicate_template_nodes"] += duplicate_template_nodes
        stats["rank_bindings"] += len(bundle["bindings"])

    def write_template_bindings(self, p, cached):
        """Send a cached bundle's rank bindings against templates ASTRA holds.

        Writes the same line `write_template_bundle` would for a bundle with
        no new template definitions; the bindings JSON was encoded once,
        when the batch was first converted.
        """
        encoded = '{"templates":{},"bindings":' + cached.bindings_json + '}'
        p.stdin.write("ET_TEMPLATE_BUNDLE " + encoded + "\n")
        p.stdin.flush()
        stats = self.template_transport_stats
        stats["bundles"] += 1
        stats["wire_bytes"] += len(encoded)
        stats["rank_bindings"] += cached.rank_count

    def get_template_transport_stats(self):
        return {
            **self.template_transport_stats,
            "cached_template_definitions": len(self.sent_template_ids),
        }

    def parse_output(self, output):
        """Turn one ASTRA-Sim stdout line into a completion record.

        Returns the record, or None for a line that carries no completion
        (template bookkeeping, or anything unrecognised). Called for every
        line the frontend reads, so the ordering below is by frequency and
        each rare branch is behind a prefix test rather than a regex.
        """
        if output.startswith("READY "):
            compact = _READY_RE.fullmatch(output)
            if compact:
                sys, iteration, cycle, exposed = map(int, compact.groups())
                if self.end_dict[sys] != iteration:
                    self.logger.info(
                        "NPU[%d] iteration %d finished, %d cycles, exposed communication %d cycles.",
                        sys, iteration, cycle, exposed,
                    )
                    self.end_dict[sys] = iteration
                return {'sys': sys, 'id': iteration, 'cycle': cycle}
            return

        if output.startswith("TEMPLATE_"):
            released = _TEMPLATE_RELEASE_RE.fullmatch(output)
            if released:
                self.sent_template_ids.discard(released.group(1))
                self.template_transport_stats["template_releases"] += 1
                return
            cache_stats = _TEMPLATE_CACHE_RE.fullmatch(output)
            if cache_stats:
                stats = self.template_transport_stats
                (
                    stats["astra_cache_entries"],
                    stats["astra_cache_nodes"],
                    stats["astra_cache_high_water_entries"],
                    stats["astra_cache_high_water_nodes"],
                    stats["astra_cache_evictions"],
                    stats["astra_cache_blocked_evictions"],
                ) = map(int, cache_stats.groups())
                return
            profile_stats = _TEMPLATE_PROFILE_RE.fullmatch(output)
            if profile_stats:
                stats = self.template_transport_stats
                (
                    stats["astra_template_decode_ns"],
                    stats["astra_binding_parse_ns"],
                    stats["astra_direct_feeder_init_ns"],
                    stats["astra_direct_feeder_inits"],
                ) = map(int, profile_stats.groups())
            return

        match = _ITERATION_RE.search(output)
        if match:
            sys = int(match.group(1))
            id = int(match.group(2))
            cycle = int(match.group(3))
            com_cycle = int(match.group(4))

            if self.end_dict[sys] != id:
                self.logger.info(
                    "NPU[%d] iteration %d finished, %d cycles, exposed communication %d cycles.",
                    sys,
                    id,
                    cycle,
                    com_cycle,
                )
                self.end_dict[sys] = id
            return {'sys': sys, 'id': id, 'cycle': cycle}
        return
