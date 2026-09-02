import base64
import json
import re
from .logger import get_logger

class Controller():
    def __init__(self, total_num):
        self.end_dict = {}
        self.total_num = total_num
        self.logger = get_logger(self.__class__)
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
        }
        for i in range(total_num):
            self.end_dict[i] = -1


    def read_wait(self, p):
        out = [""]
        while "Waiting" not in out[-1] and out[-1] != "Checking Non-Exited Systems ...\n":
            line = p.stdout.readline()
            # For debugging
            # print(line, end='')
            out.append(line)
            p.stdout.flush()
        return out

    def read_completion(self, p):
        """Return one compact ASTRA completion record, discarding diagnostics."""
        while True:
            line = p.stdout.readline()
            if line == "":
                raise RuntimeError("ASTRA closed stdout before a READY record")
            parsed = self.parse_output(line)
            if parsed is not None:
                return parsed
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
        # Send rank-indexed ET bytes over the existing line-oriented pipe.
        encoded = {
            str(rank): base64.b64encode(payload).decode("ascii")
            for rank, payload in payloads.items()
        }
        p.stdin.write("ET_PAYLOADS " + json.dumps(encoded, separators=(",", ":")) + "\n")
        p.stdin.flush()
    def write_template_bundle(self, p, bundle):
        """Send structural ET templates plus sparse rank overlays to ASTRA."""
        duplicate_template_ids = (
            self.sent_template_ids.intersection(bundle["templates"].keys())
        )
        duplicate_template_nodes = sum(
            len(bundle["templates"][template_id])
            for template_id in duplicate_template_ids
        )
        if duplicate_template_ids:
            # The ASTRA process owns a long-lived immutable template cache.
            # A content ID acknowledged by this controller need not cross the
            # pipe again, even if an upstream builder re-emits it.
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
        self.template_transport_stats["bundles"] += 1
        self.template_transport_stats["wire_bytes"] += len(encoded.encode("utf-8"))
        self.template_transport_stats["template_definitions"] += len(bundle["templates"])
        self.template_transport_stats["template_nodes"] += sum(
            len(nodes) for nodes in bundle["templates"].values()
        )
        self.template_transport_stats["duplicate_template_definitions"] += len(
            duplicate_template_ids
        )
        self.template_transport_stats["duplicate_template_nodes"] += duplicate_template_nodes
        self.template_transport_stats["rank_bindings"] += len(bundle["bindings"])

    def get_template_transport_stats(self):
        return {
            **self.template_transport_stats,
            "cached_template_definitions": len(self.sent_template_ids),
        }


    def parse_output(self, output):
        released = re.fullmatch(r"TEMPLATE_RELEASE ([0-9a-f]{64})\n?", output)
        if released:
            self.sent_template_ids.discard(released.group(1))
            self.template_transport_stats["template_releases"] += 1
            return None
        cache_stats = re.fullmatch(
            r"TEMPLATE_CACHE (\d+) (\d+) (\d+) (\d+) (\d+) (\d+)\n?",
            output,
        )
        if cache_stats:
            (
                self.template_transport_stats["astra_cache_entries"],
                self.template_transport_stats["astra_cache_nodes"],
                self.template_transport_stats["astra_cache_high_water_entries"],
                self.template_transport_stats["astra_cache_high_water_nodes"],
                self.template_transport_stats["astra_cache_evictions"],
                self.template_transport_stats["astra_cache_blocked_evictions"],
            ) = map(int, cache_stats.groups())
            return None
        compact = re.fullmatch(r"READY (\d+) (\d+) (\d+) (\d+)\n?", output)
        if compact:
            sys, iteration, cycle, exposed = map(int, compact.groups())
            if self.end_dict[sys] != iteration:
                self.logger.info(
                    "NPU[%d] iteration %d finished, %d cycles, exposed communication %d cycles.",
                    sys, iteration, cycle, exposed,
                )
                self.end_dict[sys] = iteration
            return {'sys': sys, 'id': iteration, 'cycle': cycle}
        pattern = r"sys\[(\d+)\] iteration (\d+) finished, (\d+) cycles, exposed communication (\d+) cycles."
        match = re.search(pattern, output)
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
