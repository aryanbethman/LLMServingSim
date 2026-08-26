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

    def check_end(self, p):
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
        p.stdin.write("ET_TEMPLATE_BUNDLE " + json.dumps(bundle, separators=(",", ":")) + "\n")
        p.stdin.flush()
        self.sent_template_ids.update(bundle["templates"].keys())


    def parse_output(self, output):
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