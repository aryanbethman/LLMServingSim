import os
import tempfile
import unittest

from inference_serving.graph_generator import cleanup_batch_artifacts


class TraceCleanupTest(unittest.TestCase):
    def test_consumed_batch_artifacts_are_removed(self):
        previous = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                trace = "inputs/trace/H100/model/instance0_batch7.txt"
                workload = "inputs/workload/H100/model/instance0_batch7"
                os.makedirs(os.path.dirname(trace))
                os.makedirs(workload)
                with open(trace, "w", encoding="utf-8") as handle:
                    handle.write("trace")
                with open(os.path.join(workload, "llm.0.et"), "w", encoding="utf-8") as handle:
                    handle.write("workload")

                cleanup_batch_artifacts("H100", "model", 0, 7)
                self.assertFalse(os.path.exists(trace))
                self.assertFalse(os.path.exists(workload))
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
