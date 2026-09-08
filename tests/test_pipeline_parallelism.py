import unittest
import os

from inference_serving.memory_model import GB_TO_BYTE, MemoryModel
from inference_serving.request import Batch, Request
from inference_serving.trace_generator import generate_trace
from llm_profile.profile_projection import REPO


class PipelineParallelMemoryTest(unittest.TestCase):
    def make_memory(self, pipeline_parallel_degree):
        return MemoryModel(
            "meta-llama/Llama-3.1-405B", 0, 0, 16, 2, 80, 1024, 16, 16,
            False, False, None, None, 0, pipeline_parallel_degree,
        )

    def test_tp8_pp2_accounts_for_largest_stage_not_full_model(self):
        memory = self.make_memory(2)
        self.assertEqual(memory.stage_layers, [63, 63])
        self.assertLess(memory.weight, 80 * GB_TO_BYTE)
        self.assertEqual(memory.get_kv(1), 32256)

    def test_tp8_without_pipeline_parallelism_is_not_memory_feasible(self):
        with self.assertRaisesRegex(RuntimeError, "Model size"):
            self.make_memory(1)

    def test_trace_declares_transformer_block_aligned_boundaries(self):
        request = Request("pp-smoke", "meta-llama/Llama-3.1-405B", 16, 17, 0, 0)
        batch = Batch(0, request.model, 16, 0, 0, [16], [0], 1, 0, [16], [0], [], 0, 0)
        batch.requests = [request]
        placement = {"default": {"weights": "LOCAL", "kv_loc": "LOCAL", "kv_evict_loc": "LOCAL"},
                     "block": [], "layer": {}}
        original_dir = os.getcwd()
        try:
            os.chdir(REPO / "inference_serving")
            trace = generate_trace(batch, "H100", 16, 2, placement=placement,
                                   return_text=True, pipeline_parallel_degree=2)
        finally:
            os.chdir(original_dir)
        self.assertEqual(trace.splitlines()[0],
                         "COLOCATED\t\tmodel_parallel_NPU_group: 2\t\tpipeline_block_boundaries: 757,1515")


if __name__ == "__main__":
    unittest.main()
