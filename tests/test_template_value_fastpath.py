import json
import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from serving.core.template_fastpath import ValueFastPath
from serving.core.trace_generator import indexed_cols

from chakra.src.converter.llm_converter import LLMConverter

TP_HEADER = "COLOCATED model_parallel_NPU_group: 1"
PP_HEADER = "COLOCATED model_parallel_NPU_group: 2 pp_stage_boundaries: 2"
PREFILL_HEADER = "PREFILL model_parallel_NPU_group: 1"


def layer(name, comp, size=128, comm_type="ALLREDUCE", comm=4096, weight_loc="LOCAL", weight=0):
    # Fields as the trace emitters produce them: text.
    return (name, str(comp), "LOCAL", str(size), weight_loc, str(weight), "LOCAL", str(size),
            comm_type, str(comm), "NONE")


def dense_rows(scale=1):
    return [
        layer("qkv_proj", 12 * scale, comm=4096 * scale),
        layer("v_proj", 7 * scale, comm=512 * scale),
        layer("o_proj", 9 * scale, comm=2048 * scale),
        layer("sampler", 3 * scale, size=4 * scale, comm_type="NONE", comm=0),
    ]


def moe_rows(scale=1):
    return [
        layer("gate", 5 * scale, comm_type="NONE", comm=0),
        (f"EXPERT 0 ALLTOALL {1024 * scale}",),
        layer("expert_0", 20 * scale, comm_type="NONE", comm=0),
        ("EXPERT 1",),
        layer("expert_1", 22 * scale, comm_type="NONE", comm=0),
        (f"EXPERT END ALLTOALL {1024 * scale}",),
        layer("sampler", 3 * scale, size=4 * scale, comm_type="NONE", comm=0),
    ]


def full(header, cols, num_npus, npu_offset, known):
    return LLMConverter(None, None, num_npus, npu_offset).convert_rows_to_template_bundle(
        header, cols, known_template_ids=known, template_values=True)


def run(fastpath, header, rows, num_npus=2, npu_offset=0, known=frozenset()):
    cols = indexed_cols(rows)
    convert = lambda c, k: full(header, c, num_npus, npu_offset, k)
    return fastpath.bundle(header, cols, num_npus, npu_offset, False, set(known), convert)


def wire(bundle):
    return json.dumps(bundle, separators=(",", ":"))


class FastPathMatchesFullConversionTest(unittest.TestCase):
    def check_sequence(self, header, make_rows, num_npus, npu_offset=0):
        fastpath = ValueFastPath()
        known = set()
        for scale in (1, 2, 3, 5):
            cols = indexed_cols(make_rows(scale))
            expected, _ = full(header, cols, num_npus, npu_offset, set(known))
            got, _ = run(fastpath, header, make_rows(scale), num_npus, npu_offset, known)
            self.assertEqual(wire(got), wire(expected), (header, scale))
            known |= set(got["templates"])
        self.assertEqual(fastpath.stats["learned"], 1)
        self.assertEqual(fastpath.stats["hit"], 3)

    def test_tp(self):
        self.check_sequence(TP_HEADER, dense_rows, 4)

    def test_pp_with_rank_overlays(self):
        self.check_sequence(PP_HEADER, dense_rows, 4, npu_offset=4)

    def test_prefill_kv_sends(self):
        self.check_sequence(PREFILL_HEADER, dense_rows, 2)

    def test_moe_expert_markers(self):
        self.check_sequence(TP_HEADER, moe_rows, 2)

    def test_memory_sizes_can_cross_zero_after_learning(self):
        fastpath = ValueFastPath(verify=True)
        for size in (0, 128, 0, 256):
            rows = [list(row) for row in dense_rows()]
            rows[0][2:4] = ["REMOTE:0", str(size)]
            rows[-1][6:8] = ["REMOTE:0", str(size)]
            got, _ = run(fastpath, TP_HEADER, rows)
            expected, _ = full(TP_HEADER, indexed_cols(rows), 2, 0, set())
            self.assertEqual(wire(got), wire(expected))
        self.assertEqual(fastpath.stats["learned"], 1)
        self.assertEqual(fastpath.stats["hit"], 3)

    def test_evicted_templates_are_sent_again(self):
        fastpath = ValueFastPath()
        first, _ = run(fastpath, TP_HEADER, dense_rows(1))
        again, _ = run(fastpath, TP_HEADER, dense_rows(2), known=set())
        self.assertEqual(again["templates"], first["templates"])
        expected, _ = full(TP_HEADER, indexed_cols(dense_rows(2)), 2, 0, set())
        self.assertEqual(wire(again), wire(expected))


class FastPathShapeTest(unittest.TestCase):
    def test_a_zero_that_drops_a_node_is_a_new_shape(self):
        fastpath = ValueFastPath()
        run(fastpath, TP_HEADER, dense_rows(1))
        rows = dense_rows(1)
        rows[2] = layer("o_proj", 0, comm=2048)  # comp_time 0 removes the compute node
        got, _ = run(fastpath, TP_HEADER, rows)
        expected, _ = full(TP_HEADER, indexed_cols(rows), 2, 0, set())
        self.assertEqual(wire(got), wire(expected))
        self.assertEqual(fastpath.stats["learned"], 2)
        self.assertEqual(fastpath.stats["hit"], 0)

    def test_offset_is_part_of_the_shape(self):
        fastpath = ValueFastPath()
        run(fastpath, PP_HEADER, dense_rows(1), num_npus=4, npu_offset=0)
        got, _ = run(fastpath, PP_HEADER, dense_rows(2), num_npus=4, npu_offset=4)
        expected, _ = full(PP_HEADER, indexed_cols(dense_rows(2)), 4, 4, set())
        self.assertEqual(wire(got), wire(expected))
        self.assertEqual(fastpath.stats["hit"], 0)

    def test_pim_rows_always_use_full_conversion(self):
        fastpath = ValueFastPath()
        rows = dense_rows(1)[:1] + [("PIM 0",), layer("attn", 6), ("PIM END",)] + dense_rows(1)[2:]
        for _ in range(2):
            got, _ = run(fastpath, TP_HEADER, rows)
            expected, _ = full(TP_HEADER, indexed_cols(rows), 2, 0, set())
            self.assertEqual(wire(got), wire(expected))
        self.assertEqual(fastpath.stats["hit"], 0)
        self.assertEqual(fastpath.stats["ineligible"], 2)

    def test_entries_are_bounded(self):
        fastpath = ValueFastPath(max_entries=1)
        run(fastpath, TP_HEADER, dense_rows(1))
        run(fastpath, PP_HEADER, dense_rows(1), num_npus=4)
        run(fastpath, TP_HEADER, dense_rows(2))
        self.assertEqual(fastpath.stats["learned"], 3)
        self.assertEqual(len(fastpath._entries), 1)


class GenerateGraphFastPathTest(unittest.TestCase):
    def setUp(self):
        from serving.core import graph_generator
        self.gg = graph_generator
        graph_generator._BINDINGS_CACHE.clear()
        graph_generator._BINDINGS_CACHE_BYTES = 0
        graph_generator.set_template_value_overlay(True)

    def tearDown(self):
        self.gg.set_template_value_overlay(False)

    def test_second_batch_of_a_shape_skips_the_converter_and_matches_it(self):
        from serving.core.trace_generator import TraceData
        known = set()
        for scale in (1, 2):
            t = TraceData(header_line=TP_HEADER, rows=dense_rows(scale), path=None)
            got, _ = self.gg.generate_graph(None, None, 4, template_mode="shared-template",
                                            known_template_ids=set(known), trace=t)
            expected, _ = full(TP_HEADER, indexed_cols(dense_rows(scale)), 4, 0, set(known))
            self.assertEqual(wire(got), wire(expected))
            known |= set(got["templates"])
        stats = self.gg.graph_cache_stats()["value_fastpath"]
        self.assertEqual((stats["learned"], stats["hit"]), (1, 1))


class FastPathVerifyTest(unittest.TestCase):
    def test_verify_mode_reconverts_every_hit(self):
        fastpath = ValueFastPath(verify=True)
        for scale in (1, 2, 3):
            run(fastpath, TP_HEADER, dense_rows(scale))
        self.assertEqual(fastpath.stats["verified"], 2)

    def test_verify_mode_fails_loudly_on_a_difference(self):
        fastpath = ValueFastPath(verify=True)
        run(fastpath, TP_HEADER, dense_rows(1))
        cols = indexed_cols(dense_rows(2))
        wrong = lambda c, k: full(TP_HEADER, indexed_cols(dense_rows(3)), 2, 0, k)
        with self.assertRaises(RuntimeError):
            fastpath.bundle(TP_HEADER, cols, 2, 0, False, set(), wrong)


if __name__ == "__main__":
    unittest.main()
