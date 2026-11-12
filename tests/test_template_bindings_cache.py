import io
import json
import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from serving.core import graph_generator
from serving.core.controller import Controller
from serving.core.graph_generator import CachedTemplateBindings, generate_graph
from serving.core.trace_generator import TraceData

from chakra.src.converter.llm_converter import LLMConverter

TP_HEADER = "COLOCATED model_parallel_NPU_group: 1"
PP_HEADER = "COLOCATED model_parallel_NPU_group: 2 pp_stage_boundaries: 1"


def trace(header, comp_a="12"):
    rows = [[
        "layer_a", comp_a, "LOCAL", "128", "LOCAL", "0", "LOCAL", "128",
        "ALLREDUCE", "4096", "NONE",
    ], [
        "layer_b", "17", "LOCAL", "128", "LOCAL", "0", "LOCAL", "128",
        "ALLREDUCE", "4096", "NONE",
    ]]
    return TraceData(header_line=header, rows=rows, path=None)


def shared(t, num_npus, npu_offset, known):
    return generate_graph(None, None, num_npus, npu_offset=npu_offset,
                          template_mode="shared-template",
                          known_template_ids=known, trace=t)


def fresh_bundle(t, num_npus, npu_offset, known):
    return LLMConverter(None, None, num_npus, npu_offset).convert_rows_to_template_bundle(
        t.header_line, graph_generator.indexed_cols(t.rows), known_template_ids=known)[0]


def encode(bindings):
    return json.dumps(bindings, separators=(",", ":"))


class _Process:
    def __init__(self):
        self.stdin = io.StringIO()


class TemplateBindingsCacheTest(unittest.TestCase):
    def setUp(self):
        graph_generator._BINDINGS_CACHE.clear()
        graph_generator._BINDINGS_CACHE_BYTES = 0
        for key in graph_generator._BINDINGS_CACHE_STATS:
            graph_generator._BINDINGS_CACHE_STATS[key] = 0
        self.stats = graph_generator._BINDINGS_CACHE_STATS

    def test_tp_hit_relocates_to_another_instance_exactly(self):
        t = trace(TP_HEADER)
        bundle, _ = shared(t, 4, 0, set())
        known = set(bundle["templates"])
        self.assertTrue(known)

        hit = shared(t, 4, 4, known)
        self.assertIsInstance(hit, CachedTemplateBindings)
        expected = fresh_bundle(t, 4, 4, known)
        self.assertEqual(expected["templates"], {})
        self.assertEqual(hit.bindings_json, encode(expected["bindings"]))
        self.assertEqual(hit.rank_count, 4)
        self.assertEqual(self.stats["hit"], 1)
        self.assertEqual(self.stats["relocated_hit"], 1)

    def test_different_trace_misses(self):
        bundle, _ = shared(trace(TP_HEADER), 4, 0, set())
        result = shared(trace(TP_HEADER, comp_a="13"), 4, 0, set(bundle["templates"]))
        self.assertNotIsInstance(result, CachedTemplateBindings)
        self.assertEqual(self.stats["miss"], 2)

    def test_bindings_with_rank_overlays_only_hit_at_their_offset(self):
        t = trace(PP_HEADER)
        bundle, _ = shared(t, 4, 4, set())
        known = set(bundle["templates"])
        self.assertEqual(self.stats["fixed_entries_stored"], 1)

        moved = shared(t, 4, 8, known)
        self.assertNotIsInstance(moved, CachedTemplateBindings)
        self.assertEqual(self.stats["offset_mismatch"], 1)

        # The reconversion at offset 8 replaced the entry; offset 8 now hits.
        hit = shared(t, 4, 8, known)
        self.assertIsInstance(hit, CachedTemplateBindings)
        self.assertEqual(hit.bindings_json, encode(fresh_bundle(t, 4, 8, known)["bindings"]))

    def test_evicted_template_forces_reconversion(self):
        t = trace(TP_HEADER)
        bundle, _ = shared(t, 4, 0, set())
        result = shared(t, 4, 0, set())  # ASTRA no longer holds the template
        self.assertNotIsInstance(result, CachedTemplateBindings)
        self.assertEqual(result[0]["templates"].keys(), bundle["templates"].keys())
        self.assertEqual(self.stats["template_evicted"], 1)

    def test_controller_writes_the_line_a_template_free_bundle_would(self):
        t = trace(TP_HEADER)
        bundle, _ = shared(t, 4, 0, set())
        known = set(bundle["templates"])
        hit = shared(t, 4, 4, known)

        via_cache, via_bundle = _Process(), _Process()
        Controller(total_num=8).write_template_bindings(via_cache, hit)
        Controller(total_num=8).write_template_bundle(via_bundle, fresh_bundle(t, 4, 4, known))
        self.assertEqual(via_cache.stdin.getvalue(), via_bundle.stdin.getvalue())


if __name__ == "__main__":
    unittest.main()
