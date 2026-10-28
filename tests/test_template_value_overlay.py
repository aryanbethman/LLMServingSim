import base64
import io
import json
import os
import sys
import unittest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)
for path in (REPO_ROOT, TESTS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from serving.core import graph_generator
from serving.core.controller import Controller
from serving.core.execution_templates import (
    ExecutionTemplate,
    Node,
    NodeOverlay,
    RankAttributeOverlay,
    RankOverlay,
    _frame,
    build_template_bundle,
    materialise_rank_et,
    split_rank_et,
)
from serving.core.graph_generator import CachedTemplateBindings, generate_graph
from serving.core.trace_generator import TraceData, indexed_cols

from chakra.src.converter.llm_converter import LLMConverter
from test_execution_templates import rank_payload

TP_HEADER = "COLOCATED model_parallel_NPU_group: 1"
PP_HEADER = "COLOCATED model_parallel_NPU_group: 2 pp_stage_boundaries: 1"


def rows(comp_a="12", comm_a="4096"):
    return [[
        "layer_a", comp_a, "LOCAL", "128", "LOCAL", "0", "LOCAL", "128",
        "ALLREDUCE", comm_a, "NONE",
    ], [
        "layer_b", "17", "LOCAL", "128", "LOCAL", "0", "LOCAL", "128",
        "ALLREDUCE", "4096", "NONE",
    ]]


def payloads(header, num_npus, npu_offset, trace_rows):
    return LLMConverter(None, None, num_npus, npu_offset).convert_rows_to_payloads(
        header, indexed_cols(trace_rows))


def values_bundle(header, num_npus, npu_offset, trace_rows, known=None):
    return LLMConverter(None, None, num_npus, npu_offset).convert_rows_to_template_bundle(
        header, indexed_cols(trace_rows), known_template_ids=known, template_values=True)


def rebuild_rank_ets(bundle, templates):
    """Reconstruct every rank's ET bytes from a wire bundle, as ASTRA-Sim does."""
    templates = {**templates, **{
        template_id: ExecutionTemplate(template_id, tuple(base64.b64decode(n) for n in nodes))
        for template_id, nodes in bundle["templates"].items()
    }}
    rebuilt = {}
    for rank, binding in bundle["bindings"].items():
        template = templates[binding["template_id"]]
        nodes = []
        for index in range(len(template.node_payloads)):
            entry = binding["nodes"].get(str(index))
            if entry is None:
                nodes.append(NodeOverlay(None, ()))
                continue
            nodes.append(NodeOverlay(entry["name"], tuple(
                RankAttributeOverlay(position, base64.b64decode(payload))
                for position, payload in entry["attrs"])))
        values = None
        if "values" in binding:
            value_set = bundle["values"][binding["values"]]
            values = tuple(zip(value_set["d"], value_set["c"], value_set["t"]))
        overlay = RankOverlay(base64.b64decode(binding["metadata"]), tuple(nodes), values)
        rebuilt[int(rank)] = materialise_rank_et(template, overlay)
    return rebuilt, templates


class ValueOverlaySplitTest(unittest.TestCase):
    def test_split_and_materialise_is_byte_exact(self):
        payload = rank_payload(2, 3)
        template, overlay = split_rank_et(payload, values_mode=True)
        self.assertIsNotNone(overlay.values)
        self.assertEqual(materialise_rank_et(template, overlay), payload)

    def test_values_are_kept_out_of_the_template(self):
        template, overlay = split_rank_et(rank_payload(2, 3), values_mode=True)
        # Node 0 is the send (comm_size 4096); node 1 the compute (12 us, tensor 128).
        self.assertEqual(overlay.values, ((0, 4096, 0), (12, 0, 128)))
        send, compute = Node(), Node()
        send.ParseFromString(template.node_payloads[0])
        compute.ParseFromString(template.node_payloads[1])
        self.assertEqual([a.int64_val for a in send.attr if a.name == "comm_size"], [0])
        self.assertEqual(compute.duration_micros, 0)
        self.assertEqual([a.uint64_val for a in compute.attr if a.name == "tensor_size"], [0])

    def test_mode_off_leaves_values_in_the_template(self):
        _, overlay = split_rank_et(rank_payload(2, 3))
        self.assertIsNone(overlay.values)


class ValueOverlayBundleTest(unittest.TestCase):
    def test_batches_differing_only_in_values_share_a_template(self):
        a, _ = values_bundle(TP_HEADER, 2, 0, rows("12", "4096"))
        b, _ = values_bundle(TP_HEADER, 2, 0, rows("13", "8192"))
        self.assertEqual(a["templates"].keys(), b["templates"].keys())
        self.assertNotEqual(a["values"], b["values"])

        plain_a, _ = build_template_bundle(payloads(TP_HEADER, 2, 0, rows("12")))
        plain_b, _ = build_template_bundle(payloads(TP_HEADER, 2, 0, rows("13")))
        self.assertNotEqual(plain_a["templates"].keys(), plain_b["templates"].keys())

    def test_tp_ranks_share_one_value_set(self):
        bundle, _ = values_bundle(TP_HEADER, 4, 0, rows())
        self.assertEqual(len(bundle["values"]), 1)
        self.assertEqual({b["values"] for b in bundle["bindings"].values()}, {0})

    def test_rebuilt_ranks_match_legacy_payloads_exactly(self):
        for header, num_npus, npu_offset in ((TP_HEADER, 2, 0), (PP_HEADER, 4, 4)):
            templates = {}
            for comp, comm in (("12", "4096"), ("13", "8192")):
                expected = payloads(header, num_npus, npu_offset, rows(comp, comm))
                bundle, _ = values_bundle(header, num_npus, npu_offset, rows(comp, comm),
                                          known=set(templates))
                rebuilt, templates = rebuild_rank_ets(bundle, templates)
                self.assertEqual(rebuilt, expected, (header, comp))

    def test_fused_path_matches_payload_split_path(self):
        for header, num_npus, npu_offset in ((TP_HEADER, 2, 0), (PP_HEADER, 4, 4)):
            expected = build_template_bundle(
                payloads(header, num_npus, npu_offset, rows()), values_mode=True)
            self.assertEqual(values_bundle(header, num_npus, npu_offset, rows()), expected)

    def test_mode_off_bundle_has_no_values(self):
        bundle, _ = LLMConverter(None, None, 2, 0).convert_rows_to_template_bundle(
            TP_HEADER, indexed_cols(rows()))
        self.assertNotIn("values", bundle)
        self.assertTrue(all("values" not in b for b in bundle["bindings"].values()))


class ValueOverlayCacheTest(unittest.TestCase):
    def setUp(self):
        graph_generator._BINDINGS_CACHE.clear()
        graph_generator._BINDINGS_CACHE_BYTES = 0
        for key in graph_generator._BINDINGS_CACHE_STATS:
            graph_generator._BINDINGS_CACHE_STATS[key] = 0
        graph_generator.set_template_value_overlay(True)

    def tearDown(self):
        graph_generator.set_template_value_overlay(False)

    def shared(self, num_npus, npu_offset, known):
        t = TraceData(header_line=TP_HEADER, rows=rows(), path=None)
        return generate_graph(None, None, num_npus, npu_offset=npu_offset,
                              template_mode="shared-template", known_template_ids=known, trace=t)

    def test_values_do_not_block_relocation_and_the_line_matches_a_fresh_bundle(self):
        bundle, _ = self.shared(4, 0, set())
        known = set(bundle["templates"])
        hit = self.shared(4, 4, known)
        self.assertIsInstance(hit, CachedTemplateBindings)
        self.assertEqual(graph_generator._BINDINGS_CACHE_STATS["relocated_hit"], 1)

        fresh, _ = values_bundle(TP_HEADER, 4, 4, rows(), known=known)
        via_cache, via_bundle = io.StringIO(), io.StringIO()
        Controller(total_num=8).write_template_bindings(type("P", (), {"stdin": via_cache})(), hit)
        Controller(total_num=8).write_template_bundle(type("P", (), {"stdin": via_bundle})(), fresh)
        self.assertEqual(via_cache.getvalue(), via_bundle.getvalue())
        self.assertIn('"values":', via_cache.getvalue())

    def test_cache_key_separates_the_two_modes(self):
        bundle, _ = self.shared(4, 0, set())
        graph_generator.set_template_value_overlay(False)
        plain = self.shared(4, 0, set(bundle["templates"]))
        self.assertNotIsInstance(plain, CachedTemplateBindings)


if __name__ == "__main__":
    unittest.main()
