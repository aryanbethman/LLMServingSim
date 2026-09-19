import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from serving.core.execution_templates import (
    AttributeProto,
    build_template_bundle,
    GlobalMetadata,
    Node,
    TemplateStore,
    _frame,
    materialise_rank_et,
    split_rank_et,
)

from chakra.src.converter.llm_converter import LLMConverter


def rank_payload(rank, peer):
    metadata = GlobalMetadata(version="1.0")
    metadata.attr.add(name="input_file", string_val="same logical workload")

    node = Node(id=0, name=f"COMM_SEND_NODE_layer_ALLTOALL_{rank}_{peer}", type=6)
    node.attr.add(name="comm_type", int64_val=6)
    node.attr.add(name="comm_src", int32_val=rank)
    node.attr.add(name="comm_dst", int32_val=peer)
    node.attr.add(name="comm_size", int64_val=4096)
    node.attr.add(name="comm_tag", int32_val=17 + rank)

    compute = Node(id=1, name="COMP_NODE_layer", type=5, duration_micros=12)
    compute.attr.add(name="tensor_size", uint64_val=128)
    return (
        _frame(metadata.SerializeToString(deterministic=True))
        + _frame(node.SerializeToString(deterministic=True))
        + _frame(compute.SerializeToString(deterministic=True))
    )


def tiny_trace_rows():
    return [[
        "layer_a", "12", "LOCAL", "128", "LOCAL", "0", "LOCAL", "128",
        "ALLREDUCE", "4096", "NONE",
    ], [
        "layer_b", "17", "LOCAL", "128", "LOCAL", "0", "LOCAL", "128",
        "ALLREDUCE", "4096", "NONE",
    ]]


class ExecutionTemplateTest(unittest.TestCase):
    def test_split_and_materialise_is_byte_exact(self):
        payload = rank_payload(2, 3)
        template, overlay = split_rank_et(payload)
        self.assertEqual(materialise_rank_et(template, overlay), payload)

    def test_rank_variants_share_one_template(self):
        first = rank_payload(0, 1)
        second = rank_payload(1, 0)
        template_a, overlay_a = split_rank_et(first)
        template_b, overlay_b = split_rank_et(second)
        self.assertEqual(template_a.template_id, template_b.template_id)
        self.assertEqual(materialise_rank_et(template_a, overlay_a), first)
        self.assertEqual(materialise_rank_et(template_b, overlay_b), second)

    def test_reference_counted_store_releases_templates(self):
        store = TemplateStore()
        first = store.bind(0, rank_payload(0, 1))
        second = store.bind(1, rank_payload(1, 0))
        self.assertEqual(store.summary()["templates"], 1)
        self.assertEqual(store.summary()["references"], 2)
        self.assertGreater(store.summary()["template_bytes"], 0)
        self.assertEqual(store.materialise(first), rank_payload(0, 1))
        self.assertEqual(store.materialise(second), rank_payload(1, 0))
        store.release(first)
        self.assertEqual(store.summary()["references"], 1)
        store.release(second)
        self.assertEqual(store.summary(), {"templates": 0, "template_bytes": 0, "references": 0})

    def test_bundle_sends_one_template_and_sparse_rank_overlays(self):
        payloads = {0: rank_payload(0, 1), 1: rank_payload(1, 0)}
        bundle, stats = build_template_bundle(payloads)

        self.assertEqual(stats["ranks"], 2)
        self.assertEqual(stats["unique_templates"], 1)
        self.assertEqual(stats["templates_sent"], 1)
        self.assertEqual(len(bundle["templates"]), 1)
        self.assertEqual(set(bundle["bindings"]), {"0", "1"})
        template_id = bundle["bindings"]["0"]["template_id"]
        self.assertEqual(bundle["bindings"]["1"]["template_id"], template_id)

        cached_bundle, cached_stats = build_template_bundle(payloads, {template_id})
        self.assertEqual(cached_bundle["templates"], {})
        self.assertEqual(cached_stats["templates_sent"], 0)
        self.assertEqual(cached_bundle["bindings"], bundle["bindings"])

    def test_fused_rows_bundle_matches_payload_conversion_for_tp_and_pp_offsets(self):
        rows = tiny_trace_rows()
        for num_npus, npu_offset, header in (
            (2, 0, "COLOCATED model_parallel_NPU_group: 1"),
            (4, 4, "COLOCATED model_parallel_NPU_group: 2 pp_stage_boundaries: 1"),
        ):
            converter = LLMConverter(None, None, num_npus, npu_offset)
            payloads = converter.convert_rows_to_payloads(header, rows)
            expected, expected_stats = build_template_bundle(payloads)

            fused_converter = LLMConverter(None, None, num_npus, npu_offset)
            fused, fused_stats = fused_converter.convert_rows_to_template_bundle(header, rows)
            self.assertEqual(fused, expected)
            self.assertEqual(fused_stats, expected_stats)

            known = set(expected["templates"])
            fused_cached, fused_cached_stats = LLMConverter(
                None, None, num_npus, npu_offset
            ).convert_rows_to_template_bundle(header, rows, known_template_ids=known)
            expected_cached, expected_cached_stats = build_template_bundle(payloads, known)
            self.assertEqual(fused_cached, expected_cached)
            self.assertEqual(fused_cached_stats, expected_cached_stats)


    def test_normalisation_fast_paths_match_full_normalisation(self):
        # _normalise_node skips the copy for nodes that carry nothing
        # rank-specific; every node the converter emits must normalise
        # exactly as the full copy-and-strip path would.
        from serving.core import execution_templates as et
        from serving.core.trace_generator import indexed_cols

        def full(node):
            copy = et.Node()
            copy.CopyFrom(node)
            original_name = None
            match = et._RANK_NAME.match(copy.name)
            if match:
                original_name = copy.name
                copy.name = match.group(1) + "_<src>_<dst>"
            ranks, kept = [], []
            for position, attribute in enumerate(copy.attr):
                if attribute.name in et._RANK_ATTRIBUTES:
                    ranks.append(et.RankAttributeOverlay(
                        position, attribute.SerializeToString(deterministic=True)))
                else:
                    kept_attribute = et.AttributeProto()
                    kept_attribute.CopyFrom(attribute)
                    kept.append(kept_attribute)
            if ranks:
                del copy.attr[:]
                copy.attr.extend(kept)
            return (copy.SerializeToString(deterministic=True),
                    et.NodeOverlay(original_name, tuple(ranks)))

        seen = {"nodes": 0, "collectives": 0, "send_recv": 0}
        test = self

        class Probe:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def write_message(self, message):
                if isinstance(message, et.Node):
                    test.assertEqual(et._normalise_node(message), full(message), message.name)
                    seen["nodes"] += 1
                    seen["collectives"] += message.name.startswith("COMM_COLL")
                    seen["send_recv"] += bool(et._RANK_NAME.match(message.name))

        class Collector:
            def open_rank(self, rank, id_base=None):
                return Probe()

        rows = tiny_trace_rows() + [[
            "layer_c", "9", "LOCAL", "64", "LOCAL", "0", "LOCAL", "64",
            "ALLGATHER", "2048", "NONE",
        ]]
        for header, npus in (
            ("COLOCATED model_parallel_NPU_group: 1", 4),
            ("COLOCATED model_parallel_NPU_group: 2 pp_stage_boundaries: 1", 8),
            ("COLOCATED model_parallel_NPU_group: 3 pp_stage_boundaries: 1,2", 6),
            ("DECODE model_parallel_NPU_group: 2 pp_stage_boundaries: 1", 4),
            ("PREFILL model_parallel_NPU_group: 2 pp_stage_boundaries: 1", 4),
        ):
            converter = LLMConverter(None, None, npus, 0)
            converter._reset_conversion_state()
            converter._template_collector = Collector()
            try:
                converter.convert_rows(header, indexed_cols(rows))
            finally:
                converter._template_collector = None
        self.assertGreater(seen["collectives"], 0)
        self.assertGreater(seen["send_recv"], 0)

if __name__ == "__main__":
    unittest.main()
