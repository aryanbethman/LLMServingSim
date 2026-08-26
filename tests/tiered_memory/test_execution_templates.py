import unittest

from inference_serving.execution_templates import (
    AttributeProto,
    build_template_bundle,
    GlobalMetadata,
    Node,
    TemplateStore,
    _frame,
    materialise_rank_et,
    split_rank_et,
)


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


if __name__ == "__main__":
    unittest.main()
