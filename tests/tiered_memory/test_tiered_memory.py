import unittest

from inference_serving.tiered_memory import FabricLink, MemoryTier, TopologyAwareMemory


def tier(name, endpoint, capacity=8):
    return MemoryTier(name, capacity * 1024 ** 3, 1000, 0, endpoint)


class TopologyAwareMemoryTest(unittest.TestCase):
    def test_one_link_transfer_timing(self):
        memory = TopologyAwareMemory(
            [tier("source", "a"), tier("destination", "b")],
            [FabricLink("a", "b", 100, 10, "ab")],
        )
        end, path = memory.transfer("a", "b", 1_000, 0)
        self.assertEqual(end, 20)
        self.assertEqual(path, ["a->b"])

    def test_multihop_and_contention(self):
        memory = TopologyAwareMemory(
            [tier("source", "a"), tier("destination", "c")],
            [FabricLink("a", "b", 100, 10, "ab"), FabricLink("b", "c", 100, 10, "bc")],
        )
        first, _ = memory.transfer("a", "c", 1_000, 0)
        second, _ = memory.transfer("a", "c", 1_000, 0)
        self.assertEqual(first, 40)
        self.assertEqual(second, 60)
        self.assertGreater(memory.summary()["stats"]["transfer_stall_ns"], 0)

    def test_nonlocal_kv_read_charges_only_extra_cost(self):
        local = MemoryTier("local_hbm", 8 * 1024 ** 3, 1000, 0, "compute")
        remote = MemoryTier("cxl_pool", 8 * 1024 ** 3, 100, 5, "pool")
        memory = TopologyAwareMemory(
            [local, remote], [FabricLink("pool", "compute", 100, 10, "cxl")]
        )
        extra, path = memory.additional_kv_read_latency(
            "cxl_pool", "local_hbm", "compute", 1_000, 0
        )
        self.assertEqual(extra, 34)
        self.assertEqual(path, ["pool->compute"])
        self.assertEqual(memory.summary()["stats"]["kv_read_bytes"], 1_000)
        self.assertEqual(
            memory.additional_kv_read_latency("local_hbm", "local_hbm", "compute", 1_000, 0),
            (0, []),
        )

    def test_capacity_reservation(self):
        memory = TopologyAwareMemory([tier("destination", "a", capacity=1)], [])
        memory.reserve("destination", 1024 ** 3)
        with self.assertRaises(RuntimeError):
            memory.reserve("destination", 1)

    def test_block_prefetch_handoff(self):
        memory = TopologyAwareMemory(
            [tier("source", "a"), tier("destination", "b")],
            [FabricLink("a", "b", 100, 10, "ab")],
        )
        plan = memory.plan_kv_handoff("source", "destination", 1000, 100, 4, 2, 0)
        self.assertEqual(len(plan.blocks), 10)
        self.assertEqual(plan.ready_at_ns, 14)
        self.assertEqual(plan.completion_ns, 40)
        self.assertEqual(memory.used_bytes["destination"], 1000)


if __name__ == "__main__":
    unittest.main()
