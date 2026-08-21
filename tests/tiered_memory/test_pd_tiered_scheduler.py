import unittest

from inference_serving.request import Request
from inference_serving.scheduler import Scheduler
from inference_serving.tiered_memory import FabricLink, MemoryTier, TopologyAwareMemory


class MemoryStub:
    block_size = 16
    fp = 2

    def get_kv(self, tokens):
        return tokens * 10

    def get_total_kv(self, req):
        return 1_000


class TieredPDHandoffTest(unittest.TestCase):
    def test_legacy_decode_admission_stays_compatible(self):
        scheduler = object.__new__(Scheduler)
        scheduler.tiered_memory = None
        scheduler.kv_tier = None
        scheduler.memory = MemoryStub()
        scheduler.memory.allocated = []
        scheduler.memory.allocate = lambda amount, device: scheduler.memory.allocated.append(amount)
        scheduler.request = []

        req = Request(0, "model", 32, 64, 0, 0, is_init=False)
        scheduler.add_decode(req)

        self.assertEqual(scheduler.request, [req])
        self.assertEqual(scheduler.memory.allocated, [1_000])

    def test_handoff_transfers_ownership_and_gates_decode(self):
        tiers = [
            MemoryTier("prefill_hbm", 4_000, 1_000, 2, "prefill"),
            MemoryTier("decode_hbm", 4_000, 1_000, 3, "decode"),
        ]
        memory = TopologyAwareMemory(
            tiers, [FabricLink("prefill", "decode", 100, 10, "fabric")]
        )
        memory.reserve("prefill_hbm", 1_000)

        scheduler = object.__new__(Scheduler)
        scheduler.tiered_memory = memory
        scheduler.kv_tier = "decode_hbm"
        scheduler.pd_transfer = {"chunk_blocks": 4, "prefetch_blocks": 2}
        scheduler.memory = MemoryStub()
        scheduler.request = []

        req = Request(0, "model", 32, 64, 0, 0, is_init=False)
        req.kv_tier = "prefill_hbm"
        req.kv_reserved_bytes = 1_000
        scheduler.add_decode(req, current=100, source_tier="prefill_hbm")

        self.assertEqual(memory.used_bytes["prefill_hbm"], 0)
        self.assertEqual(memory.used_bytes["decode_hbm"], 1_000)
        self.assertEqual(req.kv_tier, "decode_hbm")
        self.assertEqual(req.kv_reserved_bytes, 1_000)
        self.assertGreater(req.pd_ready_at, 100)
        self.assertIsNotNone(req.kv_transfer_plan)
        self.assertEqual(scheduler.request, [req])


if __name__ == "__main__":
    unittest.main()
