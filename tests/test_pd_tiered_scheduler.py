import unittest

from serving.core.request import Request
from serving.core.scheduler import Scheduler
from serving.core.tiered_memory import FabricLink, MemoryTier, TopologyAwareMemory


class MemoryStub:
    block_size = 16
    kv_fp = 2

    def get_kv(self, tokens):
        return tokens * 10

    def get_total_kv(self, req):
        return 1_000


class KVStub:
    def get_computed_blocks(self, req):
        return [], 0, 0

    def allocate_slots(self, req, num_new, *args):
        return ["block"]

    def take_traffic(self):
        return 0, 0


def decode_scheduler(tiered_memory, kv_tier):
    scheduler = object.__new__(Scheduler)
    scheduler.instance_id = 1
    scheduler.pd_type = "decode"
    scheduler.tiered_memory = tiered_memory
    scheduler.kv_tier = kv_tier
    scheduler.tiered_kv = tiered_memory is not None and kv_tier is not None
    scheduler.pd_transfer = {"chunk_blocks": 4, "prefetch_blocks": 2}
    scheduler.memory = MemoryStub()
    scheduler.kv = KVStub()
    scheduler.running = []
    scheduler.inflight = []
    scheduler.long_prefill_token_threshold = 0
    return scheduler


def prefilled_request():
    req = Request(0, "model", 32, 64, 0, 0, is_init=False)
    req.num_computed_tokens = 32
    req.num_tokens_reached = 33
    return req


class TieredPDHandoffTest(unittest.TestCase):
    def test_handoff_moves_the_reservation_and_gates_decode(self):
        tiers = [MemoryTier("prefill_hbm", 4_000, 1_000, 2, "prefill"),
                 MemoryTier("decode_hbm", 4_000, 1_000, 3, "decode")]
        memory = TopologyAwareMemory(tiers, [FabricLink("prefill", "decode", 100, 10, "fabric")])
        memory.reserve("prefill_hbm", 1_000)
        scheduler = decode_scheduler(memory, "decode_hbm")
        req = prefilled_request()
        req.kv_tier = "prefill_hbm"
        req.kv_reserved_bytes = 1_000

        scheduler.add_decode(req, 100)

        self.assertEqual(memory.used_bytes["prefill_hbm"], 0)
        self.assertEqual(memory.used_bytes["decode_hbm"], 1_000)
        self.assertEqual(req.kv_tier, "decode_hbm")
        self.assertGreater(req.pd_ready_at, 100)

        scheduled = []
        scheduler._schedule_running(req.pd_ready_at - 1, scheduled, [], 2048)
        self.assertEqual(scheduled, [])
        scheduler._schedule_running(req.pd_ready_at, scheduled, [], 2048)
        self.assertEqual([r for r, _, _ in scheduled], [req])

        scheduler._release_tier_kv(req)
        self.assertEqual(memory.used_bytes["decode_hbm"], 0)

    def test_decode_without_a_tier_starts_immediately(self):
        scheduler = decode_scheduler(None, None)
        req = prefilled_request()

        scheduler.add_decode(req, 100)

        self.assertEqual(req.pd_ready_at, 0)
        scheduled = []
        scheduler._schedule_running(0, scheduled, [], 2048)
        self.assertEqual([r for r, _, _ in scheduled], [req])


if __name__ == "__main__":
    unittest.main()
