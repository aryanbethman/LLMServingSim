from typing import Any, List, Optional

from ..memory_model import Device
from .base import EvictionAction, EvictionPolicy
from .registry import register_policy


@register_policy("evicpress")
class EvicPressPolicy(EvictionPolicy):
    """Online EVICPRESS-inspired greedy policy.

    This policy uses an online utility-drop heuristic at eviction time to select
    request/device/ratio actions, matching the paper's greedy design direction
    while staying compatible with this simulator's request scheduler.
    """

    def __init__(self, evicpress_alpha: float = 1.0, evicpress_ratios=None, **kwargs):
        del kwargs
        self.evicpress_alpha = float(evicpress_alpha)
        if evicpress_ratios is None:
            evicpress_ratios = [1.0, 0.75, 0.5, 0.25]

        self.evicpress_ratios = sorted(
            {max(1e-6, min(float(r), 1.0)) for r in evicpress_ratios if float(r) > 0},
            reverse=True,
        )
        if not self.evicpress_ratios:
            self.evicpress_ratios = [1.0]

    def build_pool(self, gen_req: List[Any], scheduler: Any) -> List[Any]:
        del scheduler
        return list(gen_req)

    def _frequency(self, req: Any) -> float:
        remaining = max(0, req.output - req.input)
        # Keep the scaling bounded so one long request does not dominate.
        return 1.0 + min(10.0, remaining / 32.0)

    def _quality(self, req: Any, ratio: float, scheduler: Any) -> float:
        max_pos = max(1, scheduler.config.get("max_position_embeddings", 1))
        remaining = max(0, req.output - req.input)
        total = max(req.output, req.input + 1)
        remaining_ratio = remaining / total
        length_ratio = min(1.0, req.input / max_pos)
        sensitivity = min(1.0, 0.25 + 0.5 * remaining_ratio + 0.25 * length_ratio)
        quality = 1.0 - (1.0 - ratio) * sensitivity
        return max(0.0, min(1.0, quality))

    def _ttft_seconds(self, moved_bytes: int, device: Device, scheduler: Any) -> float:
        if device == Device.CPU:
            bw = scheduler.cpu_tier_bw
            latency_ns = scheduler.cpu_tier_latency
        else:
            bw = scheduler.external_tier_bw
            latency_ns = scheduler.external_tier_latency

        bytes_per_sec = max(1e-9, bw * 1_000_000_000.0)
        return (latency_ns * 1e-9) + (moved_bytes / bytes_per_sec)

    def _utility(self, req: Any, ratio: float, moved_bytes: int, device: Device, scheduler: Any) -> float:
        freq = self._frequency(req)
        quality = self._quality(req, ratio, scheduler)
        ttft_seconds = self._ttft_seconds(moved_bytes, device, scheduler)
        return (self.evicpress_alpha * quality - ttft_seconds) * freq

    def select_action(self, evict_pool: List[Any], scheduler: Any) -> Optional[EvictionAction]:
        best_action = None
        best_drop_density = None

        device_candidates = []
        if scheduler.memory.cxl_mem > 0:
            device_candidates.append(Device.CXL)
        device_candidates.append(Device.CPU)

        for req in evict_pool:
            if req.evict:
                continue

            raw_bytes = scheduler.memory.get_evict_kv(req)
            if raw_bytes <= 0:
                continue

            baseline = self.evicpress_alpha * self._frequency(req)

            for ratio in self.evicpress_ratios:
                stored_bytes = max(1, int(raw_bytes * ratio))

                for device in device_candidates:
                    if not scheduler.memory.is_avail(stored_bytes, device):
                        continue

                    option_utility = self._utility(req, ratio, stored_bytes, device, scheduler)
                    utility_drop = baseline - option_utility
                    # We evict entire request KV from NPU in this simulator.
                    freed_bytes = max(raw_bytes, 1)
                    drop_density = utility_drop / freed_bytes

                    action = EvictionAction(
                        req=req,
                        raw_bytes=raw_bytes,
                        stored_bytes=stored_bytes,
                        device=device,
                        ratio=ratio,
                        utility=option_utility,
                    )

                    if best_action is None:
                        best_action = action
                        best_drop_density = drop_density
                        continue

                    if drop_density < best_drop_density:
                        best_action = action
                        best_drop_density = drop_density
                    elif drop_density == best_drop_density and action.stored_bytes < best_action.stored_bytes:
                        best_action = action

        return best_action
