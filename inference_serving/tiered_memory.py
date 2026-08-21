"""Topology-aware memory tiers and KV-transfer accounting."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from typing import Any, Dict, Iterable, List, Optional, Tuple

GB = 1024 ** 3


@dataclass(frozen=True)
class MemoryTier:
    name: str
    capacity_bytes: int
    service_bw_gbps: float
    access_latency_ns: int
    endpoint: str
    sharing_scope: str = "cluster"
    shared: bool = True


@dataclass(frozen=True)
class FabricLink:
    src: str
    dst: str
    bandwidth_gbps: float
    latency_ns: int
    contention_group: str


@dataclass
class KVBlock:
    index: int
    bytes: int
    owner: str
    precision_bytes: int = 0
    transfer_state: str = "complete"
    ready_at_ns: int = 0
    transfer_complete_ns: int = 0


@dataclass
class KVTransferPlan:
    source: str
    destination: str
    blocks: List[KVBlock]
    chunk_blocks: int
    prefetch_blocks: int
    reservation_bytes: int
    ready_at_ns: int
    completion_ns: int
    path: List[str]


class TopologyAwareMemory:
    """Shared tier capacity and deterministic, contention-aware fabric timing."""

    def __init__(self, tiers: Iterable[MemoryTier], links: Iterable[FabricLink]):
        self.tiers: Dict[str, MemoryTier] = {tier.name: tier for tier in tiers}
        self.used_bytes: Dict[str, int] = {tier.name: 0 for tier in tiers}
        self.links = list(links)
        self.adj: Dict[str, List[FabricLink]] = {}
        self.next_free_ns: Dict[str, int] = {}
        self.tier_next_free_ns: Dict[str, int] = {tier.name: 0 for tier in tiers}
        for link in self.links:
            self.adj.setdefault(link.src, []).append(link)
            self.next_free_ns.setdefault(link.contention_group, 0)
        self.stats: Dict[str, int] = {
            "transfers": 0,
            "transfer_bytes": 0,
            "transfer_stall_ns": 0,
            "tier_access_stall_ns": 0,
            "transfer_latency_ns": 0,
            "reservation_failures": 0,
            "decode_admission_wait_ns": 0,
            "blocks_transferred": 0,
            "prefetch_ready_blocks": 0,
        }
        self.link_bytes: Dict[str, int] = {link.contention_group: 0 for link in self.links}
        self.link_busy_ns: Dict[str, int] = {link.contention_group: 0 for link in self.links}
        self.tier_peak_bytes: Dict[str, int] = {tier.name: 0 for tier in tiers}

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]]) -> Optional["TopologyAwareMemory"]:
        if not config:
            return None
        tiers = []
        for raw in config.get("memory_tiers", []):
            capacity = raw.get("capacity_gb", raw.get("capacity", 0))
            tiers.append(MemoryTier(
                name=raw["name"],
                capacity_bytes=int(float(capacity) * GB),
                service_bw_gbps=float(raw["service_bw_gbps"]),
                access_latency_ns=int(raw.get("access_latency_ns", 0)),
                endpoint=raw.get("endpoint", raw["name"]),
                sharing_scope=raw.get("sharing_scope", "cluster" if raw.get("shared", True) else "instance"),
                shared=bool(raw.get("shared", True)),
            ))
        links = []
        for raw in config.get("fabric", {}).get("links", []):
            links.append(FabricLink(
                src=raw["src"],
                dst=raw["dst"],
                bandwidth_gbps=float(raw["bandwidth_gbps"]),
                latency_ns=int(raw.get("latency_ns", 0)),
                contention_group=raw.get("contention_group", raw["src"] + "->" + raw["dst"]),
            ))
        return cls(tiers, links) if tiers else None

    def reserve(self, tier: str, bytes_: int) -> None:
        if tier not in self.tiers:
            raise KeyError("Unknown memory tier: " + tier)
        if self.used_bytes[tier] + bytes_ > self.tiers[tier].capacity_bytes:
            self.stats["reservation_failures"] += 1
            free = self.tiers[tier].capacity_bytes - self.used_bytes[tier]
            raise RuntimeError("Tier {} cannot reserve {} bytes; only {} remain".format(tier, bytes_, free))
        self.used_bytes[tier] += bytes_
        self.tier_peak_bytes[tier] = max(self.tier_peak_bytes[tier], self.used_bytes[tier])

    def release(self, tier: str, bytes_: int) -> None:
        self.used_bytes[tier] = max(0, self.used_bytes[tier] - bytes_)

    def _tier_access(self, tier: str, bytes_: int, current_ns: int) -> int:
        """Serialize a read or write at a tier, including its base access latency."""
        config = self.tiers[tier]
        begin = max(current_ns, self.tier_next_free_ns[tier])
        self.stats["tier_access_stall_ns"] += begin - current_ns
        service = int(bytes_ / config.service_bw_gbps) if config.service_bw_gbps else 0
        done = begin + config.access_latency_ns + service
        self.tier_next_free_ns[tier] = done
        return done

    def _path(self, source: str, destination: str) -> List[FabricLink]:
        if source == destination:
            return []
        serial = count()
        queue: List[Tuple[int, int, str, List[FabricLink]]] = [(0, next(serial), source, [])]
        seen: Dict[str, int] = {}
        while queue:
            cost, _, node, path = heappop(queue)
            if node == destination:
                return path
            if cost >= seen.get(node, 1 << 62):
                continue
            seen[node] = cost
            for link in self.adj.get(node, []):
                heappush(queue, (cost + link.latency_ns, next(serial), link.dst, path + [link]))
        raise RuntimeError("No fabric path from {} to {}".format(source, destination))

    def transfer(self, source: str, destination: str, bytes_: int, start_ns: int) -> Tuple[int, List[str]]:
        path = self._path(source, destination)
        current = start_ns
        names: List[str] = []
        for link in path:
            serialization = int(bytes_ / link.bandwidth_gbps) if link.bandwidth_gbps else 0
            begin = max(current, self.next_free_ns[link.contention_group])
            self.stats["transfer_stall_ns"] += begin - current
            current = begin + link.latency_ns + serialization
            self.next_free_ns[link.contention_group] = current
            self.link_bytes[link.contention_group] += bytes_
            self.link_busy_ns[link.contention_group] += link.latency_ns + serialization
            names.append(link.src + "->" + link.dst)
        self.stats["transfers"] += 1
        self.stats["transfer_bytes"] += bytes_
        self.stats["transfer_latency_ns"] += current - start_ns
        return current, names

    def plan_kv_handoff(self, source_tier: str, destination_tier: str, total_bytes: int,
                        block_bytes: int, chunk_blocks: int, prefetch_blocks: int,
                        start_ns: int, precision_bytes: int = 0) -> KVTransferPlan:
        self.reserve(destination_tier, total_bytes)
        source = self.tiers[source_tier].endpoint
        destination = self.tiers[destination_tier].endpoint
        blocks: List[KVBlock] = []
        ready_at = start_ns
        completion = start_ns
        path_names: List[str] = []
        offset = 0
        chunk_blocks = max(1, chunk_blocks)
        while offset < total_bytes:
            chunk = min(total_bytes - offset, block_bytes * chunk_blocks)
            read_done = self._tier_access(source_tier, chunk, completion)
            fabric_done, path_names = self.transfer(source, destination, chunk, read_done)
            completion = self._tier_access(destination_tier, chunk, fabric_done)
            block_count = max(1, (chunk + block_bytes - 1) // block_bytes)
            for _ in range(block_count):
                size = min(block_bytes, total_bytes - len(blocks) * block_bytes)
                blocks.append(KVBlock(len(blocks), size, destination_tier,
                                      precision_bytes=precision_bytes,
                                      transfer_complete_ns=completion))
            if len(blocks) >= max(1, prefetch_blocks) and ready_at == start_ns:
                ready_at = completion
            offset += chunk
        self.stats["blocks_transferred"] += len(blocks)
        self.stats["prefetch_ready_blocks"] += min(len(blocks), max(1, prefetch_blocks))
        self.stats["decode_admission_wait_ns"] += ready_at - start_ns
        return KVTransferPlan(source_tier, destination_tier, blocks, chunk_blocks,
                              prefetch_blocks, total_bytes, ready_at, completion, path_names)

    def summary(self, simulation_end_ns: Optional[int] = None) -> Dict[str, Any]:
        return {
            "stats": dict(self.stats),
            "tiers": {
                name: {
                    "used_bytes": self.used_bytes[name],
                    "peak_bytes": self.tier_peak_bytes[name],
                    "capacity_bytes": tier.capacity_bytes,
                }
                for name, tier in self.tiers.items()
            },
            "link_bytes": dict(self.link_bytes),
            "link_busy_ns": dict(self.link_busy_ns),
            "link_utilization": {
                group: (busy / simulation_end_ns if simulation_end_ns else None)
                for group, busy in self.link_busy_ns.items()
            },
        }
