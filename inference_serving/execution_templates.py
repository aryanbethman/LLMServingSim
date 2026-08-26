"""Lossless shared execution-template manifests for Chakra ET payloads.

This module is intentionally not wired into ASTRA yet. It proves that a
rank-specific ET can be split into an immutable structural template plus a small
rank overlay and reconstructed exactly before changing the simulator protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import re
import sys
from typing import Dict, Iterable, List, Optional, Tuple


_RANK_ATTRIBUTES = {"comm_src", "comm_dst", "comm_tag"}
_RANK_NAME = re.compile(r"^(COMM_(?:SEND|RECV)_NODE_.*)_\d+_\d+$")


def _load_chakra_types():
    chakra_root = Path(__file__).resolve().parents[1] / "astra-sim" / "extern" / "graph_frontend"
    if str(chakra_root) not in sys.path:
        sys.path.insert(0, str(chakra_root))
    from chakra.schema.protobuf.et_def_pb2 import AttributeProto, GlobalMetadata, Node
    return AttributeProto, GlobalMetadata, Node


AttributeProto, GlobalMetadata, Node = _load_chakra_types()


def _read_varint(stream: BytesIO) -> Optional[int]:
    value = 0
    shift = 0
    while True:
        raw = stream.read(1)
        if not raw:
            if shift == 0:
                return None
            raise ValueError("Truncated ET frame length")
        byte = raw[0]
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value
        shift += 7
        if shift >= 64:
            raise ValueError("Invalid ET frame length")


def _write_varint(value: int) -> bytes:
    out = bytearray()
    while value > 0x7F:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _read_frames(payload: bytes) -> Iterable[bytes]:
    stream = BytesIO(payload)
    while True:
        size = _read_varint(stream)
        if size is None:
            return
        frame = stream.read(size)
        if len(frame) != size:
            raise ValueError("Truncated ET frame payload")
        yield frame


def _frame(payload: bytes) -> bytes:
    return _write_varint(len(payload)) + payload


@dataclass(frozen=True)
class RankAttributeOverlay:
    position: int
    payload: bytes


@dataclass(frozen=True)
class NodeOverlay:
    original_name: Optional[str]
    rank_attributes: Tuple[RankAttributeOverlay, ...]


@dataclass(frozen=True)
class RankOverlay:
    metadata_payload: bytes
    nodes: Tuple[NodeOverlay, ...]


@dataclass(frozen=True)
class ExecutionTemplate:
    template_id: str
    node_payloads: Tuple[bytes, ...]

    @property
    def bytes(self) -> int:
        return sum(len(payload) + len(_write_varint(len(payload))) for payload in self.node_payloads)


@dataclass(frozen=True)
class TemplateBinding:
    rank_id: int
    template_id: str
    overlay: RankOverlay


@dataclass
class _TemplateRecord:
    template: ExecutionTemplate
    references: int = 0


def _normalise_node(node) -> Tuple[bytes, NodeOverlay]:
    normalised = Node()
    normalised.CopyFrom(node)
    original_name = None
    match = _RANK_NAME.match(normalised.name)
    if match:
        original_name = normalised.name
        normalised.name = match.group(1) + "_<src>_<dst>"

    rank_attributes: List[RankAttributeOverlay] = []
    kept = []
    for position, attribute in enumerate(normalised.attr):
        if attribute.name in _RANK_ATTRIBUTES:
            rank_attributes.append(
                RankAttributeOverlay(position, attribute.SerializeToString(deterministic=True))
            )
        else:
            copy = AttributeProto()
            copy.CopyFrom(attribute)
            kept.append(copy)
    if rank_attributes:
        del normalised.attr[:]
        normalised.attr.extend(kept)

    return (
        normalised.SerializeToString(deterministic=True),
        NodeOverlay(original_name, tuple(rank_attributes)),
    )


def split_rank_et(payload: bytes) -> Tuple[ExecutionTemplate, RankOverlay]:
    """Split one length-delimited Chakra ET stream into structure and rank overlay."""
    frames = list(_read_frames(payload))
    if not frames:
        raise ValueError("ET payload is empty")

    metadata = GlobalMetadata()
    metadata.ParseFromString(frames[0])
    node_payloads: List[bytes] = []
    overlays: List[NodeOverlay] = []

    for frame_payload in frames[1:]:
        node = Node()
        node.ParseFromString(frame_payload)
        normalised_payload, overlay = _normalise_node(node)
        node_payloads.append(normalised_payload)
        overlays.append(overlay)

    template_bytes = b"".join(_frame(node_payload) for node_payload in node_payloads)
    template = ExecutionTemplate(
        template_id=sha256(template_bytes).hexdigest(),
        node_payloads=tuple(node_payloads),
    )
    return template, RankOverlay(frames[0], tuple(overlays))


def materialise_rank_et(template: ExecutionTemplate, overlay: RankOverlay) -> bytes:
    """Reconstruct the original rank ET payload from a template and its overlay."""
    if len(template.node_payloads) != len(overlay.nodes):
        raise ValueError("Template node count does not match rank overlay")

    output = bytearray(_frame(overlay.metadata_payload))
    for template_payload, node_overlay in zip(template.node_payloads, overlay.nodes):
        node = Node()
        node.ParseFromString(template_payload)
        if node_overlay.original_name is not None:
            node.name = node_overlay.original_name
        if node_overlay.rank_attributes:
            total_attributes = len(node.attr) + len(node_overlay.rank_attributes)
            rank_by_position = {entry.position: entry.payload for entry in node_overlay.rank_attributes}
            retained = list(node.attr)
            rebuilt = []
            retained_index = 0
            for position in range(total_attributes):
                rank_payload = rank_by_position.get(position)
                if rank_payload is not None:
                    attribute = AttributeProto()
                    attribute.ParseFromString(rank_payload)
                    rebuilt.append(attribute)
                else:
                    rebuilt.append(retained[retained_index])
                    retained_index += 1
            del node.attr[:]
            node.attr.extend(rebuilt)
        output.extend(_frame(node.SerializeToString(deterministic=True)))
    return bytes(output)


class TemplateStore:
    """Reference-counted in-memory store for structural templates."""

    def __init__(self):
        self._records: Dict[str, _TemplateRecord] = {}

    def bind(self, rank_id: int, payload: bytes) -> TemplateBinding:
        template, overlay = split_rank_et(payload)
        record = self._records.get(template.template_id)
        if record is None:
            record = _TemplateRecord(template)
            self._records[template.template_id] = record
        elif record.template.node_payloads != template.node_payloads:
            raise RuntimeError("SHA-256 collision in execution-template store")
        record.references += 1
        return TemplateBinding(rank_id, template.template_id, overlay)

    def materialise(self, binding: TemplateBinding) -> bytes:
        return materialise_rank_et(self._records[binding.template_id].template, binding.overlay)

    def release(self, binding: TemplateBinding) -> None:
        record = self._records[binding.template_id]
        record.references -= 1
        if record.references < 0:
            raise RuntimeError("Execution-template reference count underflow")
        if record.references == 0:
            del self._records[binding.template_id]

    def summary(self) -> Dict[str, int]:
        return {
            "templates": len(self._records),
            "template_bytes": sum(record.template.bytes for record in self._records.values()),
            "references": sum(record.references for record in self._records.values()),
        }
