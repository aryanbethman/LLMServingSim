"""Split a rank's Chakra ET into a shared structural template plus a
per-rank overlay.

Every rank in a batch runs the same graph; what differs between them is
small and local -- the comm peer ids and tags on send/recv nodes. Writing
one full .et per rank therefore writes the same structure N times. This
splits it once: the structure becomes an immutable, content-addressed
template, and each rank keeps only its deltas.

That is a different axis from the converted-graph cache in
`graph_generator.py`, which reuses a whole conversion when a later batch
produces byte-identical rows. This one deduplicates *across ranks within
one batch*, so what it saves grows with NPU count; the cache saves across
batches over time. Both apply.

The round trip is exact: `materialise_rank_et(*split_rank_et(payload))`
reproduces the input bytes. `TemplateBundleCollector` skips the round trip
entirely by sitting in the converter's output path -- see
`LLMConverter.convert_to_template_bundle`.

With ``values_mode`` (``--template-value-overlay``) the per-node numbers that
change with token counts -- compute ``duration_micros`` and the
``comm_size``/``tensor_size`` attributes -- are zeroed in the template and
carried per batch instead, so batches that differ only in those numbers share
one template. Without it the template and the wire format are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import base64
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import re
import sys
from typing import Dict, Iterable, List, Optional, Tuple


_RANK_ATTRIBUTES = {"comm_src", "comm_dst", "comm_tag"}
_RANK_NAME = re.compile(r"^(COMM_(?:SEND|RECV)_NODE_.*)_\d+_\d+$")
# Attributes whose value moves to the per-batch value overlay, in the order a
# node's values are stored after its duration.
_VALUE_ATTRIBUTES = ("comm_size", "tensor_size")


def _load_chakra_types():
    chakra_root = Path(__file__).resolve().parents[2] / "astra-sim" / "extern" / "graph_frontend"
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


def _framed(payloads) -> bytearray:
    """Length-delimit and concatenate payloads, as _frame would one by one.

    Node payloads are almost always under 128 bytes, whose varint length is
    the single byte itself; this skips a Python-level varint per node.
    """
    out = bytearray()
    for payload in payloads:
        size = len(payload)
        if size < 0x80:
            out.append(size)
        else:
            out += _write_varint(size)
        out += payload
    return out


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


_EMPTY_NODE_OVERLAY = NodeOverlay(None, ())


@dataclass(frozen=True)
class RankOverlay:
    metadata_payload: bytes
    nodes: Tuple[NodeOverlay, ...]
    # One (duration_micros, comm_size, tensor_size) per node in values mode.
    values: Optional[Tuple[Tuple[int, int, int], ...]] = None


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


class TemplateCaptureStream:
    """Capture one rank's ET as template structure instead of framed bytes.

    ``LLMConverter`` creates protobuf messages one at a time.  The historical
    shared-template path first wrote every message to a complete rank ET byte
    stream and then parsed that stream back into protobufs here.  This sink
    accepts those messages directly, preserving the exact normalisation and
    overlay encoding used by :func:`split_rank_et` without the intermediate
    rank payload.
    """

    def __init__(self, values_mode: bool = False):
        self._metadata_payload: Optional[bytes] = None
        self._node_payloads: List[bytes] = []
        self._overlays: List[NodeOverlay] = []
        self._values: Optional[List[Tuple[int, int, int]]] = [] if values_mode else None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def write_message(self, message) -> None:
        if message.__class__ is Node:
            self._append_node(message)
            return
        if isinstance(message, GlobalMetadata):
            if self._metadata_payload is not None:
                raise ValueError("ET stream contains more than one global metadata record")
            self._metadata_payload = message.SerializeToString()
            return
        if not isinstance(message, Node):
            raise TypeError(f"Unsupported Chakra ET message: {type(message)!r}")
        self._append_node(message)

    # Normalises one node, moving its values aside first in values mode. Called by: write_message. Calls: _take_values, _normalise_node, _restore_values.
    def _append_node(self, node) -> None:
        if self._values is None:
            node_payload, overlay = _normalise_node(node)
        else:
            values, holders = _take_values(node)
            try:
                node_payload, overlay = _normalise_node(node)
            finally:
                _restore_values(node, values, holders)
            self._values.append(values)
        self._node_payloads.append(node_payload)
        self._overlays.append(overlay)

    def split(self) -> Tuple[ExecutionTemplate, RankOverlay]:
        if self._metadata_payload is None:
            raise ValueError("ET stream has no global metadata record")
        template_bytes = _framed(self._node_payloads)
        template = ExecutionTemplate(
            template_id=sha256(template_bytes).hexdigest(),
            node_payloads=tuple(self._node_payloads),
        )
        values = None if self._values is None else tuple(self._values)
        return template, RankOverlay(self._metadata_payload, tuple(self._overlays), values)


class TemplateBundleCollector:
    """Build ASTRA's shared-template bundle directly from converter messages."""

    def __init__(self, known_template_ids: Optional[set[str]] = None, values_mode: bool = False):
        # Preserve a caller-provided empty set by reference.  Using ``or`` here
        # substitutes a new set during the first batch and breaks the intended
        # long-lived controller cache relationship.
        self._known_template_ids = (
            set() if known_template_ids is None else known_template_ids
        )
        self._values_mode = values_mode
        self._streams: Dict[int, TemplateCaptureStream] = {}

    def open_rank(self, rank_id: int) -> TemplateCaptureStream:
        if rank_id in self._streams:
            raise ValueError(f"Duplicate ET stream for rank {rank_id}")
        stream = TemplateCaptureStream(self._values_mode)
        self._streams[rank_id] = stream
        return stream

    def build(self) -> Tuple[Dict[str, object], Dict[str, int]]:
        splits = {
            rank_id: stream.split() for rank_id, stream in self._streams.items()
        }
        return _build_template_bundle_from_splits(splits, self._known_template_ids)


# Zeroes a node's duration and value attributes in place and returns what was there. Called by: TemplateCaptureStream._append_node, split_rank_et.
def _take_values(node) -> Tuple[Tuple[int, int, int], List[Tuple[object, str]]]:
    duration = node.duration_micros
    if duration:
        node.duration_micros = 0
    sizes = [0, 0]
    holders: List[Tuple[object, str]] = []
    for attribute in node.attr:
        if attribute.name not in _VALUE_ATTRIBUTES:
            continue
        slot = _VALUE_ATTRIBUTES.index(attribute.name)
        if any(held.name == attribute.name for held, _ in holders):
            raise ValueError(f"node {node.name!r} has more than one {attribute.name} attribute")
        field = attribute.WhichOneof("value")
        sizes[slot] = getattr(attribute, field)
        setattr(attribute, field, 0)
        holders.append((attribute, field))
    return (duration, sizes[0], sizes[1]), holders


# Puts back what _take_values zeroed. Called by: TemplateCaptureStream._append_node, split_rank_et.
def _restore_values(node, values: Tuple[int, int, int], holders: List[Tuple[object, str]]) -> None:
    if values[0]:
        node.duration_micros = values[0]
    for attribute, field in holders:
        setattr(attribute, field, values[1 + _VALUE_ATTRIBUTES.index(attribute.name)])


# Writes one node's overlay values into a parsed template node. Called by: materialise_rank_et.
def _apply_values(node, values: Tuple[int, int, int]) -> None:
    if values[0]:
        node.duration_micros = values[0]
    for attribute in node.attr:
        if attribute.name in _VALUE_ATTRIBUTES:
            setattr(attribute, attribute.WhichOneof("value"),
                    values[1 + _VALUE_ATTRIBUTES.index(attribute.name)])


def _normalise_node(node) -> Tuple[bytes, NodeOverlay]:
    # Only communication nodes can carry the rank-varying name and
    # comm_src/comm_dst/comm_tag attributes.  Most ET nodes are compute or
    # memory operations, so serialise them directly rather than making a full
    # protobuf copy and rebuilding an empty overlay for every rank.
    if not node.name.startswith("COMM_"):
        return node.SerializeToString(deterministic=True), _EMPTY_NODE_OVERLAY
    # Collectives are COMM_ nodes too, but carry neither a rank-specific name
    # nor rank attributes. When neither is present the copy below would change
    # nothing, so serialise the node as it is.
    if _RANK_NAME.match(node.name) is None:
        for attribute in node.attr:
            if attribute.name in _RANK_ATTRIBUTES:
                break
        else:
            return node.SerializeToString(deterministic=True), _EMPTY_NODE_OVERLAY

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


def split_rank_et(payload: bytes, values_mode: bool = False) -> Tuple[ExecutionTemplate, RankOverlay]:
    """Split one length-delimited Chakra ET stream into structure and rank overlay."""
    frames = list(_read_frames(payload))
    if not frames:
        raise ValueError("ET payload is empty")

    metadata = GlobalMetadata()
    metadata.ParseFromString(frames[0])
    node_payloads: List[bytes] = []
    overlays: List[NodeOverlay] = []
    values: Optional[List[Tuple[int, int, int]]] = [] if values_mode else None

    for frame_payload in frames[1:]:
        node = Node()
        node.ParseFromString(frame_payload)
        if values is not None:
            node_values, _ = _take_values(node)
            values.append(node_values)
        normalised_payload, overlay = _normalise_node(node)
        node_payloads.append(normalised_payload)
        overlays.append(overlay)

    template_bytes = _framed(node_payloads)
    template = ExecutionTemplate(
        template_id=sha256(template_bytes).hexdigest(),
        node_payloads=tuple(node_payloads),
    )
    return template, RankOverlay(frames[0], tuple(overlays),
                                 None if values is None else tuple(values))


def materialise_rank_et(template: ExecutionTemplate, overlay: RankOverlay) -> bytes:
    """Reconstruct the original rank ET payload from a template and its overlay."""
    if len(template.node_payloads) != len(overlay.nodes):
        raise ValueError("Template node count does not match rank overlay")
    if overlay.values is not None and len(overlay.values) != len(overlay.nodes):
        raise ValueError("Template node count does not match value overlay")

    output = bytearray(_frame(overlay.metadata_payload))
    for index, (template_payload, node_overlay) in enumerate(zip(template.node_payloads, overlay.nodes)):
        node = Node()
        node.ParseFromString(template_payload)
        if overlay.values is not None:
            _apply_values(node, overlay.values[index])
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

def _build_template_bundle_from_splits(
    splits: Dict[int, Tuple[ExecutionTemplate, RankOverlay]],
    known_template_ids: Optional[set[str]] = None,
) -> Tuple[Dict[str, object], Dict[str, int]]:
    """Encode pre-split ET streams into ASTRA's JSON-safe template bundle."""
    # An empty caller cache is still the controller's cache; do not replace it.
    known_template_ids = set() if known_template_ids is None else known_template_ids
    templates: Dict[str, List[str]] = {}
    bindings: Dict[str, object] = {}
    unique_template_ids = set()
    # Ranks of one TP group carry identical values, so each distinct set is
    # sent once and bindings refer to it by index.
    value_sets: Dict[Tuple[Tuple[int, int, int], ...], int] = {}

    for rank_id, (template, overlay) in sorted(splits.items()):
        unique_template_ids.add(template.template_id)
        if template.template_id not in known_template_ids:
            templates.setdefault(
                template.template_id,
                [base64.b64encode(node).decode("ascii") for node in template.node_payloads],
            )

        nodes: Dict[str, object] = {}
        for node_index, node_overlay in enumerate(overlay.nodes):
            if node_overlay.original_name is None and not node_overlay.rank_attributes:
                continue
            nodes[str(node_index)] = {
                "name": node_overlay.original_name,
                "attrs": [
                    [entry.position, base64.b64encode(entry.payload).decode("ascii")]
                    for entry in node_overlay.rank_attributes
                ],
            }
        binding = {
            "template_id": template.template_id,
            "metadata": base64.b64encode(overlay.metadata_payload).decode("ascii"),
            "nodes": nodes,
        }
        if overlay.values is not None:
            binding["values"] = value_sets.setdefault(overlay.values, len(value_sets))
        bindings[str(rank_id)] = binding

    bundle = {"templates": templates, "bindings": bindings}
    if value_sets:
        bundle["values"] = [
            {"d": [v[0] for v in values], "c": [v[1] for v in values], "t": [v[2] for v in values]}
            for values in value_sets
        ]
    stats = {
        "ranks": len(splits),
        "unique_templates": len(unique_template_ids),
        "templates_sent": len(templates),
        "wire_bytes_estimate": len(str(bundle).encode("utf-8")),
    }
    return bundle, stats


def build_template_bundle(
    payloads: Dict[int, bytes], known_template_ids: Optional[set[str]] = None,
    values_mode: bool = False,
) -> Tuple[Dict[str, object], Dict[str, int]]:
    """Create a JSON-safe structural-template bundle from framed ET payloads.

    This compatibility entry point retains the original payload-based API.
    New converter code uses :class:`TemplateBundleCollector` to avoid creating
    and reparsing those intermediate rank payloads.
    """
    splits = {
        rank_id: split_rank_et(payload, values_mode) for rank_id, payload in payloads.items()
    }
    return _build_template_bundle_from_splits(splits, known_template_ids)
