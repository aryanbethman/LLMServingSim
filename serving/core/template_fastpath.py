"""Value-overlay bundles without re-running the converter.

With ``--template-value-overlay`` a bundle is the structural templates plus
one list of numbers per value set. The converter only copies those numbers
out of the trace rows -- compute time, memory sizes, comm sizes -- and the
rest of what it emits depends on the rows' structure: layer names, memory
locations, comm types, expert/PIM markers, the header, and whether each
number is zero (a zero compute time drops the compute node, a zero weight
size drops the weight load, a zero comm size drops the expert all-to-all and
the KV send). So two batches whose rows agree on all of that, the *shape*,
produce the same templates and bindings, and differ only in which row field
lands in which value slot.

The first batch of a shape is converted normally. The rows are also
converted once with every non-zero number replaced by a tag naming its
(row, field), which shows where each slot's value comes from. That map is
kept only if it reproduces the real conversion exactly; a shape whose values
are not plain copies (PIM adds two sizes) always uses the converter. Every
later batch of the shape fills the value sets straight from its rows.

The bundle is identical to a fresh conversion's, byte for byte, including
which ranks share a value set. LLMSS_VERIFY_VALUE_FASTPATH=1 re-converts
every hit and raises on any difference.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

_LAYER_FIELDS = 11
# Numeric columns of a layer row: comp_time and the input/weight/output
# memory sizes and the comm size. EXPERT markers carry a comm size at 3.
_LAYER_NUMERIC = (1, 3, 5, 7, 9)
_EXPERT_COMM = 3
_TAG_BASE = 1 << 40
_TAG_FIELDS = 16

Slot = Optional[Tuple[int, int]]          # (row, column), or None for a constant 0
Mapping = Tuple[Tuple[Slot, ...], Tuple[Slot, ...], Tuple[Slot, ...]]


def _tag(row: int, column: int) -> int:
    return _TAG_BASE + row * _TAG_FIELDS + column


def _untag(value: int, cols: List[list]) -> Slot:
    offset = value - _TAG_BASE
    if offset < 0:
        return False
    row, column = divmod(offset, _TAG_FIELDS)
    if row >= len(cols) or column not in _numeric_columns(cols[row]):
        return False
    return row, column


def _numeric_columns(col: list) -> Tuple[int, ...]:
    if len(col) == _LAYER_FIELDS:
        return _LAYER_NUMERIC
    if col and col[0] == "EXPERT" and len(col) > _EXPERT_COMM:
        return (_EXPERT_COMM,)
    return ()


# Structure of a trace as the converter's branches see it, or None when the fast path cannot apply. Called by: ValueFastPath.bundle.
def shape_key(header_line: str, cols: List[list], num_npus: int, npu_offset: int,
              local_offloading: bool) -> Optional[tuple]:
    if header_line.split(None, 1)[0] == "EVENT":
        return None
    shape = [header_line, num_npus, npu_offset, bool(local_offloading)]
    for col in cols:
        if len(col) == _LAYER_FIELDS:
            shape.append((col[0], col[2], col[4], col[6], col[8], col[10],
                           int(col[1]) != 0, int(col[5]) != 0, int(col[9]) != 0))
        elif col and col[0] == "EXPERT":
            comm = int(col[_EXPERT_COMM]) != 0 if len(col) > _EXPERT_COMM else None
            shape.append(("EXPERT", tuple(col[1:_EXPERT_COMM]), comm, len(col)))
        else:
            # PIM sums two sizes into one value; anything else is unknown.
            return None
    return tuple(shape)


def _tagged(cols: List[list]) -> List[list]:
    tagged = []
    for row, col in enumerate(cols):
        copy = list(col)
        for column in _numeric_columns(col):
            if int(col[column]) != 0:
                copy[column] = _tag(row, column)
        tagged.append(copy)
    return tagged


def _rank_values(bundle: dict) -> Dict[str, tuple]:
    """Each rank's value set as one (d, c, t) tuple, independent of set indices."""
    sets = [(tuple(s["d"]), tuple(s["c"]), tuple(s["t"])) for s in bundle.get("values", [])]
    return {rank: sets[b["values"]] for rank, b in bundle["bindings"].items()}


class _Entry:
    __slots__ = ("templates", "ranks", "mappings")

    def __init__(self, templates, ranks, mappings):
        self.templates = templates      # template_id -> base64 node list
        self.ranks = ranks              # ((rank, template_id, metadata, nodes, mapping index), ...)
        self.mappings = mappings        # (Mapping, ...)


def _fill(mapping: Mapping, cols: List[list]) -> tuple:
    return tuple(
        tuple(0 if slot is None else int(cols[slot[0]][slot[1]]) for slot in slots)
        for slots in mapping
    )


class ValueFastPath:
    """Shape-keyed store of value maps for --template-value-overlay bundles."""

    def __init__(self, verify: bool = False, max_entries: int = 256):
        self.verify = verify
        self.max_entries = max_entries
        self._entries: "OrderedDict[tuple, Optional[_Entry]]" = OrderedDict()
        self.stats = {"hit": 0, "learned": 0, "unsafe": 0, "ineligible": 0,
                      "verified": 0, "evictions": 0}

    # Returns (bundle, stats) for one batch, filled from the rows when the shape is known. Called by: graph_generator.generate_graph. Calls: shape_key, _learn, _build.
    def bundle(self, header_line: str, cols: List[list], num_npus: int, npu_offset: int,
               local_offloading: bool, known_template_ids: set,
               convert: Callable[[List[list], set], Tuple[dict, dict]]) -> Tuple[dict, dict]:
        key = shape_key(header_line, cols, num_npus, npu_offset, local_offloading)
        if key is None:
            self.stats["ineligible"] += 1
            return convert(cols, known_template_ids)
        if key in self._entries:
            entry = self._entries[key]
            self._entries.move_to_end(key)
            if entry is None:
                self.stats["unsafe"] += 1
                return convert(cols, known_template_ids)
            self.stats["hit"] += 1
            result = self._build(entry, cols, known_template_ids)
            if self.verify:
                expected, _ = convert(cols, known_template_ids)
                if _wire(expected) != _wire(result[0]):
                    raise RuntimeError("value fast path bundle differs from a fresh conversion")
                self.stats["verified"] += 1
            return result
        result = convert(cols, known_template_ids)
        self._store(key, self._learn(cols, result[0], convert))
        return result

    def _store(self, key: tuple, entry: Optional[_Entry]) -> None:
        self.stats["learned" if entry is not None else "unsafe"] += 1
        self._entries[key] = entry
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
            self.stats["evictions"] += 1

    # Derives the value map from a tagged conversion and keeps it only if it reproduces `real`. Called by: bundle. Calls: _tagged, _untag, _fill.
    def _learn(self, cols: List[list], real: dict, convert) -> Optional[_Entry]:
        tagged, _ = convert(_tagged(cols), set())
        if ("values" not in tagged or "values" not in real
                or set(tagged["bindings"]) != set(real["bindings"])):
            return None
        mappings: List[Mapping] = []
        for value_set in tagged["values"]:
            mapping = []
            for slots in (value_set["d"], value_set["c"], value_set["t"]):
                resolved = []
                for value in slots:
                    slot = None if value == 0 else _untag(value, cols)
                    if slot is False:
                        return None
                    resolved.append(slot)
                mapping.append(tuple(resolved))
            mappings.append(tuple(mapping))

        ranks = []
        for rank, binding in tagged["bindings"].items():
            real_binding = real["bindings"][rank]
            if any(real_binding[k] != binding[k] for k in ("template_id", "metadata", "nodes")):
                return None
            ranks.append((rank, binding["template_id"], binding["metadata"],
                          binding["nodes"], binding["values"]))
        for template_id, nodes in real["templates"].items():
            if tagged["templates"].get(template_id) != nodes:
                return None
        if list(real["bindings"]) != [r[0] for r in ranks]:
            return None

        filled = [_fill(m, cols) for m in mappings]
        real_values = _rank_values(real)
        if any(filled[r[4]] != real_values[r[0]] for r in ranks):
            return None
        return _Entry(dict(tagged["templates"]), tuple(ranks), tuple(mappings))

    # Assembles the bundle a fresh conversion would produce for these rows. Called by: bundle. Calls: _fill.
    @staticmethod
    def _build(entry: _Entry, cols: List[list], known_template_ids: set) -> Tuple[dict, dict]:
        filled = [_fill(m, cols) for m in entry.mappings]
        templates: Dict[str, list] = {}
        bindings: Dict[str, dict] = {}
        value_sets: Dict[tuple, int] = {}
        for rank, template_id, metadata, nodes, mapping_index in entry.ranks:
            if template_id not in known_template_ids:
                templates.setdefault(template_id, entry.templates[template_id])
            values = filled[mapping_index]
            bindings[rank] = {"template_id": template_id, "metadata": metadata, "nodes": nodes,
                              "values": value_sets.setdefault(values, len(value_sets))}
        bundle = {"templates": templates, "bindings": bindings,
                  "values": [{"d": list(d), "c": list(c), "t": list(t)} for d, c, t in value_sets]}
        stats = {"ranks": len(bindings), "unique_templates": len({r[1] for r in entry.ranks}),
                 "templates_sent": len(templates), "fast_path": True}
        return bundle, stats


def _wire(bundle: dict) -> str:
    return json.dumps(bundle, separators=(",", ":"))
