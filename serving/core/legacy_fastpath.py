"""Cached mutable rank graphs for the file-based legacy ablation.

Uses the shared fast path's shape key and tagged value mapping, but retains
complete protobuf graphs per rank. A hit updates existing node objects and
serializes full legacy ET bytes; ASTRA-Sim still opens and parses rank files.
"""

from collections import OrderedDict

from .execution_templates import Node, _frame, _read_frames, _restore_values, _take_values
from .template_fastpath import _tagged, _untag, shape_key


class LegacyValueFastPath:
    def __init__(self, max_entries=256, verify=False):
        if max_entries <= 0:
            raise ValueError("legacy fast-path entry bound must be positive")
        self.max_entries = max_entries
        self.verify = verify
        self.entries = OrderedDict()
        self.stats = {"hit": 0, "learned": 0, "unsafe": 0, "ineligible": 0,
                      "verified": 0, "evictions": 0}

    def payloads(self, header, cols, num_npus, offset, local_offloading,
                 metadata, convert):
        key = shape_key(header, cols, num_npus, offset, local_offloading)
        if key is None:
            self.stats["ineligible"] += 1
            return convert(cols)
        if key not in self.entries:
            real = convert(cols)
            entry = self._learn(cols, real, convert(_tagged(cols)))
            self.entries[key] = entry
            self.stats["learned" if entry is not None else "unsafe"] += 1
            while len(self.entries) > self.max_entries:
                self.entries.popitem(last=False)
                self.stats["evictions"] += 1
            return real
        entry = self.entries[key]
        self.entries.move_to_end(key)
        if entry is None:
            self.stats["unsafe"] += 1
            return convert(cols)
        self.stats["hit"] += 1
        result = {}
        prefix = _frame(metadata)
        for rank, nodes in entry.items():
            output = bytearray(prefix)
            for node, targets in nodes:
                for holder, field, slot in targets:
                    value = 0 if slot is None else int(cols[slot[0]][slot[1]])
                    setattr(holder, field, value)
                output.extend(_frame(node.SerializeToString(deterministic=True)))
            result[rank] = bytes(output)
        if self.verify:
            if result != convert(cols):
                raise RuntimeError("legacy fast path differs from full conversion")
            self.stats["verified"] += 1
        return result

    @staticmethod
    def _learn(cols, real, tagged):
        if real.keys() != tagged.keys():
            return None
        entry = {}
        for rank, payload in real.items():
            frames = list(_read_frames(payload))
            tag_frames = list(_read_frames(tagged[rank]))
            if len(frames) != len(tag_frames) or frames[0] != tag_frames[0]:
                return None
            nodes = []
            for frame, tag_frame in zip(frames[1:], tag_frames[1:]):
                node, tag_node = Node(), Node()
                node.ParseFromString(frame)
                tag_node.ParseFromString(tag_frame)
                real_values, holders = _take_values(node)
                tag_values, _ = _take_values(tag_node)
                if node.SerializeToString(deterministic=True) != tag_node.SerializeToString(deterministic=True):
                    return None
                slots = [None if value == 0 else _untag(value, cols) for value in tag_values]
                if any(slot is False for slot in slots):
                    return None
                filled = tuple(0 if slot is None else int(cols[slot[0]][slot[1]]) for slot in slots)
                if filled != real_values:
                    return None
                targets = [(node, "duration_micros", slots[0])]
                for holder, field in holders:
                    position = 1 if holder.name == "comm_size" else 2
                    targets.append((holder, field, slots[position]))
                _restore_values(node, real_values, holders)
                nodes.append((node, targets))
            entry[rank] = nodes
        return entry
