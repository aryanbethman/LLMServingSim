import glob
import hashlib
import json
import os
from collections import OrderedDict
from time import time
from .request import *
from .logger import get_logger
from .run_paths import input_path
from .trace_generator import indexed_cols, write_trace

logger = get_logger("GraphGenerator")

# ----------------------------------------------------------------------
# Content-addressed cache of converted graphs.
#
# A DP group whose members are unevenly loaded spends half its waves on
# dummy batches: an idle member emits a 1-token placeholder so the round's
# ALLTOALL still has a partner, and _pad_batch_to_max then inflates it to
# the group's max, so its trace is the same shape and cost as a real one.
# On the swe-bench MoE DP+EP example that is 4,405 of 8,810 batches -- and
# only 22 of those 4,405 traces are distinct, with one accounting for
# 4,382. Converting that same graph 4,382 times cost ~12 s of a 63 s run.
#
# Keyed on the trace bytes plus every other input the converter reads
# (num_npus, npu_offset, local_offloading -- its whole CLI surface besides
# the paths), so a hit is byte-identical to a miss by construction. The
# PREFILL path writes two files per rank and both land next to each other,
# so the cache stores whatever `llm.*.et` a conversion produced rather
# than assuming a count.
#
# Held as bytes rather than paths because each DP wave needs its own copy:
# ASTRA-Sim is handed one folder per wave and every member reads its own
# `llm.<npu>.et` out of it.
# ----------------------------------------------------------------------
_ET_CACHE = OrderedDict()          # key -> [(basename, bytes), ...]
_ET_CACHE_BYTES = 0
_ET_CACHE_MAX_BYTES = 64 * 1024 * 1024
_ET_CACHE_STATS = {"hit": 0, "miss": 0, "skipped": 0}

# Traces seen exactly once. A graph is only worth holding after it repeats:
# most *real* batches produce a unique trace (3,482 distinct out of 4,405 on
# the swe-bench MoE DP+EP example), and caching those on first sight filled
# 64 MB with entries that were never read again, evicting the dummy-wave
# graph that actually repeats. Storing on second sight costs one extra
# conversion per distinct trace and keeps the cache to what earns its keep.
_ET_SEEN = OrderedDict()
_ET_SEEN_MAX = 200_000


# ----------------------------------------------------------------------
# Shared-template bindings cache: the _ET_CACHE's counterpart for the
# file-free path.
#
# A converter is built fresh per batch, so a shared-template bundle is a
# pure function of the trace rows plus (num_npus, npu_offset,
# local_offloading) -- the same key the _ET_CACHE uses. The global
# metadata carries input_file=None in this mode, so nothing batch-specific
# leaks in.
#
# npu_offset is left out of the key. Identical instances produce identical
# traces, and when no rank carries a comm/name overlay (TP-only graphs:
# collectives, no send/recv) the bindings differ only in their global rank
# ids, so a hit relocates them to the requesting instance's offset. On the
# 16-NPU ShareGPT-750 run (4 x TP4) keying on the offset made the same
# batch shape four separate entries. Bindings that do carry overlays hold
# global peer ids inside them, so they only hit at the offset that produced
# them.
#
# Only the rank bindings are held, pre-encoded per local rank as the exact
# JSON the controller would have written, never the template bodies:
# ASTRA-Sim owns those. A hit is served only when every template it references is still
# in the controller's sent set -- the same condition under which a fresh
# conversion would have produced a bundle with no template definitions --
# so the wire bytes are identical to a miss. Otherwise the batch is
# converted again, which re-sends what ASTRA-Sim evicted.
#
# LLMSS_TEMPLATE_BINDINGS_CACHE_BYTES bounds it (0 disables);
# LLMSS_VERIFY_TEMPLATE_BINDINGS_CACHE=1 re-converts every hit and fails
# on any difference.
# ----------------------------------------------------------------------
_BINDINGS_CACHE = OrderedDict()    # key -> CachedTemplateBindings
_BINDINGS_CACHE_BYTES = 0
_BINDINGS_CACHE_MAX_BYTES = int(
    os.environ.get("LLMSS_TEMPLATE_BINDINGS_CACHE_BYTES", 16 * 1024 * 1024))
_BINDINGS_CACHE_VERIFY = os.environ.get(
    "LLMSS_VERIFY_TEMPLATE_BINDINGS_CACHE") == "1"
_BINDINGS_CACHE_STATS = {"hit": 0, "relocated_hit": 0, "miss": 0,
                         "offset_mismatch": 0, "template_evicted": 0,
                         "verified": 0, "evictions": 0,
                         "relocatable_entries_stored": 0,
                         "fixed_entries_stored": 0}


class CachedTemplateBindings:
    """A cache hit: rank bindings to send against templates ASTRA holds."""

    __slots__ = ("template_ids", "bindings_json", "rank_count")

    def __init__(self, template_ids, bindings_json, rank_count):
        self.template_ids = template_ids
        self.bindings_json = bindings_json
        self.rank_count = rank_count


class _BindingsEntry:
    """Per-local-rank encoded bindings for one trace."""

    __slots__ = ("template_ids", "ranks", "npu_offset", "relocatable", "nbytes")

    def __init__(self, template_ids, ranks, npu_offset, relocatable):
        self.template_ids = template_ids
        self.ranks = ranks              # ((local_rank, value_json), ...)
        self.npu_offset = npu_offset
        self.relocatable = relocatable
        self.nbytes = (sum(len(v) + 16 for _, v in ranks)
                       + 64 * len(template_ids) + 128)

    def encode(self, npu_offset):
        body = ",".join('"%d":%s' % (npu_offset + local, value)
                        for local, value in self.ranks)
        return CachedTemplateBindings(self.template_ids, "{" + body + "}",
                                      len(self.ranks))


def _bindings_entry(bundle, npu_offset):
    bindings = bundle["bindings"]
    template_ids = tuple(sorted({b["template_id"] for b in bindings.values()}))
    ranks = tuple(
        (int(rank) - npu_offset, json.dumps(value, separators=(",", ":")))
        for rank, value in bindings.items()
    )
    relocatable = all(not value["nodes"] for value in bindings.values())
    return _BindingsEntry(template_ids, ranks, npu_offset, relocatable)


def _bindings_cache_store(key, entry):
    global _BINDINGS_CACHE_BYTES
    if entry.nbytes > _BINDINGS_CACHE_MAX_BYTES:
        return
    old = _BINDINGS_CACHE.pop(key, None)
    if old is not None:
        _BINDINGS_CACHE_BYTES -= old.nbytes
    _BINDINGS_CACHE[key] = entry
    _BINDINGS_CACHE_BYTES += entry.nbytes
    while _BINDINGS_CACHE_BYTES > _BINDINGS_CACHE_MAX_BYTES:
        _, evicted = _BINDINGS_CACHE.popitem(last=False)
        _BINDINGS_CACHE_BYTES -= evicted.nbytes
        _BINDINGS_CACHE_STATS["evictions"] += 1


def graph_cache_stats():
    """Hit/miss counts for the converted-graph and template-bindings caches."""
    return dict(_ET_CACHE_STATS, entries=len(_ET_CACHE), bytes=_ET_CACHE_BYTES,
                template_bindings=dict(_BINDINGS_CACHE_STATS,
                                       entries=len(_BINDINGS_CACHE),
                                       bytes=_BINDINGS_CACHE_BYTES,
                                       max_bytes=_BINDINGS_CACHE_MAX_BYTES))


def _rows_digest(trace):
    """Digest a synthesized trace without formatting it.

    Keyed on the same content the formatter would emit -- the header line and
    every field of every row, in order -- but joined with separators instead
    of padded into columns, which is far cheaper than the real format and
    just as discriminating. Cryptographic rather than ``hash()`` because a
    collision here would silently hand back the wrong graph.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(trace.header_line.encode())
    h.update(b"\n")
    h.update("\n".join("\t".join(row) for row in trace.rows).encode())
    return h.hexdigest()


def _et_names(workload_dir):
    return set(glob.glob(os.path.join(workload_dir, "llm.*.et")))


def _cache_store(key, paths):
    """Read the freshly converted files into the cache, evicting LRU.

    Only caches a trace that has been seen before; see _ET_SEEN.
    """
    global _ET_CACHE_BYTES
    if key not in _ET_SEEN:
        _ET_SEEN[key] = None
        while len(_ET_SEEN) > _ET_SEEN_MAX:
            _ET_SEEN.popitem(last=False)
        return
    entry = []
    total = 0
    for path in sorted(paths):
        with open(path, "rb") as f:
            blob = f.read()
        entry.append((os.path.basename(path), blob))
        total += len(blob)
    if not entry or total > _ET_CACHE_MAX_BYTES:
        _ET_CACHE_STATS["skipped"] += 1
        return
    _ET_CACHE[key] = entry
    _ET_CACHE_BYTES += total
    while _ET_CACHE_BYTES > _ET_CACHE_MAX_BYTES and len(_ET_CACHE) > 1:
        _, evicted = _ET_CACHE.popitem(last=False)
        _ET_CACHE_BYTES -= sum(len(b) for _, b in evicted)

# Chakra's LLMConverter, imported once and reused for the whole run.
#
# This used to be `python -m chakra.src.converter.converter LLM ...` in a
# fresh subprocess per batch. Measured on the sim container, that cost
# ~56 ms per call, of which ~52 ms was interpreter startup plus the
# protobuf import and ~1.3 ms was the actual conversion — which made
# graph generation 73-85% of simulator wall-clock across every config
# profiled (1 NPU, 8 NPUs, and MoE DP+EP alike). The subprocess path is
# gone; `git log` has it if it is ever needed to isolate a converter crash.
#
# Calling the converter in-process is safe because it keeps *all* of its
# mutable state on the instance (`next_node_id`, `next_comm_tag`,
# `comm_tag_dict`) — there is no module-level state to leak between
# batches, so a fresh LLMConverter per batch is equivalent to a fresh
# process. Verified byte-identical over 146 `.et` files.
#
# The import resolves to the *installed* chakra in site-packages, not the
# checked-out tree, so editing the tree requires `pip3 install .` to take
# effect — see AGENTS.md.
_LLMConverter = None


def _get_llm_converter():
    """Import LLMConverter on first use and cache the class.

    Deferred rather than imported at module scope so a broken or
    not-yet-installed chakra fails when a graph is first converted, rather
    than at simulator import time.
    """
    global _LLMConverter
    if _LLMConverter is None:
        from chakra.src.converter.llm_converter import LLMConverter
        _LLMConverter = LLMConverter
    return _LLMConverter


def generate_graph(batch, hardware, num_npus, node_id=0, instance_id=0, npu_offset=0, enable_local_offloading=False, event=False, workload_name=None, inputs_root=None, save_trace_text=False, template_mode='legacy', known_template_ids=None, *, trace):

    cwd = os.getcwd()
    if inputs_root is None:
        inputs_root = os.path.join(cwd, "inputs")

    # File-free modes hand the graph straight to ASTRA-Sim over the pipe.
    # They return the payload instead of writing llm.<npu>.et per rank,
    # so none of the directory, cache or trace-text handling below runs.
    #
    # This is a different axis from the _ET_CACHE: that reuses one whole
    # conversion when a later batch repeats a trace, while these split the
    # structure every rank shares from the little each rank differs by. The
    # cache saves across time, templates save across ranks, and at 512 NPUs
    # it is the per-rank writes that dominate.
    if template_mode != "legacy":
        converter = _get_llm_converter()(
            None, None, num_npus, npu_offset, enable_local_offloading,
        )
        if template_mode == "shared-template":
            if _BINDINGS_CACHE_MAX_BYTES <= 0:
                return converter.convert_rows_to_template_bundle(
                    trace.header_line, indexed_cols(trace.rows),
                    known_template_ids=known_template_ids,
                )
            key = (_rows_digest(trace), num_npus, enable_local_offloading)
            entry = _BINDINGS_CACHE.get(key)
            if entry is None:
                _BINDINGS_CACHE_STATS["miss"] += 1
            elif not (entry.relocatable or entry.npu_offset == npu_offset):
                _BINDINGS_CACHE_STATS["offset_mismatch"] += 1
            elif known_template_ids is None or not all(
                    t in known_template_ids for t in entry.template_ids):
                _BINDINGS_CACHE_STATS["template_evicted"] += 1
            else:
                _BINDINGS_CACHE.move_to_end(key)
                _BINDINGS_CACHE_STATS["hit"] += 1
                if entry.npu_offset != npu_offset:
                    _BINDINGS_CACHE_STATS["relocated_hit"] += 1
                cached = entry.encode(npu_offset)
                if _BINDINGS_CACHE_VERIFY:
                    bundle, _ = converter.convert_rows_to_template_bundle(
                        trace.header_line, indexed_cols(trace.rows),
                        known_template_ids=known_template_ids,
                    )
                    fresh = json.dumps(bundle["bindings"], separators=(",", ":"))
                    if bundle["templates"] or fresh != cached.bindings_json:
                        raise RuntimeError(
                            "template-bindings cache hit differs from a "
                            "fresh conversion")
                    _BINDINGS_CACHE_STATS["verified"] += 1
                return cached
            result = converter.convert_rows_to_template_bundle(
                trace.header_line, indexed_cols(trace.rows),
                known_template_ids=known_template_ids,
            )
            fresh_entry = _bindings_entry(result[0], npu_offset)
            _BINDINGS_CACHE_STATS["relocatable_entries_stored" if fresh_entry.relocatable
                                  else "fixed_entries_stored"] += 1
            _bindings_cache_store(key, fresh_entry)
            return result
        return converter.convert_rows_to_payloads(
            trace.header_line, indexed_cols(trace.rows),
        )

    if event:
        file_name = 'event_handler'
    else:
        file_name = f'{hardware}/{batch.model}/instance{instance_id}_batch{batch.batch_id}'

    # For DP groups, all instances write .et files to a shared workload folder
    output_name = workload_name if workload_name else file_name

    trace_path = input_path(inputs_root, "trace", f"{file_name}.txt")
    output_path = input_path(inputs_root, "workload", output_name, "llm")
    workload_dir = os.path.dirname(output_path)
    os.makedirs(workload_dir, exist_ok=True)

    # Every trace arrives as rows now, the event handler's included, so the
    # text is never an input -- only an artifact somebody asked for.
    if save_trace_text:
        write_trace(trace)

    cache_key = (_rows_digest(trace), num_npus, npu_offset,
                 enable_local_offloading)

    cached = _ET_CACHE.get(cache_key)
    if cached is not None:
        _ET_CACHE.move_to_end(cache_key)
        _ET_CACHE_STATS["hit"] += 1
        logger.debug("Graph cache hit for %s", trace_path,
                     extra={"node_id": node_id, "instance_id": instance_id})
        for name, blob in cached:
            with open(os.path.join(workload_dir, name), "wb") as g:
                g.write(blob)
        return
    _ET_CACHE_STATS["miss"] += 1

    before = _et_names(workload_dir)

    logger.debug("Converting graph: %s -> %s", trace_path, output_path,
                 extra={"node_id": node_id, "instance_id": instance_id})
    converter = _get_llm_converter()(
        trace_path, output_path, num_npus, npu_offset, enable_local_offloading,
    )
    converter.convert_rows(trace.header_line, indexed_cols(trace.rows))

    _cache_store(cache_key, _et_names(workload_dir) - before)
    return
