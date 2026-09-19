# File-Free Wall-Time Optimization Progress

## Objective

Make the file-free shared-template simulator faster than the matched upstream file-mode baseline (4:07.34) without giving back correctness, filesystem, or host-memory advantages.

## Non-negotiable gates

- 750-request CSV byte-identical to the matched legacy control.
- Simulated clock exactly 89,678,987,869 ns.
- No dynamic rank ET or text-trace files.
- Keep filesystem output and transient result storage at the file-free level.
- Keep process-tree RSS at or below the 217 MiB matched legacy result.

## Matched 16-NPU ShareGPT-750 results

| Configuration | Wall time | Tree RSS | Key result |
| --- | ---: | ---: | --- |
| New legacy / file mode | 4:07.34 | 217 MiB | Baseline; 5,632,088 filesystem output blocks |
| New shared template / bounded-128 | 7:20.97 | 201 MiB | Exact; 5,056 blocks, 80.8 KB result peak |
| New shared template / unbounded | 7:10.60 | 1.45 GiB | Faster than bounded but unacceptable cache growth |
| Raw in-memory ET / normal feeder | 9:57.42 | 152 MiB | Exact and file-free; not the speed fix |
| Shared template / normalization fast path | 6:27.89 | 203 MiB | Exact and file-free; 53.08 s (12.0%) faster than bounded-128 |

The fast-path run emitted zero dynamic graph files, 4,952 filesystem output blocks, a 79,181-byte result-directory peak, and the exact legacy CSV and simulated clock.

## Profile evidence

The sampled bounded shared-template profile took 632.5 s with profiling overhead; use it for attribution, not absolute runtime.

- generate_graph: 358.1 s (56.6%).
- Controller.read_completion / ASTRA wait: 243.9 s (38.6%).
- Controller send: 23.6 s.
- Template-capture write_message: 166.0 s.
- _normalise_node: 128.6 s.

The first fix bypasses protobuf copying and empty-overlay construction for non-communication nodes. Focused execution-template equivalence tests pass.

## Next implementation sequence

1. **Rank-leader structural capture.** Canonicalize and serialize common TP structure once; derive only sparse communication/name overlays for remaining ranks. Keep a safe fallback for non-equivalent structures, including PP stages.
2. **Exact tests first.** Require fused leader/overlay bundles to match payload-derived bundles for TP and TP×PP offsets, then run the 750-request comparator.
3. **ASTRA instantiation attribution.** Add aggregate timings for template decode/cache, rank binding, protobuf cloning, and feeder graph construction.
4. **Compact mutable rank state.** If cloning/allocation dominates, retain immutable descriptors and allocate only mutable dependency state in contiguous resettable storage.
5. **Release gate.** Require wall time below 4:07.34 plus every non-negotiable gate; repeat cold and warm runs before claiming a win.

## Decisions

- Do not use unbounded cache as the performance result.
- Do not restore legacy files as a speed fix.
- Raw in-memory ET is a diagnostic control, not the final architecture.
- The final claim must be filesystem-optimized and faster than new legacy file mode.

## Instrumented ASTRA result (2026-09-19)

The bounded file-free shared-template control completed successfully in 6:30.51 with the exact 750-row legacy CSV and simulated clock. Filesystem output was 4,840 blocks.

| ASTRA stage | Total |
| --- | ---: |
| Template decode/cache insertion | 3.52 s |
| Rank-binding parse | 5.29 s |
| Direct feeder construction/cloning | 63.27 s across 63,744 initializations |

Decision: ASTRA decode and binding parsing are not the wall-time bottleneck. Even eliminating all feeder construction would leave the run above the 4:07.34 target. The next implementation is guarded rank-leader structural capture: canonicalize TP-shared structure once, produce follower overlays only, and retain full capture as the structural-mismatch fallback. This implementation has not started yet.

## Target met (2026-09-19, afternoon)

Shared-template mode now beats the 4:07.34 legacy baseline on the matched 16-NPU ShareGPT-750 workload. Three consecutive runs, each alone on the host:

| Run | Wall | CSV | Clock | Tree RSS | FS output blocks |
| --- | ---: | --- | --- | ---: | ---: |
| shared r1 | 3:59.26 | exact | exact | 208.8 MiB | 4,872 |
| shared r2 | 3:59.46 | exact | exact | 208.5 MiB | 4,584 |
| shared r3 | 4:00.03 | exact | exact | 208.7 MiB | 4,560 |
| legacy, same new binary | 4:08.33 | exact | exact | 213.8 MiB | 5,632,080 |

The ASTRA changes below also run in legacy mode. Legacy on the new binary is unchanged (4:08.33 vs 4:07.34), so the comparison is fair. Results: `~/hipc-results/new16npu-gate-*-20260919`.

### What the time actually was

`/usr/bin/time` does not count ASTRA's CPU in shared mode (it counts it in legacy), so its user time is not comparable across modes. Per-process CPU read from `/proc` (`~/hipc-results/cpu_split.sh`):

| Configuration | Python CPU | ASTRA CPU | Wall |
| --- | ---: | ---: | ---: |
| legacy | 130 s | 125 s | 4:07–4:19 |
| shared + bindings cache | 97 s | 214 s | 4:58 |
| shared + cache + getline + shared nodes | 95 s | 142 s | 3:58 |

The processes take turns over the pipe with no measurable idle time, so wall time ≈ Python CPU + ASTRA CPU.

### Changes

1. **Template-bindings cache** (`serving/core/graph_generator.py`, `controller.write_template_bindings`, dispatch in `serving/__main__.py`). The key is (trace-row digest, num_npus, local_offloading), with `npu_offset` deliberately left out. A hit sends the cached rank bindings, relocated to the requesting instance's NPU offset, with no conversion. It is served only when every referenced template is still in the controller's sent set, so the wire bytes equal a fresh conversion's (359,302,038 either way). Bindings with rank-specific overlays (send/recv, PP) are not relocatable and hit only at their original offset. 8,347 hits out of 15,936 batches, 5,611 of them relocated across instances. Size-bounded LRU of pre-encoded JSON, 8.2 MB at the end of the run. `LLMSS_TEMPLATE_BINDINGS_CACHE_BYTES` sets the bound (default 16 MiB; 0 disables); `LLMSS_VERIFY_TEMPLATE_BINDINGS_CACHE=1` re-converts every hit and fails on any difference. Effect: 6:28 → 4:58.
   - Keying on the offset, as `_ET_CACHE` does, gives only ~30% repeats (11,156 distinct keys), because the four identical TP4 instances each count separately. Legacy's `_ET_CACHE` hits only 7% here (1,055 / 15,937).
2. **Bulk stdin read in ASTRA** (`read_command` in both analytical `main.cc` files). `std::getline` on the stdio-synced `std::cin` read each ~11 KB bundle one character at a time (getc + lock + ungetc), about 19% of ASTRA CPU. POSIX `getline(3)` on `stdin` reads the same buffer in bulk. Effect: 4:58 → 4:46.
3. **Shared immutable template nodes in the direct feeder** (chakra `et_feeder.cpp`, `et_feeder_node.{h,cpp}`). The feeder used to copy every template node per rank and batch (`make_shared` + `CopyFrom`) only because `freeChildrenNodes` erased entries from the proto's `data_deps`. Consumed dependencies are now tracked in `ETFeederNode` (a counter plus a 64-bit consumed mask, spilling to a vector above 64 deps) with the same "erase first matching entry" semantics. Template nodes without overlays are shared via `const_pointer_cast`; nodes with overlays are still copied. Effect: 4:46 → 3:58 (ASTRA 214 → 142 s CPU).

### Verification

- `validate.sh --clocks-only`: 58/58 legacy and 46/46 shared-template scenarios, the shared runs with cache verification on.
- 750-request run with cache verification: all 8,347 hits matched a fresh conversion; CSV and clock exact.
- Unit tests: 50/51. The one failure, `test_controller_encodes_template_bundle_and_tracks_cached_ids`, predates this work: it doesn't expect the uncommitted `astra_*_ns` profiling fields in `controller.py`.

### Gates not cleanly met / caveats

- **FDs:** the tree steady state is 21, the same as the fast-path run before these changes. One startup sample reaches 25 at 1-second sampling. The ≤18 figure came from the original bounded run under a different launcher. Not a regression from this work, but not ≤18 either.
- **Result-directory peak:** 92 KB in r1/r3. The extra over 80.8 KB is the monitor's own `host_resources.csv` at 1-second sampling; r2 sampled right after the 2.1 MB `requests.csv` was written. The simulator's in-run output is `stdout.log` at ~80.6 KB, as before.
- **Cold-cache run:** not done. Dropping the page cache needs root. All reported runs are warm.
- **Margin:** about 8 s (3.2%). Run-to-run spread across the three runs was under 1 s.
- **Build hygiene:** the ASTRA binary links `libprotobuf.so.17` and `libstdc++` from the old fork's env (`/home/marvell/LLMServingSim/env/lib`).
- The ASTRA binaries were rebuilt in place. The pre-change binaries are in `~/hipc-results/astra-bin-backup-20260919/`. Nothing is committed.

### Profiling without perf

`perf_event_paranoid=4`, `ptrace_scope=1` and no sudo mean perf and gdb attach are unavailable. `~/hipc-results/libsampler.so` (source `sampler.c`) is an LD_PRELOAD SIGPROF stack sampler. Run with `SAMPLER_DIR=... SAMPLER_MATCH=AstraSim_Analytical_Congestion_Unaware` and read the output with `symbolize.py`. Note that `AnalyticalAstra` is a symlink to that binary. The sampler's proportions are usable, but it undercounts absolute CPU.

### Remaining headroom

- ASTRA is still 17 s of CPU above legacy: `issue_dep_free_nodes` and general malloc pressure (`_int_malloc`/`malloc_consolidate`).
- Every NPU that asks for a batch receives the full 4-rank bundle and ASTRA parses all four bindings: 31,872 bundles for 15,936 batches.
- The 7,323 cache misses still convert all four TP ranks; node IDs are offset per rank, so rank-leader capture needs an ID-rebase in ASTRA.

## Release gate, toolchain and scaling (2026-09-19, evening)

### Toolchain

ASTRA-Sim now builds against `env/cpp`, a repo-owned conda prefix holding byte-identical protobuf 3.6.1 and libstdc++ builds. It no longer depends on the fork's `~/LLMServingSim/env`; MIGRATION.md has the recipe. The rebuilt binary passes 58/58 legacy and 46/46 shared-template clock scenarios, and 56/56 unit tests pass. The Python side already used this repo's venv; its installed `chakra` matches the checked-out Python sources.

### 16-NPU gate on the rebuilt binary

| Run | Wall | CSV / clock | Tree RSS |
| --- | ---: | --- | ---: |
| shared, warm r1–r3 | 3:59.20 / 3:58.64 / 3:59.70 | exact | 208.8–209.5 MiB |
| legacy, warm | 4:07.08 | exact | 214.1 MiB |
| shared, cold | 3:58.45 | exact | 208.2 MiB |
| legacy, cold | 4:07.78 | exact | 212.9 MiB |

"Cold" means the simulator's files (repo, both envs, binary, profiles, dataset, Python interpreter and stdlib; 42,251 files, 1.98 GiB) were evicted with `posix_fadvise(DONTNEED)` before each run (`~/hipc-results/evict.py`). A full page-cache drop needs root. Startup I/O makes no measurable difference in either mode.

### FD gate: replace the tree total

The ≤18 figure summed the launcher wrappers (`/usr/bin/time`, `timeout`) with the simulator. Measured per process, mid-run:

| | Python | ASTRA-Sim |
| --- | ---: | ---: |
| shared, 16 or 72 NPUs | 7 | 6 |
| legacy, 72 NPUs | 7 | 78 (one open `.et` per NPU) |

Python's fd 3 is `stderr_time.log`, inherited from `/usr/bin/time`. Proposed gate: the simulator processes hold a constant number of FDs independent of NPU count (13 in shared mode). Legacy grows by one per NPU.

### Scaling (ShareGPT-750, upstream defaults, the two modes run concurrently on the host)

| Workload | Mode | Wall | Python CPU | ASTRA CPU | Tree RSS | FS output blocks | CSV / clock |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 70B TP8, 72 NPUs | legacy | 31:04 | 728 s | 1,122 s | 366 MiB | 33,068,968 | match |
| 70B TP8, 72 NPUs | shared | **23:02** | 209 s | 1,165 s | 349 MiB | 5,424 | match |
| 405B TP8×PP2 | legacy | **4:10** | 85 s | 164 s | 216 MiB | 5,079,976 | match |
| 405B TP8×PP2 | shared | 5:10 | 108 s | 201 s | 215 MiB | 4,592 | match |
| 405B TP8×PP2 | shared, 2,048-template cache | 5:39 | 106 s | 233 s | 769 MiB | 4,960 | match |

Before today's work, shared mode on these workloads took 42:38 (72-NPU TP8) and 7:08 (405B).

- **72-NPU TP8: shared is 26% faster than legacy.** Nine identical TP8 instances give an 85% bindings-cache hit rate (39,158 of 45,911 batches, 35,101 of them relocated across instances), and Python CPU falls 3.5×. ASTRA simulation dominates and is about equal in both modes.
- **405B TP8×PP2: shared is still 24% slower than legacy.** There is one instance, and its PP send/recv overlays make bindings non-relocatable, so the cache hits only 40%. Its templates are mostly unique (12,064 definitions even with a 2,048-entry ASTRA cache), so a larger ASTRA cache costs memory and helps nothing. ASTRA stage timers: feeder construction 40 s, binding parse 14 s (every NPU parses all 16 ranks' bindings), template decode 9 s.

### Open items for the next round

1. Send each NPU only its own rank's binding. The 405B binding parse is 14 s, and 72-NPU is 93,768 bundles for 45,911 batches.
2. Convert one rank per PP stage on cache misses (rank-leader capture with an ASTRA-side node-ID rebase). Per-rank node IDs are what make 405B templates rank-unique.
3. ASTRA allocator pressure in feeder construction: 127 s at 72 NPUs, 40 s at 405B.

## Round 2 (2026-09-19/20)

### Output equivalence, re-checked

- 72-NPU TP8 and 405B TP8×PP2 `requests.csv` in both modes are byte-identical to the 2026-09-15 runs, which predate all of this work (`de762118…`, `9107f1ad…`; clocks 85,066,556,121 and 160,375,250,899 ns).
- Regenerating all four upstream `bench/examples` gives byte-identical `sim.csv` and `validation/summary.txt`. The `sim.log` files differ only in run ID, install path and wall time.

### Kept: decode only the bindings an NPU will build (astra-sim)

Each batch reaches ASTRA-Sim twice: once at the instance's start NPU, which builds itself and its managed ranks, and once at the end NPU, which builds only itself. Both times ASTRA parsed every rank's binding. It now decodes only the ranks it will instantiate. Skipped bindings still refresh their template's LRU position exactly as a parse did, so evictions, releases and wire bytes are unchanged (405B: 13,712 definitions, 13,584 releases and 717,166,715 bytes both before and after). `decode_base64` also uses a lookup table instead of searching the alphabet for each character.

| Run (solo) | Wall | Python CPU | ASTRA CPU | ASTRA binding parse / template decode |
| --- | ---: | ---: | ---: | ---: |
| 405B legacy | 3:45 | 79 s | 147 s | — |
| 405B shared | 4:25 | 101 s | 165 s | 6.1 s / 4.2 s |
| 16-NPU shared ×2 | 3:57.66 / 3:57.34 | | | |

The 405B figures before this change came from four concurrent runs (legacy 4:10, shared 5:10), so they are not directly comparable. The solo gap is now 40 s, about half Python and half ASTRA.

### Tried and reverted: rank-invariant templates

The converter numbers nodes with one counter across all ranks, so every rank is its own template: 16 per 405B bundle, 4 per 16-NPU bundle. I built exact-by-construction rebasing. Templates stored rank-relative ids, the binding carried `id_base`, ASTRA's feeder mapped ids back, and every rank's ET rebuilt from the bundle matched direct conversion byte for byte. Clocks and CSVs matched.

It worked as transport: 405B definitions fell 13,712 → 1,570 and wire bytes 717 MB → 135 MB; 16-NPU fell 5,416 → 1,067 and 359 MB → 101 MB. ASTRA binding parse and decode dropped to about 1.8 s. But it was slower overall:

| | Before | Rebase, rewriting in the capture stream | Rebase, assigned in the converter |
| --- | ---: | ---: | ---: |
| 405B shared wall / Python CPU | 4:25 / 101 s | 5:06 / 149 s | 4:43 / 122 s |
| 16-NPU shared wall | 3:57–3:59 | — | 4:02 |

Python still builds and normalises every rank. Exact rebasing needs per-node bookkeeping: either rewriting and restoring ids and deps, or tracking which rank section created each node so cross-section use can be rejected. At about 78 M node creations per 405B run, that costs more than the ASTRA transport it saves. Rebasing only pays off together with converting one rank per stage, which would mean re-implementing the converter's per-rank side effects (the shared comm-tag counter) outside the converter. That is not worth the exactness risk for the remaining gap.

### Where 405B's remaining 40 s is

- **Python, +22 s:** per-rank node generation plus normalisation and hashing on the 60% of batches that miss the bindings cache. The misses are unique traces, and the single instance's PP overlays prevent relocation.
- **ASTRA, +18 s:** mostly feeder construction (21 s) and allocator pressure. The same work costs legacy about half, because file-mode nodes are parsed straight into fresh objects.

## Round 3: closing the 405B gap (2026-09-20)

All pairs below ran both modes side by side under the same host load. The overnight 256-NPU runs were also active, so absolute walls read a few percent higher than solo runs.

### The ASTRA side was allocator behaviour, not feeder work

With the lazy binding decode in place, a paired ASTRA profile (`~/hipc-results/prof405-*-20260920`) showed feeder construction costing about the same in both modes (legacy `add_workload` 8.8%, shared `add_templates` 7.5%). The shared-only cost was `malloc`. Removing one allocation site (a fresh command string per read, then a `substr` copy of each bundle before parsing) only moved the time to other sites; total ASTRA CPU stayed about 190–200 s.

Cause: shared mode interleaves large allocations (each ~80 KB bundle line, its JSON parse, decoded templates) with millions of small node allocations. With glibc fastbins populated, every large request runs `malloc_consolidate` over all of them.

| 405B, same load | ASTRA CPU, default | `GLIBC_TUNABLES=glibc.malloc.mxfast=0` | + trim/mmap tuning |
| --- | ---: | ---: | ---: |
| shared | 200 s | 150 s | 150 s |
| legacy | 164 s | 162 s | 158 s |

Now built in as `mallopt(M_MXFAST, 0)` at the top of both analytical `main()`s (astra-sim `b87c9e2`), together with the reused command buffer and in-place JSON parse. Shared-mode ASTRA-Sim is now **cheaper than legacy's** on 405B (146–151 s vs 154–160 s).

One legacy run launched together with three others exited at startup ("ASTRA closed stdout before a Waiting prompt", 0.4 s). An identical re-run succeeded. It was not reproduced; this may be the same family as MIGRATION.md's unexplained 2026-09-13 hang.

### Python side

- `_framed`: frame all node payloads in one bytearray, with a one-byte varint fast path. Checked byte-identical to the old per-node `_frame` join on random payloads across every varint boundary. `build` fell from 15.8 s to 7.4 s (profiled).
- Collectives (`COMM_COLL_*`) used to take the full copy-and-strip normalisation. They now serialise directly whenever they carry neither a send/recv name nor a rank attribute, the exact case where that path changes nothing. A new unit test checks both fast paths against full normalisation for every node of TP, TP×PP (2 and 3 stages), decode and prefill conversions.
- Tried and dropped: inlining the non-COMM case into `write_message` with sparse overlays. Output was identical but there was no measurable effect (Python 93 s either way).

### Result

| 405B TP8×PP2, same load | legacy | shared |
| --- | ---: | ---: |
| pair 1 | 3:57.94 | 3:57.28 |
| pair 2 | 4:04.98 | 4:05.94 |
| pair 3 | 4:05.45 | 4:04.77 |
| pair 4 | 4:05.97 | 4:10.39 |
| Python / ASTRA CPU (typical) | 85 s / 159 s | 93 s / 151 s |

**405B is now at parity**: within ±2% across four pairs, compared with 40–60 s behind before this round. CSV and clock were identical in every run. Under the same load, 16-NPU shared ran 4:10.57 vs legacy 4:36.36.

What remains on 405B is Python's per-node work across 16 ranks for the ~60% of batches that are unique traces (+8 s CPU), offset by ASTRA-Sim now being 8 s cheaper.

### 256 NPUs (64 × TP4), overnight, frozen snapshots

- **Shared-template, snapshot `48e7955` (before this round's allocator fix): 49:15.** Exit 0. Python 251 s, ASTRA-Sim 2,697 s CPU, 563 MiB tree RSS, 9,224 FS output blocks, minimum disk free 46.5 GiB. The old fork took 1:56:16 at this scale.
- Legacy file mode from the same snapshot runs afterwards, with a disk watchdog (stop below 12 GiB free). Legacy keeps every per-NPU ET until exit, about 16 GB at 72 NPUs.
- Then shared-template from snapshot `07dd7b1` (this round's code), queued automatically.

Results: `~/hipc-results/scale256-*-20260920/summary.txt`; driver: `~/hipc-results/overnight.sh`, `overnight2.sh`, `queue2.sh`.
