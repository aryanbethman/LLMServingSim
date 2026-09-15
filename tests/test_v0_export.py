"""Invariants for the v0 -> current profile conversion.

The conversion carries no fitting, so every emitted number must be
reproducible from the v0 source by hand. These tests assert exactly
that, in both directions:

  * the CSVs the exporter writes equal the v0 arithmetic, and
  * the numbers the *simulator's own loader* reads back out of those
    CSVs still equal the v0 arithmetic.

The second half matters more than the first. A bundle can be written
correctly and still be read wrong -- wrong column order, a unit slip, an
axis the loader brackets differently than the exporter assumed. Going
through `_load_perf_db` and the real `_lookup_*` functions is what
catches that.

No hardware and no vLLM: everything here reads committed CSVs.

Run:  python -m unittest tests.test_v0_export -v
"""

import csv
import os
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from profiler.v0 import export as v0x
from serving.core import trace_generator as tg


V0_ROOT = os.path.join(REPO_ROOT, "profiler", "v0", "perf_models")
SRC_TP4 = os.path.join(V0_ROOT, "H100", "meta-llama", "Llama-3.1-70B", "tp4")


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


class V0ExportTestCase(unittest.TestCase):
    """Shared fixture: convert the measured H100/70B TP=4 bundle once
    into a temp dir, then assert against it.
    """

    @classmethod
    def setUpClass(cls):
        if not os.path.isdir(SRC_TP4):
            raise unittest.SkipTest(f"v0 source bundle missing: {SRC_TP4}")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.layers = v0x.read_layers(os.path.join(SRC_TP4, "layers.csv"))
        cls.prefill = v0x.read_prefill(
            os.path.join(SRC_TP4, "predictions", "attn_prefill_predictions.csv"))
        cls.decode = v0x.read_decode(
            os.path.join(SRC_TP4, "predictions", "attn_decode_predictions.csv"))
        cls.variant_root, cls.counts = v0x.convert_bundle(
            src_dir=SRC_TP4, out_root=cls.tmp.name, hardware="H100",
            model="meta-llama/Llama-3.1-70B", variant="bf16", tp=4,
        )
        cls.tp_root = os.path.join(cls.variant_root, "tp4")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()


class TestDense(V0ExportTestCase):

    def test_fused_operators_equal_their_component_sums(self):
        """qkv_proj == q+k+v, gate_up_proj == gate+up, at every token count."""
        emitted = {}
        for row in _rows(os.path.join(self.tp_root, "dense.csv")):
            emitted[(row["layer"], int(row["tokens"]))] = float(row["time_us"])

        checked = 0
        for canonical, (sources, rule) in v0x.OPERATOR_MAP.items():
            for tokens in sorted(self.layers[sources[0]]):
                want_ns = sum(self.layers[s][tokens] for s in sources)
                if rule == "mean":
                    want_ns /= len(sources)
                got_us = emitted[(canonical, tokens)]
                self.assertAlmostEqual(
                    got_us, want_ns / 1000.0, places=9,
                    msg=f"{canonical} @ {tokens} tokens",
                )
                checked += 1
        self.assertGreater(checked, 10_000)

    def test_layernorm_preserves_the_per_block_total(self):
        """The trace emits `layernorm` twice per block, so 2 x the mean
        must equal v0's input_layernorm + post_layernorm."""
        for tokens in (1, 17, 512, 2048):
            emitted = v0x._combine(
                self.layers, ["input_layernorm", "post_layernorm"], "mean", tokens)
            want = (self.layers["input_layernorm"][tokens]
                    + self.layers["post_layernorm"][tokens])
            self.assertAlmostEqual(2 * emitted, want, places=6)

    def test_no_v0_operator_is_silently_dropped(self):
        """Every dense layer v0 profiled lands somewhere in the new bundle."""
        mapped = {s for srcs, _ in v0x.OPERATOR_MAP.values() for s in srcs}
        mapped |= {s for srcs, _ in v0x.PER_SEQUENCE_MAP.values() for s in srcs}
        # `attn` is the one v0 name that moves into attention.csv instead.
        unmapped = set(self.layers) - mapped - {"attn"}
        self.assertEqual(unmapped, set(), f"v0 operators with no home: {unmapped}")


class TestAttention(V0ExportTestCase):

    def test_pure_prefill_equals_the_v0_prefill_table(self):
        """n_decode = 0 rows must be the prefill table verbatim."""
        checked = 0
        for row in _rows(os.path.join(self.tp_root, "attention.csv")):
            if int(row["n_decode"]) != 0:
                continue
            pc, kvp = int(row["prefill_chunk"]), int(row["kv_prefill"])
            self.assertAlmostEqual(
                float(row["time_us"]), self.prefill[(kvp, pc)] / 1000.0,
                places=9, msg=f"pure prefill pc={pc} kv={kvp}")
            checked += 1
        self.assertGreater(checked, 0)

    def test_pure_decode_equals_the_v0_decode_table(self):
        """prefill_chunk = 0 rows must be the decode table verbatim."""
        checked = 0
        for row in _rows(os.path.join(self.tp_root, "attention.csv")):
            if int(row["prefill_chunk"]) != 0:
                continue
            nd, kvd = int(row["n_decode"]), int(row["kv_decode"])
            self.assertAlmostEqual(
                float(row["time_us"]), self.decode[(nd, kvd)] / 1000.0,
                places=9, msg=f"pure decode n={nd} kv={kvd}")
            checked += 1
        self.assertGreater(checked, 0)

    def test_mixed_equals_the_v0_additive_composition(self):
        """Every mixed row is prefill(kv_p, chunk) + decode(n, kv_d), which
        is what the v0 simulator computed at run time."""
        checked = 0
        for row in _rows(os.path.join(self.tp_root, "attention.csv")):
            pc, kvp = int(row["prefill_chunk"]), int(row["kv_prefill"])
            nd, kvd = int(row["n_decode"]), int(row["kv_decode"])
            if pc == 0 or nd == 0:
                continue
            want = self.prefill[(kvp, pc)] + self.decode[(nd, kvd)]
            self.assertAlmostEqual(float(row["time_us"]), want / 1000.0,
                                   places=9, msg=f"mixed {pc}/{kvp}/{nd}/{kvd}")
            checked += 1
        self.assertGreater(checked, 1000)

    def test_grid_is_a_full_cross_product(self):
        """The loader brackets each axis assuming every (chunk, n_decode)
        slice carries the same kv grid. A hole would interpolate wrong
        rather than raise, so assert the product is complete."""
        rows = _rows(os.path.join(self.tp_root, "attention.csv"))
        keys = {(int(r["prefill_chunk"]), int(r["kv_prefill"]),
                 int(r["n_decode"]), int(r["kv_decode"])) for r in rows}
        pcs = [0] + v0x.DEFAULT_PREFILL_CHUNKS
        nds = [0] + v0x.DEFAULT_N_DECODE
        expected = {(pc, kvp, nd, kvd)
                    for pc in pcs for kvp in v0x.DEFAULT_KV
                    for nd in nds for kvd in v0x.DEFAULT_KV
                    if not (pc == 0 and nd == 0)}
        self.assertEqual(keys, expected)


class TestLoaderRoundTrip(V0ExportTestCase):
    """Read the exported bundle back through the simulator's own loader."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # The loader resolves bundles against a path relative to the
        # cwd the simulator chdirs into at run time (astra-sim/). Point
        # it at this test's own temp export instead, so the round trip
        # covers the bytes these tests just wrote rather than whatever
        # happens to be committed under profiler/perf/.
        cls._saved_root = tg._PROFILER_ROOT_REL
        tg._PROFILER_ROOT_REL = os.path.join(cls.tmp.name, "_root")
        os.makedirs(tg._PROFILER_ROOT_REL, exist_ok=True)
        perf_link = os.path.join(tg._PROFILER_ROOT_REL, "perf")
        if not os.path.exists(perf_link):
            os.symlink(cls.tmp.name, perf_link)
        tg._perf_db_cache.clear()
        cls.db = tg._load_perf_db(
            "H100", "meta-llama/Llama-3.1-70B", "bf16",
            tp_needed={4}, model_type="llama",
        )

    @classmethod
    def tearDownClass(cls):
        tg._PROFILER_ROOT_REL = cls._saved_root
        tg._perf_db_cache.clear()
        super().tearDownClass()

    def test_loader_dense_matches_v0_component_sums(self):
        """_lookup_dense returns ns; compare against the v0 sum directly."""
        for tokens in (1, 8, 64, 512, 2048):
            got = tg._lookup_dense(self.db, "qkv_proj", 4, tokens)
            want = sum(self.layers[s][tokens] for s in ("q_proj", "k_proj", "v_proj"))
            self.assertEqual(got, want, f"qkv_proj @ {tokens}")

            got = tg._lookup_dense(self.db, "gate_up_proj", 4, tokens)
            want = self.layers["gate_proj"][tokens] + self.layers["up_proj"][tokens]
            self.assertEqual(got, want, f"gate_up_proj @ {tokens}")

    def test_loader_attention_matches_v0_on_grid_points(self):
        """On an exact grid point the loader must not interpolate at all."""
        cases = [
            (0, 0, 1, 64), (0, 0, 32, 1024), (0, 0, 256, 2048),
            (256, 0, 0, 0), (1024, 512, 0, 0), (2048, 2048, 0, 0),
            (512, 256, 16, 512), (2048, 1024, 128, 2048),
        ]
        for pc, kvp, nd, kvd in cases:
            got = tg._lookup_attention(self.db, 4, pc, kvp, nd, kvd)
            want = v0x.compose_attention_ns(self.prefill, self.decode,
                                            pc, kvp, nd, kvd)
            self.assertAlmostEqual(got, want, delta=1.0,
                                   msg=f"attention {pc}/{kvp}/{nd}/{kvd}")

    def test_sampler_is_present_and_declared_unmeasured(self):
        """The head sequence requires a sampler row; it must exist and the
        meta must say it was asserted rather than measured."""
        got = tg._lookup_per_sequence(self.db, "sampler", 4, 8)
        self.assertGreaterEqual(got, 1)
        with open(os.path.join(self.variant_root, "meta.yaml")) as f:
            meta_text = f.read()
        self.assertIn("sampler_note", meta_text)
        self.assertIn("not measured", meta_text)

    def test_skew_correction_is_off_for_converted_bundles(self):
        """v0 never swept heterogeneous decode batches, so alpha must
        resolve to 0 -- no correction invented from nothing."""
        alpha = tg._skew_alpha(self.db, 4, pc=512, n=32,
                               skew_rate=0.5, kv_big=4096, kp=0)
        self.assertEqual(alpha, 0.0)


class TestProvenance(V0ExportTestCase):

    def test_meta_records_source_digests(self):
        with open(os.path.join(self.variant_root, "meta.yaml")) as f:
            meta_text = f.read()
        for name in ("layers.csv", "attn_prefill_predictions.csv",
                     "attn_decode_predictions.csv"):
            self.assertIn(name, meta_text)
        self.assertIn("v0_sources", meta_text)
        self.assertIn("converted from a v0 profile bundle", meta_text)

    def test_digests_match_the_source_files(self):
        with open(os.path.join(self.variant_root, "meta.yaml")) as f:
            meta_text = f.read()
        want = v0x._sha256(os.path.join(SRC_TP4, "layers.csv"))
        self.assertIn(want, meta_text)


if __name__ == "__main__":
    unittest.main()
