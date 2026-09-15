"""Converted-profile timing compatibility; native 4D profiles stay native."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml
from profiler.v0 import export as ex
from serving.core import trace_generator as tg


class V0AttentionCompatibilityTest(unittest.TestCase):
    def test_complete_source_tables_preserve_projected_profile_lookups(self):
        root = Path(__file__).resolve().parents[1]
        bundles = [
            ("Llama-3.1-70B", "tp8", "bf16"),
            ("Llama-3.1-405B", "tp8", "bf16"),
            ("Llama-3.1-405B", "variants/low", "bf16-low"),
            ("Llama-3.1-405B", "variants/high", "bf16-high"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            for model, source_suffix, variant in bundles:
                src = root / "profiler/v0/perf_models/H100/meta-llama" / model / source_suffix
                # Variant layouts are described by their provenance, not assumed.
                existing_meta = root / "profiler/perf/H100/meta-llama" / model / variant / "meta.yaml"
                if existing_meta.exists():
                    meta = yaml.safe_load(existing_meta.read_text())
                    src = Path(meta["v0_sources"]["tp8"]["source_dir"])
                    if not src.is_absolute():
                        src = root / src
                self.assertTrue(src.is_dir(), str(src))
                dest, _ = ex.convert_bundle(str(src), temporary, "H100",
                                             "meta-llama/" + model, variant, 8)
                db = {"root": dest, "meta": yaml.safe_load(
                    Path(dest, "meta.yaml").read_text())}
                prefill = ex.read_prefill(src / "predictions/attn_prefill_predictions.csv")
                decode = ex.read_decode(src / "predictions/attn_decode_predictions.csv")
                for pc, kp, nd, kd in [
                    (1536, 0, 0, 0), (4096, 0, 0, 0),
                    (8192, 0, 0, 0), (16384, 0, 0, 0),
                    (96, 192, 3, 192), (0, 0, 3, 192),
                    (17, 1, 3, 65),
                ]:
                    qpc = ((pc + 31) // 32) * 32
                    qkp = ((kp + 63) // 64) * 64
                    qkd = ((kd + 63) // 64) * 64
                    expected = ex.compose_attention_ns(prefill, decode, qpc, qkp, nd, qkd)
                    with self.subTest(model=model, variant=variant, keys=(pc, kp, nd, kd)):
                        self.assertEqual(tg._lookup_attention(db, 8, pc, kp, nd, kd), expected)
                # The bundle remains usable without its v0 source directory.
                self.assertTrue(Path(dest, "tp8/attention_prefill_v0.csv").is_file())

    def test_bilinear_and_edge_extrapolation_do_not_mutate_source(self):
        surface = {"xs": [0, 10], "ys": [32, 64],
                   "rows": {0: {32: 100, 64: 200}, 10: {32: 200, 64: 300}}}
        self.assertEqual(tg._v0_surface_lookup(surface, 5, 48), 200)
        self.assertEqual(tg._v0_surface_lookup(surface, 20, 96), 500)
        self.assertEqual(tg._v0_surface_lookup(surface, -10, 0), 1)
        self.assertEqual(sum(map(len, surface["rows"].values())), 4)

    def test_prefill_timing_uses_legacy_l2_key_without_changing_sizes(self):
        ctx = SimpleNamespace(perf_db={"meta": {"v0_export": {
            "attention_lookup": "v0-additive"}}}, tp_size=8)
        bctx = SimpleNamespace(batch=SimpleNamespace(q_list=[96, 128, 1, 1, 1]),
                               prefill_chunk=224, kv_prefill=100,
                               n_decode=3, kv_decode_mean=65)
        with patch.object(tg, "_lookup_attention", return_value=123) as lookup:
            self.assertEqual(tg._batch_attention_latency(ctx, bctx), 123)
            lookup.assert_called_once_with(ctx.perf_db, 8, 160, 100, 3, 65)
        self.assertEqual(bctx.prefill_chunk, 224)

    def test_native_profiles_do_not_use_legacy_tables(self):
        db = {"meta": {}}
        with patch.object(tg, "_tp_tables", return_value={}) as native, \
                patch.object(tg, "_lookup_v0_attention") as legacy:
            with self.assertRaises(KeyError):
                tg._lookup_attention(db, 1, 32, 0, 0, 0)
            native.assert_called_once_with(db, 1)
            legacy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
