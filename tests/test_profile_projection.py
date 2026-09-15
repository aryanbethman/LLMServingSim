import json
import pickle
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from profiler.v0.profile_projection import (
    ROOT,
    load_model,
    memory_feasibility,
    physical_limit_check,
    project_405b,
    project_layers_tp8,
    project_tp8,
    read_rows,
    validate_profile,
)

BASE = ROOT / "H100" / "meta-llama"
TABLES = (
    "layers.csv",
    "predictions/attn_prefill_predictions.csv",
    "predictions/attn_decode_predictions.csv",
)


class ProfileProjectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tp8 = Path(cls._tmp.name) / "tp8"
        cls.p405 = Path(cls._tmp.name) / "405b"
        project_tp8(Namespace(hardware="H100", model="meta-llama/Llama-3.1-70B",
                              output=str(cls.tp8), method_version="v1"))
        project_405b(Namespace(output=str(cls.p405), method_version="v1",
                               runtime_reserve_gb=8.0, comm_reserve_gb=1.0))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_405b_regenerates_the_committed_tables_byte_for_byte(self):
        committed = BASE / "Llama-3.1-405B" / "tp8"
        for arm in (".", "variants/low", "variants/high"):
            for name in TABLES:
                self.assertEqual((self.p405 / arm / name).read_bytes(),
                                 (committed / arm / name).read_bytes(), f"{arm} {name}")
        self.assertEqual((self.p405 / "memory_feasibility.json").read_bytes(),
                         (committed / "memory_feasibility.json").read_bytes())

    def test_tp8_layers_and_decode_match_the_committed_profile(self):
        # The committed tp8 is the fork's frozen v0 profile. Prefill attention
        # above 2,048 tokens differs on purpose; see PROFILE_PROJECTIONS.md.
        committed = BASE / "Llama-3.1-70B" / "tp8"
        for name in ("layers.csv", "predictions/attn_decode_predictions.csv"):
            self.assertEqual((self.tp8 / name).read_bytes(),
                             (committed / name).read_bytes(), name)

    def test_tp4_backtest_meets_documented_gate(self):
        model = BASE / "Llama-3.1-70B"
        _projected, backtest, _bias = project_layers_tp8(
            read_rows(model / "tp1" / "layers.csv"),
            read_rows(model / "tp2" / "layers.csv"),
            read_rows(model / "tp4" / "layers.csv"),
        )
        errors = sorted(float(row["absolute_error_pct"]) for row in backtest)
        self.assertLessEqual(errors[len(errors) // 2], 10.0)
        self.assertLessEqual(errors[round((len(errors) - 1) * 0.90)], 25.0)

    def test_405b_memory_accounting(self):
        report = memory_feasibility(load_model("meta-llama/Llama-3.1-405B"), 8.0, 1.0)
        self.assertEqual(report["parameter_count_analytical"], 405853372416)
        self.assertEqual(report["kv_bytes_per_token_per_gpu"], 32256)
        self.assertTrue(report["fits_weights_and_reserves"])
        self.assertGreater(report["max_kv_tokens_per_gpu"], 131072)

    def test_405b_profile_respects_h100_physical_bounds(self):
        report = physical_limit_check(read_rows(BASE / "Llama-3.1-405B" / "tp8" / "layers.csv"),
                                      load_model("meta-llama/Llama-3.1-405B"), 8)
        self.assertTrue(report["valid"], report["violations"])

    def test_generated_profiles_are_valid_and_explicitly_projected(self):
        for profile in (self.tp8, self.p405, self.p405 / "variants" / "low",
                        self.p405 / "variants" / "high"):
            self.assertTrue(validate_profile(profile, 8)["valid"], profile)
        manifest = json.loads((self.p405 / "profile_manifest.json").read_text())
        self.assertEqual(manifest["measurement_status"], "calibrated_projection_not_measured")
        self.assertEqual(manifest["target_tensor_parallel_degree"], 8)

    def test_attention_pickles_follow_astra_schema(self):
        for profile in (self.tp8, self.p405):
            for filename in ("attn_prefill_prediction_dict.pkl", "attn_decode_prediction_dict.pkl"):
                with (profile / "predictions" / filename).open("rb") as handle:
                    dictionary = pickle.load(handle)
                self.assertIn("latency(ns)", next(iter(dictionary.values())))


if __name__ == "__main__":
    unittest.main()
