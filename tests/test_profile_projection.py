import json
import pickle
import unittest
from pathlib import Path

from llm_profile.profile_projection import (
    REPO,
    load_model,
    memory_feasibility,
    physical_limit_check,
    project_layers_tp8,
    read_rows,
    validate_profile,
)


class ProfileProjectionTest(unittest.TestCase):
    def setUp(self):
        self.base = REPO / "llm_profile" / "perf_models" / "H100" / "meta-llama"

    def test_tp8_layer_rows_semantically_match_frozen_reference(self):
        model = self.base / "Llama-3.1-70B"
        projected, _backtest, _bias = project_layers_tp8(
            read_rows(model / "tp1" / "layers.csv"),
            read_rows(model / "tp2" / "layers.csv"),
            read_rows(model / "tp4" / "layers.csv"),
        )
        reference = read_rows(model / "tp8_v0_reference" / "layers.csv")
        normalized = [
            {
                "layer_name": row["layer_name"],
                "input": int(row["input"]),
                "kv_cache": int(row["kv_cache"]),
                "tp_size": int(row["tp_size"]),
                "latency(ns)": int(row["latency(ns)"]),
            }
            for row in reference
        ]
        self.assertEqual(projected, normalized)

    def test_tp4_backtest_meets_documented_gate(self):
        model = self.base / "Llama-3.1-70B"
        _projected, backtest, _bias = project_layers_tp8(
            read_rows(model / "tp1" / "layers.csv"),
            read_rows(model / "tp2" / "layers.csv"),
            read_rows(model / "tp4" / "layers.csv"),
        )
        errors = sorted(float(row["absolute_error_pct"]) for row in backtest)
        median = errors[len(errors) // 2]
        p90 = errors[round((len(errors) - 1) * 0.90)]
        self.assertLessEqual(median, 10.0)
        self.assertLessEqual(p90, 25.0)

    def test_405b_memory_accounting(self):
        report = memory_feasibility(load_model("meta-llama/Llama-3.1-405B"), 8.0, 1.0)
        self.assertEqual(report["parameter_count_analytical"], 405853372416)
        self.assertEqual(report["kv_bytes_per_token_per_gpu"], 32256)
        self.assertTrue(report["fits_weights_and_reserves"])
        self.assertGreater(report["max_kv_tokens_per_gpu"], 131072)

    def test_405b_profile_respects_h100_physical_bounds(self):
        profile = self.base / "Llama-3.1-405B" / "tp8"
        report = physical_limit_check(read_rows(profile / "layers.csv"),
                                      load_model("meta-llama/Llama-3.1-405B"), 8)
        self.assertTrue(report["valid"], report["violations"])

    def test_generated_profiles_are_valid_and_explicitly_projected(self):
        tp8 = self.base / "Llama-3.1-70B" / "tp8_v1_generated"
        p405 = self.base / "Llama-3.1-405B" / "tp8"
        self.assertTrue(validate_profile(tp8, 8)["valid"])
        self.assertTrue(validate_profile(p405, 8)["valid"])
        self.assertTrue(validate_profile(p405 / "variants" / "low", 8)["valid"])
        self.assertTrue(validate_profile(p405 / "variants" / "high", 8)["valid"])
        manifest = json.loads((p405 / "profile_manifest.json").read_text())
        self.assertEqual(manifest["measurement_status"], "calibrated_projection_not_measured")
        self.assertEqual(manifest["target_tensor_parallel_degree"], 8)

    def test_attention_pickles_follow_astra_schema(self):
        profiles = (
            self.base / "Llama-3.1-70B" / "tp8_v1_generated",
            self.base / "Llama-3.1-405B" / "tp8",
        )
        for profile in profiles:
            for filename in ("attn_prefill_prediction_dict.pkl", "attn_decode_prediction_dict.pkl"):
                with (profile / "predictions" / filename).open("rb") as handle:
                    dictionary = pickle.load(handle)
                self.assertIn("latency(ns)", next(iter(dictionary.values())))


if __name__ == "__main__":
    unittest.main()
