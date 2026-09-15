import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from analysis import write_run_manifest as manifest


class ManifestTests(unittest.TestCase):
    def test_uniform_profiles_preserve_scalar_fields(self):
        config = {
            "nodes": [{"instances": [
                {"model_name": "m", "hardware": "H100", "tp_size": 4, "num_npus": 4},
                {"model_name": "m", "hardware": "H100", "tp_size": 4, "num_npus": 4},
            ]}]
        }
        profiles = manifest.config_profiles(config, 8)
        self.assertEqual(manifest.uniform_or_list(profiles, "model"), "m")
        self.assertEqual(manifest.uniform_or_list(profiles, "tensor_parallel_degree"), 4)

    def test_heterogeneous_profiles_are_explicit(self):
        config = {"nodes": [{"instances": [
            {"model_name": "m1", "hardware": "H100", "tp_size": 4, "num_npus": 4},
            {"model_name": "m2", "hardware": "A100", "tp_size": 2, "num_npus": 2},
        ]}]}
        profiles = manifest.config_profiles(config, 6)
        self.assertEqual(manifest.uniform_or_list(profiles, "model"), ["m1", "m2"])
        self.assertEqual(manifest.uniform_or_list(profiles, "hardware"), ["A100", "H100"])

    def test_logical_npu_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            manifest.config_profiles({"nodes": [{"instances": [
                {"model_name": "m", "hardware": "H100", "tp_size": 4, "num_npus": 4}
            ]}]}, 8)

    def test_manifest_contains_chakra_and_untracked_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            dataset = root / "dataset.jsonl"
            config.write_text(json.dumps({"nodes": [{"instances": [
                {"model_name": "m", "hardware": "H100", "tp_size": 4, "num_npus": 4}
            ]}]}))
            dataset.write_text("{}\n")
            args = argparse.Namespace(
                logical_npus=4, template_cache_max_entries=2, profile_variant="bf16-low")
            metadata = {"head": "h", "status": "", "working_diff_sha256": "d"}
            with patch.object(manifest, "git_metadata", return_value=metadata), \
                 patch.object(manifest, "untracked_source_hashes", return_value={"serving/new.py": "x"}):
                result = manifest.manifest_for(root, config, dataset, args)
            self.assertIn("chakra", result["astra_sim"])
            self.assertEqual(result["profile_variant"], "bf16-low")
            self.assertEqual(result["untracked_source_hashes"]["serving/new.py"], "x")


if __name__ == "__main__":
    unittest.main()
