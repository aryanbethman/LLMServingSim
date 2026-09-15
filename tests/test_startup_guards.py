import unittest

from serving.__main__ import _startup_guard_error


def _config(*instances):
    return {"nodes": [{"instances": list(instances)}]}


class StartupGuardTest(unittest.TestCase):
    def test_independent_replicas_are_allowed(self):
        config = _config({"instance_id": 0}, {"instance_id": 1})
        self.assertIsNone(_startup_guard_error("in-memory", "analytical", config))

    def test_tensor_and_pipeline_parallel_instance_is_allowed(self):
        config = _config({"instance_id": 0, "tp_size": 8, "pp_size": 2})
        self.assertIsNone(_startup_guard_error("shared-template", "analytical", config))

    def test_dp_group_is_rejected_before_template_transport(self):
        config = _config({"instance_id": 3, "dp_group": "workers"})
        error = _startup_guard_error("shared-template", "analytical", config)
        self.assertIn("explicit dp_group", error)
        self.assertIn("legacy", error)

    def test_ns3_is_rejected_before_template_transport(self):
        error = _startup_guard_error("in-memory", "ns3", _config({"instance_id": 0}))
        self.assertIn("requires --network-backend=analytical", error)

    def test_legacy_mode_keeps_existing_backend_and_dp_support(self):
        config = _config({"instance_id": 0, "dp_group": "workers"})
        self.assertIsNone(_startup_guard_error("legacy", "ns3", config))


if __name__ == "__main__":
    unittest.main()
