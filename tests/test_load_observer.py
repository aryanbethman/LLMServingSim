from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from serving.core.load_observer import LoadObserver


class LoadObserverTest(unittest.TestCase):
    def test_clips_idle_time_and_integrates_busy_intervals(self):
        with tempfile.TemporaryDirectory() as tmp:
            observer = LoadObserver(Path(tmp) / "load.json", 2, 10, 30, 10)
            a = SimpleNamespace(batch_id=0, batch_time=5, requests=[1, 2])
            b = SimpleNamespace(batch_id=0, batch_time=20, requests=[3])
            observer.begin(0, a)
            observer.begin(1, b)
            observer.complete(0, a, 20)
            observer.complete(1, b, 35)
            result = observer.finish(40)
            self.assertEqual(result["per_replica_busy_fraction"], [.5, .5])
            self.assertEqual([r["mean_active_replicas"] for r in result["bins"]], [1, 1])
            self.assertEqual(result["sampled_peak_active_replicas"], 1)
            self.assertEqual(result["mean_batch_size"], 1.5)

    def test_incomplete_window_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = LoadObserver(Path(tmp) / "load.json", 1, 10, 30).finish(20)
            self.assertFalse(result["window_complete"])
            self.assertEqual(result["observed_window_ns"], [10, 20])


if __name__ == "__main__":
    unittest.main()
