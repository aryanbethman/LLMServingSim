from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from serving.core import graph_generator as graph
from serving.core.trace_generator import TraceData
from test_template_value_overlay import TP_HEADER, rows


class RepeatCacheAblationTest(unittest.TestCase):
    def tearDown(self):
        graph.set_template_value_overlay(False)
        graph.set_legacy_value_fastpath(False)

    def test_legacy_keeps_rank_file_delivery_without_repeat_digest_or_cache(self):
        trace = TraceData(header_line=TP_HEADER, rows=rows(), path=None)
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(graph, "_REPEAT_CACHES_ENABLED", False), \
                patch.object(graph, "_rows_digest", side_effect=AssertionError("repeat digest used")), \
                patch.object(graph, "_cache_store") as store:
            for enabled in (False, True):
                graph.set_legacy_value_fastpath(enabled)
                for _ in range(2):
                    graph.generate_graph(SimpleNamespace(model="test", batch_id=0),
                                         "test", 2, inputs_root=tmp, trace=trace)
                paths = list(Path(tmp).rglob("*.et"))
                self.assertEqual(len(paths), 2)
                self.assertTrue(all(p.stat().st_size > 0 for p in paths))
            store.assert_not_called()

    def test_shared_uses_structural_fast_path_without_encoded_bindings_cache(self):
        trace = TraceData(header_line=TP_HEADER, rows=rows(), path=None)
        graph.set_template_value_overlay(True)
        with patch.object(graph, "_REPEAT_CACHES_ENABLED", False), \
                patch.object(graph, "_rows_digest", side_effect=AssertionError("repeat digest used")), \
                patch.object(graph, "_bindings_cache_store") as store:
            first = graph.generate_graph(None, None, 2, template_mode="shared-template",
                                         known_template_ids=set(), trace=trace)
            second = graph.generate_graph(None, None, 2, template_mode="shared-template",
                                          known_template_ids=set(), trace=trace)
            self.assertEqual(first[0], second[0])
            self.assertEqual(graph._VALUE_FASTPATH.stats["hit"], 1)
            store.assert_not_called()


if __name__ == "__main__":
    unittest.main()
