import unittest

from serving.core.legacy_fastpath import LegacyValueFastPath
from serving.core.trace_generator import indexed_cols
from chakra.src.converter.llm_converter import LLMConverter
from test_template_value_fastpath import dense_rows, moe_rows, TP_HEADER, PP_HEADER, PREFILL_HEADER


class LegacyFastPathTest(unittest.TestCase):
    def test_rank_bytes_equal_full_conversion_with_changing_values_and_metadata(self):
        for header, rows, ranks in ((TP_HEADER, dense_rows, 4),
                                     (PP_HEADER, dense_rows, 4),
                                     (PREFILL_HEADER, dense_rows, 2),
                                     (TP_HEADER, moe_rows, 4)):
            fast = LegacyValueFastPath(verify=True)
            for scale in (1, 2, 3, 5):
                converter = LLMConverter(f"trace-{scale}.txt", None, ranks, 4)
                cols = indexed_cols(rows(scale))
                convert = lambda c: converter.convert_rows_to_payloads(header, c)
                expected = convert(cols)
                actual = fast.payloads(header, cols, ranks, 4, False,
                                       converter.get_global_metadata().SerializeToString(deterministic=True),
                                       convert)
                self.assertEqual(actual, expected)
            self.assertEqual(fast.stats["hit"], 3)
            self.assertEqual(fast.stats["verified"], 3)

    def test_node_dropping_zero_learns_another_shape_and_cache_is_bounded(self):
        fast = LegacyValueFastPath(max_entries=1)
        for zero in (False, True, False):
            rows = dense_rows()
            if zero:
                rows[0] = tuple("0" if i == 1 else value for i, value in enumerate(rows[0]))
            cols = indexed_cols(rows)
            converter = LLMConverter("trace", None, 2)
            convert = lambda c: converter.convert_rows_to_payloads(TP_HEADER, c)
            result = fast.payloads(TP_HEADER, cols, 2, 0, False,
                                   converter.get_global_metadata().SerializeToString(), convert)
            self.assertEqual(result, convert(cols))
        self.assertEqual(fast.stats["learned"], 3)
        self.assertEqual(fast.stats["evictions"], 2)


if __name__ == "__main__":
    unittest.main()
