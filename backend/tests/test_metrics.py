"""Unit tests for dependency-free evaluation metrics."""

from __future__ import annotations

import math
import unittest

from backend.ir_system.baseline import fixed_character_chunks
from backend.ir_system.metrics import (
    hit_at_k,
    ndcg_at_k,
    percentile,
    reciprocal_rank,
    rouge_l_f1,
    token_f1,
    unique_documents,
)


class RetrievalMetricTests(unittest.TestCase):
    def test_legacy_chunker_does_not_add_an_overlap_only_tail(self) -> None:
        chunks = fixed_character_chunks("x" * 1_000, chunk_size=600, overlap=150)

        self.assertEqual([len(chunk) for chunk in chunks], [600, 550])

    def test_unique_documents_keeps_first_rank(self) -> None:
        self.assertEqual(unique_documents(["a.pdf", "a.pdf", "b.pdf"]), ["a.pdf", "b.pdf"])

    def test_hit_at_k_obeys_cutoff(self) -> None:
        ranked = ["a.pdf", "b.pdf", "c.pdf"]
        relevant = {"c.pdf"}
        self.assertEqual(hit_at_k(ranked, relevant, 2), 0.0)
        self.assertEqual(hit_at_k(ranked, relevant, 3), 1.0)

    def test_reciprocal_rank_obeys_cutoff(self) -> None:
        ranked = ["a.pdf", "b.pdf", "c.pdf"]
        relevant = {"c.pdf"}
        self.assertAlmostEqual(reciprocal_rank(ranked, relevant, cutoff=3), 1.0 / 3.0)
        self.assertEqual(reciprocal_rank(ranked, relevant, cutoff=2), 0.0)

    def test_binary_ndcg_supports_multiple_relevant_documents(self) -> None:
        ranked = ["relevant-a.pdf", "other.pdf", "relevant-b.pdf"]
        relevant = {"relevant-a.pdf", "relevant-b.pdf"}
        dcg = 1.0 + 1.0 / math.log2(4.0)
        ideal = 1.0 + 1.0 / math.log2(3.0)
        self.assertAlmostEqual(ndcg_at_k(ranked, relevant, 3), dcg / ideal)

    def test_empty_qrels_have_zero_ndcg(self) -> None:
        self.assertEqual(ndcg_at_k(["a.pdf"], set(), 5), 0.0)


class AnswerMetricTests(unittest.TestCase):
    def test_token_f1_is_one_for_exact_answer_with_citation(self) -> None:
        self.assertEqual(token_f1("The policy applies.", "The policy applies. [1]"), 1.0)

    def test_token_f1_counts_repeated_tokens(self) -> None:
        self.assertAlmostEqual(token_f1("a a b", "a b b"), 2.0 / 3.0)

    def test_token_f1_returns_zero_without_overlap(self) -> None:
        self.assertEqual(token_f1("English", "Hindi"), 0.0)

    def test_empty_answers_match_only_each_other(self) -> None:
        self.assertEqual(token_f1("", ""), 1.0)
        self.assertEqual(token_f1("answer", ""), 0.0)
        self.assertEqual(rouge_l_f1("", ""), 1.0)

    def test_rouge_l_uses_longest_common_subsequence(self) -> None:
        self.assertAlmostEqual(rouge_l_f1("a b c", "a x c"), 2.0 / 3.0)


class TimingMetricTests(unittest.TestCase):
    def test_percentile_uses_linear_interpolation(self) -> None:
        values = [0.0, 10.0, 20.0, 30.0]
        self.assertEqual(percentile(values, 0.5), 15.0)
        self.assertAlmostEqual(percentile(values, 0.95), 28.5)

    def test_empty_percentile_is_zero(self) -> None:
        self.assertEqual(percentile([], 0.5), 0.0)


if __name__ == "__main__":
    unittest.main()
