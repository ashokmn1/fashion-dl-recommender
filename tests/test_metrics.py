"""Tests for evaluation metrics."""

import sys
from pathlib import Path

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import (
    fitb_accuracy,
    compatibility_auc,
    hit_rate_at_k,
    ndcg_at_k,
    outfit_coherence_score,
    diversity_score,
)


class TestFITBAccuracy:
    def test_correct_prediction(self):
        # Context and correct answer are similar
        context = [np.array([1.0, 0.0, 0.0])]
        choices = [
            np.array([0.9, 0.1, 0.0]),   # correct (most similar)
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([-1.0, 0.0, 0.0]),
        ]
        assert fitb_accuracy(context, choices, 0) == True

    def test_empty_context(self):
        assert fitb_accuracy([], [np.zeros(3)], 0) is False


class TestCompatibilityAUC:
    def test_perfect_separation(self):
        pos = np.array([0.9, 0.8, 0.95])
        neg = np.array([0.1, 0.2, 0.05])
        auc = compatibility_auc(pos, neg)
        assert auc == 1.0

    def test_random_separation(self):
        np.random.seed(42)
        pos = np.random.rand(100)
        neg = np.random.rand(100)
        auc = compatibility_auc(pos, neg)
        assert 0.3 < auc < 0.7  # roughly random


class TestHitRate:
    def test_all_hits(self):
        recs = [[1, 2, 3], [4, 5, 6]]
        truth = [1, 4]
        assert hit_rate_at_k(recs, truth, k=3) == 1.0

    def test_no_hits(self):
        recs = [[1, 2, 3], [4, 5, 6]]
        truth = [7, 8]
        assert hit_rate_at_k(recs, truth, k=3) == 0.0


class TestNDCG:
    def test_perfect_ranking(self):
        recs = [[1, 2, 3]]
        truth = [1]  # at position 0
        ndcg = ndcg_at_k(recs, truth, k=3)
        assert ndcg == 1.0

    def test_position_matters(self):
        recs1 = [[1, 2, 3]]
        recs2 = [[2, 3, 1]]
        truth = [1]
        ndcg1 = ndcg_at_k(recs1, truth, k=3)
        ndcg2 = ndcg_at_k(recs2, truth, k=3)
        assert ndcg1 > ndcg2  # earlier position = higher NDCG


class TestOutfitCoherence:
    def test_identical_items(self):
        embs = [np.array([1.0, 0.0]), np.array([1.0, 0.0])]
        score = outfit_coherence_score(embs)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_single_item(self):
        assert outfit_coherence_score([np.array([1.0, 0.0])]) == 0.0


class TestDiversity:
    def test_identical_outfits(self):
        outfits = [
            [np.array([1.0, 0.0])],
            [np.array([1.0, 0.0])],
        ]
        score = diversity_score(outfits)
        assert score == pytest.approx(0.0, abs=0.01)

    def test_diverse_outfits(self):
        outfits = [
            [np.array([1.0, 0.0])],
            [np.array([0.0, 1.0])],
        ]
        score = diversity_score(outfits)
        assert score > 0.5
