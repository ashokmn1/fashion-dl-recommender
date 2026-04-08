"""
Evaluation Metrics
===================
Metrics for outfit compatibility and recommendation quality.
"""

import numpy as np
from typing import Optional


def fitb_accuracy(
    context_embeddings: list[np.ndarray],
    choice_embeddings: list[np.ndarray],
    answer_index: int,
) -> bool:
    """Fill-in-the-Blank accuracy for a single question.

    Given context items and multiple choices, predict which choice
    best completes the outfit.

    Args:
        context_embeddings: List of embeddings for context items.
        choice_embeddings: List of embeddings for answer choices.
        answer_index: Index of the correct answer.

    Returns:
        True if the model's top pick is correct.
    """
    if not context_embeddings or not choice_embeddings:
        return False

    # Average context embedding
    context_avg = np.mean(context_embeddings, axis=0)
    context_norm = context_avg / (np.linalg.norm(context_avg) + 1e-8)

    # Score each choice by similarity to context
    scores = []
    for emb in choice_embeddings:
        emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
        score = np.dot(context_norm, emb_norm)
        scores.append(score)

    predicted = np.argmax(scores)
    return predicted == answer_index


def compatibility_auc(
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
) -> float:
    """Compute AUC for compatibility scoring.

    Args:
        positive_scores: Compatibility scores for positive pairs.
        negative_scores: Compatibility scores for negative pairs.

    Returns:
        AUC score between 0 and 1.
    """
    from sklearn.metrics import roc_auc_score

    labels = np.concatenate([
        np.ones(len(positive_scores)),
        np.zeros(len(negative_scores)),
    ])
    scores = np.concatenate([positive_scores, negative_scores])

    try:
        return roc_auc_score(labels, scores)
    except ValueError:
        return 0.5


def hit_rate_at_k(
    recommended_ids: list[list[int]],
    ground_truth_ids: list[int],
    k: int = 10,
) -> float:
    """Compute Hit Rate @ K.

    Args:
        recommended_ids: List of recommended item ID lists per query.
        ground_truth_ids: List of correct item IDs per query.
        k: Number of top recommendations to consider.

    Returns:
        Hit rate between 0 and 1.
    """
    hits = 0
    for recs, truth in zip(recommended_ids, ground_truth_ids):
        if truth in recs[:k]:
            hits += 1
    return hits / len(ground_truth_ids) if ground_truth_ids else 0.0


def ndcg_at_k(
    recommended_ids: list[list[int]],
    ground_truth_ids: list[int],
    k: int = 10,
) -> float:
    """Compute NDCG @ K.

    Args:
        recommended_ids: List of recommended item ID lists.
        ground_truth_ids: List of correct item IDs.
        k: Cutoff position.

    Returns:
        NDCG score between 0 and 1.
    """
    ndcg_scores = []
    for recs, truth in zip(recommended_ids, ground_truth_ids):
        dcg = 0.0
        for i, rec in enumerate(recs[:k]):
            if rec == truth:
                dcg += 1.0 / np.log2(i + 2)  # position is 1-indexed
                break

        # Ideal DCG (truth item at position 1)
        idcg = 1.0 / np.log2(2)
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    return np.mean(ndcg_scores) if ndcg_scores else 0.0


def outfit_coherence_score(
    item_embeddings: list[np.ndarray],
) -> float:
    """Score overall outfit coherence (average pairwise cosine similarity).

    Args:
        item_embeddings: List of item embeddings in the outfit.

    Returns:
        Coherence score between -1 and 1.
    """
    if len(item_embeddings) < 2:
        return 0.0

    scores = []
    for i in range(len(item_embeddings)):
        for j in range(i + 1, len(item_embeddings)):
            a = item_embeddings[i]
            b = item_embeddings[j]
            sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
            scores.append(sim)

    return float(np.mean(scores))


def diversity_score(
    outfits: list[list[np.ndarray]],
) -> float:
    """Measure diversity across recommended outfits.

    Higher score = more diverse recommendations.

    Args:
        outfits: List of outfits, each outfit is a list of item embeddings.

    Returns:
        Diversity score between 0 and 1.
    """
    if len(outfits) < 2:
        return 0.0

    # Average embedding per outfit
    outfit_avgs = []
    for outfit in outfits:
        if outfit:
            avg = np.mean(outfit, axis=0)
            avg = avg / (np.linalg.norm(avg) + 1e-8)
            outfit_avgs.append(avg)

    if len(outfit_avgs) < 2:
        return 0.0

    # Average pairwise distance between outfit averages
    distances = []
    for i in range(len(outfit_avgs)):
        for j in range(i + 1, len(outfit_avgs)):
            sim = np.dot(outfit_avgs[i], outfit_avgs[j])
            distances.append(1 - sim)  # distance = 1 - similarity

    return float(np.mean(distances))


def evaluate_pipeline(
    pipeline,
    test_outfits: list[dict],
    item_embeddings: dict[int, np.ndarray],
    num_choices: int = 4,
) -> dict:
    """Run full evaluation suite on the recommendation pipeline.

    Args:
        pipeline: RecommendationPipeline instance.
        test_outfits: Test set outfits.
        item_embeddings: Item ID to embedding mapping.
        num_choices: Number of FITB choices.

    Returns:
        Dict of metric name → score.
    """
    fitb_correct = 0
    fitb_total = 0
    coherence_scores = []
    recommended_ids_list = []
    truth_ids_list = []

    items_by_cat = {}
    for iid, meta in pipeline.item_metadata.items():
        items_by_cat.setdefault(meta["category"], []).append(iid)

    for outfit in test_outfits:
        item_ids = outfit["item_ids"]
        if len(item_ids) < 2:
            continue

        # FITB evaluation
        for remove_idx in range(len(item_ids)):
            answer_id = item_ids[remove_idx]
            context_ids = [iid for i, iid in enumerate(item_ids) if i != remove_idx]

            answer_cat = pipeline.item_metadata.get(answer_id, {}).get("category")
            if not answer_cat:
                continue

            # Get embeddings
            context_embs = [item_embeddings[cid] for cid in context_ids if cid in item_embeddings]
            answer_emb = item_embeddings.get(answer_id)
            if not context_embs or answer_emb is None:
                continue

            # Generate wrong choices
            cat_items = items_by_cat.get(answer_cat, [])
            wrong = [iid for iid in np.random.choice(cat_items, size=min(num_choices * 2, len(cat_items)), replace=False) if iid != answer_id][:num_choices - 1]

            choices = [answer_id] + wrong
            choice_embs = [item_embeddings[cid] for cid in choices if cid in item_embeddings]

            if len(choice_embs) == len(choices):
                correct = fitb_accuracy(context_embs, choice_embs, 0)
                fitb_correct += int(correct)
                fitb_total += 1

        # Outfit coherence
        outfit_embs = [item_embeddings[iid] for iid in item_ids if iid in item_embeddings]
        if len(outfit_embs) >= 2:
            coherence_scores.append(outfit_coherence_score(outfit_embs))

    results = {
        "fitb_accuracy": fitb_correct / max(fitb_total, 1),
        "fitb_total": fitb_total,
        "avg_coherence": float(np.mean(coherence_scores)) if coherence_scores else 0.0,
        "num_test_outfits": len(test_outfits),
    }

    return results
