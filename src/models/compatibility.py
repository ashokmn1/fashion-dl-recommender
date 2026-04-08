"""
Compatibility Learning Model
==============================
Type-Aware Embedding Network that learns outfit compatibility
between fashion items using BPR (Bayesian Personalized Ranking) loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations


class TypeAwareCompatibilityModel(nn.Module):
    """Learn compatibility between fashion items with type-aware projections.

    Each category pair (e.g., top-bottom, top-shoes) gets its own projection
    head, allowing the model to learn different compatibility patterns for
    different item combinations.

    Architecture:
        item_embedding → type_projection[cat_pair] → compatibility_score

    Args:
        embedding_dim: Item embedding dimension (from fusion module).
        num_categories: Number of product categories.
        hidden_dim: Projection hidden dimension.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        num_categories: int = 4,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_categories = num_categories

        # Generate all category pair keys
        self.category_pairs = list(combinations(range(num_categories), 2))
        # Also add same-category pairs for completeness
        self.category_pairs += [(i, i) for i in range(num_categories)]

        # Type-aware projection heads — one per category pair
        self.projections = nn.ModuleDict()
        for cat_a, cat_b in self.category_pairs:
            key = f"{cat_a}_{cat_b}"
            self.projections[key] = nn.Sequential(
                nn.Linear(embedding_dim * 2, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim // 2, 1),
            )

        # Fallback general compatibility head
        self.general_projection = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def _get_pair_key(self, cat_a: int, cat_b: int) -> str:
        """Get the projection key for a category pair (order-invariant)."""
        return f"{min(cat_a, cat_b)}_{max(cat_a, cat_b)}"

    def compute_compatibility(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        cat_a: int,
        cat_b: int,
    ) -> torch.Tensor:
        """Compute compatibility score between two items.

        Args:
            emb_a: [B, embedding_dim] embedding of item A.
            emb_b: [B, embedding_dim] embedding of item B.
            cat_a: Category index of item A.
            cat_b: Category index of item B.

        Returns:
            [B, 1] compatibility scores.
        """
        pair_input = torch.cat([emb_a, emb_b], dim=-1)

        key = self._get_pair_key(cat_a, cat_b)
        if key in self.projections:
            return self.projections[key](pair_input)
        else:
            return self.general_projection(pair_input)

    def forward(
        self,
        anchor_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        negative_embs: list[torch.Tensor],
        anchor_cat: int,
        positive_cat: int,
    ) -> dict:
        """Compute BPR loss for compatibility training.

        Args:
            anchor_emb: [B, dim] anchor item embedding.
            positive_emb: [B, dim] compatible item embedding.
            negative_embs: List of [B, dim] incompatible item embeddings.
            anchor_cat: Anchor category index.
            positive_cat: Positive/negative category index.

        Returns:
            Dict with 'loss', 'pos_score', 'neg_score' tensors.
        """
        # Positive compatibility score
        pos_score = self.compute_compatibility(anchor_emb, positive_emb, anchor_cat, positive_cat)

        # BPR loss: maximize margin between positive and negative scores
        total_loss = torch.tensor(0.0, device=anchor_emb.device)
        neg_scores = []

        for neg_emb in negative_embs:
            neg_score = self.compute_compatibility(anchor_emb, neg_emb, anchor_cat, positive_cat)
            neg_scores.append(neg_score)

            # BPR loss: -log(sigmoid(pos - neg))
            diff = pos_score - neg_score
            total_loss += -F.logsigmoid(diff).mean()

        total_loss /= len(negative_embs)  # average over negatives

        avg_neg_score = torch.stack(neg_scores).mean(0) if neg_scores else pos_score

        return {
            "loss": total_loss,
            "pos_score": pos_score.mean(),
            "neg_score": avg_neg_score.mean(),
            "margin": (pos_score - avg_neg_score).mean(),
        }

    def score_outfit(
        self,
        item_embeddings: list[torch.Tensor],
        item_categories: list[int],
    ) -> torch.Tensor:
        """Score overall outfit compatibility (average pairwise scores).

        Args:
            item_embeddings: List of [1, dim] item embeddings.
            item_categories: List of category indices.

        Returns:
            Scalar outfit compatibility score.
        """
        scores = []
        for i in range(len(item_embeddings)):
            for j in range(i + 1, len(item_embeddings)):
                score = self.compute_compatibility(
                    item_embeddings[i],
                    item_embeddings[j],
                    item_categories[i],
                    item_categories[j],
                )
                scores.append(score)

        if not scores:
            return torch.tensor(0.0)
        return torch.stack(scores).mean()
