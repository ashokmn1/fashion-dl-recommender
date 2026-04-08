"""
User Profile & Personalization
================================
Build user preference profiles from interaction history
and apply personalized re-ranking to recommendations.
"""

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class UserProfileBuilder:
    """Build user preference vectors from interaction history.

    Aggregates item embeddings from user interactions with temporal
    decay weighting (recent interactions matter more).

    Args:
        embedding_dim: Dimension of item embeddings.
        decay_halflife: Half-life in days for temporal decay.
        interaction_weights: Weight per interaction type.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        decay_halflife: float = 30.0,
        interaction_weights: Optional[dict] = None,
    ):
        self.embedding_dim = embedding_dim
        self.decay_halflife = decay_halflife
        self.interaction_weights = interaction_weights or {
            "view": 1.0,
            "click": 2.0,
            "add_to_cart": 4.0,
            "purchase": 8.0,
            "save": 5.0,
        }
        self.user_profiles = {}

    def _temporal_weight(self, days_ago: float) -> float:
        """Exponential decay weight based on recency."""
        return math.exp(-math.log(2) * days_ago / self.decay_halflife)

    def build_profile(
        self,
        user_id: int,
        interactions: list[dict],
        item_embeddings: dict[int, np.ndarray],
        current_date: Optional[str] = None,
    ) -> np.ndarray:
        """Build user preference vector from interactions.

        Args:
            user_id: User identifier.
            interactions: List of {item_id, interaction_type, timestamp}.
            item_embeddings: Dict mapping item_id → embedding vector.
            current_date: Reference date for temporal decay.

        Returns:
            User preference vector (normalized).
        """
        if current_date is None:
            current_date = datetime.now().isoformat()

        try:
            ref_date = datetime.fromisoformat(current_date.replace("Z", "+00:00"))
        except Exception:
            ref_date = datetime.now()

        weighted_sum = np.zeros(self.embedding_dim)
        total_weight = 0.0

        for interaction in interactions:
            item_id = interaction["item_id"]
            if item_id not in item_embeddings:
                continue

            # Interaction type weight
            itype = interaction.get("interaction_type", "view")
            type_weight = self.interaction_weights.get(itype, 1.0)

            # Temporal decay
            try:
                ts = datetime.fromisoformat(interaction["timestamp"].replace("Z", "+00:00"))
                days_ago = (ref_date - ts).days
            except Exception:
                days_ago = 30  # default

            temporal_weight = self._temporal_weight(max(0, days_ago))
            weight = type_weight * temporal_weight

            weighted_sum += weight * item_embeddings[item_id]
            total_weight += weight

        if total_weight > 0:
            profile = weighted_sum / total_weight
            # L2 normalize
            norm = np.linalg.norm(profile)
            if norm > 0:
                profile = profile / norm
        else:
            profile = np.zeros(self.embedding_dim)

        self.user_profiles[user_id] = profile
        return profile

    def build_all_profiles(
        self,
        users: list[dict],
        interactions: list[dict],
        item_embeddings: dict[int, np.ndarray],
    ) -> dict[int, np.ndarray]:
        """Build profiles for all users."""
        # Group interactions by user
        user_interactions = {}
        for inter in interactions:
            uid = inter["user_id"]
            user_interactions.setdefault(uid, []).append(inter)

        profiles = {}
        for user in users:
            uid = user["user_id"]
            user_inters = user_interactions.get(uid, [])
            profiles[uid] = self.build_profile(uid, user_inters, item_embeddings)

        self.user_profiles = profiles
        return profiles

    def get_profile(self, user_id: int) -> Optional[np.ndarray]:
        return self.user_profiles.get(user_id)


class PersonalizedReranker(nn.Module):
    """Re-rank candidate items based on user preferences.

    Combines compatibility score with personalization score:
        final_score = (1 - alpha) * compatibility_score + alpha * personalization_score

    Args:
        embedding_dim: Item/user embedding dimension.
        alpha: Personalization weight (0 = no personalization, 1 = full).
    """

    def __init__(self, embedding_dim: int = 512, alpha: float = 0.3):
        super().__init__()
        self.alpha = alpha

        self.preference_scorer = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        item_embeddings: torch.Tensor,
        user_embedding: torch.Tensor,
        compatibility_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Re-rank items based on compatibility + personalization.

        Args:
            item_embeddings: [N, dim] candidate item embeddings.
            user_embedding: [1, dim] user preference vector.
            compatibility_scores: [N] base compatibility scores.

        Returns:
            [N] final scores after personalization.
        """
        # Expand user embedding to match items
        user_expanded = user_embedding.expand(item_embeddings.size(0), -1)
        combined = torch.cat([item_embeddings, user_expanded], dim=-1)

        personalization_scores = self.preference_scorer(combined).squeeze(-1)

        # Weighted combination
        final_scores = (
            (1 - self.alpha) * compatibility_scores
            + self.alpha * personalization_scores
        )
        return final_scores

    def rerank_simple(
        self,
        item_embeddings: np.ndarray,
        user_profile: np.ndarray,
        compatibility_scores: np.ndarray,
        alpha: Optional[float] = None,
    ) -> np.ndarray:
        """Simple cosine-similarity based re-ranking (no learned params).

        Args:
            item_embeddings: [N, dim] candidate items.
            user_profile: [dim] user preference vector.
            compatibility_scores: [N] base scores.
            alpha: Override personalization weight.

        Returns:
            [N] re-ranked scores.
        """
        a = alpha if alpha is not None else self.alpha

        # Cosine similarity as personalization signal
        if np.linalg.norm(user_profile) > 0:
            cos_sim = item_embeddings @ user_profile
            cos_sim = (cos_sim + 1) / 2  # normalize to [0, 1]
        else:
            cos_sim = np.zeros(len(item_embeddings))

        return (1 - a) * compatibility_scores + a * cos_sim
