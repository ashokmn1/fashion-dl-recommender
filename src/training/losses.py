"""
Training Loss Functions
========================
BPR, Triplet, and Contrastive losses for compatibility learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BPRLoss(nn.Module):
    """Bayesian Personalized Ranking loss.

    Maximizes the margin between positive and negative pair scores:
        L = -log(sigmoid(pos_score - neg_score))
    """

    def forward(
        self, pos_scores: torch.Tensor, neg_scores: torch.Tensor
    ) -> torch.Tensor:
        return -F.logsigmoid(pos_scores - neg_scores).mean()


class TripletMarginLoss(nn.Module):
    """Triplet margin loss with configurable margin.

    L = max(0, margin + d(anchor, positive) - d(anchor, negative))

    Args:
        margin: Minimum margin between positive and negative distances.
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        pos_dist = 1 - F.cosine_similarity(anchor, positive)
        neg_dist = 1 - F.cosine_similarity(anchor, negative)
        loss = F.relu(self.margin + pos_dist - neg_dist)
        return loss.mean()


class ContrastiveLoss(nn.Module):
    """Contrastive loss (SimCLR-style) for outfit items.

    Treats items in the same outfit as positives and items
    from different outfits as negatives.

    Args:
        temperature: Softmax temperature parameter.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            embeddings: [B, dim] normalized embeddings.
            labels: [B] outfit IDs (items with same ID are positives).
        """
        # Cosine similarity matrix
        sim_matrix = torch.mm(embeddings, embeddings.t()) / self.temperature

        # Mask out self-similarity
        mask = torch.eye(len(labels), device=labels.device).bool()
        sim_matrix.masked_fill_(mask, -float("inf"))

        # Positive mask: same outfit
        pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        pos_mask.masked_fill_(mask, False)

        # For each sample, compute loss over its positives
        loss = 0.0
        count = 0
        for i in range(len(labels)):
            positives = pos_mask[i]
            if positives.sum() == 0:
                continue

            # Log-softmax over all pairs
            log_probs = F.log_softmax(sim_matrix[i], dim=0)
            loss -= log_probs[positives].mean()
            count += 1

        return loss / max(count, 1)
