"""
Multimodal Fusion Module
==========================
Combines visual, text, and attribute features into a unified item embedding.
"""

import torch
import torch.nn as nn


class MultimodalFusion(nn.Module):
    """Late fusion of visual, text, and attribute features.

    Architecture:
        [visual_emb || text_emb || attr_emb] → MLP → unified embedding

    Args:
        visual_dim: Dimension of visual features.
        text_dim: Dimension of text features.
        attr_dim: Dimension of attribute features.
        hidden_dim: Hidden layer dimension.
        output_dim: Final unified embedding dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        visual_dim: int = 2048,
        text_dim: int = 384,
        attr_dim: int = 64,
        hidden_dim: int = 1024,
        output_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        input_dim = visual_dim + text_dim + attr_dim

        self.fusion_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        # L2 normalization for cosine similarity compatibility
        self.normalize = nn.functional.normalize

        self.output_dim = output_dim
        self._init_weights()

    def _init_weights(self):
        for m in self.fusion_network:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        attribute_features: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Fuse multimodal features into unified embedding.

        Args:
            visual_features: [B, visual_dim] from VisualEncoder.
            text_features: [B, text_dim] from TextEncoder.
            attribute_features: [B, attr_dim] from AttributeEncoder.
            normalize: L2-normalize output embeddings.

        Returns:
            [B, output_dim] unified item embedding.
        """
        combined = torch.cat([visual_features, text_features, attribute_features], dim=-1)
        embeddings = self.fusion_network(combined)

        if normalize:
            embeddings = self.normalize(embeddings, p=2, dim=-1)

        return embeddings
