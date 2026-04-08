"""
Attribute Feature Encoder
==========================
Learned embeddings for categorical fashion attributes.
"""

import torch
import torch.nn as nn


class AttributeEncoder(nn.Module):
    """Encode categorical fashion attributes into dense embeddings.

    Attributes encoded: category, color, material, pattern, season.
    Each attribute gets its own learned embedding table.

    Args:
        embedding_dim: Dimension per attribute embedding.
        num_categories: Number of product categories.
        num_colors: Number of color options.
        num_materials: Number of material types.
        num_patterns: Number of pattern types.
        num_seasons: Number of seasons.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        num_categories: int = 4,
        num_colors: int = 12,
        num_materials: int = 8,
        num_patterns: int = 6,
        num_seasons: int = 4,
    ):
        super().__init__()

        self.category_emb = nn.Embedding(num_categories, embedding_dim)
        self.color_emb = nn.Embedding(num_colors, embedding_dim)
        self.material_emb = nn.Embedding(num_materials, embedding_dim)
        self.pattern_emb = nn.Embedding(num_patterns, embedding_dim)
        self.season_emb = nn.Embedding(num_seasons, embedding_dim)

        # Combine all attribute embeddings
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 5, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        self.output_dim = embedding_dim
        self._init_weights()

    def _init_weights(self):
        """Initialize embedding weights."""
        for emb in [self.category_emb, self.color_emb, self.material_emb,
                     self.pattern_emb, self.season_emb]:
            nn.init.xavier_uniform_(emb.weight)

    def forward(self, attributes: torch.Tensor) -> torch.Tensor:
        """Encode attribute tensor into dense embedding.

        Args:
            attributes: [batch_size, 5] tensor with columns:
                        [category, color, material, pattern, season]

        Returns:
            [batch_size, embedding_dim] attribute embedding.
        """
        cat_emb = self.category_emb(attributes[:, 0])
        col_emb = self.color_emb(attributes[:, 1])
        mat_emb = self.material_emb(attributes[:, 2])
        pat_emb = self.pattern_emb(attributes[:, 3])
        sea_emb = self.season_emb(attributes[:, 4])

        combined = torch.cat([cat_emb, col_emb, mat_emb, pat_emb, sea_emb], dim=-1)
        return self.fusion(combined)
