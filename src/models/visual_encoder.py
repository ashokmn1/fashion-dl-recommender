"""
Visual Feature Encoder
=======================
CNN backbone for extracting visual features from fashion item images.
Uses pre-trained ResNet-50 with optional fine-tuning of later layers.
"""

import torch
import torch.nn as nn
from torchvision import models


class VisualEncoder(nn.Module):
    """ResNet-50 based visual feature extractor for fashion images.

    Architecture:
        ResNet-50 (ImageNet pretrained) → AdaptiveAvgPool → FC projection
        Output: [batch_size, embedding_dim] visual embedding

    Args:
        embedding_dim: Output embedding dimension (default: 2048).
        pretrained: Use ImageNet pretrained weights.
        fine_tune_layers: Number of ResNet blocks to fine-tune (0-4).
    """

    def __init__(
        self,
        embedding_dim: int = 2048,
        pretrained: bool = True,
        fine_tune_layers: int = 2,
    ):
        super().__init__()

        # Load pretrained ResNet-50
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)

        # Remove final FC layer — we use our own projection
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # up to avgpool
        self.resnet_dim = 2048  # ResNet-50 output dim

        # Projection head
        if embedding_dim != self.resnet_dim:
            self.projection = nn.Sequential(
                nn.Linear(self.resnet_dim, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
            )
        else:
            self.projection = nn.Identity()

        # Freeze early layers, fine-tune only later blocks
        self._set_fine_tune(fine_tune_layers)

    def _set_fine_tune(self, num_layers: int):
        """Freeze all layers except the last `num_layers` ResNet blocks."""
        # ResNet blocks are children 4-7 (layer1-layer4)
        all_children = list(self.features.children())
        freeze_until = max(0, 8 - num_layers)  # 8 = total children before avgpool

        for i, child in enumerate(all_children):
            if i < freeze_until:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract visual features from fashion item images.

        Args:
            images: [batch_size, 3, 224, 224] normalized image tensor.

        Returns:
            [batch_size, embedding_dim] visual embedding.
        """
        features = self.features(images)           # [B, 2048, 1, 1]
        features = features.flatten(1)             # [B, 2048]
        embeddings = self.projection(features)     # [B, embedding_dim]
        return embeddings

    def get_feature_dim(self) -> int:
        """Return the output feature dimension."""
        if isinstance(self.projection, nn.Identity):
            return self.resnet_dim
        return self.projection[0].out_features

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features in inference mode (no gradient)."""
        self.eval()
        return self.forward(images)
