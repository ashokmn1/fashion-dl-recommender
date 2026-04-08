"""Tests for model modules."""

import sys
from pathlib import Path

import pytest
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestVisualEncoder:
    def test_output_shape(self):
        from src.models.visual_encoder import VisualEncoder
        model = VisualEncoder(embedding_dim=512, pretrained=False, fine_tune_layers=0)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 512)

    def test_feature_dim(self):
        from src.models.visual_encoder import VisualEncoder
        model = VisualEncoder(embedding_dim=256, pretrained=False)
        assert model.get_feature_dim() == 256


class TestAttributeEncoder:
    def test_output_shape(self):
        from src.models.attribute_encoder import AttributeEncoder
        model = AttributeEncoder(embedding_dim=64)
        attrs = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 4, 1]])
        out = model(attrs)
        assert out.shape == (2, 64)


class TestMultimodalFusion:
    def test_fusion_output(self):
        from src.models.fusion import MultimodalFusion
        model = MultimodalFusion(
            visual_dim=512, text_dim=384, attr_dim=64,
            hidden_dim=256, output_dim=128
        )
        vis = torch.randn(4, 512)
        txt = torch.randn(4, 384)
        attr = torch.randn(4, 64)
        out = model(vis, txt, attr)
        assert out.shape == (4, 128)

    def test_normalization(self):
        from src.models.fusion import MultimodalFusion
        model = MultimodalFusion(
            visual_dim=64, text_dim=64, attr_dim=64,
            hidden_dim=64, output_dim=32
        )
        vis = torch.randn(2, 64)
        txt = torch.randn(2, 64)
        attr = torch.randn(2, 64)
        out = model(vis, txt, attr, normalize=True)
        norms = torch.norm(out, dim=1)
        assert torch.allclose(norms, torch.ones(2), atol=1e-5)


class TestCompatibilityModel:
    def test_forward_pass(self):
        from src.models.compatibility import TypeAwareCompatibilityModel
        model = TypeAwareCompatibilityModel(embedding_dim=128, num_categories=4, hidden_dim=64)
        anchor = torch.randn(4, 128)
        positive = torch.randn(4, 128)
        negatives = [torch.randn(4, 128) for _ in range(3)]
        result = model(anchor, positive, negatives, anchor_cat=0, positive_cat=1)
        assert "loss" in result
        assert "pos_score" in result
        assert "neg_score" in result
        assert result["loss"].requires_grad

    def test_outfit_scoring(self):
        from src.models.compatibility import TypeAwareCompatibilityModel
        model = TypeAwareCompatibilityModel(embedding_dim=64, num_categories=4)
        embs = [torch.randn(1, 64) for _ in range(4)]
        cats = [0, 1, 2, 3]
        score = model.score_outfit(embs, cats)
        assert isinstance(score, torch.Tensor)


class TestLosses:
    def test_bpr_loss(self):
        from src.training.losses import BPRLoss
        loss_fn = BPRLoss()
        pos = torch.tensor([2.0, 1.5, 3.0])
        neg = torch.tensor([1.0, 0.5, 1.0])
        loss = loss_fn(pos, neg)
        assert loss.item() > 0
        assert loss.item() < 1  # margin is positive, loss should be small

    def test_triplet_loss(self):
        from src.training.losses import TripletMarginLoss
        loss_fn = TripletMarginLoss(margin=0.3)
        anchor = torch.randn(4, 64)
        positive = anchor + 0.1 * torch.randn(4, 64)  # similar
        negative = torch.randn(4, 64)  # random
        loss = loss_fn(anchor, positive, negative)
        assert loss.item() >= 0
