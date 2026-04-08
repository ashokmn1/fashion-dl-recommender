"""Tests for data pipeline and datasets."""

import json
import sys
from pathlib import Path

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_data_dir(tmp_path):
    """Create minimal test dataset."""
    items = [
        {"item_id": i, "category": cat, "subcategory": "test",
         "color": "black", "material": "cotton", "pattern": "solid",
         "season": "summer", "gender": "unisex", "occasion": "casual",
         "price": 50.0, "image_path": f"images/test_{i}.png",
         "description": f"A nice {cat} item for testing"}
        for i, cat in enumerate(["top", "bottom", "shoes", "accessory"] * 5)
    ]

    outfits_train = [
        {"outfit_id": 0, "item_ids": [0, 1, 2, 3], "style": "casual", "season": "summer"},
        {"outfit_id": 1, "item_ids": [4, 5, 6, 7], "style": "casual", "season": "summer"},
    ]
    outfits_val = [
        {"outfit_id": 2, "item_ids": [8, 9, 10, 11], "style": "casual", "season": "summer"},
    ]
    outfits_test = [
        {"outfit_id": 3, "item_ids": [12, 13, 14, 15], "style": "casual", "season": "summer"},
    ]

    with open(tmp_path / "items.json", "w") as f:
        json.dump(items, f)
    with open(tmp_path / "outfits_train.json", "w") as f:
        json.dump(outfits_train, f)
    with open(tmp_path / "outfits_val.json", "w") as f:
        json.dump(outfits_val, f)
    with open(tmp_path / "outfits_test.json", "w") as f:
        json.dump(outfits_test, f)

    # Create dummy images
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    from PIL import Image
    for i in range(20):
        img = Image.new("RGB", (224, 224), (100, 100, 100))
        img.save(img_dir / f"test_{i}.png")

    return tmp_path


class TestFashionItemDataset:
    def test_dataset_length(self, sample_data_dir):
        from src.data.dataset import FashionItemDataset
        ds = FashionItemDataset(str(sample_data_dir), split="train")
        assert len(ds) == 20

    def test_dataset_item_format(self, sample_data_dir):
        from src.data.dataset import FashionItemDataset
        ds = FashionItemDataset(str(sample_data_dir), split="train")
        item = ds[0]
        assert "item_id" in item
        assert "image" in item
        assert "category" in item
        assert "attributes" in item
        assert item["image"].shape == (3, 224, 224)

    def test_category_mapping(self, sample_data_dir):
        from src.data.dataset import FashionItemDataset
        ds = FashionItemDataset(str(sample_data_dir), split="train")
        assert ds.category_map["top"] == 0
        assert ds.category_map["accessory"] == 3


class TestCompatibilityPairDataset:
    def test_pair_dataset(self, sample_data_dir):
        from src.data.dataset import CompatibilityPairDataset
        ds = CompatibilityPairDataset(str(sample_data_dir), split="train", num_negatives=2)
        assert len(ds) > 0

    def test_pair_format(self, sample_data_dir):
        from src.data.dataset import CompatibilityPairDataset
        ds = CompatibilityPairDataset(str(sample_data_dir), split="train", num_negatives=2)
        pair = ds[0]
        assert "anchor_id" in pair
        assert "positive_id" in pair
        assert "negative_ids" in pair
        assert len(pair["negative_ids"]) == 2


class TestOutfitDataset:
    def test_fitb_format(self, sample_data_dir):
        from src.data.dataset import OutfitDataset
        ds = OutfitDataset(str(sample_data_dir), split="test", num_choices=4)
        sample = ds[0]
        assert "context_ids" in sample
        assert "choices" in sample
        assert "answer_position" in sample
        assert len(sample["choices"]) == 4
