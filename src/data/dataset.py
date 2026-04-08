"""
Fashion Dataset Classes
========================
PyTorch datasets for outfit compatibility training and recommendation.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FashionItemDataset(Dataset):
    """Dataset for individual fashion items with multimodal features."""

    def __init__(
        self,
        data_dir: str,
        transform: Optional[transforms.Compose] = None,
        split: str = "train",
    ):
        self.data_dir = Path(data_dir)
        self.split = split

        # Load items
        with open(self.data_dir / "items.json") as f:
            self.items = json.load(f)

        self.item_by_id = {item["item_id"]: item for item in self.items}

        # Default image transform
        self.transform = transform or self._default_transform(split)

        # Category mapping
        self.category_map = {"top": 0, "bottom": 1, "shoes": 2, "accessory": 3}
        self.color_map = {
            c: i for i, c in enumerate([
                "black", "white", "navy", "red", "blue", "green",
                "beige", "grey", "brown", "pink", "olive", "burgundy"
            ])
        }
        self.material_map = {
            m: i for i, m in enumerate([
                "cotton", "polyester", "denim", "leather",
                "silk", "wool", "linen", "synthetic"
            ])
        }
        self.pattern_map = {
            p: i for i, p in enumerate([
                "solid", "striped", "plaid", "floral", "graphic", "abstract"
            ])
        }
        self.season_map = {"spring": 0, "summer": 1, "fall": 2, "winter": 3}

    @staticmethod
    def _default_transform(split: str) -> transforms.Compose:
        if split == "train":
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self) -> int:
        return len(self.items)

    def get_item_by_id(self, item_id: int) -> dict:
        return self.item_by_id[item_id]

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]

        # Load and transform image
        img_path = self.data_dir / item["image_path"]
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            image = Image.new("RGB", (224, 224), (128, 128, 128))

        if self.transform:
            image = self.transform(image)

        # Encode attributes
        attributes = torch.tensor([
            self.category_map.get(item["category"], 0),
            self.color_map.get(item["color"], 0),
            self.material_map.get(item["material"], 0),
            self.pattern_map.get(item["pattern"], 0),
            self.season_map.get(item["season"], 0),
        ], dtype=torch.long)

        return {
            "item_id": item["item_id"],
            "image": image,
            "category": self.category_map[item["category"]],
            "attributes": attributes,
            "description": item["description"],
            "color": item["color"],
            "price": item["price"],
        }


class CompatibilityPairDataset(Dataset):
    """Dataset for outfit compatibility pair training (BPR loss)."""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        num_negatives: int = 4,
    ):
        self.data_dir = Path(data_dir)
        self.num_negatives = num_negatives

        # Load items and outfits
        with open(self.data_dir / "items.json") as f:
            self.items = json.load(f)
        self.item_by_id = {item["item_id"]: item for item in self.items}

        # Load outfit split
        with open(self.data_dir / f"outfits_{split}.json") as f:
            self.outfits = json.load(f)

        # Build compatibility pairs from outfits
        self.positive_pairs = []
        for outfit in self.outfits:
            ids = outfit["item_ids"]
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    self.positive_pairs.append((ids[i], ids[j]))

        # Build category index for negative sampling
        self.items_by_category = {}
        for item in self.items:
            cat = item["category"]
            self.items_by_category.setdefault(cat, []).append(item["item_id"])

        # Build outfit membership set for hard negative filtering
        self.outfit_pairs = set()
        for outfit in self.outfits:
            ids = outfit["item_ids"]
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    self.outfit_pairs.add((min(ids[i], ids[j]), max(ids[i], ids[j])))

    def __len__(self) -> int:
        return len(self.positive_pairs)

    def _sample_negative(self, anchor_id: int, positive_id: int) -> int:
        """Sample a negative item from the same category as the positive."""
        pos_cat = self.item_by_id[positive_id]["category"]
        candidates = self.items_by_category[pos_cat]

        for _ in range(100):  # max attempts
            neg_id = candidates[np.random.randint(len(candidates))]
            pair_key = (min(anchor_id, neg_id), max(anchor_id, neg_id))
            if neg_id != positive_id and pair_key not in self.outfit_pairs:
                return neg_id
        return candidates[0]  # fallback

    def __getitem__(self, idx: int) -> dict:
        anchor_id, positive_id = self.positive_pairs[idx]

        # Sample negatives
        negative_ids = [
            self._sample_negative(anchor_id, positive_id)
            for _ in range(self.num_negatives)
        ]

        return {
            "anchor_id": anchor_id,
            "positive_id": positive_id,
            "negative_ids": negative_ids,
            "anchor_category": self.item_by_id[anchor_id]["category"],
            "positive_category": self.item_by_id[positive_id]["category"],
        }


class OutfitDataset(Dataset):
    """Dataset for full outfit evaluation (FITB task)."""

    def __init__(self, data_dir: str, split: str = "test", num_choices: int = 4):
        self.data_dir = Path(data_dir)
        self.num_choices = num_choices

        with open(self.data_dir / "items.json") as f:
            self.items = json.load(f)
        self.item_by_id = {item["item_id"]: item for item in self.items}

        with open(self.data_dir / f"outfits_{split}.json") as f:
            self.outfits = json.load(f)

        self.items_by_category = {}
        for item in self.items:
            self.items_by_category.setdefault(item["category"], []).append(item["item_id"])

    def __len__(self) -> int:
        return len(self.outfits)

    def __getitem__(self, idx: int) -> dict:
        """Generate a FITB question: given partial outfit, pick the correct missing item."""
        outfit = self.outfits[idx]
        item_ids = outfit["item_ids"]

        # Remove one item (the answer)
        remove_idx = np.random.randint(len(item_ids))
        answer_id = item_ids[remove_idx]
        context_ids = [iid for i, iid in enumerate(item_ids) if i != remove_idx]

        # Generate wrong choices from same category
        answer_cat = self.item_by_id[answer_id]["category"]
        candidates = self.items_by_category[answer_cat]

        wrong_choices = []
        for _ in range(self.num_choices - 1):
            neg = candidates[np.random.randint(len(candidates))]
            while neg == answer_id or neg in wrong_choices:
                neg = candidates[np.random.randint(len(candidates))]
            wrong_choices.append(neg)

        # Shuffle choices
        choices = [answer_id] + wrong_choices
        np.random.shuffle(choices)
        answer_position = choices.index(answer_id)

        return {
            "outfit_id": outfit["outfit_id"],
            "context_ids": context_ids,
            "choices": choices,
            "answer_position": answer_position,
            "answer_id": answer_id,
        }
