"""
Training Script
=================
Train the compatibility model end-to-end.

Usage:
    python scripts/train.py --data_dir data/processed --epochs 30
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import CompatibilityPairDataset
from src.models.compatibility import TypeAwareCompatibilityModel
from src.training.trainer import CompatibilityTrainer
from src.recommendation.pipeline import RecommendationPipeline


def collate_fn(batch):
    """Custom collate for compatibility pairs."""
    return {
        "anchor_id": torch.tensor([b["anchor_id"] for b in batch]),
        "positive_id": torch.tensor([b["positive_id"] for b in batch]),
        "negative_ids": torch.tensor([b["negative_ids"] for b in batch]),
        "anchor_category": [b["anchor_category"] for b in batch],
        "positive_category": [b["positive_category"] for b in batch],
    }


def main():
    parser = argparse.ArgumentParser(description="Train compatibility model")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--num_negatives", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print("=" * 60)
    print("Fashion Compatibility Model Training")
    print("=" * 60)

    # Step 1: Initialize pipeline to compute embeddings
    print("\n[1/4] Initializing pipeline and computing embeddings...")
    pipeline = RecommendationPipeline(data_dir=args.data_dir)
    pipeline.initialize()

    # Step 2: Create datasets
    print("\n[2/4] Creating datasets...")
    train_dataset = CompatibilityPairDataset(
        data_dir=args.data_dir, split="train", num_negatives=args.num_negatives
    )
    val_dataset = CompatibilityPairDataset(
        data_dir=args.data_dir, split="val", num_negatives=args.num_negatives
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )

    print(f"  Train: {len(train_dataset)} pairs")
    print(f"  Val:   {len(val_dataset)} pairs")

    # Step 3: Build model
    print("\n[3/4] Building model...")
    actual_dim = next(iter(pipeline.item_embeddings.values())).shape[0]
    print(f"  Embedding dimension: {actual_dim}")

    model = TypeAwareCompatibilityModel(
        embedding_dim=actual_dim,
        num_categories=4,
        hidden_dim=256,
    )

    # Category mapping
    category_map = {"top": 0, "bottom": 1, "shoes": 2, "accessory": 3}

    def embedding_lookup(item_id):
        emb = pipeline.item_embeddings.get(item_id)
        if emb is not None:
            return emb
        return np.zeros(actual_dim)

    def category_lookup(item_id):
        meta = pipeline.item_metadata.get(item_id, {})
        return category_map.get(meta.get("category", "top"), 0)

    # Step 4: Train
    print("\n[4/4] Training...")
    trainer = CompatibilityTrainer(
        model=model,
        embedding_lookup=embedding_lookup,
        category_lookup=category_lookup,
        device=args.device,
        config={
            "learning_rate": args.lr,
            "num_epochs": args.epochs,
            "early_stopping_patience": 5,
            "mixed_precision": args.device != "cpu",
        },
    )

    history = trainer.train(train_loader, val_loader, args.checkpoint_dir)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
