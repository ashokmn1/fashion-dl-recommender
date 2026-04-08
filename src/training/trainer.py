"""
Training Loop
===============
Compatibility model trainer with early stopping, mixed precision,
and experiment tracking.
"""

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from src.training.losses import BPRLoss


class CompatibilityTrainer:
    """Trainer for the outfit compatibility model.

    Features:
        - Mixed precision training (AMP)
        - Early stopping on validation loss
        - Learning rate scheduling
        - Experiment logging

    Args:
        model: The compatibility model to train.
        embedding_lookup: Callable(item_id) -> embedding tensor.
        category_lookup: Callable(item_id) -> category_index.
        device: Torch device.
        config: Training configuration dict.
    """

    def __init__(
        self,
        model: nn.Module,
        embedding_lookup: callable,
        category_lookup: callable,
        device: str = "cpu",
        config: Optional[dict] = None,
    ):
        self.model = model.to(device)
        self.embedding_lookup = embedding_lookup
        self.category_lookup = category_lookup
        self.device = torch.device(device)
        self.config = config or {}

        # Training params
        self.lr = self.config.get("learning_rate", 5e-4)
        self.weight_decay = self.config.get("weight_decay", 1e-4)
        self.num_epochs = self.config.get("num_epochs", 50)
        self.patience = self.config.get("early_stopping_patience", 5)
        self.use_amp = self.config.get("mixed_precision", False) and device != "cpu"

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.num_epochs, eta_min=self.lr * 0.01
        )

        # Mixed precision
        self.scaler = GradScaler(enabled=self.use_amp)
        self.loss_fn = BPRLoss()

        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def _get_batch_embeddings(self, item_ids: list) -> torch.Tensor:
        """Look up embeddings for a batch of item IDs."""
        embeddings = []
        for iid in item_ids:
            emb = self.embedding_lookup(iid)
            if isinstance(emb, np.ndarray):
                emb = torch.from_numpy(emb).float()
            embeddings.append(emb)
        return torch.stack(embeddings).to(self.device)

    def train_epoch(self, train_loader: DataLoader) -> dict:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        total_margin = 0.0
        num_batches = 0

        for batch in train_loader:
            anchor_ids = batch["anchor_id"]
            positive_ids = batch["positive_id"]
            negative_id_lists = batch["negative_ids"]  # [B, num_negatives]

            # Get embeddings
            anchor_embs = self._get_batch_embeddings(anchor_ids.tolist())
            positive_embs = self._get_batch_embeddings(positive_ids.tolist())

            negative_embs_list = []
            for neg_idx in range(negative_id_lists.shape[1]):
                neg_embs = self._get_batch_embeddings(negative_id_lists[:, neg_idx].tolist())
                negative_embs_list.append(neg_embs)

            # Get category indices (use first item's category for batch)
            anchor_cat = self.category_lookup(anchor_ids[0].item())
            pos_cat = self.category_lookup(positive_ids[0].item())

            # Forward pass
            self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                result = self.model(
                    anchor_embs, positive_embs, negative_embs_list,
                    anchor_cat, pos_cat
                )
                loss = result["loss"]

            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            total_margin += result["margin"].item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_margin = total_margin / max(num_batches, 1)

        return {"loss": avg_loss, "margin": avg_margin}

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> dict:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        total_pos_score = 0.0
        total_neg_score = 0.0
        num_batches = 0

        for batch in val_loader:
            anchor_ids = batch["anchor_id"]
            positive_ids = batch["positive_id"]
            negative_id_lists = batch["negative_ids"]

            anchor_embs = self._get_batch_embeddings(anchor_ids.tolist())
            positive_embs = self._get_batch_embeddings(positive_ids.tolist())

            negative_embs_list = []
            for neg_idx in range(negative_id_lists.shape[1]):
                neg_embs = self._get_batch_embeddings(negative_id_lists[:, neg_idx].tolist())
                negative_embs_list.append(neg_embs)

            anchor_cat = self.category_lookup(anchor_ids[0].item())
            pos_cat = self.category_lookup(positive_ids[0].item())

            result = self.model(
                anchor_embs, positive_embs, negative_embs_list,
                anchor_cat, pos_cat
            )

            total_loss += result["loss"].item()
            total_pos_score += result["pos_score"].item()
            total_neg_score += result["neg_score"].item()
            num_batches += 1

        n = max(num_batches, 1)
        return {
            "loss": total_loss / n,
            "pos_score": total_pos_score / n,
            "neg_score": total_neg_score / n,
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_dir: str = "checkpoints",
    ) -> dict:
        """Full training loop with early stopping.

        Returns:
            Training history dict.
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        history = {"train_loss": [], "val_loss": [], "margin": []}
        print(f"\nTraining for up to {self.num_epochs} epochs (patience={self.patience})")
        print("-" * 60)

        for epoch in range(self.num_epochs):
            start = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader)
            # Validate
            val_metrics = self.validate(val_loader)

            # Step scheduler
            self.scheduler.step()

            elapsed = time.time() - start

            # Track history
            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["margin"].append(train_metrics["margin"])

            print(
                f"Epoch {epoch+1:3d}/{self.num_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Margin: {train_metrics['margin']:.4f} | "
                f"LR: {self.scheduler.get_last_lr()[0]:.6f} | "
                f"Time: {elapsed:.1f}s"
            )

            # Early stopping check
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.patience_counter = 0
                # Save best model
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_metrics["loss"],
                }, checkpoint_dir / "best_model.pt")
                print(f"  ✓ New best model saved (val_loss={val_metrics['loss']:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break

        # Save training history
        with open(checkpoint_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        print(f"\nTraining complete. Best val loss: {self.best_val_loss:.4f}")
        return history
