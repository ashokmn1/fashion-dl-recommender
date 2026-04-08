"""
Text Feature Encoder
=====================
Sentence-BERT based encoder for fashion product descriptions.
Falls back to a lightweight TF-IDF + MLP encoder when sentence-transformers is unavailable.
"""

import torch
import torch.nn as nn
import numpy as np


class TextEncoder(nn.Module):
    """Text encoder using Sentence-BERT or fallback TF-IDF embeddings.

    Architecture:
        Option A: Sentence-BERT → projection MLP
        Option B: TF-IDF → MLP (lightweight fallback)
        Output: [batch_size, embedding_dim]

    Args:
        embedding_dim: Output embedding dimension.
        model_name: Sentence-BERT model name.
        max_length: Maximum token length for descriptions.
        use_sbert: Try to use Sentence-BERT (falls back to TF-IDF if unavailable).
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        model_name: str = "all-MiniLM-L6-v2",
        max_length: int = 128,
        use_sbert: bool = True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.sbert_model = None
        self._use_sbert = False

        if use_sbert:
            try:
                from sentence_transformers import SentenceTransformer
                self.sbert_model = SentenceTransformer(model_name)
                self._sbert_dim = self.sbert_model.get_sentence_embedding_dimension()
                self._use_sbert = True
                # Projection if dimensions don't match
                if self._sbert_dim != embedding_dim:
                    self.projection = nn.Sequential(
                        nn.Linear(self._sbert_dim, embedding_dim),
                        nn.LayerNorm(embedding_dim),
                        nn.ReLU(inplace=True),
                    )
                else:
                    self.projection = nn.Identity()
            except ImportError:
                print("sentence-transformers not available, using TF-IDF fallback")

        if not self._use_sbert:
            # Fallback: simple word embedding bag
            self.vocab_size = 10000
            self.word_embedding = nn.EmbeddingBag(self.vocab_size, embedding_dim, mode="mean")
            self.projection = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.ReLU(inplace=True),
            )

    def _simple_tokenize(self, texts: list[str]) -> torch.Tensor:
        """Simple hash-based tokenizer for fallback mode."""
        batch_indices = []
        batch_offsets = [0]

        for text in texts:
            words = text.lower().split()[:self.max_length]
            indices = [hash(w) % self.vocab_size for w in words]
            if not indices:
                indices = [0]
            batch_indices.extend(indices)
            batch_offsets.append(len(batch_indices))

        return (
            torch.tensor(batch_indices, dtype=torch.long),
            torch.tensor(batch_offsets[:-1], dtype=torch.long),
        )

    def forward(self, texts: list[str]) -> torch.Tensor:
        """Encode text descriptions into embeddings.

        Args:
            texts: List of product description strings.

        Returns:
            [batch_size, embedding_dim] text embeddings.
        """
        if self._use_sbert:
            # Use Sentence-BERT
            with torch.no_grad():
                sbert_embs = self.sbert_model.encode(
                    texts,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                )
            return self.projection(sbert_embs)
        else:
            # Fallback tokenization
            indices, offsets = self._simple_tokenize(texts)
            device = next(self.parameters()).device
            indices = indices.to(device)
            offsets = offsets.to(device)
            embs = self.word_embedding(indices, offsets)
            return self.projection(embs)

    @torch.no_grad()
    def encode_batch(self, texts: list[str], batch_size: int = 64) -> torch.Tensor:
        """Encode texts in batches for large-scale feature extraction."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embs = self.forward(batch)
            all_embeddings.append(embs.cpu())
        return torch.cat(all_embeddings, dim=0)
