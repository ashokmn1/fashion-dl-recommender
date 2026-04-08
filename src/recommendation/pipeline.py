"""
End-to-End Recommendation Pipeline
=====================================
Orchestrates the full recommendation flow from query to outfit.
"""

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.models.visual_encoder import VisualEncoder
from src.models.text_encoder import TextEncoder
from src.models.attribute_encoder import AttributeEncoder
from src.models.fusion import MultimodalFusion
from src.models.compatibility import TypeAwareCompatibilityModel
from src.retrieval.faiss_index import FashionFAISSIndex
from src.recommendation.outfit_generator import OutfitGenerator
from src.personalization.user_profile import UserProfileBuilder


class RecommendationPipeline:
    """Full recommendation pipeline: query item → complete outfits.

    Orchestrates:
        1. Feature extraction (visual + text + attributes)
        2. Embedding computation
        3. FAISS retrieval
        4. Compatibility scoring
        5. Personalized re-ranking
        6. Outfit generation

    Args:
        data_dir: Path to processed dataset.
        model_dir: Path to saved model checkpoints.
        device: Torch device.
    """

    def __init__(
        self,
        data_dir: str = "data/processed",
        model_dir: str = "checkpoints",
        device: str = "cpu",
        config: Optional[dict] = None,
    ):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.device = torch.device(device)
        self.config = config or {}

        self.item_metadata = {}
        self.item_embeddings = {}
        self.faiss_index = None
        self.outfit_generator = None
        self.user_profile_builder = None
        self._initialized = False

    def initialize(self):
        """Load data, models, and build indices."""
        print("Initializing recommendation pipeline...")
        start = time.time()

        # Load item metadata
        with open(self.data_dir / "items.json") as f:
            items = json.load(f)
        self.item_metadata = {item["item_id"]: item for item in items}
        print(f"  Loaded {len(items)} items")

        # Load or compute embeddings
        emb_path = self.data_dir / "embeddings.npy"
        ids_path = self.data_dir / "embedding_ids.npy"

        if emb_path.exists() and ids_path.exists():
            embeddings = np.load(emb_path)
            item_ids = np.load(ids_path).tolist()
            print(f"  Loaded cached embeddings: {embeddings.shape}")
        else:
            print("  Computing embeddings (this may take a while)...")
            embeddings, item_ids = self._compute_all_embeddings(items)
            np.save(emb_path, embeddings)
            np.save(ids_path, np.array(item_ids))
            print(f"  Computed and cached embeddings: {embeddings.shape}")

        # Build item embedding lookup
        for i, iid in enumerate(item_ids):
            self.item_embeddings[iid] = embeddings[i]

        # Build FAISS index
        categories = [self.item_metadata[iid]["category"] for iid in item_ids]
        self.faiss_index = FashionFAISSIndex(
            embedding_dim=embeddings.shape[1],
            index_type="flat",  # Use flat for smaller datasets
        )
        self.faiss_index.build_index(embeddings, item_ids, categories)

        # Initialize outfit generator
        self.outfit_generator = OutfitGenerator(
            faiss_index=self.faiss_index,
            compatibility_scorer=None,  # Uses cosine similarity fallback
            item_embeddings=self.item_embeddings,
            item_metadata=self.item_metadata,
            beam_width=self.config.get("beam_width", 5),
            diversity_lambda=self.config.get("diversity_lambda", 0.3),
        )

        # Initialize user profile builder
        self.user_profile_builder = UserProfileBuilder(
            embedding_dim=embeddings.shape[1],
        )

        # Load user data if available
        users_path = self.data_dir / "users.json"
        interactions_path = self.data_dir / "interactions.json"
        if users_path.exists() and interactions_path.exists():
            with open(users_path) as f:
                users = json.load(f)
            with open(interactions_path) as f:
                interactions = json.load(f)
            self.user_profile_builder.build_all_profiles(
                users, interactions, self.item_embeddings
            )
            print(f"  Built {len(users)} user profiles")

        self._initialized = True
        elapsed = time.time() - start
        print(f"Pipeline initialized in {elapsed:.1f}s")

    def _compute_all_embeddings(
        self, items: list[dict]
    ) -> tuple[np.ndarray, list[int]]:
        """Compute embeddings for all items using a simple feature approach.

        For the MVP, we use a lightweight embedding strategy:
        - Attribute-based features (category, color, material encoded)
        - TF-IDF-like text features from descriptions
        This avoids requiring GPU for the initial setup.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder

        # Text features
        descriptions = [item["description"] for item in items]
        tfidf = TfidfVectorizer(max_features=256, stop_words="english")
        text_features = tfidf.fit_transform(descriptions).toarray()

        # Categorical features
        cat_columns = ["category", "color", "material", "pattern", "season", "occasion", "gender"]
        cat_features = []
        for col in cat_columns:
            le = LabelEncoder()
            encoded = le.fit_transform([item.get(col, "unknown") for item in items])
            n_classes = len(le.classes_)
            one_hot = np.zeros((len(items), n_classes))
            one_hot[np.arange(len(items)), encoded] = 1.0
            cat_features.append(one_hot)

        cat_features = np.hstack(cat_features)

        # Price feature (normalized)
        prices = np.array([item.get("price", 50) for item in items]).reshape(-1, 1)
        prices = (prices - prices.mean()) / (prices.std() + 1e-8)

        # Combine all features
        embeddings = np.hstack([text_features, cat_features, prices]).astype(np.float32)

        # Reduce dimensionality with SVD
        from sklearn.decomposition import TruncatedSVD
        target_dim = min(128, embeddings.shape[1] - 1)
        svd = TruncatedSVD(n_components=target_dim, random_state=42)
        embeddings = svd.fit_transform(embeddings)

        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)

        item_ids = [item["item_id"] for item in items]
        return embeddings, item_ids

    def recommend(
        self,
        item_id: int,
        user_id: Optional[int] = None,
        num_outfits: int = 3,
    ) -> dict:
        """Generate outfit recommendations for a query item.

        Args:
            item_id: Query item ID.
            user_id: Optional user ID for personalization.
            num_outfits: Number of outfits to return.

        Returns:
            Recommendation response dict.
        """
        if not self._initialized:
            self.initialize()

        start = time.time()

        if item_id not in self.item_metadata:
            return {"error": f"Item {item_id} not found", "outfits": []}

        # Get user profile for personalization
        user_profile = None
        if user_id is not None:
            user_profile = self.user_profile_builder.get_profile(user_id)

        # Generate outfits
        outfits = self.outfit_generator.generate(
            query_item_id=item_id,
            num_outfits=num_outfits,
            user_profile=user_profile,
        )

        elapsed = time.time() - start

        return {
            "query_item": self.item_metadata[item_id],
            "outfits": outfits,
            "num_results": len(outfits),
            "latency_ms": round(elapsed * 1000, 1),
            "personalized": user_id is not None,
        }

    def get_item(self, item_id: int) -> Optional[dict]:
        """Get item metadata by ID."""
        return self.item_metadata.get(item_id)

    def search_items(
        self, category: Optional[str] = None, color: Optional[str] = None, limit: int = 20
    ) -> list[dict]:
        """Search items by category and/or color."""
        results = []
        for item in self.item_metadata.values():
            if category and item["category"] != category:
                continue
            if color and item["color"] != color:
                continue
            results.append(item)
            if len(results) >= limit:
                break
        return results
