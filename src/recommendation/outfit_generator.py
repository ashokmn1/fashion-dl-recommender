"""
Complete-the-Look Outfit Generator
====================================
Given a query item, generates complete outfit recommendations
using beam search with compatibility scoring and diversity.
"""

from typing import Optional

import numpy as np


class OutfitGenerator:
    """Generate complete outfit recommendations from a query item.

    Algorithm:
        1. Get query item embedding & category
        2. Determine missing categories
        3. For each missing category, retrieve top-K candidates via FAISS
        4. Beam search across categories for globally optimal outfit
        5. Apply MMR diversity re-ranking
        6. Return top-N complete outfits

    Args:
        faiss_index: FashionFAISSIndex instance.
        compatibility_model: Trained compatibility scorer.
        item_embeddings: Dict[item_id, embedding_vector].
        item_metadata: Dict[item_id, item_dict].
        beam_width: Beam search width.
        diversity_lambda: MMR diversity parameter (0=pure relevance, 1=pure diversity).
    """

    ALL_CATEGORIES = ["top", "bottom", "shoes", "accessory"]

    def __init__(
        self,
        faiss_index,
        compatibility_scorer,
        item_embeddings: dict[int, np.ndarray],
        item_metadata: dict[int, dict],
        beam_width: int = 5,
        diversity_lambda: float = 0.3,
        top_k_candidates: int = 50,
    ):
        self.faiss_index = faiss_index
        self.compatibility_scorer = compatibility_scorer
        self.item_embeddings = item_embeddings
        self.item_metadata = item_metadata
        self.beam_width = beam_width
        self.diversity_lambda = diversity_lambda
        self.top_k = top_k_candidates

    def _get_missing_categories(self, query_category: str) -> list[str]:
        """Determine which categories need to be filled."""
        return [c for c in self.ALL_CATEGORIES if c != query_category]

    def _score_compatibility(
        self, item_a_id: int, item_b_id: int
    ) -> float:
        """Score compatibility between two items."""
        emb_a = self.item_embeddings.get(item_a_id)
        emb_b = self.item_embeddings.get(item_b_id)

        if emb_a is None or emb_b is None:
            return 0.0

        # Cosine similarity as compatibility proxy
        sim = np.dot(emb_a, emb_b) / (
            np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-8
        )
        return float(sim)

    def _score_outfit(self, item_ids: list[int]) -> float:
        """Score overall outfit compatibility (average pairwise)."""
        if len(item_ids) < 2:
            return 0.0

        scores = []
        for i in range(len(item_ids)):
            for j in range(i + 1, len(item_ids)):
                scores.append(self._score_compatibility(item_ids[i], item_ids[j]))

        return np.mean(scores) if scores else 0.0

    def _mmr_diversify(
        self,
        candidates: list[dict],
        selected: list[dict],
        lambda_param: float,
    ) -> list[dict]:
        """Apply Maximal Marginal Relevance for diversity.

        Balances relevance (compatibility score) with diversity
        (dissimilarity to already-selected items).
        """
        if not candidates or not selected:
            return candidates

        remaining = list(candidates)
        result = []

        while remaining and len(result) < len(candidates):
            best_score = -float("inf")
            best_idx = 0

            for i, cand in enumerate(remaining):
                # Relevance score
                relevance = cand.get("score", 0)

                # Diversity: max similarity to already selected
                max_sim = 0.0
                cand_emb = self.item_embeddings.get(cand["item_id"])
                if cand_emb is not None:
                    for sel in selected + result:
                        sel_emb = self.item_embeddings.get(sel["item_id"])
                        if sel_emb is not None:
                            sim = np.dot(cand_emb, sel_emb) / (
                                np.linalg.norm(cand_emb) * np.linalg.norm(sel_emb) + 1e-8
                            )
                            max_sim = max(max_sim, sim)

                # MMR score
                mmr = (1 - lambda_param) * relevance - lambda_param * max_sim
                if mmr > best_score:
                    best_score = mmr
                    best_idx = i

            result.append(remaining.pop(best_idx))

        return result

    def generate(
        self,
        query_item_id: int,
        num_outfits: int = 3,
        user_profile: Optional[np.ndarray] = None,
        personalization_alpha: float = 0.2,
    ) -> list[dict]:
        """Generate complete outfit recommendations for a query item.

        Args:
            query_item_id: The item to build an outfit around.
            num_outfits: Number of outfit recommendations to return.
            user_profile: Optional user preference vector for personalization.
            personalization_alpha: Weight for personalization (0-1).

        Returns:
            List of outfit dicts with items and scores.
        """
        query_meta = self.item_metadata.get(query_item_id)
        if query_meta is None:
            return []

        query_emb = self.item_embeddings.get(query_item_id)
        if query_emb is None:
            return []

        query_category = query_meta["category"]
        missing_categories = self._get_missing_categories(query_category)

        # Step 1: Retrieve candidates for each missing category
        category_candidates = {}
        for cat in missing_categories:
            results = self.faiss_index.search(
                query_emb,
                top_k=self.top_k,
                category_filter=cat,
                exclude_ids={query_item_id},
            )

            # Score candidates
            for r in results:
                compat_score = self._score_compatibility(query_item_id, r["item_id"])

                # Add personalization bonus
                if user_profile is not None:
                    item_emb = self.item_embeddings.get(r["item_id"])
                    if item_emb is not None:
                        personal_score = np.dot(user_profile, item_emb) / (
                            np.linalg.norm(user_profile) * np.linalg.norm(item_emb) + 1e-8
                        )
                        r["score"] = (
                            (1 - personalization_alpha) * compat_score
                            + personalization_alpha * float(personal_score)
                        )
                    else:
                        r["score"] = compat_score
                else:
                    r["score"] = compat_score

            category_candidates[cat] = sorted(results, key=lambda x: -x["score"])

        # Step 2: Beam search for optimal outfits
        # Initialize beams with query item
        beams = [{"items": [query_item_id], "score": 0.0, "categories_filled": {query_category}}]

        for cat in missing_categories:
            candidates = category_candidates.get(cat, [])
            if not candidates:
                continue

            # Apply diversity
            new_beams = []
            for beam in beams:
                top_candidates = candidates[: self.beam_width * 2]

                for cand in top_candidates:
                    new_items = beam["items"] + [cand["item_id"]]
                    outfit_score = self._score_outfit(new_items)

                    new_beams.append({
                        "items": new_items,
                        "score": outfit_score,
                        "categories_filled": beam["categories_filled"] | {cat},
                    })

            # Keep top beam_width beams
            new_beams.sort(key=lambda x: -x["score"])
            beams = new_beams[: self.beam_width]

        # Step 3: Format results
        outfits = []
        seen_combinations = set()

        for beam in sorted(beams, key=lambda x: -x["score"]):
            combo_key = tuple(sorted(beam["items"]))
            if combo_key in seen_combinations:
                continue
            seen_combinations.add(combo_key)

            outfit_items = []
            for item_id in beam["items"]:
                meta = self.item_metadata.get(item_id, {})
                outfit_items.append({
                    "item_id": item_id,
                    "category": meta.get("category", "unknown"),
                    "subcategory": meta.get("subcategory", "unknown"),
                    "color": meta.get("color", "unknown"),
                    "description": meta.get("description", ""),
                    "price": meta.get("price", 0),
                    "image_path": meta.get("image_path", ""),
                })

            outfits.append({
                "items": outfit_items,
                "compatibility_score": round(beam["score"], 4),
                "num_items": len(beam["items"]),
                "style_tags": self._infer_style_tags(beam["items"]),
            })

            if len(outfits) >= num_outfits:
                break

        return outfits

    def _infer_style_tags(self, item_ids: list[int]) -> list[str]:
        """Infer style tags from outfit items."""
        tags = set()
        occasions = set()
        seasons = set()

        for iid in item_ids:
            meta = self.item_metadata.get(iid, {})
            occasions.add(meta.get("occasion", ""))
            seasons.add(meta.get("season", ""))

        tags.update(occasions - {""})
        tags.update(seasons - {""})
        return list(tags)[:5]
