"""
Evaluation Script
==================
Run full evaluation suite on the trained model.

Usage:
    python scripts/evaluate.py --data_dir data/processed
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.recommendation.pipeline import RecommendationPipeline
from src.evaluation.metrics import (
    evaluate_pipeline,
    outfit_coherence_score,
    diversity_score,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate recommendation system")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print("=" * 60)
    print("Fashion Recommendation System Evaluation")
    print("=" * 60)

    # Initialize pipeline
    print("\n[1/3] Initializing pipeline...")
    pipeline = RecommendationPipeline(data_dir=args.data_dir)
    pipeline.initialize()

    # Load test outfits
    print("\n[2/3] Loading test data...")
    with open(data_dir / "outfits_test.json") as f:
        test_outfits = json.load(f)
    print(f"  Test outfits: {len(test_outfits)}")

    # Run evaluation
    print("\n[3/3] Running evaluation...")
    metrics = evaluate_pipeline(
        pipeline, test_outfits, pipeline.item_embeddings
    )

    # Run recommendation quality evaluation
    print("\nGenerating sample recommendations for quality check...")
    rec_coherences = []
    rec_diversities = []
    latencies = []

    # Sample items for evaluation
    sample_items = list(pipeline.item_metadata.keys())[:100]

    for item_id in sample_items:
        result = pipeline.recommend(item_id, num_outfits=3)
        latencies.append(result.get("latency_ms", 0))

        for outfit in result.get("outfits", []):
            # Get outfit embeddings
            outfit_embs = []
            for item in outfit["items"]:
                emb = pipeline.item_embeddings.get(item["item_id"])
                if emb is not None:
                    outfit_embs.append(emb)
            if len(outfit_embs) >= 2:
                rec_coherences.append(outfit_coherence_score(outfit_embs))

        # Diversity across outfits
        all_outfit_embs = []
        for outfit in result.get("outfits", []):
            o_embs = [
                pipeline.item_embeddings[item["item_id"]]
                for item in outfit["items"]
                if item["item_id"] in pipeline.item_embeddings
            ]
            if o_embs:
                all_outfit_embs.append(o_embs)
        if len(all_outfit_embs) >= 2:
            rec_diversities.append(diversity_score(all_outfit_embs))

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\n--- Offline Metrics ---")
    print(f"  FITB Accuracy:           {metrics['fitb_accuracy']:.4f} ({metrics['fitb_total']} questions)")
    print(f"  Avg Outfit Coherence:    {metrics['avg_coherence']:.4f}")

    print(f"\n--- Recommendation Quality ---")
    if rec_coherences:
        print(f"  Avg Rec Coherence:       {np.mean(rec_coherences):.4f}")
    if rec_diversities:
        print(f"  Avg Rec Diversity:       {np.mean(rec_diversities):.4f}")

    print(f"\n--- Latency ---")
    if latencies:
        print(f"  P50 Latency:             {np.percentile(latencies, 50):.1f} ms")
        print(f"  P95 Latency:             {np.percentile(latencies, 95):.1f} ms")
        print(f"  P99 Latency:             {np.percentile(latencies, 99):.1f} ms")

    # Sample recommendation
    print(f"\n--- Sample Recommendation ---")
    sample_id = sample_items[0]
    result = pipeline.recommend(sample_id, num_outfits=1)
    query = result["query_item"]
    print(f"  Query: {query['subcategory']} ({query['color']} {query['material']})")
    if result["outfits"]:
        outfit = result["outfits"][0]
        print(f"  Outfit (score={outfit['compatibility_score']}):")
        for item in outfit["items"]:
            print(f"    - {item['category']:10s} | {item['subcategory']:18s} | {item['color']}")

    # Save results
    results_file = data_dir / "evaluation_results.json"
    all_results = {
        **metrics,
        "avg_rec_coherence": float(np.mean(rec_coherences)) if rec_coherences else 0,
        "avg_rec_diversity": float(np.mean(rec_diversities)) if rec_diversities else 0,
        "p50_latency_ms": float(np.percentile(latencies, 50)) if latencies else 0,
        "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0,
    }
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
