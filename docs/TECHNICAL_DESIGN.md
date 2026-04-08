# Technical Design Document: Fashion DL Recommender

## 1. Purpose

This system is a **"Complete the Look"** fashion recommendation engine. Given a single fashion item (e.g., a navy t-shirt), it generates complete outfit suggestions by selecting complementary items across categories (tops, bottoms, shoes, accessories). The recommendations are personalized per user based on interaction history, temporally decayed to reflect evolving preferences.

### Goals

- Produce style-coherent, diverse outfit recommendations in real time
- Learn cross-category compatibility (e.g., which shoes go with which tops) from outfit data
- Personalize results using user interaction signals (views, clicks, purchases)
- Serve recommendations via a REST API with sub-200ms latency

---

## 2. System Architecture

```
                        ┌──────────────────────────────────────┐
                        │          Query Item Input             │
                        └──────────────┬───────────────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              ▼                        ▼                        ▼
   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
   │  Visual Encoder   │   │   Text Encoder    │   │ Attribute Encoder │
   │  (ResNet-50)      │   │ (Sentence-BERT)   │   │ (Learned Embeds)  │
   │  Output: 2048-d   │   │  Output: 384-d    │   │   Output: 64-d    │
   └────────┬──────────┘   └────────┬──────────┘   └────────┬──────────┘
            └────────────────────────┼────────────────────────┘
                                     ▼
                        ┌──────────────────────────┐
                        │   Multimodal Fusion MLP   │
                        │  2496-d → 512-d (L2 norm) │
                        └────────────┬─────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                                  ▼
          ┌──────────────────┐              ┌──────────────────┐
          │   FAISS Index     │              │  Compatibility    │
          │  (IVF256,PQ32)    │              │  Model (BPR)      │
          │  ANN Retrieval    │              │  Type-Aware Heads  │
          └────────┬─────────┘              └────────┬─────────┘
                   │                                  │
                   └──────────────┬───────────────────┘
                                  ▼
                     ┌─────────────────────────┐
                     │  Beam Search (width=5)   │
                     │  + MMR Diversity Rerank   │
                     └────────────┬────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼                            ▼
          ┌──────────────────┐        ┌──────────────────┐
          │  User Profile     │        │  Final Outfit     │
          │  (Temporal Decay)  │──────▶│  Recommendations  │
          └──────────────────┘        └────────┬─────────┘
                                               │
                                               ▼
                                    ┌──────────────────┐
                                    │   FastAPI Server   │
                                    │  REST Endpoints    │
                                    └──────────────────┘
```

---

## 3. Data Pipeline

### 3.1 Synthetic Dataset Generator

**Script:** `scripts/generate_dataset.py`

The generator produces a self-contained dataset with realistic fashion taxonomy and compatibility rules.

| Asset              | Volume  | Details                                                     |
|--------------------|---------|-------------------------------------------------------------|
| Items              | 5,000   | 4 categories: tops (30%), bottoms (25%), shoes (25%), accessories (20%) |
| Outfits            | 1,200   | Built from 6 style profiles (casual, formal, streetwear, summer, winter, sporty) |
| Users              | 500     | Each with 1-3 preferred styles and 3-6 color preferences    |
| Interaction events | 50,000  | Types: view (50%), click (25%), add_to_cart (12%), purchase (8%), save (5%) |
| Compatibility pairs| ~15,000 | Positive pairs from outfits + hard negatives from same category |

**Item metadata fields:** category, subcategory, color, material, pattern, season, gender, occasion, price, description.

**Product images:** 224x224 synthetic PNGs with category-specific silhouettes and color fills.

**Color compatibility matrix:** Enforces realistic color pairings (e.g., navy+beige is valid, red+green is not) when composing outfits.

### 3.2 Train/Validation/Test Split

Splitting is performed at the **outfit level** (80/10/10) to prevent data leakage between sets.

### 3.3 Compatibility Pairs

For each outfit:
- **Positive pairs:** All pairwise item combinations within the outfit
- **Negative pairs:** Replace one item with a random same-category item not in the outfit

These pairs are used to train the BPR compatibility objective.

---

## 4. Feature Extraction (Multimodal Encoders)

### 4.1 Visual Encoder

**Module:** `src/models/visual_encoder.py`

| Property         | Value                              |
|------------------|------------------------------------|
| Backbone         | ResNet-50 (ImageNet pretrained)    |
| Frozen layers    | Blocks 0-2 (configurable)         |
| Fine-tuned       | layer3, layer4                     |
| Projection head  | Linear → BatchNorm → ReLU → Dropout |
| Output dimension | 2048-d                             |

**Purpose:** Extracts visual features capturing silhouette, texture, and color patterns from product images.

### 4.2 Text Encoder

**Module:** `src/models/text_encoder.py`

| Property         | Value                              |
|------------------|------------------------------------|
| Model            | Sentence-BERT (`all-MiniLM-L6-v2`) |
| Fallback         | Word embedding bag (if SBERT unavailable) |
| Projection       | Linear → LayerNorm → ReLU         |
| Output dimension | 384-d                              |

**Purpose:** Encodes product descriptions to capture semantic attributes like material, style cues, and occasion mentions.

### 4.3 Attribute Encoder

**Module:** `src/models/attribute_encoder.py`

| Property           | Value                                               |
|--------------------|-----------------------------------------------------|
| Input attributes   | category (4), color (12), material (8), pattern (6), season (4) |
| Embedding per attr | 64-d each                                           |
| Fusion             | Concatenate (5x64=320) → MLP → 64-d                |
| MLP layers         | Linear(320→64) → LayerNorm → ReLU → Dropout        |
| Output dimension   | 64-d                                                |

**Purpose:** Learns dense representations of discrete fashion properties (color, material, pattern, etc.).

### 4.4 Multimodal Fusion

**Module:** `src/models/fusion.py`

Concatenates the three encoder outputs (2048 + 384 + 64 = 2496-d) and projects them into a unified embedding space:

```
Input (2496-d)
  → Linear(2496→1024) + BatchNorm + ReLU + Dropout(0.3)
  → Linear(1024→512)  + BatchNorm + ReLU + Dropout(0.15)
  → Linear(512→512)
  → L2 Normalization
```

**Output:** 512-d unit-normalized item embedding used for both retrieval and compatibility scoring.

---

## 5. Compatibility Model

**Module:** `src/models/compatibility.py`

### 5.1 Type-Aware Architecture

Instead of a single compatibility function, the model learns **separate projection heads** for each category pair:

| Pair Index | Categories            |
|------------|-----------------------|
| 0          | top ↔ bottom          |
| 1          | top ↔ shoes           |
| 2          | top ↔ accessory       |
| 3          | bottom ↔ shoes        |
| 4          | bottom ↔ accessory    |
| 5          | shoes ↔ accessory     |

Each head is a 3-layer MLP:

```
Concatenate [emb_a, emb_b] (1024-d)
  → Linear(1024→256) + ReLU + Dropout(0.2)
  → Linear(256→128)  + ReLU
  → Linear(128→1)    → compatibility score
```

A **general fallback head** handles any unseen category pairs.

**Rationale:** Different category pairs have fundamentally different compatibility patterns. A formal shirt pairs with trousers through color/formality matching, while shoe-accessory pairing depends more on material/tone coherence.

### 5.2 Outfit Scoring

The overall outfit compatibility score is the **mean of all pairwise compatibility scores** among items in the outfit.

### 5.3 Loss Function: Bayesian Personalized Ranking (BPR)

**Module:** `src/training/losses.py`

```
L_BPR = -log(σ(s_pos - s_neg))
```

Where:
- `s_pos` = compatibility score for a compatible pair
- `s_neg` = compatibility score for an incompatible pair
- `σ` = sigmoid function

For each positive pair, **4 negative items** are sampled from the same category. The loss is averaged across all negatives.

**Why BPR over alternatives:**
- Directly optimizes the ranking objective (compatible > incompatible)
- More robust than pointwise BCE for recommendation tasks
- Naturally handles implicit feedback signals

Additional losses available in `src/training/losses.py`:
- **Triplet loss** with configurable margin
- **Contrastive loss** for embedding separation

---

## 6. Training Pipeline

**Module:** `src/training/trainer.py`  
**Script:** `scripts/train.py`

### 6.1 Training Configuration

| Parameter              | Default   | Purpose                              |
|------------------------|-----------|--------------------------------------|
| Batch size             | 64        | Compatibility pair batches           |
| Learning rate          | 5e-4      | AdamW optimizer                      |
| Weight decay           | 1e-4      | L2 regularization                    |
| Epochs                 | 50        | Maximum training epochs              |
| Early stopping         | 5 epochs  | Patience on validation loss          |
| Gradient clipping      | max_norm=1.0 | Prevents gradient explosion       |
| Mixed precision        | Enabled   | AMP (float16) for speed/memory      |
| LR scheduler           | Cosine annealing | Decays to 1% of initial LR    |

### 6.2 Training Loop

1. Load precomputed item embeddings
2. Create compatibility pair datasets from outfit data (train/val splits)
3. For each epoch:
   - **Forward:** Compute BPR loss on training pairs
   - **Backward:** Gradient scaling (AMP) + clipping + AdamW step
   - **Validate:** Compute loss and positive-negative score margin
   - **Checkpoint:** Save model if validation loss improves
   - **Early stop:** Halt if no improvement for 5 consecutive epochs
4. Save training history (JSON) and best model checkpoint

### 6.3 Monitored Metrics

- **Training loss** (BPR)
- **Validation loss** (BPR)
- **Margin** (mean positive score - mean negative score; should be positive and increasing)
- **Learning rate** (tracks cosine schedule)

---

## 7. Retrieval Engine

**Module:** `src/retrieval/faiss_index.py`

### 7.1 Index Types

| Index Type    | When Used       | Method                                              |
|---------------|-----------------|-----------------------------------------------------|
| Flat          | <1K items       | Brute-force inner product (`IndexFlatIP`)           |
| IVF256,PQ32   | ≥1K items       | Inverted file (256 centroids) + product quantization |

### 7.2 IVF-PQ Details

- **Training:** Clusters 5K item embeddings into 256 Voronoi cells
- **Quantization:** Compresses each 512-d vector to 32 bytes via PQ
- **Search:** Probes 32 clusters (nprobe=32) at query time
- **Complexity:** Sub-linear search, ~10x faster than brute force at 5K items

### 7.3 Category-Filtered Search

When generating outfits, retrieval is filtered by target category. FAISS returns top-100 candidates per category, which are then scored by the compatibility model.

---

## 8. Outfit Generation

**Module:** `src/recommendation/outfit_generator.py`

### 8.1 Beam Search Algorithm

```
Input:  query_item (e.g., item_id=42, category="top")
Output: top-N complete outfits (4 items each)

1. Identify missing categories: [bottom, shoes, accessory]
2. Initialize 5 beams, each containing only the query item

3. For each missing category:
   a. Retrieve top-100 candidates via FAISS (filtered by category)
   b. For each beam × candidate:
      - Compute average pairwise compatibility with existing beam items
      - Apply personalization bonus (if user_id provided)
   c. Keep top-5 expanded beams (pruning)

4. Return top-N beams as complete outfits
```

**Beam width = 5** balances exploration (trying different combinations) with computational cost.

### 8.2 MMR Diversity Re-ranking

**Maximal Marginal Relevance** prevents recommending N nearly identical outfits:

```
MMR(item) = (1 - λ) * relevance(item) - λ * max_similarity(item, selected)
```

Where `λ = 0.3` (configurable). Higher λ pushes toward more diverse selections within each category's candidate pool.

### 8.3 Personalization Blending

When a user_id is provided:

```
final_score = (1 - α) * compatibility_score + α * user_preference_score
```

Where `α = 0.2`. The user preference score is the cosine similarity between the candidate item embedding and the user's profile vector.

---

## 9. User Personalization

**Module:** `src/personalization/user_profile.py`

### 9.1 Profile Construction

User profiles are built as **weighted averages of interacted item embeddings**:

```
profile_vector = Σ(weight_i * embedding_i) / Σ(weight_i)
```

Weights combine two factors:

**Interaction type weights:**

| Type        | Weight |
|-------------|--------|
| View        | 1.0    |
| Click       | 2.0    |
| Add to cart | 4.0    |
| Save        | 5.0    |
| Purchase    | 8.0    |

**Temporal decay:**

```
decay(t) = exp(-ln(2) * days_since_interaction / half_life)
```

Default half-life = 30 days. An interaction from 30 days ago gets half the weight of today's interaction. This reflects evolving user taste.

### 9.2 Profile Output

The final profile is L2-normalized to a unit vector in the same 512-d embedding space as items, enabling direct cosine similarity comparisons.

---

## 10. Evaluation

**Module:** `src/evaluation/metrics.py`  
**Script:** `scripts/evaluate.py`

### 10.1 Offline Metrics (Ground Truth)

| Metric             | What It Measures                                    | Method                                                    |
|--------------------|-----------------------------------------------------|-----------------------------------------------------------|
| FITB Accuracy      | Can the model pick the correct missing item?        | Given 3 outfit items + 4 choices, select the best match   |
| Compatibility AUC  | How well does the model separate compatible vs. not? | ROC-AUC on positive/negative pair scores                  |
| Outfit Coherence   | Are recommended items visually/stylistically similar? | Mean pairwise cosine similarity within outfits            |
| Diversity Score    | Are different outfit suggestions sufficiently varied? | Pairwise distance between outfit centroids                |

### 10.2 Online Metrics (Recommendation Quality)

| Metric      | What It Measures                             |
|-------------|----------------------------------------------|
| Hit Rate@K  | % of ground truth items in top-K results     |
| NDCG@K      | Rank-aware relevance (rewards higher ranks)  |
| Latency     | P50, P95, P99 response time in milliseconds  |

### 10.3 Evaluation Flow

1. Initialize the recommendation pipeline (loads data, builds index)
2. Run FITB on held-out test outfits
3. Sample 100 items and generate recommendations
4. Compute coherence, diversity, and latency across samples
5. Export all metrics to `evaluation_results.json`

---

## 11. Serving Layer

**Module:** `src/api/main.py`, `src/api/models.py`

### 11.1 Framework

FastAPI with Uvicorn ASGI server. CORS enabled for all origins.

### 11.2 Endpoints

| Method | Endpoint                    | Purpose                           | Response                        |
|--------|-----------------------------|-----------------------------------|---------------------------------|
| POST   | `/recommend/complete-look`  | Generate outfit recommendations   | Outfits with scores + latency   |
| GET    | `/items/{item_id}`          | Retrieve item metadata            | Item JSON                       |
| GET    | `/items`                    | Search items by category/color    | Paginated item list             |
| GET    | `/items/{item_id}/image`    | Serve product image               | PNG file                        |
| GET    | `/categories`               | List categories with item counts  | Category summary                |
| GET    | `/health`                   | Health check                      | Status + item/user counts       |

### 11.3 Startup Sequence

On server startup, the pipeline automatically:
1. Loads item metadata from `data/processed/items.json`
2. Computes or loads cached embeddings (`data/processed/embeddings.npy`)
3. Builds the FAISS index
4. Loads user interactions and builds all user profiles
5. Server is ready to accept requests

### 11.4 Request/Response Example

**Request:**
```json
POST /recommend/complete-look
{
  "item_id": 42,
  "user_id": 1,
  "num_outfits": 3
}
```

**Response:**
```json
{
  "query_item": { "item_id": 42, "category": "top", "color": "navy", ... },
  "outfits": [
    {
      "items": [
        { "item_id": 42,  "category": "top",       "color": "navy"  },
        { "item_id": 156, "category": "bottom",    "color": "beige" },
        { "item_id": 289, "category": "shoes",     "color": "white" },
        { "item_id": 512, "category": "accessory", "color": "brown" }
      ],
      "compatibility_score": 0.7823,
      "num_items": 4,
      "style_tags": ["casual", "summer"]
    }
  ],
  "num_results": 3,
  "latency_ms": 145.3,
  "personalized": true
}
```

---

## 12. MVP Embedding Strategy

**Module:** `src/recommendation/pipeline.py`

For efficient CPU-only deployment without GPU inference, the pipeline uses a lightweight embedding approach:

| Feature Source      | Method              | Dimensions |
|---------------------|---------------------|------------|
| Product description | TF-IDF vectorization | 256        |
| Categorical attrs   | One-hot encoding    | ~40        |
| Price               | Z-score normalization | 1         |

These are concatenated (~297-d) and reduced via **TruncatedSVD** to **128-d**, then L2-normalized. Embeddings are cached to `data/processed/embeddings.npy` to avoid recomputation.

This strategy enables the full recommendation pipeline to run without loading deep learning models, while the neural encoders (Section 4) can be swapped in when GPU resources are available.

---

## 13. Configuration

**File:** `configs/model.yaml`

Key parameters organized by component:

```yaml
visual_encoder:
  backbone: "resnet50"
  embedding_dim: 2048
  fine_tune_layers: 2

text_encoder:
  model_name: "all-MiniLM-L6-v2"
  embedding_dim: 384

attribute_encoder:
  embedding_dim: 64

fusion:
  input_dim: 2496               # 2048 + 384 + 64
  output_dim: 512

compatibility:
  embedding_dim: 512
  num_category_pairs: 6
  margin: 0.3
  num_negatives: 4

training:
  batch_size: 64
  learning_rate: 0.0005
  num_epochs: 50
  early_stopping_patience: 5
  mixed_precision: true

retrieval:
  faiss_index_type: "IVF256,PQ32"
  top_k_candidates: 100
  beam_width: 5
  diversity_lambda: 0.3

personalization:
  temporal_decay_halflife: 30   # days
  num_style_clusters: 20
```

---

## 14. End-to-End Flow Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│ Step 1: Data Generation                                             │
│   python scripts/generate_dataset.py                                │
│   → 5K items + 1.2K outfits + 500 users + 50K interactions         │
│   → Synthetic images + JSON metadata + compatibility pairs          │
├─────────────────────────────────────────────────────────────────────┤
│ Step 2: Training                                                    │
│   python scripts/train.py --epochs 30                               │
│   → Load item embeddings → Create BPR pairs from outfits           │
│   → Train type-aware compatibility model with early stopping        │
│   → Save best checkpoint to checkpoints/                            │
├─────────────────────────────────────────────────────────────────────┤
│ Step 3: Evaluation                                                  │
│   python scripts/evaluate.py                                        │
│   → FITB accuracy on test outfits                                   │
│   → Coherence, diversity, latency on sampled recommendations        │
│   → Export metrics to evaluation_results.json                       │
├─────────────────────────────────────────────────────────────────────┤
│ Step 4: Serving                                                     │
│   uvicorn src.api.main:app --host 0.0.0.0 --port 8000              │
│   → Startup: load data → compute embeddings → build FAISS index    │
│   → Build user profiles → ready to serve                            │
├─────────────────────────────────────────────────────────────────────┤
│ Step 5: Recommendation Request                                      │
│   POST /recommend/complete-look {item_id: 42, user_id: 1}          │
│                                                                     │
│   1. Look up query item embedding                                   │
│   2. Identify 3 missing categories                                  │
│   3. FAISS retrieval: top-100 candidates per category               │
│   4. Score candidates with compatibility model                      │
│   5. Blend with user profile similarity (α=0.2)                     │
│   6. Beam search (width=5) to assemble outfits                      │
│   7. MMR diversity re-ranking (λ=0.3)                               │
│   8. Return top-N outfits with scores                               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 15. Dependencies

| Category       | Libraries                                           |
|----------------|-----------------------------------------------------|
| Deep Learning  | PyTorch ≥2.0, torchvision ≥0.15, Transformers ≥4.30, Sentence-Transformers ≥2.2 |
| Retrieval      | faiss-cpu ≥1.7.4                                    |
| API            | FastAPI ≥0.100, Uvicorn ≥0.23, Pydantic ≥2.0       |
| Data           | NumPy ≥1.24, Pandas ≥2.0, scikit-learn ≥1.3, Pillow ≥10.0 |
| Tracking       | MLflow ≥2.5                                         |
| Deployment     | Docker, docker-compose                              |

**Python version:** 3.9+

---

## 16. Testing

**Location:** `tests/`

| Test File           | Coverage                                                    |
|---------------------|-------------------------------------------------------------|
| `test_models.py`    | Encoder output shapes, normalization, fusion integration, compatibility scoring, BPR loss |
| `test_dataset.py`   | Dataset loading, batch creation, FITB dataset format        |
| `test_metrics.py`   | FITB accuracy, AUC, coherence, diversity metric computation |

Run all tests:
```bash
pytest tests/ -v
```
