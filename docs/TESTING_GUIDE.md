# Testing Guide

## Overview

This project uses **pytest** as its testing framework. The test suite validates the core ML components, data pipeline, and evaluation metrics to ensure correctness of model architectures, data loading, and scoring logic.

## Prerequisites

Install dev dependencies (includes pytest and pytest-cov):

```bash
pip install -e ".[dev]"
```

## Running Tests

### Run all tests

```bash
pytest tests/ -v
```

### Run a specific test file

```bash
pytest tests/test_models.py -v
pytest tests/test_dataset.py -v
pytest tests/test_metrics.py -v
```

### Run a specific test class or method

```bash
pytest tests/test_models.py::TestVisualEncoder -v
pytest tests/test_models.py::TestVisualEncoder::test_output_shape -v
```

### Run with coverage report

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

### Run with short summary on failures

```bash
pytest tests/ -v --tb=short
```

---

## Test Structure

```
tests/
├── __init__.py
├── test_models.py      # Neural network architectures and loss functions
├── test_dataset.py     # Data loading, datasets, and data formats
└── test_metrics.py     # Evaluation metrics and scoring
```

---

## What's Covered

### 1. Model Tests (`test_models.py`)

Tests all neural network modules to verify output shapes, gradient flow, and numerical properties.

#### Visual Encoder

| Test | What It Validates |
|------|-------------------|
| `test_output_shape` | Input batch of images (2, 3, 224, 224) produces embeddings of shape (2, 512) |
| `test_feature_dim` | `get_feature_dim()` returns the correct embedding dimension |

**Why it matters:** Ensures the ResNet-50 backbone and projection head produce correctly shaped embeddings that feed into the fusion module.

#### Attribute Encoder

| Test | What It Validates |
|------|-------------------|
| `test_output_shape` | Attribute tensor (2, 5) maps to embeddings (2, 64) |

**Why it matters:** Verifies that the 5 categorical attributes (category, color, material, pattern, season) are correctly embedded and fused into a single vector.

#### Multimodal Fusion

| Test | What It Validates |
|------|-------------------|
| `test_fusion_output` | Combines visual (512-d), text (384-d), and attribute (64-d) inputs into 128-d output |
| `test_normalization` | With `normalize=True`, output vectors have unit L2 norm (~1.0) |

**Why it matters:** The fusion module is the central bottleneck — all three modalities must combine correctly, and L2 normalization is required for cosine similarity in retrieval.

#### Compatibility Model

| Test | What It Validates |
|------|-------------------|
| `test_forward_pass` | Triplet-based training with anchor/positive/negative items produces a differentiable loss tensor |
| `test_outfit_scoring` | Scoring a full outfit (multiple items + categories) returns a valid compatibility score |

**Why it matters:** Validates the type-aware projection heads and the end-to-end gradient flow from embeddings through pairwise scoring to loss computation.

#### Loss Functions

| Test | What It Validates |
|------|-------------------|
| `test_bpr_loss` | BPR loss produces a positive scalar less than 1 for well-separated scores |
| `test_triplet_loss` | Triplet margin loss (margin=0.3) produces a non-negative loss value |

**Why it matters:** Incorrect loss computation directly breaks training. These tests verify the mathematical properties that ensure the optimizer receives correct gradients.

---

### 2. Dataset Tests (`test_dataset.py`)

Tests data loading and dataset classes using a temporary synthetic fixture.

#### Test Fixture: `sample_data_dir`

Creates a temporary directory with:
- 20 items across 4 categories (top, bottom, shoes, accessory)
- 4 outfits split into train/val/test JSON files
- 224x224 dummy PNG images

This fixture isolates dataset tests from the full generated data.

#### Fashion Item Dataset

| Test | What It Validates |
|------|-------------------|
| `test_dataset_length` | All 20 items are loaded correctly |
| `test_dataset_item_format` | Each item contains required fields: `item_id`, `image`, `category`, `attributes` |
| `test_dataset_item_format` | Image tensors have shape (3, 224, 224) — channels-first format for PyTorch |
| `test_category_mapping` | Categories map to correct indices (top=0, bottom=1, shoes=2, accessory=3) |

**Why it matters:** The item dataset feeds all encoders. Wrong shapes or missing fields would cause silent errors downstream.

#### Compatibility Pair Dataset

| Test | What It Validates |
|------|-------------------|
| `test_pair_dataset` | Dataset is non-empty when configured with `num_negatives=2` |
| `test_pair_format` | Each sample contains `anchor_id`, `positive_id`, and `negative_ids` fields |

**Why it matters:** BPR training requires correctly structured triplets. Missing negatives or malformed pairs would produce meaningless gradients.

#### Outfit Dataset (FITB)

| Test | What It Validates |
|------|-------------------|
| `test_fitb_format` | Each sample has `context_ids` (3 items), `choices` (4 options), and `answer_position` |

**Why it matters:** The Fill-in-the-Blank evaluation format must have exactly 4 choices with one correct answer. Incorrect format would invalidate evaluation metrics.

---

### 3. Metric Tests (`test_metrics.py`)

Tests all evaluation metrics with controlled inputs to verify mathematical correctness.

#### FITB Accuracy

| Test | What It Validates |
|------|-------------------|
| `test_correct_prediction` | Returns correct accuracy when model selects the right item |
| `test_empty_context` | Handles edge case of empty context without crashing |

#### Compatibility AUC

| Test | What It Validates |
|------|-------------------|
| `test_perfect_separation` | AUC = 1.0 when positive scores are all higher than negative scores |
| `test_random_separation` | AUC ≈ 0.5 when scores are random (no discriminative power) |

**Why it matters:** AUC measures how well the compatibility model separates good from bad pairs. These boundary conditions verify the metric itself is trustworthy.

#### Hit Rate @ K

| Test | What It Validates |
|------|-------------------|
| `test_all_hits` | Returns 1.0 when all ground truth items appear in recommendations |
| `test_no_hits` | Returns 0.0 when no ground truth items appear |

#### NDCG @ K

| Test | What It Validates |
|------|-------------------|
| `test_perfect_ranking` | NDCG = 1.0 when the target item is ranked first |
| `test_position_matters` | Items ranked higher produce higher NDCG than items ranked lower |

**Why it matters:** NDCG is position-sensitive — ranking a correct item at position 1 is far better than position 10. This test verifies that the rank discounting works correctly.

#### Outfit Coherence

| Test | What It Validates |
|------|-------------------|
| `test_identical_items` | Identical embeddings produce coherence ≈ 1.0 |
| `test_single_item` | Single-item outfit returns 0.0 (no pairs to compare) |

#### Diversity Score

| Test | What It Validates |
|------|-------------------|
| `test_identical_outfits` | Identical outfit sets produce diversity ≈ 0.0 |
| `test_diverse_outfits` | Dissimilar outfit sets produce diversity > 0.5 |

---

## Test Coverage Summary

| Component | Module | Tests | What's Validated |
|-----------|--------|-------|------------------|
| Visual Encoder | `src/models/visual_encoder.py` | 2 | Output shape, feature dimension |
| Attribute Encoder | `src/models/attribute_encoder.py` | 1 | Output shape from categorical inputs |
| Multimodal Fusion | `src/models/fusion.py` | 2 | Combined output shape, L2 normalization |
| Compatibility Model | `src/models/compatibility.py` | 2 | Forward pass gradient flow, outfit scoring |
| BPR Loss | `src/training/losses.py` | 1 | Loss value range and properties |
| Triplet Loss | `src/training/losses.py` | 1 | Non-negative loss with margin |
| Item Dataset | `src/data/dataset.py` | 3 | Length, item format, category mapping |
| Pair Dataset | `src/data/dataset.py` | 2 | Non-empty pairs, triplet structure |
| FITB Dataset | `src/data/dataset.py` | 1 | Context/choices/answer format |
| FITB Accuracy | `src/evaluation/metrics.py` | 2 | Correct prediction, empty context edge case |
| Compatibility AUC | `src/evaluation/metrics.py` | 2 | Perfect and random separation bounds |
| Hit Rate | `src/evaluation/metrics.py` | 2 | Full hit and zero hit boundaries |
| NDCG | `src/evaluation/metrics.py` | 2 | Perfect ranking, position sensitivity |
| Coherence | `src/evaluation/metrics.py` | 2 | Identical items, single item edge case |
| Diversity | `src/evaluation/metrics.py` | 2 | Identical vs. diverse outfit sets |

**Total: 27 test cases across 15 components**

---

## API Testing

### Step 1: Start the server

Make sure the dataset is generated, then start the API:

```bash
python scripts/generate_dataset.py   # skip if already generated
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Wait for the startup log:
```
Pipeline ready!
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 2: Open the interactive docs

Open **http://localhost:8000/docs** in your browser. This is the Swagger UI where you can test every endpoint directly from the browser by clicking **"Try it out"** on any endpoint.

In Codespaces, go to the **Ports** tab and click the globe icon next to port 8000, then append `/docs` to the URL.

### Step 3: Test each endpoint

Open a second terminal and run the following commands.

#### Root — `GET /`

Returns the API overview and list of available endpoints.

```bash
curl http://localhost:8000/
```

**Expected response:**
```json
{
  "name": "Fashion Styling Recommendations API",
  "version": "1.0.0",
  "docs": "/docs",
  "endpoints": {
    "POST /recommend/complete-look": "Generate outfit recommendations",
    "GET /items/{item_id}": "Get item details",
    "GET /items": "Search items by category/color",
    "GET /items/{item_id}/image": "Get product image",
    "GET /categories": "List categories with counts",
    "GET /health": "Health check"
  }
}
```

---

#### Health Check — `GET /health`

Verifies the server is running and the pipeline is initialized.

```bash
curl http://localhost:8000/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "num_items": 5000,
  "version": "1.0.0"
}
```

**What to check:**
- `status` should be `"healthy"` (not `"initializing"`)
- `num_items` should be `5000`

---

#### List Categories — `GET /categories`

Returns all item categories and their counts.

```bash
curl http://localhost:8000/categories
```

**Expected response:**
```json
{
  "categories": {
    "top": 1500,
    "bottom": 1250,
    "shoes": 1250,
    "accessory": 1000
  }
}
```

**What to check:**
- All 4 categories are present
- Counts add up to 5000

---

#### Get Item Details — `GET /items/{item_id}`

Retrieves full metadata for a single item.

```bash
curl http://localhost:8000/items/42
```

**Expected response:**
```json
{
  "item_id": 42,
  "category": "top",
  "subcategory": "t-shirt",
  "color": "navy",
  "material": "cotton",
  "pattern": "solid",
  "season": "summer",
  "gender": "unisex",
  "occasion": "casual",
  "price": 29.99,
  "image_path": "images/item_0042.png",
  "description": "A navy cotton t-shirt..."
}
```

**What to check:**
- All fields are present (category, color, material, pattern, season, gender, occasion, price, description)
- `item_id` matches the requested ID

**Error case — invalid item:**
```bash
curl http://localhost:8000/items/99999
```
Expected: `404` with `{"detail": "Item 99999 not found"}`

---

#### Search Items — `GET /items`

Search and filter items by category and/or color.

```bash
# Get all tops
curl "http://localhost:8000/items?category=top&limit=5"

# Get black shoes
curl "http://localhost:8000/items?category=shoes&color=black&limit=5"

# Get items with default limit (20)
curl http://localhost:8000/items
```

**Expected response:**
```json
{
  "items": [
    {"item_id": 1, "category": "top", "color": "white", ...},
    {"item_id": 5, "category": "top", "color": "navy", ...}
  ],
  "total": 5
}
```

**What to check:**
- `total` matches the number of items in the `items` array
- All returned items match the filter criteria (correct category/color)
- `limit` parameter is respected

---

#### Get Item Image — `GET /items/{item_id}/image`

Returns the product image as a PNG file.

```bash
# Download image
curl -o item_42.png http://localhost:8000/items/42/image

# Check it's a valid PNG
file item_42.png
```

**What to check:**
- Response content-type is `image/png`
- Downloaded file is a valid 224x224 PNG image

**Error case:**
```bash
curl http://localhost:8000/items/99999/image
```
Expected: `404` with `{"detail": "Item 99999 not found"}`

---

#### Complete the Look — `POST /recommend/complete-look`

The core endpoint. Generates outfit recommendations from a single item.

**Basic request (no personalization):**
```bash
curl -X POST http://localhost:8000/recommend/complete-look \
  -H "Content-Type: application/json" \
  -d '{"item_id": 42, "num_outfits": 3}'
```

**Personalized request (with user ID):**
```bash
curl -X POST http://localhost:8000/recommend/complete-look \
  -H "Content-Type: application/json" \
  -d '{"item_id": 42, "user_id": 1, "num_outfits": 3}'
```

**Expected response:**
```json
{
  "query_item": {
    "item_id": 42,
    "category": "top",
    "color": "navy",
    ...
  },
  "outfits": [
    {
      "items": [
        {"item_id": 42, "category": "top", ...},
        {"item_id": 156, "category": "bottom", ...},
        {"item_id": 289, "category": "shoes", ...},
        {"item_id": 512, "category": "accessory", ...}
      ],
      "compatibility_score": 0.7823,
      "num_items": 4,
      "style_tags": ["casual", "summer"]
    }
  ],
  "num_results": 3,
  "latency_ms": 145.3,
  "personalized": false
}
```

**What to check:**
- `query_item.item_id` matches the requested item
- Each outfit has exactly 4 items (one per category: top, bottom, shoes, accessory)
- The query item is included in each outfit
- `compatibility_score` is between 0 and 1
- `num_results` matches the requested `num_outfits`
- `personalized` is `true` when `user_id` is provided, `false` otherwise
- `latency_ms` is reported (should be under ~500ms)
- No duplicate items within a single outfit

**Error cases:**
```bash
# Invalid item
curl -X POST http://localhost:8000/recommend/complete-look \
  -H "Content-Type: application/json" \
  -d '{"item_id": 99999}'
```
Expected: `404` with `{"detail": "Item 99999 not found"}`

```bash
# Missing required field
curl -X POST http://localhost:8000/recommend/complete-look \
  -H "Content-Type: application/json" \
  -d '{}'
```
Expected: `422` validation error (item_id is required)

```bash
# Invalid num_outfits range
curl -X POST http://localhost:8000/recommend/complete-look \
  -H "Content-Type: application/json" \
  -d '{"item_id": 42, "num_outfits": 50}'
```
Expected: `422` validation error (num_outfits max is 10)

---

### API Test Summary

| Endpoint | Method | Test | Expected |
|----------|--------|------|----------|
| `/` | GET | API overview | Returns name, version, endpoint list |
| `/health` | GET | Pipeline status | `status: "healthy"`, `num_items: 5000` |
| `/categories` | GET | Category counts | 4 categories, total = 5000 |
| `/items/{id}` | GET | Valid item | Full metadata with all fields |
| `/items/{id}` | GET | Invalid item | 404 error |
| `/items` | GET | No filters | Returns up to 20 items |
| `/items` | GET | Category filter | Only matching category returned |
| `/items` | GET | Category + color | Both filters applied |
| `/items/{id}/image` | GET | Valid item | 224x224 PNG image |
| `/items/{id}/image` | GET | Invalid item | 404 error |
| `/recommend/complete-look` | POST | Basic request | 3 outfits, 4 items each |
| `/recommend/complete-look` | POST | With user_id | `personalized: true` |
| `/recommend/complete-look` | POST | Invalid item | 404 error |
| `/recommend/complete-look` | POST | Missing item_id | 422 validation error |
| `/recommend/complete-look` | POST | Invalid num_outfits | 422 validation error |

---

## Adding New Tests

When adding tests, follow these conventions:

1. **Group by module** — one test class per source module (e.g., `TestVisualEncoder` for `visual_encoder.py`)
2. **Use descriptive names** — `test_output_shape` is better than `test_1`
3. **Test boundaries** — include edge cases (empty inputs, single items, identical values)
4. **Use fixtures for data** — see `sample_data_dir` in `test_dataset.py` for creating temporary test data
5. **Check numerical properties** — use `pytest.approx()` for floating-point comparisons

Example of adding a new test:

```python
class TestNewModule:
    def test_expected_behavior(self):
        module = NewModule(param=value)
        output = module(input_tensor)
        assert output.shape == (batch_size, expected_dim)

    def test_edge_case(self):
        module = NewModule(param=value)
        output = module(empty_input)
        assert output is not None
```
