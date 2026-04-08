# Deep Learning Framework for Personalized Fashion Styling Recommendations

A "Complete the Look" recommendation system that uses deep learning to generate personalized outfit suggestions. Given a single fashion item, the system recommends complementary items (tops, bottoms, shoes, accessories) to create a cohesive outfit.

## Architecture

```
Query Item → Visual/Text/Attribute Encoders → Multimodal Fusion → FAISS Retrieval
                                                                       ↓
User Profile ← Interaction History              Beam Search ← Compatibility Model
      ↓                                              ↓
  Personalized Re-ranking → Complete Outfit Recommendations → FastAPI
```

**Core modules:**

- **Multimodal Feature Extraction** — ResNet-50 (visual), Sentence-BERT (text), learned embeddings (attributes), late fusion into unified 512-dim item vectors
- **Compatibility Learning** — Type-aware embedding network with BPR loss. Separate projection heads per category pair (top→bottom, top→shoes, etc.)
- **Personalization** — User profile builder with temporal decay, style clustering, and personalized re-ranking
- **Outfit Generation** — FAISS nearest-neighbor retrieval + beam search + MMR diversity re-ranking
- **Serving** — FastAPI REST API with Docker deployment

## Quick Start

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

### 2. Generate the synthetic dataset

```bash
python scripts/generate_dataset.py
```

This creates 5,000 fashion items, 1,200 outfits, 500 user profiles, and 50K interaction events with synthetic product images.

### 3. Train the compatibility model

```bash
python scripts/train.py --epochs 30 --batch_size 64
```

### 4. Evaluate

```bash
python scripts/evaluate.py
```

### 5. Start the API server

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000/docs for the interactive API documentation.

### Docker

```bash
docker-compose up --build
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/recommend/complete-look` | Generate outfit recommendations |
| `GET` | `/items/{item_id}` | Get item details |
| `GET` | `/items` | Search items by category/color |
| `GET` | `/items/{item_id}/image` | Get product image |
| `GET` | `/categories` | List categories with counts |
| `GET` | `/health` | Health check |

**Example request:**

```bash
curl -X POST http://localhost:8000/recommend/complete-look \
  -H "Content-Type: application/json" \
  -d '{"item_id": 42, "user_id": 1, "num_outfits": 3}'
```

## Dataset

This project includes a synthetic dataset generator (`scripts/generate_dataset.py`) that produces:

- **5,000 items** across 4 categories (tops, bottoms, shoes, accessories)
- **1,200 outfits** composed using 6 style profiles (casual, formal, streetwear, summer, winter, sporty)
- **500 users** with interaction histories
- **50,000 interaction events** (views, clicks, purchases)
- **Style-aware compatibility pairs** for training

The generator uses real fashion taxonomy (colors, materials, patterns) and compatibility rules to ensure realistic outfit compositions.

## Project Structure

```
fashion-dl-recommender/
├── src/
│   ├── data/           # Dataset classes, transforms, data loaders
│   ├── models/         # Visual encoder, text encoder, attribute encoder,
│   │                   # multimodal fusion, compatibility model
│   ├── personalization/# User profiles, style clustering, re-ranking
│   ├── recommendation/ # Outfit generator, end-to-end pipeline
│   ├── retrieval/      # FAISS index management
│   ├── training/       # Training loop, loss functions
│   ├── evaluation/     # FITB accuracy, AUC, NDCG, coherence metrics
│   └── api/            # FastAPI endpoints and Pydantic models
├── scripts/
│   ├── generate_dataset.py  # Synthetic dataset generator
│   ├── train.py             # Training script
│   └── evaluate.py          # Evaluation script
├── tests/              # Unit tests
├── configs/            # YAML configuration files
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| FITB Accuracy | Fill-in-the-blank: predict missing outfit item from 4 choices |
| Compatibility AUC | Pairwise compatibility scoring accuracy |
| Hit Rate @ K | Whether correct item appears in top-K recommendations |
| NDCG @ K | Ranking quality (position-aware) |
| Outfit Coherence | Average pairwise similarity within recommended outfits |
| Diversity Score | Variation across multiple outfit recommendations |

## Tech Stack

- **Deep Learning:** PyTorch, torchvision, Sentence-Transformers
- **Retrieval:** FAISS (Facebook AI Similarity Search)
- **API:** FastAPI, Uvicorn, Pydantic
- **Data:** NumPy, Pandas, scikit-learn
- **Deployment:** Docker, docker-compose
- **Tracking:** MLflow

## License

MIT
