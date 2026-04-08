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

## Prerequisites

- **Python:** 3.9 or higher
- **OS:** macOS, Linux, or Windows
- **Hardware:** CPU-only is supported; GPU (CUDA) optional for faster training

## Environment Setup

### 1. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### 2. Install the package with all dependencies

```bash
pip install -e ".[dev]"
```

This installs the following dependencies:

#### Core Dependencies

| Package              | Version   | Purpose                                      |
|----------------------|-----------|----------------------------------------------|
| torch                | ≥2.0.0    | Deep learning framework (model training/inference) |
| torchvision          | ≥0.15.0   | Image transforms and pretrained ResNet-50 backbone |
| transformers         | ≥4.30.0   | Hugging Face model hub integration           |
| sentence-transformers| ≥2.2.0    | Sentence-BERT text encoder (`all-MiniLM-L6-v2`) |
| faiss-cpu            | ≥1.7.4    | Approximate nearest neighbor search (FAISS index) |
| fastapi              | ≥0.100.0  | REST API framework for serving recommendations |
| uvicorn              | ≥0.23.0   | ASGI server to run the FastAPI application   |
| pydantic             | ≥2.0.0    | Request/response validation and serialization |
| numpy                | ≥1.24.0   | Numerical computing for embeddings and arrays |
| pandas               | ≥2.0.0    | Data manipulation for dataset and interactions |
| scikit-learn         | ≥1.3.0    | TF-IDF, SVD, and preprocessing utilities     |
| Pillow               | ≥10.0.0   | Synthetic product image generation (224x224 PNGs) |
| mlflow               | ≥2.5.0    | Experiment tracking and metric logging       |
| matplotlib           | ≥3.7.0    | Training visualization and metric plots      |
| seaborn              | ≥0.12.0   | Statistical data visualization               |
| tqdm                 | ≥4.65.0   | Progress bars for training and data generation |
| pyyaml               | ≥6.0      | YAML configuration file parsing              |
| redis                | ≥4.6.0    | Caching layer for production deployment      |
| implicit             | ≥0.7.0    | Collaborative filtering utilities            |
| python-multipart     | ≥0.0.6    | File upload support for FastAPI              |

#### Dev Dependencies

| Package    | Version  | Purpose                        |
|------------|----------|--------------------------------|
| pytest     | ≥7.4.0   | Test framework                 |
| pytest-cov | ≥4.1.0   | Code coverage reporting        |
| ruff       | ≥0.1.0   | Linter and formatter           |
| pre-commit | ≥3.3.0   | Git pre-commit hook management |

## Quick Start

### 1. Generate the synthetic dataset

```bash
python scripts/generate_dataset.py
```

This creates 5,000 fashion items, 1,200 outfits, 500 user profiles, and 50K interaction events with synthetic product images.

### 2. Train the compatibility model

```bash
python scripts/train.py --epochs 30 --batch_size 64
```

### 3. Evaluate

```bash
python scripts/evaluate.py
```

### 4. Start the API server

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000/docs for the interactive API documentation.

### Docker

```bash
docker-compose up --build
```

## Running in GitHub Codespaces

The project includes a `.devcontainer/devcontainer.json` that automatically configures the environment.

### 1. Create a Codespace

- Go to your GitHub repository
- Click the green **Code** button → **Codespaces** tab → **Create codespace on main**
- Wait for the Codespace to build (the devcontainer will automatically install Python 3.11 and all dependencies)

The devcontainer handles the following on creation:
- Sets up **Python 3.11** base image
- Runs `pip install -e ".[dev]"` to install all dependencies
- Generates the synthetic dataset on first start
- Forwards port **8000** (API) and **6379** (Redis)
- Installs Python and Pylance VS Code extensions

### 2. Train the model

Once the Codespace is ready and the terminal is available:

```bash
python scripts/train.py --epochs 30 --batch_size 64
```

### 3. Run evaluation

```bash
python scripts/evaluate.py
```

### 4. Start the API server

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Codespaces will show a notification to **Open in Browser** when port 8000 becomes available. Click it to access the interactive API docs at the `/docs` endpoint.

> You can also go to the **Ports** tab in the Codespaces terminal panel, find port 8000, and click the globe icon to open it.

### 5. Test the API

```bash
curl -X POST http://localhost:8000/recommend/complete-look \
  -H "Content-Type: application/json" \
  -d '{"item_id": 42, "user_id": 1, "num_outfits": 3}'
```

### 6. Run tests

```bash
pytest tests/ -v
```

### Alternative: Run with Docker in Codespaces

Docker is available inside the Codespace (via Docker-in-Docker), so you can also run:

```bash
docker-compose up --build
```

This starts both the API server (port 8000) and Redis (port 6379). Codespaces will automatically forward both ports.

> **Note:** Codespaces provides a minimum 2-core / 8 GB machine. This project runs on CPU and fits comfortably within those resources. For faster training, select a 4-core machine type from the Codespace creation options.

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
