FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
    fastapi uvicorn pydantic numpy pandas scikit-learn \
    Pillow tqdm pyyaml faiss-cpu

# Copy application
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
COPY data/ data/

# Generate dataset if not present
RUN python scripts/generate_dataset.py || true

ENV DATA_DIR=data/processed
EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
