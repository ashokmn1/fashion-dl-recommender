"""
FastAPI Serving Layer
======================
REST API for fashion outfit recommendations.

Usage:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""

import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.models import (
    RecommendationRequest,
    RecommendationResponse,
    ItemResponse,
    ItemSearchResponse,
    HealthResponse,
)
from src.recommendation.pipeline import RecommendationPipeline

# ─── App Setup ────────────────────────────────────────────────────

app = FastAPI(
    title="Fashion Styling Recommendations API",
    description=(
        "Deep Learning Framework for Personalized Fashion Styling. "
        "Generate complete outfit recommendations from a single query item."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline: Optional[RecommendationPipeline] = None


@app.on_event("startup")
async def startup_event():
    """Initialize the recommendation pipeline on startup."""
    global pipeline
    data_dir = os.environ.get("DATA_DIR", "data/processed")
    print(f"Loading pipeline from {data_dir}...")
    pipeline = RecommendationPipeline(data_dir=data_dir)
    pipeline.initialize()
    print("Pipeline ready!")


# ─── Endpoints ────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Root endpoint with API overview."""
    return {
        "name": "Fashion Styling Recommendations API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "POST /recommend/complete-look": "Generate outfit recommendations",
            "GET /items/{item_id}": "Get item details",
            "GET /items": "Search items by category/color",
            "GET /items/{item_id}/image": "Get product image",
            "GET /categories": "List categories with counts",
            "GET /health": "Health check",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if pipeline and pipeline._initialized else "initializing",
        num_items=len(pipeline.item_metadata) if pipeline else 0,
        version="1.0.0",
    )


@app.post("/recommend/complete-look", response_model=RecommendationResponse)
async def complete_the_look(request: RecommendationRequest):
    """Generate complete outfit recommendations for a query item.

    Given a single fashion item, returns complete outfit suggestions
    with compatibility scores and personalization.
    """
    if not pipeline or not pipeline._initialized:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if request.item_id not in pipeline.item_metadata:
        raise HTTPException(status_code=404, detail=f"Item {request.item_id} not found")

    result = pipeline.recommend(
        item_id=request.item_id,
        user_id=request.user_id,
        num_outfits=request.num_outfits,
    )

    return RecommendationResponse(**result)


@app.get("/items/{item_id}", response_model=ItemResponse)
async def get_item(item_id: int):
    """Get item details by ID."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    item = pipeline.get_item(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")

    return ItemResponse(**item)


@app.get("/items", response_model=ItemSearchResponse)
async def search_items(
    category: Optional[str] = Query(None, description="Filter by category"),
    color: Optional[str] = Query(None, description="Filter by color"),
    limit: int = Query(20, ge=1, le=100, description="Max results"),
):
    """Search items by category and/or color."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    items = pipeline.search_items(category=category, color=color, limit=limit)
    return ItemSearchResponse(items=items, total=len(items))


@app.get("/items/{item_id}/image")
async def get_item_image(item_id: int):
    """Serve the item's product image."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    item = pipeline.get_item(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")

    image_path = pipeline.data_dir / item["image_path"]
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(str(image_path), media_type="image/png")


@app.get("/categories")
async def list_categories():
    """List available categories and their item counts."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    counts = {}
    for item in pipeline.item_metadata.values():
        cat = item["category"]
        counts[cat] = counts.get(cat, 0) + 1

    return {"categories": counts}
