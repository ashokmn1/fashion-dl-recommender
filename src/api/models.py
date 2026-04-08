"""
Pydantic Models for API Request/Response
==========================================
"""

from typing import Optional
from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    """Request body for outfit recommendations."""
    item_id: int = Field(..., description="Query item ID to build outfit around")
    user_id: Optional[int] = Field(None, description="User ID for personalization")
    num_outfits: int = Field(3, ge=1, le=10, description="Number of outfits to return")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"item_id": 42, "user_id": 1, "num_outfits": 3}
            ]
        }
    }


class OutfitItem(BaseModel):
    item_id: int
    category: str
    subcategory: str
    color: str
    description: str = ""
    price: float = 0
    image_path: str = ""


class Outfit(BaseModel):
    items: list[OutfitItem]
    compatibility_score: float
    num_items: int
    style_tags: list[str] = []


class RecommendationResponse(BaseModel):
    """Response body for outfit recommendations."""
    query_item: dict
    outfits: list[Outfit]
    num_results: int
    latency_ms: float
    personalized: bool


class ItemResponse(BaseModel):
    """Single item details."""
    item_id: int
    category: str
    subcategory: str
    color: str
    material: str
    pattern: str
    season: str
    gender: str
    occasion: str
    price: float
    image_path: str
    description: str


class ItemSearchResponse(BaseModel):
    """Item search results."""
    items: list[dict]
    total: int


class HealthResponse(BaseModel):
    """API health check response."""
    status: str
    num_items: int
    version: str
