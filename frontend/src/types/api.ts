// ─── Item Types ──────────────────────────────────────────────────

export interface Item {
  item_id: number;
  category: Category;
  subcategory: string;
  color: string;
  material: string;
  pattern: string;
  season: string;
  gender: string;
  occasion: string;
  price: number;
  image_path: string;
  description: string;
}

export type Category = "top" | "bottom" | "shoes" | "accessory";

export const CATEGORIES: Category[] = ["top", "bottom", "shoes", "accessory"];

export const COLORS = [
  "black", "white", "navy", "blue", "grey", "beige",
  "brown", "red", "burgundy", "green", "olive", "pink",
] as const;

// ─── Outfit Types ────────────────────────────────────────────────

export interface OutfitItem {
  item_id: number;
  category: string;
  subcategory: string;
  color: string;
  description: string;
  price: number;
  image_path: string;
}

export interface Outfit {
  items: OutfitItem[];
  compatibility_score: number;
  num_items: number;
  style_tags: string[];
}

// ─── Request / Response Types ────────────────────────────────────

export interface RecommendationRequest {
  item_id: number;
  user_id?: number;
  num_outfits?: number;
}

export interface RecommendationResponse {
  query_item: Item;
  outfits: Outfit[];
  num_results: number;
  latency_ms: number;
  personalized: boolean;
}

export interface ItemSearchResponse {
  items: Item[];
  total: number;
}

export interface CategoriesResponse {
  categories: Record<string, number>;
}

export interface HealthResponse {
  status: "healthy" | "initializing";
  num_items: number;
  version: string;
}
