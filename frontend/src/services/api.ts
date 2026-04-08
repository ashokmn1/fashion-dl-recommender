import axios from "axios";
import type {
  Item,
  ItemSearchResponse,
  CategoriesResponse,
  HealthResponse,
  RecommendationRequest,
  RecommendationResponse,
} from "../types/api";

const API_BASE = import.meta.env.VITE_API_URL ?? "/api";

const client = axios.create({
  baseURL: API_BASE,
});

// ─── Items ───────────────────────────────────────────────────────

export async function getItem(itemId: number): Promise<Item> {
  const { data } = await client.get<Item>(`/items/${itemId}`);
  return data;
}

export async function searchItems(params: {
  category?: string;
  color?: string;
  limit?: number;
}): Promise<ItemSearchResponse> {
  const { data } = await client.get<ItemSearchResponse>("/items", { params });
  return data;
}

export function getItemImageUrl(itemId: number): string {
  return `${API_BASE}/items/${itemId}/image`;
}

// ─── Recommendations ─────────────────────────────────────────────

export async function getRecommendations(
  req: RecommendationRequest
): Promise<RecommendationResponse> {
  const { data } = await client.post<RecommendationResponse>(
    "/recommend/complete-look",
    req
  );
  return data;
}

// ─── Meta ────────────────────────────────────────────────────────

export async function getCategories(): Promise<CategoriesResponse> {
  const { data } = await client.get<CategoriesResponse>("/categories");
  return data;
}

export async function getHealth(): Promise<HealthResponse> {
  const { data } = await client.get<HealthResponse>("/health");
  return data;
}
