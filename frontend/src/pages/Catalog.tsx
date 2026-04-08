import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import {
  Box,
  Typography,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Stack,
  CircularProgress,
  Alert,
  Pagination,
} from "@mui/material";
import type { Item, Category } from "../types/api";
import { CATEGORIES, COLORS } from "../types/api";
import { searchItems } from "../services/api";
import ItemCard from "../components/ItemCard";

const PAGE_SIZE = 20;

export default function Catalog() {
  const [params, setParams] = useSearchParams();
  const [items, setItems] = useState<Item[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [page, setPage] = useState(1);

  const category = (params.get("category") ?? "") as Category | "";
  const color = params.get("color") ?? "";

  useEffect(() => {
    setLoading(true);
    setError("");
    searchItems({
      category: category || undefined,
      color: color || undefined,
      limit: 100,
    })
      .then((res) => {
        setItems(res.items);
        setTotal(res.total);
        setPage(1);
      })
      .catch(() => setError("Failed to load items."))
      .finally(() => setLoading(false));
  }, [category, color]);

  const updateFilter = (key: string, value: string) => {
    const next = new URLSearchParams(params);
    if (value) next.set(key, value);
    else next.delete(key);
    setParams(next);
  };

  const pageItems = items.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);
  const pageCount = Math.ceil(items.length / PAGE_SIZE);

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Catalog
      </Typography>

      {/* Filters */}
      <Stack direction="row" spacing={2} sx={{ mb: 3 }}>
        <FormControl size="small" sx={{ minWidth: 140 }}>
          <InputLabel>Category</InputLabel>
          <Select
            value={category}
            label="Category"
            onChange={(e) => updateFilter("category", e.target.value)}
          >
            <MenuItem value="">All</MenuItem>
            {CATEGORIES.map((c) => (
              <MenuItem key={c} value={c} sx={{ textTransform: "capitalize" }}>
                {c}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        <FormControl size="small" sx={{ minWidth: 140 }}>
          <InputLabel>Color</InputLabel>
          <Select
            value={color}
            label="Color"
            onChange={(e) => updateFilter("color", e.target.value)}
          >
            <MenuItem value="">All</MenuItem>
            {COLORS.map((c) => (
              <MenuItem key={c} value={c} sx={{ textTransform: "capitalize" }}>
                {c}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Stack>

      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

      {loading ? (
        <Box sx={{ textAlign: "center", py: 6 }}><CircularProgress /></Box>
      ) : items.length === 0 ? (
        <Alert severity="info">No items found. Try adjusting filters.</Alert>
      ) : (
        <>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Showing {pageItems.length} of {total} items
          </Typography>
          <Grid container spacing={2}>
            {pageItems.map((item) => (
              <Grid size={{ xs: 6, sm: 4, md: 3 }} key={item.item_id}>
                <ItemCard item={item} />
              </Grid>
            ))}
          </Grid>
          {pageCount > 1 && (
            <Box sx={{ display: "flex", justifyContent: "center", mt: 4 }}>
              <Pagination
                count={pageCount}
                page={page}
                onChange={(_, v) => setPage(v)}
                color="primary"
              />
            </Box>
          )}
        </>
      )}
    </Box>
  );
}
