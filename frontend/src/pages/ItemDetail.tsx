import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import {
  Box,
  Typography,
  Grid,
  Chip,
  Stack,
  Button,
  CircularProgress,
  Alert,
  Card,
  CardContent,
  CardMedia,
  TextField,
  Divider,
  Slider,
} from "@mui/material";
import AutoAwesomeIcon from "@mui/icons-material/AutoAwesome";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import type { Item, RecommendationResponse } from "../types/api";
import { getItem, getRecommendations, getItemImageUrl } from "../services/api";
import OutfitCard from "../components/OutfitCard";

export default function ItemDetail() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const [item, setItem] = useState<Item | null>(null);
  const [rec, setRec] = useState<RecommendationResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [recLoading, setRecLoading] = useState(false);
  const [error, setError] = useState("");
  const [recError, setRecError] = useState("");
  const [userId, setUserId] = useState("");
  const [numOutfits, setNumOutfits] = useState(3);

  const itemId = parseInt(id ?? "", 10);

  useEffect(() => {
    if (isNaN(itemId)) {
      setError("Invalid item ID");
      setLoading(false);
      return;
    }
    setLoading(true);
    setError("");
    setRec(null);
    getItem(itemId)
      .then(setItem)
      .catch(() => setError("Item not found."))
      .finally(() => setLoading(false));
  }, [itemId]);

  const handleComplete = async () => {
    setRecLoading(true);
    setRecError("");
    try {
      const uid = parseInt(userId, 10);
      const result = await getRecommendations({
        item_id: itemId,
        user_id: isNaN(uid) ? undefined : uid,
        num_outfits: numOutfits,
      });
      setRec(result);
    } catch {
      setRecError("Failed to generate recommendations.");
    } finally {
      setRecLoading(false);
    }
  };

  if (loading) {
    return <Box sx={{ textAlign: "center", py: 8 }}><CircularProgress /></Box>;
  }

  if (error || !item) {
    return (
      <Box>
        <Button startIcon={<ArrowBackIcon />} onClick={() => navigate(-1)} sx={{ mb: 2 }}>Back</Button>
        <Alert severity="error">{error || "Item not found."}</Alert>
      </Box>
    );
  }

  return (
    <Box>
      <Button startIcon={<ArrowBackIcon />} onClick={() => navigate(-1)} sx={{ mb: 2 }}>Back</Button>

      {/* Item Detail */}
      <Grid container spacing={4} sx={{ mb: 4 }}>
        <Grid size={{ xs: 12, md: 5 }}>
          <Card>
            <CardMedia
              component="img"
              image={getItemImageUrl(item.item_id)}
              alt={item.description}
              sx={{ height: 350, objectFit: "cover", bgcolor: "#f0f0f0" }}
            />
          </Card>
        </Grid>
        <Grid size={{ xs: 12, md: 7 }}>
          <Typography variant="h4" gutterBottom>
            {item.subcategory.charAt(0).toUpperCase() + item.subcategory.slice(1)}
          </Typography>
          <Typography variant="h5" color="secondary" gutterBottom>
            ${item.price.toFixed(2)}
          </Typography>
          <Typography variant="body1" color="text.secondary" sx={{ mb: 2 }}>
            {item.description}
          </Typography>

          <Stack direction="row" spacing={1} sx={{ mb: 2, flexWrap: "wrap", gap: 0.5 }}>
            <Chip label={item.category} color="primary" />
            <Chip label={item.color} variant="outlined" />
            <Chip label={item.material} variant="outlined" />
            <Chip label={item.pattern} variant="outlined" />
            <Chip label={item.season} variant="outlined" />
            <Chip label={item.occasion} variant="outlined" />
            <Chip label={item.gender} variant="outlined" />
          </Stack>

          <Typography variant="caption" color="text.secondary">
            Item ID: {item.item_id}
          </Typography>
        </Grid>
      </Grid>

      <Divider sx={{ mb: 4 }} />

      {/* Complete the Look Controls */}
      <Card sx={{ mb: 4, bgcolor: "background.default" }}>
        <CardContent>
          <Typography variant="h5" gutterBottom>
            <AutoAwesomeIcon sx={{ mr: 1, verticalAlign: "middle", color: "secondary.main" }} />
            Complete the Look
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            Generate outfit recommendations that complement this item.
          </Typography>

          <Stack direction={{ xs: "column", sm: "row" }} spacing={2} sx={{ alignItems: "center" }}>
            <TextField
              size="small"
              label="User ID (optional)"
              placeholder="e.g. 1"
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
              sx={{ width: 160 }}
            />
            <Box sx={{ width: 200 }}>
              <Typography variant="caption" color="text.secondary">
                Outfits: {numOutfits}
              </Typography>
              <Slider
                value={numOutfits}
                onChange={(_, v) => setNumOutfits(v as number)}
                min={1}
                max={10}
                step={1}
                marks
                size="small"
              />
            </Box>
            <Button
              variant="contained"
              color="secondary"
              size="large"
              startIcon={<AutoAwesomeIcon />}
              onClick={handleComplete}
              disabled={recLoading}
            >
              {recLoading ? "Generating..." : "Generate Outfits"}
            </Button>
          </Stack>
        </CardContent>
      </Card>

      {recError && <Alert severity="error" sx={{ mb: 2 }}>{recError}</Alert>}

      {recLoading && (
        <Box sx={{ textAlign: "center", py: 4 }}>
          <CircularProgress />
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            Finding the best outfit combinations...
          </Typography>
        </Box>
      )}

      {/* Results */}
      {rec && (
        <Box>
          <Stack direction="row" sx={{ justifyContent: "space-between", alignItems: "center", mb: 2 }}>
            <Typography variant="h5">
              {rec.num_results} Outfit{rec.num_results !== 1 ? "s" : ""} Found
            </Typography>
            <Stack direction="row" spacing={1}>
              {rec.personalized && <Chip label="Personalized" color="secondary" size="small" />}
              <Chip label={`${rec.latency_ms.toFixed(0)}ms`} size="small" variant="outlined" />
            </Stack>
          </Stack>

          <Stack spacing={3}>
            {rec.outfits.map((outfit, i) => (
              <OutfitCard key={i} outfit={outfit} index={i} />
            ))}
          </Stack>
        </Box>
      )}
    </Box>
  );
}
