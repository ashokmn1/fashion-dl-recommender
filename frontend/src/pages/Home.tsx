import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Box,
  Typography,
  TextField,
  Button,
  Grid,
  Card,
  CardActionArea,
  CardContent,
  Stack,
  CircularProgress,
  Alert,
  InputAdornment,
} from "@mui/material";
import SearchIcon from "@mui/icons-material/Search";
import CheckroomIcon from "@mui/icons-material/Checkroom";
import AutoAwesomeIcon from "@mui/icons-material/AutoAwesome";
import SpeedIcon from "@mui/icons-material/Speed";
import PersonIcon from "@mui/icons-material/Person";
import type { CategoriesResponse } from "../types/api";
import { getCategories } from "../services/api";

const CATEGORY_ICONS: Record<string, string> = {
  top: "\uD83D\uDC55",
  bottom: "\uD83D\uDC56",
  shoes: "\uD83D\uDC5F",
  accessory: "\uD83D\uDC5C",
};

const FEATURES = [
  { icon: <AutoAwesomeIcon fontSize="large" />, title: "AI-Powered", desc: "Deep learning compatibility model scores outfit combinations" },
  { icon: <PersonIcon fontSize="large" />, title: "Personalized", desc: "Recommendations adapt to your browsing and purchase history" },
  { icon: <SpeedIcon fontSize="large" />, title: "Real-Time", desc: "FAISS-powered retrieval delivers results in milliseconds" },
];

export default function Home() {
  const navigate = useNavigate();
  const [categories, setCategories] = useState<CategoriesResponse | null>(null);
  const [itemId, setItemId] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    getCategories()
      .then(setCategories)
      .catch(() => setError("Could not connect to API. Is the backend running?"))
      .finally(() => setLoading(false));
  }, []);

  const handleQuickLook = () => {
    const id = parseInt(itemId, 10);
    if (!isNaN(id) && id >= 0) navigate(`/items/${id}`);
  };

  return (
    <Box>
      {/* Hero */}
      <Box sx={{ textAlign: "center", py: { xs: 4, md: 8 } }}>
        <Typography variant="h3" gutterBottom>
          Complete the Look
        </Typography>
        <Typography variant="h6" color="text.secondary" sx={{ maxWidth: 600, mx: "auto", mb: 4 }}>
          Pick any fashion item and our deep learning engine will build a full outfit around it.
        </Typography>

        <Stack direction="row" spacing={1} sx={{ justifyContent: "center", maxWidth: 420, mx: "auto" }}>
          <TextField
            size="small"
            placeholder="Enter item ID (e.g. 42)"
            value={itemId}
            onChange={(e) => setItemId(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleQuickLook()}
            slotProps={{
              input: {
                startAdornment: (
                  <InputAdornment position="start">
                    <SearchIcon />
                  </InputAdornment>
                ),
              },
            }}
            sx={{ flex: 1 }}
          />
          <Button variant="contained" color="secondary" onClick={handleQuickLook}>
            Go
          </Button>
        </Stack>
      </Box>

      {error && <Alert severity="warning" sx={{ mb: 3 }}>{error}</Alert>}

      {/* Category Cards */}
      <Typography variant="h5" sx={{ mb: 2 }}>
        Browse by Category
      </Typography>

      {loading ? (
        <Box sx={{ textAlign: "center", py: 4 }}><CircularProgress /></Box>
      ) : (
        <Grid container spacing={2} sx={{ mb: 6 }}>
          {categories &&
            Object.entries(categories.categories).map(([cat, count]) => (
              <Grid size={{ xs: 6, sm: 3 }} key={cat}>
                <Card>
                  <CardActionArea onClick={() => navigate(`/catalog?category=${cat}`)}>
                    <CardContent sx={{ textAlign: "center", py: 3 }}>
                      <Typography variant="h3" sx={{ mb: 1 }}>
                        {CATEGORY_ICONS[cat] ?? <CheckroomIcon />}
                      </Typography>
                      <Typography variant="h6" sx={{ textTransform: "capitalize" }}>
                        {cat === "top" ? "Tops" : cat === "bottom" ? "Bottoms" : cat === "accessory" ? "Accessories" : "Shoes"}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {count.toLocaleString()} items
                      </Typography>
                    </CardContent>
                  </CardActionArea>
                </Card>
              </Grid>
            ))}
        </Grid>
      )}

      {/* Features */}
      <Typography variant="h5" sx={{ mb: 2 }}>
        How It Works
      </Typography>
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {FEATURES.map((f) => (
          <Grid size={{ xs: 12, md: 4 }} key={f.title}>
            <Card sx={{ height: "100%" }}>
              <CardContent sx={{ textAlign: "center", py: 4 }}>
                <Box sx={{ color: "secondary.main", mb: 1 }}>{f.icon}</Box>
                <Typography variant="h6" gutterBottom>{f.title}</Typography>
                <Typography variant="body2" color="text.secondary">{f.desc}</Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
}
