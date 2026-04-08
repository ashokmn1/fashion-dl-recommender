import {
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  Stack,
  LinearProgress,
  CardMedia,
} from "@mui/material";
import { useNavigate } from "react-router-dom";
import type { Outfit } from "../types/api";
import { getItemImageUrl } from "../services/api";

interface Props {
  outfit: Outfit;
  index: number;
}

export default function OutfitCard({ outfit, index }: Props) {
  const navigate = useNavigate();
  const score = Math.round(outfit.compatibility_score * 100);
  const totalPrice = outfit.items.reduce((sum, i) => sum + i.price, 0);

  return (
    <Card>
      <CardContent>
        <Stack direction="row" sx={{ justifyContent: "space-between", alignItems: "center", mb: 2 }}>
          <Typography variant="h6">Outfit {index + 1}</Typography>
          <Stack direction="row" spacing={0.5}>
            {outfit.style_tags.map((tag) => (
              <Chip key={tag} label={tag} size="small" color="secondary" variant="outlined" />
            ))}
          </Stack>
        </Stack>

        <Box sx={{ display: "flex", gap: 1.5, overflowX: "auto", pb: 1 }}>
          {outfit.items.map((item) => (
            <Box
              key={item.item_id}
              sx={{
                minWidth: 140,
                cursor: "pointer",
                textAlign: "center",
                "&:hover": { opacity: 0.85 },
              }}
              onClick={() => navigate(`/items/${item.item_id}`)}
            >
              <CardMedia
                component="img"
                image={getItemImageUrl(item.item_id)}
                alt={item.subcategory}
                sx={{
                  height: 150,
                  width: 140,
                  objectFit: "cover",
                  borderRadius: 2,
                  bgcolor: "#f0f0f0",
                }}
              />
              <Typography variant="caption" sx={{ display: "block", mt: 0.5 }} noWrap>
                {item.subcategory}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {item.color} &middot; ${item.price.toFixed(2)}
              </Typography>
            </Box>
          ))}
        </Box>

        <Stack direction="row" spacing={2} sx={{ alignItems: "center", mt: 2 }}>
          <Typography variant="body2" color="text.secondary" sx={{ minWidth: 100 }}>
            Match: {score}%
          </Typography>
          <LinearProgress
            variant="determinate"
            value={score}
            sx={{ flex: 1, height: 8, borderRadius: 4 }}
            color={score >= 70 ? "success" : score >= 50 ? "warning" : "error"}
          />
        </Stack>

        <Typography variant="body2" color="text.secondary" sx={{ mt: 1, textAlign: "right" }}>
          Total: ${totalPrice.toFixed(2)}
        </Typography>
      </CardContent>
    </Card>
  );
}
