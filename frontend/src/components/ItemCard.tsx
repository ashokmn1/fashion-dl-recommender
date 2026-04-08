import {
  Card,
  CardActionArea,
  CardMedia,
  CardContent,
  Typography,
  Chip,
  Stack,
} from "@mui/material";
import { useNavigate } from "react-router-dom";
import type { Item } from "../types/api";
import { getItemImageUrl } from "../services/api";

interface Props {
  item: Item;
}

export default function ItemCard({ item }: Props) {
  const navigate = useNavigate();

  return (
    <Card>
      <CardActionArea onClick={() => navigate(`/items/${item.item_id}`)}>
        <CardMedia
          component="img"
          height="220"
          image={getItemImageUrl(item.item_id)}
          alt={item.description}
          sx={{ objectFit: "cover", bgcolor: "#f0f0f0" }}
        />
        <CardContent>
          <Typography variant="subtitle1" noWrap>
            {item.subcategory.charAt(0).toUpperCase() + item.subcategory.slice(1)}
          </Typography>
          <Typography variant="body2" color="text.secondary" noWrap>
            {item.description}
          </Typography>
          <Typography variant="h6" color="secondary" sx={{ mt: 1 }}>
            ${item.price.toFixed(2)}
          </Typography>
          <Stack direction="row" spacing={0.5} sx={{ mt: 1, flexWrap: "wrap", gap: 0.5 }}>
            <Chip label={item.color} size="small" variant="outlined" />
            <Chip label={item.category} size="small" color="primary" />
            <Chip label={item.season} size="small" variant="outlined" />
          </Stack>
        </CardContent>
      </CardActionArea>
    </Card>
  );
}
