import { useState } from "react";
import { Outlet, useNavigate, useLocation } from "react-router-dom";
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Box,
  Container,
  IconButton,
  Drawer,
  List,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  useMediaQuery,
  useTheme,
} from "@mui/material";
import MenuIcon from "@mui/icons-material/Menu";
import HomeIcon from "@mui/icons-material/Home";
import CheckroomIcon from "@mui/icons-material/Checkroom";
import StyleIcon from "@mui/icons-material/Style";

const NAV = [
  { label: "Home", path: "/", icon: <HomeIcon /> },
  { label: "Catalog", path: "/catalog", icon: <CheckroomIcon /> },
];

export default function Layout() {
  const navigate = useNavigate();
  const { pathname } = useLocation();
  const theme = useTheme();
  const mobile = useMediaQuery(theme.breakpoints.down("sm"));
  const [drawer, setDrawer] = useState(false);

  return (
    <Box sx={{ display: "flex", flexDirection: "column", minHeight: "100vh" }}>
      <AppBar position="sticky" elevation={0} sx={{ bgcolor: "primary.main" }}>
        <Toolbar>
          {mobile && (
            <IconButton color="inherit" onClick={() => setDrawer(true)} edge="start" sx={{ mr: 1 }}>
              <MenuIcon />
            </IconButton>
          )}
          <StyleIcon sx={{ mr: 1.5 }} />
          <Typography
            variant="h6"
            sx={{ flexGrow: 1, cursor: "pointer", letterSpacing: 0.5 }}
            onClick={() => navigate("/")}
          >
            Fashion Recommender
          </Typography>
          {!mobile &&
            NAV.map((n) => (
              <Button
                key={n.path}
                color="inherit"
                onClick={() => navigate(n.path)}
                sx={{
                  mx: 0.5,
                  borderBottom: pathname === n.path ? "2px solid white" : "2px solid transparent",
                  borderRadius: 0,
                }}
              >
                {n.label}
              </Button>
            ))}
        </Toolbar>
      </AppBar>

      <Drawer open={drawer} onClose={() => setDrawer(false)}>
        <Box sx={{ width: 240, pt: 2 }}>
          <List>
            {NAV.map((n) => (
              <ListItemButton
                key={n.path}
                selected={pathname === n.path}
                onClick={() => {
                  navigate(n.path);
                  setDrawer(false);
                }}
              >
                <ListItemIcon>{n.icon}</ListItemIcon>
                <ListItemText primary={n.label} />
              </ListItemButton>
            ))}
          </List>
        </Box>
      </Drawer>

      <Container maxWidth="lg" sx={{ flex: 1, py: 4 }}>
        <Outlet />
      </Container>

      <Box component="footer" sx={{ py: 3, textAlign: "center", bgcolor: "primary.main", color: "white" }}>
        <Typography variant="body2">
          Fashion DL Recommender &mdash; Deep Learning Powered Outfit Suggestions
        </Typography>
      </Box>
    </Box>
  );
}
