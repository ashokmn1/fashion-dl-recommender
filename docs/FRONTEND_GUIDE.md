# Frontend Technical Guide

## Overview

The frontend is a single-page application built with **React 19**, **TypeScript**, and **Material UI 9**. It provides a visual interface for browsing fashion items and generating AI-powered outfit recommendations using the backend API.

---

## Tech Stack

| Technology       | Version | Purpose                                |
|------------------|---------|----------------------------------------|
| React            | 19.x    | UI framework                           |
| TypeScript       | 5.x     | Type safety                            |
| Material UI      | 9.x     | Component library and theming          |
| React Router     | 7.x     | Client-side routing                    |
| Axios            | 1.x     | HTTP client for API calls              |
| Vite             | 8.x     | Build tool and dev server              |
| Emotion          | 11.x    | CSS-in-JS (MUI styling engine)         |

---

## Project Structure

```
frontend/src/
├── main.tsx                  # React root, renders <App /> in StrictMode
├── App.tsx                   # ThemeProvider + BrowserRouter + route definitions
├── theme.ts                  # MUI custom theme (colors, typography, card effects)
├── index.css                 # Global CSS reset (body margin, font smoothing)
├── types/
│   └── api.ts                # TypeScript interfaces matching backend Pydantic models
├── services/
│   └── api.ts                # Axios client with all API endpoint functions
├── components/
│   ├── Layout.tsx            # App shell: header, navigation, footer
│   ├── ItemCard.tsx          # Product card used in catalog grid
│   └── OutfitCard.tsx        # Outfit display with item gallery and score
└── pages/
    ├── Home.tsx              # Landing page with search and category cards
    ├── Catalog.tsx           # Item browsing with filters and pagination
    └── ItemDetail.tsx        # Item view + "Complete the Look" generator
```

---

## Application Flow

### Startup

```
main.tsx
  └── <App />
        ├── ThemeProvider (custom MUI theme)
        ├── CssBaseline (global CSS normalization)
        └── BrowserRouter
              └── <Layout />         ← persistent header + footer
                    ├── /              → <Home />
                    ├── /catalog       → <Catalog />
                    └── /items/:id     → <ItemDetail />
```

### Page Flow

```
┌──────────────────────────────────────────────────────────┐
│                        Home (/)                          │
│                                                          │
│  ┌─────────────────────────────────────────────┐         │
│  │  Quick Search: Enter item ID → Go           │         │
│  └────────────────────┬────────────────────────┘         │
│                       │ navigates to /items/:id          │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────────┐        │
│  │  Tops  │ │Bottoms │ │ Shoes  │ │Accessories │        │
│  │ 1500   │ │ 1250   │ │ 1250   │ │   1000     │        │
│  └───┬────┘ └───┬────┘ └───┬────┘ └─────┬──────┘        │
│      └──────────┴──────────┴─────────────┘               │
│                       │ navigates to /catalog?category=   │
└───────────────────────┼──────────────────────────────────┘
                        ▼
┌──────────────────────────────────────────────────────────┐
│                   Catalog (/catalog)                     │
│                                                          │
│  Filters: [Category ▼] [Color ▼]                        │
│                                                          │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                    │
│  │ Item │ │ Item │ │ Item │ │ Item │  ← ItemCard grid   │
│  │ Card │ │ Card │ │ Card │ │ Card │                     │
│  └──┬───┘ └──────┘ └──────┘ └──────┘                    │
│     │ click navigates to /items/:id                      │
│  [< 1  2  3  4  5 >]  ← client-side pagination          │
└─────┼────────────────────────────────────────────────────┘
      ▼
┌──────────────────────────────────────────────────────────┐
│                 Item Detail (/items/:id)                 │
│                                                          │
│  ┌────────────┐  Subcategory Name                        │
│  │            │  $29.99                                   │
│  │   Image    │  Description text...                     │
│  │  (350px)   │  [top] [navy] [cotton] [solid] ...       │
│  └────────────┘                                          │
│                                                          │
│  ┌─────────────────────────────────────────────┐         │
│  │  ✨ Complete the Look                       │         │
│  │                                             │         │
│  │  User ID: [___]  Outfits: ──●────── 3       │         │
│  │                                             │         │
│  │  [ ✨ Generate Outfits ]                    │         │
│  └─────────────────────┬───────────────────────┘         │
│                        │ POST /api/recommend/complete-look│
│                        ▼                                  │
│  ┌─────────────────────────────────────────────┐         │
│  │  3 Outfits Found  [Personalized] [145ms]    │         │
│  │                                             │         │
│  │  Outfit 1                    [casual]       │         │
│  │  ┌──────┐┌──────┐┌──────┐┌──────┐          │         │
│  │  │ top  ││bottom││shoes ││acces.│          │         │
│  │  │$29.99││$45.00││$65.00││$120  │          │         │
│  │  └──────┘└──────┘└──────┘└──────┘          │         │
│  │  Match: 78%  ████████████░░░░               │         │
│  │                        Total: $259.99       │         │
│  │                                             │         │
│  │  Outfit 2 ...                               │         │
│  │  Outfit 3 ...                               │         │
│  └─────────────────────────────────────────────┘         │
└──────────────────────────────────────────────────────────┘
```

---

## Routing

| Route         | Page        | Data Fetched on Mount                    |
|---------------|-------------|------------------------------------------|
| `/`           | Home        | `GET /api/categories`                    |
| `/catalog`    | Catalog     | `GET /api/items?category=&color=`        |
| `/items/:id`  | ItemDetail  | `GET /api/items/:id`                     |

Query parameters:
- `/catalog?category=top` pre-selects the category filter
- `/catalog?category=shoes&color=black` applies both filters

---

## API Integration

### Service Layer (`services/api.ts`)

All API calls go through a single Axios instance with `baseURL` set to `/api`. The Vite dev server proxies `/api/*` requests to the backend at `http://localhost:8000`, stripping the `/api` prefix.

```
Browser                    Vite Dev Server              Backend
  │                              │                         │
  │  GET /api/categories         │                         │
  │─────────────────────────────>│  GET /categories        │
  │                              │────────────────────────>│
  │                              │  { categories: {...} }  │
  │                              │<────────────────────────│
  │  { categories: {...} }       │                         │
  │<─────────────────────────────│                         │
```

### API Functions

| Function              | HTTP Method | Backend Endpoint               | Used By         |
|-----------------------|-------------|--------------------------------|-----------------|
| `getCategories()`     | GET         | `/categories`                  | Home            |
| `searchItems(params)` | GET         | `/items?category=&color=`      | Catalog         |
| `getItem(id)`         | GET         | `/items/{id}`                  | ItemDetail      |
| `getItemImageUrl(id)` | URL builder | `/items/{id}/image`            | ItemCard, OutfitCard, ItemDetail |
| `getRecommendations(req)` | POST   | `/recommend/complete-look`     | ItemDetail      |
| `getHealth()`         | GET         | `/health`                      | Available, not yet used |

### Image Loading

Product images are served by the backend at `/items/{id}/image`. The `getItemImageUrl()` function builds the full URL (e.g., `/api/items/42/image`) which is used as the `src` for `<img>` tags via MUI's `CardMedia` component.

---

## Component Details

### Layout (`components/Layout.tsx`)

The persistent app shell wrapping all pages via React Router's `<Outlet />`.

| Section    | Desktop                                   | Mobile (< 600px)                  |
|------------|-------------------------------------------|-----------------------------------|
| Header     | AppBar with inline nav buttons (Home, Catalog) | AppBar with hamburger menu icon |
| Navigation | Buttons in AppBar with active underline   | Slide-out Drawer with list items  |
| Content    | `<Container maxWidth="lg">` with padding  | Same, responsive                  |
| Footer     | Centered text bar with dark background    | Same                              |

### ItemCard (`components/ItemCard.tsx`)

Displays a single fashion item as a clickable card.

| Element      | Content                                      |
|--------------|----------------------------------------------|
| Image        | 220px height, fetched from `/api/items/{id}/image` |
| Title        | Subcategory name (capitalized)               |
| Description  | Product description (single line, truncated) |
| Price        | Formatted as `$XX.XX` in secondary color     |
| Chips        | Color (outlined), Category (primary), Season (outlined) |
| Click action | Navigates to `/items/{id}`                   |

### OutfitCard (`components/OutfitCard.tsx`)

Displays one recommended outfit with all its items.

| Element           | Content                                              |
|-------------------|------------------------------------------------------|
| Header            | "Outfit N" title + style tag chips                   |
| Item gallery      | Horizontal scrollable row of item images (140x150px) |
| Each item         | Image, subcategory, color, price — clickable to detail |
| Compatibility bar | Percentage + LinearProgress (green/yellow/red)       |
| Total price       | Sum of all item prices                               |

Compatibility score color thresholds:
- Green (`success`): score >= 70%
- Yellow (`warning`): score >= 50%
- Red (`error`): score < 50%

---

## Page Details

### Home Page

**Mount:** Calls `getCategories()` to fetch category names and item counts.

**Sections:**
1. **Hero** — Title, subtitle, and a quick-search text field. Entering an item ID and pressing Enter or Go navigates to `/items/{id}`.
2. **Category Cards** — Four cards (Tops, Bottoms, Shoes, Accessories) with emoji icons and item counts. Clicking navigates to `/catalog?category={cat}`.

**States:**
- Loading: CircularProgress spinner while categories load
- Error: Warning alert if the backend is unreachable

### Catalog Page

**Mount:** Calls `searchItems()` with current filter params from the URL query string.

**Filters:**
- Category dropdown: All / top / bottom / shoes / accessory
- Color dropdown: All / 12 color options

Changing a filter updates the URL query params (`useSearchParams`), which triggers a re-fetch. This makes filtered views shareable via URL.

**Pagination:** Client-side, 20 items per page. Backend returns up to its default limit per request.

**States:**
- Loading: Spinner during fetch
- Empty: Info alert when no items match filters
- Error: Error alert on fetch failure

### Item Detail Page

**Mount:** Calls `getItem(id)` to load full item metadata.

**Layout:**
- Left column (md:5): Item image in a Card
- Right column (md:7): Name, price, description, attribute chips, item ID

**Complete the Look Panel:**
- User ID text field (optional, enables personalization)
- Number of outfits slider (1-10, default 3)
- Generate Outfits button triggers `getRecommendations()`

**Results Section:**
- Header showing outfit count + badges for personalization status and latency
- Stack of OutfitCard components, one per recommendation
- Each outfit item is clickable, navigating to its own detail page

**States:**
- Loading item: Full-page spinner
- Item not found: Error alert with back button
- Generating outfits: Spinner with "Finding the best outfit combinations..." message
- Recommendation error: Error alert

---

## Theming (`theme.ts`)

| Token              | Value                        | Usage                      |
|--------------------|------------------------------|----------------------------|
| `primary.main`     | `#1a1a2e` (dark navy)        | AppBar, footer, chips      |
| `secondary.main`   | `#e94560` (red)              | Prices, buttons, accents   |
| `background.default` | `#f8f9fa` (light grey)     | Page background            |
| `background.paper` | `#ffffff`                    | Cards                      |
| `shape.borderRadius` | `12px`                     | All components             |
| Font family        | Inter, Roboto, Helvetica     | All text                   |

Card hover effect: `translateY(-4px)` + increased box shadow on hover.

---

## Dev Server & Proxy

**Vite config (`vite.config.ts`):**

```typescript
server: {
  port: 3000,
  proxy: {
    "/api": {
      target: "http://localhost:8000",
      rewrite: (path) => path.replace(/^\/api/, ""),
    },
  },
}
```

- Frontend dev server: `http://localhost:3000`
- All `/api/*` requests are proxied to the backend at `http://localhost:8000` with the `/api` prefix stripped
- In production, configure a reverse proxy (nginx) or set `VITE_API_URL` to the backend URL

### Environment Variables

| Variable       | Default | Purpose                              |
|----------------|---------|--------------------------------------|
| `VITE_API_URL` | `/api`  | Base URL for all API calls. Override for production deployment |

---

## Build & Run

### Development

```bash
# Terminal 1 — Backend
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Terminal 2 — Frontend
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000` in the browser.

### Production Build

```bash
cd frontend
npm run build
```

Output goes to `frontend/dist/`. Serve with any static file server, with API requests proxied to the backend.

### Available Scripts

| Script         | Command              | Purpose                         |
|----------------|----------------------|---------------------------------|
| `npm run dev`  | `vite`               | Start dev server with HMR       |
| `npm run build`| `tsc -b && vite build` | Type-check and build for production |
| `npm run preview` | `vite preview`    | Preview the production build    |
| `npm run lint` | `eslint .`           | Run ESLint                      |

---

## Data Flow Summary

```
User Action              Frontend                    API Call                      Backend
─────────────────────────────────────────────────────────────────────────────────────────────
Opens home page     →    Home mounts             →   GET /api/categories       →  Returns 4 categories + counts
Clicks "Tops"       →    Navigate /catalog?       →   GET /api/items?           →  Returns item list
                         category=top                  category=top
Clicks item card    →    Navigate /items/42       →   GET /api/items/42         →  Returns item metadata
Image renders       →    <img src="/api/          →   GET /api/items/42/image   →  Returns PNG file
                         items/42/image">
Clicks "Generate    →    ItemDetail calls         →   POST /api/recommend/      →  Returns outfits with
Outfits"                 getRecommendations()          complete-look                 compatibility scores
Clicks outfit item  →    Navigate /items/156      →   GET /api/items/156        →  Returns item metadata
```
