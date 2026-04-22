# Hospitality Use Case (Hyatt) — Design

## Goal

Add a hospitality configuration to Retail Vision. Users watch a hotel video, click on detected objects (pool, bar, coffee shop, etc.), and either **book an experience** (for amenities like pools) or **purchase products** (for bars/coffee shops). Reuse the existing multi-brand pattern — almost no backend changes.

## Approach

Extend the existing `brands.ts` multi-brand system (currently: `under-armour`, `blend360`) with a new `hyatt` brand and a per-brand **catalog** that maps YOLOE class names to cart entries. The `ShoppingCart` component becomes kind-aware so it can render both products and bookable experiences.

## Changes

### 1. `src/config/brands.ts`

Extend `BrandConfig`:

```ts
type ProductCatalogEntry = {
  kind: 'product';
  title: string;                 // panel header when this class is dominant
  items: Array<{
    name: string;
    price: number;
    size?: string;               // default size if applicable
    sizes?: string[];            // selectable sizes; defaults to existing XS..XXL if absent
    colors?: string[];           // selectable colors; absent means no color selector
  }>;
};

type ExperienceCatalogEntry = {
  kind: 'experience';
  title: string;                 // "Book an Experience"
  options: Array<{
    name: string;                // "Water aerobics"
    time: string;                // "9:00 - 10:00 AM"
    duration: string;            // "45 min"
  }>;
};

type CatalogEntry = ProductCatalogEntry | ExperienceCatalogEntry;

export interface BrandConfig {
  // ...existing fields...
  catalog?: Record<string, CatalogEntry>;  // keyed by YOLOE class_name (lowercase)
}
```

Add a new `hyatt` brand:

- `yoloeClasses`: `['pool', 'bar', 'coffee shop', 'lounge chair', 'umbrella', 'restaurant']`. If any class detects poorly in the supplied video, we'll trim the list — but the spec commits to this set as the starting point.
- `catalog`:
  - `pool` → experience with: Water aerobics (9:00–10:00 AM), Aqua yoga (11:00 AM–12:00 PM), Marco polo championship (2:00–3:00 PM), Zumba classes (5:00–6:00 PM). All 45 min duration.
  - `coffee shop` → product with: Espresso, Americano, Latte, Cappuccino. All $5.99, default size `Medium`, selectable sizes `Small/Medium/Large`. No color selector.
  - `bar` → product with: Mojito ($12), Margarita ($13), Old Fashioned ($14), House Red ($11). No size or color selector.
- `logo`: new `hyatt-logo.png` in `src/assets/` (user-supplied).
- `videoUrl` / `videoPath`: `Hyatt.mp4` under `retail-vision-ui/public/` (user-supplied).
- `tagline`: `Hyatt — Hospitality` (finalized with user if they prefer something else).

The existing `under-armour` and `blend360` brands gain no `catalog` and keep their current behavior (fallback path in `addToCart`).

### 2. `CartItem` type (in `App.tsx` + `ShoppingCart.tsx`)

Convert `CartItem` to a discriminated union:

```ts
type ProductCartItem = {
  kind: 'product';
  id: string;
  name: string;
  price: number;
  quantity: number;
  detectionId: number;
  confidence: number;
  size?: string;
  color?: string;
};

type ExperienceCartItem = {
  kind: 'experience';
  id: string;
  name: string;           // "Water aerobics"
  time: string;
  duration: string;
  selected: boolean;      // checkbox state; drives "N Activities Selected" count
  detectionId: number;
  confidence: number;
};

type CartItem = ProductCartItem | ExperienceCartItem;
```

Existing clothing items continue to work because the `kind: 'product'` default path preserves `size`/`color` behavior.

### 3. `App.tsx` — `addToCart` becomes catalog-driven

Rewrite `addToCart(detection)`:

1. Look up `brand.catalog?.[detection.class_name.toLowerCase()]`.
2. If entry is `experience` → add **one ExperienceCartItem per option** in the catalog entry (matches inspiration: clicking pool surfaces all activities as a selectable list). Alternative chosen: add the whole list so the user can check/uncheck in the cart — simpler than introducing a separate picker component.
3. If entry is `product` → add one ProductCartItem per item in the catalog entry (same rationale; user can remove unwanted ones).
4. If no entry and class is a clothing class → existing behavior ($99 + size/color).
5. Otherwise → existing behavior ($99 fallback).

Dedup rule: before adding, skip items with the same `(kind, name, detectionId)` that are already in the cart for the same inference session (reuses `addedDetectionsRef` in `InferencePanel.tsx`).

### 4. `ShoppingCart.tsx` — kind-aware rendering

- Items grouped by `kind` with two sub-sections:
  - **"Book an Experience"** section (if any experience items) — each row is a checkbox + activity name + time + duration. Checking toggles `selected`. Footer shows "N Activities Selected" (counting `selected: true`) with a "Book Now" button.
  - **"Shopping Cart"** section (if any product items) — current rendering (qty, size, color) with "Checkout" button.
- Empty state unchanged.
- Total is computed from product items only. Experience items have no price and contribute nothing to total.
- Items of the first two added activities default to `selected: true` (matches inspiration image: first two checked, last two unchecked).

Props expand with a new `onToggleSelected(itemId)` handler. `onUpdateSize` / `onUpdateColor` only apply to product items; the branch guards handle this. "Book Now" is a visual CTA only — clicking it doesn't integrate with any booking system.

### 5. Video file path override

`main.py` currently honors `VIDEO_PATH` env var (line 96). For local dev with the hospitality brand, either:

- Set `VIDEO_PATH=../public/Hyatt.mp4` when starting the backend, **or**
- Make the backend read the brand from `BRAND` env var and map to `videoPath` via a small dict (preferred — keeps frontend and backend switchable by one env var).

Chosen: backend gets a tiny `BRAND_VIDEO_MAP` dict; if `BRAND` is set and known, it overrides `VIDEO_PATH`. Frontend-driven `REACT_APP_BRAND` and backend-driven `BRAND` stay separate env vars so docker-compose can set both.

### 6. Launch ergonomics

Update `quick_launch.sh` (or add a `quick_launch_hyatt.sh` sibling) so `BRAND=hyatt REACT_APP_BRAND=hyatt ./quick_launch.sh` boots the hospitality variant with the right video and classes.

## Out of Scope

- Real booking backend / payment flow — the "Book Now" and "Checkout" buttons remain visual only.
- Per-option prices inside a single experience (all activities are free/included in the demo).
- Dynamic catalog editing from the UI.
- Changes to inference pipeline, model loading, or supervision annotation.

## Testing

- Frontend: load app with `REACT_APP_BRAND=hyatt`, verify Hyatt logo renders, verify clicking a pool in the video adds experience items (not products), clicking a coffee shop adds product items.
- Regression: run with `REACT_APP_BRAND=under-armour` and `REACT_APP_BRAND=blend360`, confirm existing clothing detection + cart behavior unchanged.
- No automated tests required for this demo-focused feature; manual check in browser per `CLAUDE.md` guidance.

## Assets Required From User

- `Hyatt.mp4` → `retail-vision-ui/public/`
- `hyatt-logo.png` → `retail-vision-ui/src/assets/`
- Final tagline string (default: `Hyatt — Hospitality`)
- Confirmation or edits to the drink/experience catalog listed above (the implementation will ship with those defaults if no feedback is given).
