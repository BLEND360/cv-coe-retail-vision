# Hospitality Use Case (Hyatt) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Hyatt hospitality configuration where clicking on hotel objects (pools, bars, coffee shops) surfaces either a bookable-experience list or a purchasable-drinks cart, reusing the existing multi-brand system.

**Architecture:** Extend `src/config/brands.ts` with an optional per-brand `catalog` that maps a YOLOE class name to either a product list or an experience list. `App.tsx`'s `addToCart` becomes catalog-driven. `ShoppingCart.tsx` renders two sections (Experiences, Products) with kind-specific UI. Backend grows a small `BRAND`→video map so a single env var can flip the whole app.

**Tech Stack:** React 18 + TypeScript + MUI (frontend), FastAPI (backend). No new dependencies.

**Conventions (from `CLAUDE.md` + memory):**
- Do NOT add a `Co-Authored-By` trailer to any commit in this project.
- No emojis in source files, commit messages, or any output.
- No automated tests for this demo feature — we verify manually in the browser per CLAUDE.md.

**Source spec:** `docs/superpowers/specs/2026-04-22-hospitality-use-case-design.md`

---

## File Structure

| Path | Action | Responsibility |
|------|--------|----------------|
| `retail-vision-ui/src/types.ts` | Create | Shared `CatalogEntry` and `CartItem` discriminated unions |
| `retail-vision-ui/src/config/brands.ts` | Modify | Add `catalog` to `BrandConfig`; register `hyatt` brand |
| `retail-vision-ui/src/App.tsx` | Modify | Use shared `CartItem` type; rewrite `addToCart` to read `brand.catalog`; add `toggleSelected` handler |
| `retail-vision-ui/src/components/ShoppingCart.tsx` | Modify | Use shared `CartItem` type; split rendering by kind; add checkbox for experiences; "Book Now" CTA |
| `retail-vision-ui/backend/main.py` | Modify | Read `BRAND` env var and map to a video file path before falling back to `VIDEO_PATH` |
| `quick_launch.sh` | Modify | Add `hyatt` brand case |
| `retail-vision-ui/public/Hyatt.mp4` | Create (user-supplied) | Demo video |
| `retail-vision-ui/src/assets/hyatt-logo.png` | Create (user-supplied) | Brand logo |

---

## Task 1: Shared type module for catalog + cart

**Files:**
- Create: `retail-vision-ui/src/types.ts`

- [ ] **Step 1: Create the shared types file**

Write `retail-vision-ui/src/types.ts`:

```ts
export type ProductCatalogItem = {
  name: string;
  price: number;
  size?: string;
  sizes?: string[];
  colors?: string[];
};

export type ExperienceCatalogOption = {
  name: string;
  time: string;
  duration: string;
};

export type ProductCatalogEntry = {
  kind: 'product';
  title: string;
  items: ProductCatalogItem[];
};

export type ExperienceCatalogEntry = {
  kind: 'experience';
  title: string;
  options: ExperienceCatalogOption[];
};

export type CatalogEntry = ProductCatalogEntry | ExperienceCatalogEntry;

export type ProductCartItem = {
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

export type ExperienceCartItem = {
  kind: 'experience';
  id: string;
  name: string;
  time: string;
  duration: string;
  selected: boolean;
  detectionId: number;
  confidence: number;
};

export type CartItem = ProductCartItem | ExperienceCartItem;
```

- [ ] **Step 2: Verify TypeScript compiles**

Run:

```bash
cd retail-vision-ui && npx tsc --noEmit
```

Expected: exits 0 (no errors — the new file is standalone and compiles cleanly).

- [ ] **Step 3: Commit**

```bash
git add retail-vision-ui/src/types.ts
git commit -m "Add shared CartItem and CatalogEntry types for hospitality"
```

---

## Task 2: Extend `BrandConfig` with optional `catalog`

**Files:**
- Modify: `retail-vision-ui/src/config/brands.ts`

- [ ] **Step 1: Import `CatalogEntry` and add the field**

Edit `retail-vision-ui/src/config/brands.ts`. Add the import and extend `BrandConfig`:

```ts
import underArmourLogo from '../assets/under-armour-logo.png';
import blendLogo from '../assets/blend-logo.png';
import type { CatalogEntry } from '../types';

export interface BrandConfig {
  name: string;
  logo: string;
  logoAlt: string;
  logoHeight: string;
  videoUrl: string;
  videoPath: string;
  tagline: string;
  yoloeClasses: string[];
  catalog?: Record<string, CatalogEntry>;
}
```

Do not change the existing `under-armour` or `blend360` entries — they retain their current behavior because `catalog` is optional.

- [ ] **Step 2: Verify TypeScript compiles**

Run:

```bash
cd retail-vision-ui && npx tsc --noEmit
```

Expected: exits 0.

- [ ] **Step 3: Commit**

```bash
git add retail-vision-ui/src/config/brands.ts
git commit -m "Add optional catalog field to BrandConfig"
```

---

## Task 3: Register the `hyatt` brand

**Files:**
- Modify: `retail-vision-ui/src/config/brands.ts`

This task references `../assets/hyatt-logo.png`. The file does not yet exist; a later task (Task 8) places the user-supplied file there. The build will fail at runtime until the asset is present, but TypeScript allows the import (webpack resolves at build time; we rely on Task 8 before `npm start`).

- [ ] **Step 1: Add the import and brand entry**

In `retail-vision-ui/src/config/brands.ts`, add the logo import near the top:

```ts
import hyattLogo from '../assets/hyatt-logo.png';
```

Inside the `brands` record, add a `hyatt` entry after `blend360`:

```ts
  hyatt: {
    name: 'Hyatt',
    logo: hyattLogo,
    logoAlt: 'Hyatt',
    logoHeight: '56px',
    videoUrl: '/videos/Hyatt.mp4',
    videoPath: '../public/Hyatt.mp4',
    tagline: 'Hyatt - Hospitality',
    yoloeClasses: [
      'pool', 'bar', 'coffee shop', 'lounge chair', 'umbrella', 'restaurant',
    ],
    catalog: {
      pool: {
        kind: 'experience',
        title: 'Book an Experience',
        options: [
          { name: 'Water aerobics',          time: '9:00 - 10:00 AM',  duration: '45 min' },
          { name: 'Aqua yoga',               time: '11:00 AM - 12:00 PM', duration: '45 min' },
          { name: 'Marco polo championship', time: '2:00 - 3:00 PM',   duration: '45 min' },
          { name: 'Zumba classes',           time: '5:00 - 6:00 PM',   duration: '45 min' },
        ],
      },
      'coffee shop': {
        kind: 'product',
        title: 'Shopping Cart',
        items: [
          { name: 'Espresso',   price: 5.99, size: 'Medium', sizes: ['Small', 'Medium', 'Large'] },
          { name: 'Americano',  price: 5.99, size: 'Medium', sizes: ['Small', 'Medium', 'Large'] },
          { name: 'Latte',      price: 5.99, size: 'Medium', sizes: ['Small', 'Medium', 'Large'] },
          { name: 'Cappuccino', price: 5.99, size: 'Medium', sizes: ['Small', 'Medium', 'Large'] },
        ],
      },
      bar: {
        kind: 'product',
        title: 'Bar Menu',
        items: [
          { name: 'Mojito',        price: 12 },
          { name: 'Margarita',     price: 13 },
          { name: 'Old Fashioned', price: 14 },
          { name: 'House Red',     price: 11 },
        ],
      },
    },
  },
```

- [ ] **Step 2: Verify TypeScript compiles**

Run:

```bash
cd retail-vision-ui && npx tsc --noEmit
```

Expected: exits 0. (TypeScript only checks the source; the missing PNG asset is a webpack concern handled later.)

- [ ] **Step 3: Commit**

```bash
git add retail-vision-ui/src/config/brands.ts
git commit -m "Register hyatt brand with hospitality catalog"
```

---

## Task 4: Rewrite `App.tsx` for catalog-driven cart

**Files:**
- Modify: `retail-vision-ui/src/App.tsx`

- [ ] **Step 1: Replace the local `CartItem` interface with the shared import**

In `retail-vision-ui/src/App.tsx`, delete the inline `interface CartItem { ... }` block (lines 10-19) and add this import near the existing imports:

```ts
import type { CartItem, ProductCartItem, ExperienceCartItem } from './types';
```

- [ ] **Step 2: Rewrite `addToCart` to consult the brand catalog**

Replace the existing `addToCart` function (currently at lines 138-155) with:

```ts
  const addToCart = (detection: { id: number; class_name: string; confidence: number }) => {
    const baseId = `${detection.id}-${Date.now()}`;
    const className = detection.class_name.toLowerCase();
    const catalogEntry = brand.catalog?.[className];

    if (catalogEntry?.kind === 'experience') {
      const newItems: ExperienceCartItem[] = catalogEntry.options.map((opt, idx) => ({
        kind: 'experience',
        id: `${baseId}-exp-${idx}`,
        name: opt.name,
        time: opt.time,
        duration: opt.duration,
        selected: idx < 2,
        detectionId: detection.id,
        confidence: detection.confidence,
      }));
      setCartItems(prev => [...prev, ...newItems]);
      return;
    }

    if (catalogEntry?.kind === 'product') {
      const newItems: ProductCartItem[] = catalogEntry.items.map((item, idx) => ({
        kind: 'product',
        id: `${baseId}-prod-${idx}`,
        name: item.name,
        price: item.price,
        quantity: 1,
        detectionId: detection.id,
        confidence: detection.confidence,
        ...(item.size !== undefined && { size: item.size }),
        ...(item.colors && item.colors.length > 0 && { color: item.colors[0] }),
      }));
      setCartItems(prev => [...prev, ...newItems]);
      return;
    }

    const clothingItems = ['blazer', 'shirt', 'shorts', 'running pants', 'running shoes', 'jacket', 'gloves'];
    const isClothing = clothingItems.includes(className);
    const fallback: ProductCartItem = {
      kind: 'product',
      id: baseId,
      name: detection.class_name,
      price: 99,
      quantity: 1,
      detectionId: detection.id,
      confidence: detection.confidence,
      ...(isClothing && { size: 'M', color: 'Black' }),
    };
    setCartItems(prev => [...prev, fallback]);
  };
```

- [ ] **Step 3: Update `updateQuantity`, `updateSize`, `updateColor` for the union type**

Still in `App.tsx`, replace the three update functions (currently at lines 161-187) with these kind-guarded versions:

```ts
  const updateQuantity = (itemId: string, newQuantity: number) => {
    if (newQuantity <= 0) {
      removeFromCart(itemId);
      return;
    }
    setCartItems(prev =>
      prev.map(item =>
        item.id === itemId && item.kind === 'product'
          ? { ...item, quantity: newQuantity }
          : item
      )
    );
  };

  const updateSize = (itemId: string, newSize: string) => {
    setCartItems(prev =>
      prev.map(item =>
        item.id === itemId && item.kind === 'product'
          ? { ...item, size: newSize }
          : item
      )
    );
  };

  const updateColor = (itemId: string, newColor: string) => {
    setCartItems(prev =>
      prev.map(item =>
        item.id === itemId && item.kind === 'product'
          ? { ...item, color: newColor }
          : item
      )
    );
  };
```

- [ ] **Step 4: Add `toggleSelected` for experience items**

Immediately after `updateColor`, add:

```ts
  const toggleSelected = (itemId: string) => {
    setCartItems(prev =>
      prev.map(item =>
        item.id === itemId && item.kind === 'experience'
          ? { ...item, selected: !item.selected }
          : item
      )
    );
  };
```

- [ ] **Step 5: Pass `onToggleSelected` to `ShoppingCart`**

In the JSX (around line 382-388), add the new prop:

```tsx
                  <ShoppingCart 
                    cartItems={cartItems}
                    onRemoveItem={removeFromCart}
                    onUpdateQuantity={updateQuantity}
                    onUpdateSize={updateSize}
                    onUpdateColor={updateColor}
                    onToggleSelected={toggleSelected}
                  />
```

- [ ] **Step 6: Verify TypeScript compiles**

Run:

```bash
cd retail-vision-ui && npx tsc --noEmit
```

Expected: `ShoppingCart.tsx` will surface type errors until Task 5 lands, so this step is expected to fail. Do not commit until after Task 5 passes type-check. Note the errors and move on to Task 5.

---

## Task 5: Refactor `ShoppingCart.tsx` to render by kind

**Files:**
- Modify: `retail-vision-ui/src/components/ShoppingCart.tsx`

- [ ] **Step 1: Replace imports and the inline `CartItem` with the shared type**

At the top of `retail-vision-ui/src/components/ShoppingCart.tsx`, update the imports to include `Checkbox` and `FormControlLabel`, and add the shared type import:

```ts
import React from 'react';
import { 
  Box, 
  Typography, 
  Card, 
  CardContent, 
  IconButton, 
  Button,
  Chip,
  Grid,
  Paper,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Checkbox,
  FormControlLabel
} from '@mui/material';
import { 
  ShoppingCart as ShoppingCartIcon, 
  Delete as DeleteIcon,
  Add as AddIcon,
  Remove as RemoveIcon,
  Event as EventIcon,
  AccessTime as AccessTimeIcon
} from '@mui/icons-material';
import type { CartItem, ProductCartItem, ExperienceCartItem } from '../types';
```

Then remove the existing `interface CartItem { ... }` block (currently lines 24-33).

- [ ] **Step 2: Update `ShoppingCartProps` to include `onToggleSelected`**

Replace the existing `ShoppingCartProps` interface (currently lines 35-41) with:

```ts
interface ShoppingCartProps {
  cartItems: CartItem[];
  onRemoveItem: (itemId: string) => void;
  onUpdateQuantity: (itemId: string, newQuantity: number) => void;
  onUpdateSize: (itemId: string, newSize: string) => void;
  onUpdateColor: (itemId: string, newColor: string) => void;
  onToggleSelected: (itemId: string) => void;
}
```

And update the component signature to destructure the new prop:

```ts
const ShoppingCart: React.FC<ShoppingCartProps> = ({ 
  cartItems, 
  onRemoveItem, 
  onUpdateQuantity,
  onUpdateSize,
  onUpdateColor,
  onToggleSelected
}) => {
```

- [ ] **Step 3: Split `cartItems` by kind and update totals**

Replace the existing `totalPrice` / `totalItems` lines (currently lines 50-51) with:

```ts
  const productItems = cartItems.filter((i): i is ProductCartItem => i.kind === 'product');
  const experienceItems = cartItems.filter((i): i is ExperienceCartItem => i.kind === 'experience');
  const totalPrice = productItems.reduce((sum, item) => sum + (item.price * item.quantity), 0);
  const totalItems = productItems.reduce((sum, item) => sum + item.quantity, 0);
  const selectedExperiences = experienceItems.filter(e => e.selected).length;
```

- [ ] **Step 4: Replace the `Cart Items` grid with two kind-specific sections**

Find the block starting with `{/* Cart Items */}` (around line 171) and the matching `{/* Summary */}` block that follows (around line 397). Replace everything from `{/* Cart Items */}` up to (but not including) the closing `</Box>` of the component's root with the following. This consolidates both rendering and footer summary logic into kind-aware sections.

```tsx
      {/* Cart Items */}
      <Box sx={{ 
        flex: 1, 
        overflow: 'auto', 
        mb: 3,
        minHeight: 0,
        '&::-webkit-scrollbar': { width: '6px' },
        '&::-webkit-scrollbar-track': { background: 'rgba(0, 0, 0, 0.02)', borderRadius: '3px' },
        '&::-webkit-scrollbar-thumb': {
          background: 'rgba(0, 0, 0, 0.15)',
          borderRadius: '3px',
          '&:hover': { background: 'rgba(0, 0, 0, 0.25)' },
        },
      }}>
        {experienceItems.length > 0 && (
          <Box sx={{ mb: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
              <EventIcon sx={{ fontSize: 20, color: 'text.primary' }} />
              <Typography variant="h6" sx={{ fontWeight: 600, color: 'text.primary' }}>
                Book an Experience
              </Typography>
            </Box>
            <Grid container spacing={2} sx={{ pr: 1 }}>
              {experienceItems.map(item => (
                <Grid item xs={12} key={item.id}>
                  <Card
                    variant="outlined"
                    sx={{
                      borderColor: item.selected ? 'primary.main' : 'grey.200',
                      borderWidth: item.selected ? 2 : 1,
                      borderRadius: 2,
                    }}
                  >
                    <CardContent sx={{ p: '16px !important' }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <FormControlLabel
                          control={
                            <Checkbox
                              checked={item.selected}
                              onChange={() => onToggleSelected(item.id)}
                            />
                          }
                          label={
                            <Box>
                              <Typography sx={{ fontWeight: 600, color: 'text.primary' }}>
                                {capitalizeFirstLetter(item.name)}
                              </Typography>
                              <Box sx={{ display: 'flex', gap: 2, mt: 0.5, alignItems: 'center' }}>
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                  <AccessTimeIcon sx={{ fontSize: 14, color: 'text.secondary' }} />
                                  <Typography variant="caption" color="text.secondary">{item.time}</Typography>
                                </Box>
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                  <AccessTimeIcon sx={{ fontSize: 14, color: 'text.secondary' }} />
                                  <Typography variant="caption" color="text.secondary">{item.duration}</Typography>
                                </Box>
                              </Box>
                            </Box>
                          }
                        />
                        <IconButton
                          size="small"
                          onClick={() => onRemoveItem(item.id)}
                          sx={{
                            bgcolor: 'rgba(255, 59, 48, 0.1)',
                            color: '#FF3B30',
                            '&:hover': { bgcolor: 'rgba(255, 59, 48, 0.2)' },
                          }}
                        >
                          <DeleteIcon fontSize="small" />
                        </IconButton>
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </Box>
        )}

        {productItems.length > 0 && (
          <Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
              <ShoppingCartIcon sx={{ fontSize: 20, color: 'text.primary' }} />
              <Typography variant="h6" sx={{ fontWeight: 600, color: 'text.primary' }}>
                Shopping Cart
              </Typography>
            </Box>
            <Grid container spacing={2} sx={{ pr: 1 }}>
              {productItems.map((item) => (
                <Grid item xs={12} key={item.id}>
                  <Card 
                    variant="outlined" 
                    sx={{ 
                      bgcolor: 'background.paper',
                      borderColor: 'grey.200',
                      borderWidth: 1,
                      borderRadius: 2,
                      '&:hover': {
                        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
                        transform: 'translateY(-1px)'
                      },
                      transition: 'all 0.2s cubic-bezier(0.25, 0.46, 0.45, 0.94)'
                    }}
                  >
                    <CardContent sx={{ p: '16px !important' }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                        <Box sx={{ flex: 1 }}>
                          <Typography 
                            variant="h6" 
                            sx={{ 
                              mb: isClothingItem(item.name) ? 1.5 : 0.5,
                              color: 'text.primary',
                              fontSize: '1.1rem',
                              fontWeight: 600
                            }}
                          >
                            {capitalizeFirstLetter(item.name)}
                          </Typography>
                          {isClothingItem(item.name) && item.size && item.color && (
                            <Box sx={{ display: 'flex', gap: 1.5, mt: 1 }}>
                              <FormControl size="small" sx={{ minWidth: 80 }}>
                                <InputLabel sx={{ fontSize: '0.75rem' }}>Size</InputLabel>
                                <Select
                                  value={item.size}
                                  label="Size"
                                  onChange={(e) => onUpdateSize(item.id, e.target.value)}
                                  sx={{ fontSize: '0.875rem', '& .MuiOutlinedInput-notchedOutline': { borderColor: 'grey.300' } }}
                                >
                                  <MenuItem value="XS">XS</MenuItem>
                                  <MenuItem value="S">S</MenuItem>
                                  <MenuItem value="M">M</MenuItem>
                                  <MenuItem value="L">L</MenuItem>
                                  <MenuItem value="XL">XL</MenuItem>
                                  <MenuItem value="XXL">XXL</MenuItem>
                                </Select>
                              </FormControl>
                              <FormControl size="small" sx={{ minWidth: 80 }}>
                                <InputLabel sx={{ fontSize: '0.75rem' }}>Color</InputLabel>
                                <Select
                                  value={item.color}
                                  label="Color"
                                  onChange={(e) => onUpdateColor(item.id, e.target.value)}
                                  renderValue={(selected) => (
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                      <ColorCircle color={selected} />
                                    </Box>
                                  )}
                                  sx={{ fontSize: '0.875rem', '& .MuiOutlinedInput-notchedOutline': { borderColor: 'grey.300' } }}
                                >
                                  {['Black','White','Gray','Navy','Red','Blue','Green','Brown','Beige'].map(c => (
                                    <MenuItem key={c} value={c}>
                                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                        <ColorCircle color={c} />
                                      </Box>
                                    </MenuItem>
                                  ))}
                                </Select>
                              </FormControl>
                            </Box>
                          )}
                          {item.size && !isClothingItem(item.name) && (
                            <FormControl size="small" sx={{ minWidth: 80, mt: 1 }}>
                              <InputLabel sx={{ fontSize: '0.75rem' }}>Size</InputLabel>
                              <Select
                                value={item.size}
                                label="Size"
                                onChange={(e) => onUpdateSize(item.id, e.target.value)}
                                sx={{ fontSize: '0.875rem' }}
                              >
                                {['Small','Medium','Large'].map(s => (
                                  <MenuItem key={s} value={s}>{s}</MenuItem>
                                ))}
                              </Select>
                            </FormControl>
                          )}
                        </Box>
                        <IconButton 
                          size="small" 
                          onClick={() => onRemoveItem(item.id)}
                          sx={{ 
                            ml: 1,
                            bgcolor: 'rgba(255, 59, 48, 0.1)',
                            color: '#FF3B30',
                            '&:hover': { bgcolor: 'rgba(255, 59, 48, 0.2)' }
                          }}
                        >
                          <DeleteIcon fontSize="small" />
                        </IconButton>
                      </Box>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <IconButton 
                            size="small"
                            onClick={() => handleQuantityChange(item.id, item.quantity, -1)}
                            disabled={item.quantity <= 1}
                            sx={{
                              bgcolor: 'grey.100', color: 'text.secondary',
                              '&:hover': { bgcolor: 'grey.200' },
                              '&:disabled': { bgcolor: 'grey.50', color: 'grey.400' }
                            }}
                          >
                            <RemoveIcon fontSize="small" />
                          </IconButton>
                          <Typography 
                            variant="h6" 
                            sx={{ minWidth: '32px', textAlign: 'center', color: 'text.primary', fontWeight: 600 }}
                          >
                            {item.quantity}
                          </Typography>
                          <IconButton 
                            size="small"
                            onClick={() => handleQuantityChange(item.id, item.quantity, 1)}
                            sx={{
                              bgcolor: 'grey.100', color: 'text.secondary',
                              '&:hover': { bgcolor: 'grey.200' }
                            }}
                          >
                            <AddIcon fontSize="small" />
                          </IconButton>
                        </Box>
                        <Typography variant="h6" sx={{ color: 'text.primary', fontSize: '1.1rem', fontWeight: 600 }}>
                          ${(item.price * item.quantity).toFixed(2)}
                        </Typography>
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </Box>
        )}
      </Box>

      {/* Footer summaries */}
      {experienceItems.length > 0 && (
        <Paper sx={{
          p: 3,
          mb: productItems.length > 0 ? 2 : 0,
          bgcolor: 'grey.50',
          border: '1px solid',
          borderColor: 'grey.200',
          borderRadius: 2,
          flexShrink: 0,
        }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6" sx={{ fontWeight: 600, color: 'text.primary' }}>
              {selectedExperiences} Activities Selected
            </Typography>
          </Box>
          <Button
            variant="contained"
            fullWidth
            disabled={selectedExperiences === 0}
            sx={{
              py: 2,
              fontSize: '1rem',
              fontWeight: 600,
              borderRadius: 2,
              textTransform: 'none',
              bgcolor: 'primary.main',
              '&:hover': { bgcolor: 'grey.800' }
            }}
          >
            Book Now
          </Button>
        </Paper>
      )}

      {productItems.length > 0 && (
        <Paper sx={{ 
          p: 3, 
          bgcolor: 'grey.50', 
          border: '1px solid', 
          borderColor: 'grey.200',
          borderRadius: 2,
          flexShrink: 0,
        }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6" sx={{ fontWeight: 600, color: 'text.primary' }}>
              Total ({totalItems} items)
            </Typography>
            <Typography variant="h6" sx={{ fontWeight: 600, color: 'text.primary' }}>
              ${totalPrice.toFixed(2)}
            </Typography>
          </Box>
          <Button 
            variant="contained" 
            fullWidth
            sx={{ 
              py: 2,
              fontSize: '1rem',
              fontWeight: 600,
              borderRadius: 2,
              textTransform: 'none',
              bgcolor: 'primary.main',
              '&:hover': { bgcolor: 'grey.800', transform: 'translateY(-1px)' }
            }}
          >
            Checkout
          </Button>
        </Paper>
      )}
```

- [ ] **Step 5: Update the `totalItems` Chip in the header**

The existing header block (around line 143-169) references `totalItems`. Since that now counts only product items, this chip remains correct but will show 0 when only experiences are present. Update the header Chip block to include both counts:

```tsx
      <Box sx={{ mb: 3, display: 'flex', alignItems: 'center', gap: 2 }}>
        <ShoppingCartIcon sx={{ fontSize: 28, color: 'text.primary' }} />
        <Typography variant="h5" component="h2" sx={{ fontWeight: 600, color: 'text.primary' }}>
          Cart
        </Typography>
        <Chip 
          label={totalItems + experienceItems.length}
          size="small"
          sx={{ ml: 'auto', bgcolor: 'primary.main', color: 'white', fontWeight: 600 }}
        />
      </Box>
```

- [ ] **Step 6: Verify TypeScript compiles across the whole frontend**

Run:

```bash
cd retail-vision-ui && npx tsc --noEmit
```

Expected: exits 0.

- [ ] **Step 7: Commit**

```bash
git add retail-vision-ui/src/App.tsx retail-vision-ui/src/components/ShoppingCart.tsx
git commit -m "Render cart by kind: experiences with checkboxes, products with totals"
```

---

## Task 6: Backend `BRAND` env var → video path mapping

**Files:**
- Modify: `retail-vision-ui/backend/main.py`

- [ ] **Step 1: Add the `BRAND_VIDEO_MAP` and resolve from `BRAND` first**

In `retail-vision-ui/backend/main.py`, replace the single line inside `lifespan` that reads the env var (line 96):

```python
    video_path = os.environ.get("VIDEO_PATH", "../public/The BLEND360 Approach.mp4")
```

with:

```python
    BRAND_VIDEO_MAP = {
        "under-armour": "../public/Under-Armour.mp4",
        "blend360":     "../public/The BLEND360 Approach.mp4",
        "hyatt":        "../public/Hyatt.mp4",
    }
    brand_key = os.environ.get("BRAND", "").lower()
    video_path = (
        BRAND_VIDEO_MAP.get(brand_key)
        or os.environ.get("VIDEO_PATH")
        or "../public/The BLEND360 Approach.mp4"
    )
    logger.info(f"Using video path: {video_path} (BRAND={brand_key or 'unset'})")
```

- [ ] **Step 2: Smoke-test the backend startup**

From the repo root:

```bash
cd retail-vision-ui/backend && source venv/bin/activate && BRAND=blend360 python -c "
import os
os.environ.setdefault('BRAND', 'blend360')
from main import app  # import side-effects include logging
print('import ok')
"
```

Expected: `import ok`, and no exception. (Full server start is deferred to Task 8.)

- [ ] **Step 3: Commit**

```bash
git add retail-vision-ui/backend/main.py
git commit -m "Backend: resolve video path from BRAND env var"
```

---

## Task 7: Add `hyatt` case to `quick_launch.sh`

**Files:**
- Modify: `quick_launch.sh`

- [ ] **Step 1: Extend the `case` statement and pass `BRAND` to the backend**

In `quick_launch.sh`, change the `case` block (lines 14-28) to:

```bash
case "$BRAND" in
  under-armour)
    VIDEO_PATH="../public/Under-Armour.mp4"
    echo "Brand: Under Armour"
    ;;
  blend360)
    VIDEO_PATH="../public/The BLEND360 Approach.mp4"
    echo "Brand: BLEND360"
    ;;
  hyatt)
    VIDEO_PATH="../public/Hyatt.mp4"
    echo "Brand: Hyatt"
    ;;
  *)
    echo "Unknown brand: $BRAND"
    echo "Available brands: under-armour, blend360, hyatt"
    exit 1
    ;;
esac
```

Update the backend launch line (line 43) to also export `BRAND` so the new map takes precedence:

```bash
BRAND="$BRAND" VIDEO_PATH="$VIDEO_PATH" nohup python run_backend.py > ../backend.log 2>&1 &
```

Update the usage comment at the top of the file (lines 3-7):

```bash
# Quick Launch Script for Retail Vision
# Usage:
#   ./quick_launch.sh                    # Launch with BLEND360 (default)
#   ./quick_launch.sh under-armour       # Launch with Under Armour
#   ./quick_launch.sh blend360           # Launch with BLEND360
#   ./quick_launch.sh hyatt              # Launch with Hyatt (hospitality)
```

- [ ] **Step 2: Lint the script**

Run:

```bash
bash -n quick_launch.sh
```

Expected: exits 0 (syntax check passes).

- [ ] **Step 3: Commit**

```bash
git add quick_launch.sh
git commit -m "Add hyatt brand to quick_launch.sh"
```

---

## Task 8: Drop in user-supplied assets and verify end-to-end

**Files:**
- Create: `retail-vision-ui/public/Hyatt.mp4` (user-supplied)
- Create: `retail-vision-ui/src/assets/hyatt-logo.png` (user-supplied)

- [ ] **Step 1: Confirm assets are present**

Ask the user to place the files. Then verify:

```bash
ls -lh retail-vision-ui/public/Hyatt.mp4 retail-vision-ui/src/assets/hyatt-logo.png
```

Expected: both files listed with non-zero size.

- [ ] **Step 2: Launch the hospitality variant**

From the repo root:

```bash
./quick_launch.sh hyatt
```

Expected: backend log at `retail-vision-ui/backend.log` contains `Using video path: ../public/Hyatt.mp4` and `YOLO-E v8l model loaded successfully`; frontend compiles without errors in `retail-vision-ui/frontend.log`.

- [ ] **Step 3: Manual verification — hospitality flow**

Open http://localhost:3000. Verify:

- Hyatt logo renders in the upper-left corner.
- Header tagline reads `Hyatt - Hospitality`.
- Video `Hyatt.mp4` plays and is seekable.
- Clicking on a visible **pool** in the video: cart's "Book an Experience" section shows 4 rows (Water aerobics, Aqua yoga, Marco polo championship, Zumba classes); the first two have checkboxes checked; footer reads `2 Activities Selected`. Toggling a checkbox updates the count.
- Clicking on a visible **coffee shop** area: "Shopping Cart" section appears with Espresso/Americano/Latte/Cappuccino, each $5.99, size `Medium` selectable; qty controls work; total reflects $5.99 × 4 = $23.96 plus any existing product items.
- Clicking on a visible **bar** area: adds Mojito/Margarita/Old Fashioned/House Red with the listed prices and no size selector.
- Removing items with the delete button works for both kinds.

- [ ] **Step 4: Regression — existing brands still work**

Stop the running services (Ctrl+C in the `quick_launch.sh` terminal) and re-launch each existing brand:

```bash
./quick_launch.sh blend360
```

Verify the BLEND360 logo renders, video plays, clicking a shirt/jacket adds a single clothing item with size/color selectors (the existing behavior). Stop and repeat:

```bash
./quick_launch.sh under-armour
```

Same verification for Under Armour.

- [ ] **Step 5: Report results and commit any tweaks**

If any UI polish is needed (spacing, colors, copy), fix it in the relevant file and commit with a descriptive message. If everything works without changes, no commit is needed for this step — declare the plan complete.

---

## Self-Review Checklist

Before declaring the plan complete, confirm:

- **Spec coverage:** Every bullet in the design spec is implemented by one of Tasks 1-8. (Types: Task 1-2; hyatt brand: Task 3; catalog-driven addToCart: Task 4; kind-aware cart: Task 5; backend env var: Task 6; launch script: Task 7; assets + verification: Task 8.)
- **Placeholders:** None. All drinks, experiences, prices, and class lists are concrete.
- **Type consistency:** `CartItem`, `ProductCartItem`, `ExperienceCartItem`, `CatalogEntry`, `ProductCatalogEntry`, `ExperienceCatalogEntry` are defined once in `types.ts` and imported everywhere.
- **No `Co-Authored-By`:** All `git commit` commands use a single-line message with no trailer. (Reconfirmed against memory rule for this project.)
