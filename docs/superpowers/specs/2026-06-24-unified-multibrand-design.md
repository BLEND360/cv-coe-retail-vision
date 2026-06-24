# Unified Multi-Brand Retail Vision - Design Spec

Date: 2026-06-24

## Goal

Serve all three Retail Vision brands - BLEND360, Hospitality (Hyatt), and
Under Armour - from a single branch and a single running app, switchable at
runtime via tabs. Every brand keeps its existing functionality and aesthetics:
its own logo, tagline, header layout, YOLOE class set, video, and shopping-cart
layout (retail "unified" vs hospitality "split" booking flow).

Today each brand is selected at **build time** via the `REACT_APP_BRAND` (frontend)
and `BRAND` (backend) environment variables, producing a single-brand build. This
work converts that to **runtime** selection without losing any per-brand behavior.

## Background / Current State

The `retail-vision-hospitality` branch is already the superset:

- All three videos exist in `retail-vision-ui/public/`: `Hyatt.mp4`,
  `The BLEND360 Approach.mp4`, `Under-Armour.mp4`.
- All three logos exist in `retail-vision-ui/src/assets/`.
- `src/config/brands.ts` already defines all three brand configs (logo, tagline,
  logoHeight, videoUrl, videoPath, yoloeClasses, optional catalog).
- `App.tsx`, `ShoppingCart.tsx`, `InferencePanel.tsx`, and `VideoPlayer.tsx` are
  already fully brand-driven; `App.tsx`'s `handleInference` already branches
  between hospitality (catalog-trigger) and retail (single-click fallback) modes,
  and `ShoppingCart` already supports both `split` and `unified` layouts.

The only thing missing is runtime switching. The active brand is currently a
module-level constant (`export const brand`) chosen from `REACT_APP_BRAND`.

### Key constraint

The backend (`retail-vision-ui/backend/main.py`) loads **one** video into a global
`video_cap` at startup and sets a **single** YOLOE class list at init. A code
comment documents that YOLOE cannot change its class **count** after init without
an internal tensor-reshape error (retail = 14 classes, hospitality = 9). Both of
these are single-brand assumptions that must be removed.

## Approach

### Frontend: build-time constant -> runtime context

1. **Brand context.** Introduce a React context exposing
   `{ brandKey, brand, setBrandKey }` and a `useBrand()` hook returning the active
   `BrandConfig`. The `brands` registry in `src/config/brands.ts` is unchanged. The
   default brand key still falls back to `REACT_APP_BRAND` (then `blend360`).

2. **Tab bar.** A full-width MUI tab bar pinned at the very top of the app, above
   the "Retail Vision" header: `[ BLEND360 ] [ Hospitality ] [ Under Armour ]`.
   Selecting a tab calls `setBrandKey`. The per-brand logo, tagline, header
   spacing, and cart layout continue to render below it, so each brand's aesthetic
   is preserved exactly.

3. **Consumers.** `VideoPlayer`, `InferencePanel`, and `ShoppingCart` replace
   `import { brand } from '../config/brands'` with `const brand = useBrand()`.

4. **Clean state on switch.** The video + inference + cart subtree is remounted on
   brand change via `key={brandKey}`, and the cart is cleared. This guarantees no
   stale detections, cart items, or frozen text prompts (e.g. `InferencePanel`'s
   `useState` initializer that captures `brand.yoloeClasses`) leak across brands.

### Backend: one process, per-class-set models + per-brand videos

1. **Separate model instances keyed by class set.** Instead of one model whose
   classes are switched per request (which risks the documented reshape crash and
   rebuilds MobileCLIP embeddings on every switch), maintain a cache of YOLOE
   instances keyed by the **tuple of classes**:
   - `get_model_for_classes(classes)` lazily loads (or returns a cached) YOLOE
     instance with those classes set exactly once at init.
   - Today this yields **two** instances: retail (14 classes, shared by `blend360`
     and `under-armour`) and hospitality (9 classes). Identical class lists share
     one instance automatically; a future brand with a new class list gets its own
     instance with no code change.
   - Each model's classes are set once and never changed -> the reshape error
     cannot occur.

2. **Shared MobileCLIP.** MobileCLIP is loaded once globally and used only to build
   text embeddings at model init, so it is not duplicated per model. The text-PE
   helper (`_get_text_pe_direct`) and inference helpers are refactored to take the
   target model instance as a parameter instead of referencing the global
   `yolo_e_model`.

3. **Per-brand video cache.** Maintain a cache of `cv2.VideoCapture` objects keyed
   by brand, opened lazily from each brand's `videoPath`. `get_frame_at_time` reads
   from the capture for the requested brand. The existing `/videos/{name}`
   streaming endpoint already serves all three files, so playback is unchanged.

4. **Request shape.** The click-inference request gains a `brand` field. The
   backend uses it to select both the model (via that brand's class set) and the
   video capture. The frontend sends the active `brandKey` with each inference
   request.

5. **Backward compatibility.** The `BRAND` / `REACT_APP_BRAND` env vars remain as
   harmless fallbacks used only to pick the default tab / default video, so Docker
   images and `quick_launch.sh` keep working. If a request omits `brand`, the
   backend falls back to the env default.

## Components and Responsibilities

| Unit | Responsibility | Depends on |
| --- | --- | --- |
| `brands.ts` registry | Static per-brand config (unchanged) | assets, `types` |
| Brand context + `useBrand()` | Hold active brand key, expose config + setter | `brands.ts` |
| Tab bar (in `App.tsx`) | Switch active brand | brand context |
| `VideoPlayer` | Play active brand's video, report clicks | `useBrand()` |
| `InferencePanel` | Send click + brand to backend, surface detections | `useBrand()` |
| `ShoppingCart` | Render cart per layout (split / unified) | layout prop from `App` |
| `App.handleInference` | Map detections to cart items per brand mode | active brand |
| Backend model cache | One YOLOE instance per class set | YOLOE, shared MobileCLIP |
| Backend video cache | One VideoCapture per brand | brand `videoPath` |
| Inference endpoint | Pick model + video by `brand`, run inference | both caches |

## Data Flow

User selects brand tab -> brand context updates -> subtree remounts, cart clears,
`VideoPlayer` loads that brand's video. User clicks video frame -> frontend sends
`POST` inference with coordinates, timestamp, and `brand` -> backend selects the
brand's video capture (extract frame) and the class-set model (run inference) ->
returns detections + annotated frame -> `handleInference` maps detections to cart
items using the active brand's mode (catalog triggers for hospitality, single-click
fallback for retail) -> cart renders in that brand's layout.

## Error Handling

- Unknown / missing `brand` in a request -> fall back to the env default brand.
- A brand's video file missing -> that brand's capture fails to open; inference
  returns the existing "frame not found" path; other brands are unaffected.
- Model load failure for one class set -> logged; other models unaffected; that
  brand surfaces the existing "no model available" error.
- Brand switch in the UI always clears the cart and remounts, so a partial /
  in-flight inference from a previous brand cannot pollute the new brand's state.

## Testing

- Frontend: switching tabs swaps logo, tagline, header layout, video, and cart
  layout; cart resets on switch; clicking a video produces brand-appropriate cart
  items (booking experiences for Hyatt, products for retail).
- Backend: inference requests with each `brand` value return detections restricted
  to that brand's classes; the retail brands share one model instance while
  hospitality uses a separate one; videos are sourced per brand.
- Regression: existing single-brand Docker build (driven by env vars) still works.

## Out of Scope

- No new brands or new detection classes.
- No changes to the YOLOE model weights or the MobileCLIP fix itself.
- No change to the cloud deployment topology (the stack is currently torn down).
- No persistence of cart state across brand switches (switching intentionally
  clears the cart).
