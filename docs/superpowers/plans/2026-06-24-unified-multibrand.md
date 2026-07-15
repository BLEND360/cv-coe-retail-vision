# Unified Multi-Brand Retail Vision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve all three brands (BLEND360, Hospitality/Hyatt, Under Armour) from one running app, switchable at runtime via a top tab bar, with one backend process serving all three videos and class sets.

**Architecture:** Frontend converts the build-time `brand` constant into a React context with a `useBrand()` hook and a top tab bar; the video/inference/cart subtree remounts and the cart clears on switch. Backend keeps a cache of YOLOE model instances keyed by class set (retail's 14 classes shared by blend+UA, hospitality's 9 separate) plus a per-brand video-capture cache; the inference request carries a `brand` field selecting both.

**Tech Stack:** React 18 + TypeScript + Material-UI (Create React App, Jest + React Testing Library), FastAPI + ultralytics YOLOE + OpenCV + MobileCLIP (pytest for backend pure-helper tests).

## Global Constraints

- No emojis in any project files or output.
- No `Co-Authored-By` line in git commits for this project.
- Backend commands run from `retail-vision-ui/backend/` with `source venv/bin/activate` (Python 3.11 venv).
- Frontend commands run from `retail-vision-ui/` (`npm test` uses CRA / react-scripts, watch mode off via `CI=true`).
- Preserve every brand's existing aesthetics: logo, `logoHeight`, tagline, Hyatt header spacing special-case, and cart layout (`split` for hospitality, `unified` for retail).
- Keep `BRAND` (backend) and `REACT_APP_BRAND` (frontend) env vars working as fallbacks for the default brand, so Docker / `quick_launch.sh` keep working.
- The three brand keys are exactly: `blend360`, `hyatt`, `under-armour`.

---

### Task 1: Brand context and `useBrand()` hook

Convert the module-level `brand` constant into runtime React state while keeping the static `brands` registry unchanged.

**Files:**
- Modify: `retail-vision-ui/src/config/brands.ts` (keep registry; keep `brand` export as the env-default for backward compatibility)
- Create: `retail-vision-ui/src/config/BrandContext.tsx`
- Test: `retail-vision-ui/src/config/BrandContext.test.tsx`

**Interfaces:**
- Consumes: `brands` (`Record<string, BrandConfig>`) and `BrandConfig` from `./brands`.
- Produces:
  - `BrandProvider: React.FC<{ children: React.ReactNode }>`
  - `useBrand(): BrandConfig` — the active brand config
  - `useBrandKey(): { brandKey: string; setBrandKey: (k: string) => void }`
  - `DEFAULT_BRAND_KEY: string` — `process.env.REACT_APP_BRAND` if it is a valid key, else `'blend360'`

- [ ] **Step 1: Write the failing test**

```tsx
// retail-vision-ui/src/config/BrandContext.test.tsx
import { render, screen, fireEvent } from '@testing-library/react';
import { BrandProvider, useBrand, useBrandKey } from './BrandContext';

function Probe() {
  const brand = useBrand();
  const { brandKey, setBrandKey } = useBrandKey();
  return (
    <div>
      <span data-testid="key">{brandKey}</span>
      <span data-testid="name">{brand.name}</span>
      <button onClick={() => setBrandKey('hyatt')}>to-hyatt</button>
    </div>
  );
}

test('defaults to blend360 and switches brand at runtime', () => {
  render(
    <BrandProvider>
      <Probe />
    </BrandProvider>
  );
  expect(screen.getByTestId('key').textContent).toBe('blend360');
  expect(screen.getByTestId('name').textContent).toBe('BLEND360');

  fireEvent.click(screen.getByText('to-hyatt'));
  expect(screen.getByTestId('key').textContent).toBe('hyatt');
  expect(screen.getByTestId('name').textContent).toBe('Hyatt');
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd retail-vision-ui && CI=true npx react-scripts test --testPathPattern BrandContext --watchAll=false`
Expected: FAIL — cannot find module `./BrandContext`.

- [ ] **Step 3: Write the context implementation**

```tsx
// retail-vision-ui/src/config/BrandContext.tsx
import React, { createContext, useContext, useMemo, useState } from 'react';
import brands, { BrandConfig } from './brands';

const envKey = process.env.REACT_APP_BRAND;
export const DEFAULT_BRAND_KEY: string =
  envKey && brands[envKey] ? envKey : 'blend360';

interface BrandContextValue {
  brandKey: string;
  setBrandKey: (k: string) => void;
}

const BrandContext = createContext<BrandContextValue | null>(null);

export const BrandProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [brandKey, setBrandKey] = useState<string>(DEFAULT_BRAND_KEY);
  const value = useMemo(() => ({ brandKey, setBrandKey }), [brandKey]);
  return <BrandContext.Provider value={value}>{children}</BrandContext.Provider>;
};

function useBrandContext(): BrandContextValue {
  const ctx = useContext(BrandContext);
  if (!ctx) throw new Error('useBrand must be used within a BrandProvider');
  return ctx;
}

export function useBrandKey(): BrandContextValue {
  return useBrandContext();
}

export function useBrand(): BrandConfig {
  const { brandKey } = useBrandContext();
  return brands[brandKey] || brands['blend360'];
}
```

- [ ] **Step 4: Ensure `brands.ts` exports `BrandConfig` and the default `brand`**

`brands.ts` already exports `BrandConfig` (interface), `brands` (default), and `brand` (const). Leave the `brand` const export in place — it is the env-default and keeps any not-yet-migrated import compiling. No change required unless the build reports a missing export.

- [ ] **Step 5: Run test to verify it passes**

Run: `cd retail-vision-ui && CI=true npx react-scripts test --testPathPattern BrandContext --watchAll=false`
Expected: PASS (2 assertions).

- [ ] **Step 6: Commit**

```bash
git add retail-vision-ui/src/config/BrandContext.tsx retail-vision-ui/src/config/BrandContext.test.tsx
git commit -m "Add runtime brand context and useBrand hook"
```

---

### Task 2: Wrap the app in BrandProvider and add the top tab bar

Add the full-width brand tab bar above the "Retail Vision" header, wire it to the context, and remount + clear the cart on brand switch.

**Files:**
- Modify: `retail-vision-ui/src/index.tsx` (wrap `<App />` in `<BrandProvider>`)
- Modify: `retail-vision-ui/src/App.tsx`
- Test: `retail-vision-ui/src/App.test.tsx`

**Interfaces:**
- Consumes: `BrandProvider`, `useBrand`, `useBrandKey` from `./config/BrandContext`; `brands` from `./config/brands`.
- Produces: an `App` that reads the active brand from context (no longer importing the `brand` constant) and renders a `Tabs` bar with one tab per brand key.

- [ ] **Step 1: Wrap the root in BrandProvider**

In `retail-vision-ui/src/index.tsx`, wrap the rendered `<App />`:

```tsx
import { BrandProvider } from './config/BrandContext';
// ...
root.render(
  <React.StrictMode>
    <BrandProvider>
      <App />
    </BrandProvider>
  </React.StrictMode>
);
```

(If `index.tsx` does not currently use `React.StrictMode`, keep its existing wrapper and just insert `<BrandProvider>` around `<App />`.)

- [ ] **Step 2: Write the failing test**

```tsx
// retail-vision-ui/src/App.test.tsx
import { render, screen, fireEvent } from '@testing-library/react';
import { BrandProvider } from './config/BrandContext';
import App from './App';

function renderApp() {
  return render(
    <BrandProvider>
      <App />
    </BrandProvider>
  );
}

test('renders a tab per brand and switches the tagline on click', () => {
  renderApp();
  // Tabs labelled by brand display name
  expect(screen.getByRole('tab', { name: /BLEND360/i })).toBeInTheDocument();
  expect(screen.getByRole('tab', { name: /Hyatt/i })).toBeInTheDocument();
  expect(screen.getByRole('tab', { name: /Under Armour/i })).toBeInTheDocument();

  // Default brand tagline visible
  expect(screen.getByText(/AI-Powered Retail Intelligence/i)).toBeInTheDocument();

  // Switch to Hyatt; its tagline replaces the default
  fireEvent.click(screen.getByRole('tab', { name: /Hyatt/i }));
  expect(screen.getByText(/Hyatt - Hospitality/i)).toBeInTheDocument();
});
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd retail-vision-ui && CI=true npx react-scripts test --testPathPattern App.test --watchAll=false`
Expected: FAIL — no `tab` roles found (tabs not yet added).

- [ ] **Step 4: Replace the brand import with context in App.tsx**

In `retail-vision-ui/src/App.tsx`:

1. Remove `import { brand } from './config/brands';`
2. Add:

```tsx
import { Tabs, Tab } from '@mui/material';
import brands from './config/brands';
import { useBrand, useBrandKey } from './config/BrandContext';
```

3. At the top of the `App` function body, read brand from context:

```tsx
const brand = useBrand();
const { brandKey, setBrandKey } = useBrandKey();
```

- [ ] **Step 5: Add the tab bar and clear the cart on switch**

Inside `App`, add a handler that switches brand and resets cart-related state:

```tsx
const handleBrandChange = useCallback((_e: React.SyntheticEvent, newKey: string) => {
  setBrandKey(newKey);
  setCartItems([]);
  setLastClickData(null);
  setMenuRefreshVersion(0);
}, [setBrandKey]);
```

Render the tab bar as the first child inside the outermost `<Box>` (above the logo / header `Box`):

```tsx
<Tabs
  value={brandKey}
  onChange={handleBrandChange}
  variant="fullWidth"
  sx={{
    borderBottom: '1px solid rgba(0,0,0,0.08)',
    flexShrink: 0,
    bgcolor: 'background.paper',
    '& .MuiTab-root': { textTransform: 'none', fontWeight: 600, fontSize: '1rem' },
  }}
>
  {Object.entries(brands).map(([key, cfg]) => (
    <Tab key={key} value={key} label={cfg.name} />
  ))}
</Tabs>
```

- [ ] **Step 6: Remount the video/inference/cart subtree on brand switch**

Wrap the "Main Content Grid" `<Box>` (the one with `flex: 1, px: 6, pb: 4`) so it remounts when the brand changes. Add `key={brandKey}` to that `<Box>`:

```tsx
<Box key={brandKey} sx={{ flex: 1, px: 6, pb: 4, minHeight: 0, overflow: 'hidden' }}>
```

This forces `VideoPlayer`, `InferencePanel`, and `ShoppingCart` to unmount/remount on switch, clearing their internal state (including `InferencePanel`'s captured text prompt).

- [ ] **Step 7: Run test to verify it passes**

Run: `cd retail-vision-ui && CI=true npx react-scripts test --testPathPattern App.test --watchAll=false`
Expected: PASS (tabs present, tagline switches).

- [ ] **Step 8: Type-check the build**

Run: `cd retail-vision-ui && CI=true npx tsc --noEmit`
Expected: no errors. (If `tsc` is not directly available, run `CI=true npm run build` and expect a successful compile.)

- [ ] **Step 9: Commit**

```bash
git add retail-vision-ui/src/index.tsx retail-vision-ui/src/App.tsx retail-vision-ui/src/App.test.tsx
git commit -m "Add brand tab bar and runtime brand switching in App"
```

---

### Task 3: Migrate VideoPlayer and InferencePanel to context; send brand to backend

Point the remaining brand consumers at `useBrand()` and include the active brand key in the inference request.

**Files:**
- Modify: `retail-vision-ui/src/components/VideoPlayer.tsx`
- Modify: `retail-vision-ui/src/components/InferencePanel.tsx`
- Test: `retail-vision-ui/src/components/InferencePanel.test.tsx`

**Interfaces:**
- Consumes: `useBrand` and `useBrandKey` from `../config/BrandContext`.
- Produces: an inference request body that includes `brand: <brandKey>` alongside the existing `video_time, x, y, frame_width, frame_height, text_prompt` fields.

- [ ] **Step 1: Migrate VideoPlayer to context**

In `retail-vision-ui/src/components/VideoPlayer.tsx`:
1. Replace `import { brand } from '../config/brands';` with `import { useBrand } from '../config/BrandContext';`
2. Inside the component body (top), add: `const brand = useBrand();`

No other change — `brand.videoUrl` usage stays the same.

- [ ] **Step 2: Write the failing test for InferencePanel request body**

```tsx
// retail-vision-ui/src/components/InferencePanel.test.tsx
import { render, act } from '@testing-library/react';
import { BrandProvider } from '../config/BrandContext';
import InferencePanel from './InferencePanel';

test('inference request includes the active brand key', async () => {
  const fetchMock = jest.fn().mockResolvedValue({
    ok: true,
    json: async () => ({
      timestamp: 1, video_time: 0, clicked_pixel: { x: 1, y: 1 },
      detections: [], frame_base64: '', annotated_frame_base64: '',
      clicked_object: null, inference_type: 'YOLO-E',
    }),
  });
  // @ts-ignore
  global.fetch = fetchMock;

  const click = { x: 1, y: 1, currentTime: 0, frameWidth: 10, frameHeight: 10 };

  await act(async () => {
    render(
      <BrandProvider>
        <InferencePanel lastClickData={click} onInference={() => {}} />
      </BrandProvider>
    );
  });

  expect(fetchMock).toHaveBeenCalled();
  const body = JSON.parse(fetchMock.mock.calls[0][1].body);
  expect(body.brand).toBe('blend360');
  expect(body.x).toBe(1);
});
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd retail-vision-ui && CI=true npx react-scripts test --testPathPattern InferencePanel --watchAll=false`
Expected: FAIL — `body.brand` is `undefined`.

- [ ] **Step 4: Migrate InferencePanel to context and add brand to the request**

In `retail-vision-ui/src/components/InferencePanel.tsx`:

1. Replace `import { brand } from '../config/brands';` with:
```tsx
import { useBrand, useBrandKey } from '../config/BrandContext';
```
2. At the top of the component body add:
```tsx
const brand = useBrand();
const { brandKey } = useBrandKey();
```
3. Change the frozen text-prompt state to track the active brand. Replace:
```tsx
const [textPrompt] = useState<string>(brand.yoloeClasses.join(', '));
```
with:
```tsx
const textPrompt = brand.yoloeClasses.join(', ');
```
4. In `fetchClickInference`, add the brand to `requestBody`:
```tsx
const requestBody: any = {
  video_time: clickData.currentTime,
  x: clickData.x,
  y: clickData.y,
  frame_width: clickData.frameWidth,
  frame_height: clickData.frameHeight,
  brand: brandKey,
};
```
5. Add `brandKey` to the `fetchClickInference` `useCallback` dependency array (it currently lists `[inferenceType, textPrompt]` -> `[inferenceType, textPrompt, brandKey]`).

- [ ] **Step 5: Run test to verify it passes**

Run: `cd retail-vision-ui && CI=true npx react-scripts test --testPathPattern InferencePanel --watchAll=false`
Expected: PASS.

- [ ] **Step 6: Type-check**

Run: `cd retail-vision-ui && CI=true npx tsc --noEmit`
Expected: no errors. Confirm no remaining `import { brand } from '../config/brands'` lines exist in components:

Run: `cd retail-vision-ui && grep -rn "import { brand }" src/ || echo "none"`
Expected: `none`.

- [ ] **Step 7: Commit**

```bash
git add retail-vision-ui/src/components/VideoPlayer.tsx retail-vision-ui/src/components/InferencePanel.tsx retail-vision-ui/src/components/InferencePanel.test.tsx
git commit -m "Migrate VideoPlayer and InferencePanel to brand context; send brand in inference request"
```

---

### Task 4: Filter detections to the active brand in handleInference

Guarantee a brand never processes another brand's detections (defense for the shared-model edge and any cross-brand noise).

**Files:**
- Modify: `retail-vision-ui/src/App.tsx` (the `handleInference` callback)
- Test: covered by the Task 2 `App.test.tsx` plus the manual verification in Task 8 (no new unit test — `handleInference` is an inline closure not exported).

**Interfaces:**
- Consumes: `brand.yoloeClasses` (lowercased) from the active brand.
- Produces: a `handleInference` that ignores detections whose `class_name` is not in the active brand's class set.

- [ ] **Step 1: Add the brand class filter at the top of handleInference**

In `retail-vision-ui/src/App.tsx`, at the start of the `handleInference` callback body (before the `isHospitality` branch), filter both the clicked detection and the detection list:

```tsx
const allowed = new Set(brand.yoloeClasses.map(c => c.toLowerCase()));
const inBrand = (c: { class_name: string }) => allowed.has(c.class_name.toLowerCase());
const brandDetections = detections.filter(inBrand);
const brandClicked = clickedDetection && inBrand(clickedDetection) ? clickedDetection : null;
```

Then use `brandDetections` everywhere the function currently uses `detections`, and `brandClicked` everywhere it uses `clickedDetection` (the `detectedClasses` set, the `clickedDetection?.id` references in the hospitality branch, and the retail fallback's early `if (!clickedDetection) return;` -> `if (!brandClicked) return;` and subsequent `clickedDetection.*` -> `brandClicked.*`).

5. Add `brand` to the `handleInference` `useCallback` dependency array (currently `[]` -> `[brand]`).

- [ ] **Step 2: Type-check**

Run: `cd retail-vision-ui && CI=true npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 3: Run the full frontend test suite**

Run: `cd retail-vision-ui && CI=true npx react-scripts test --watchAll=false`
Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add retail-vision-ui/src/App.tsx
git commit -m "Filter inference detections to the active brand's class set"
```

---

### Task 5: Backend — class-set-keyed model cache

Replace the single global model + per-request class switching with a cache of YOLOE instances keyed by class set, each with classes set once at init.

**Files:**
- Modify: `retail-vision-ui/backend/main.py`
- Test: `retail-vision-ui/backend/test_brand_routing.py`

**Interfaces:**
- Consumes: `BRAND_CLASSES_MAP` (`Dict[str, List[str]]`), shared MobileCLIP loader `_load_mobileclip()`.
- Produces:
  - `classes_for_brand(brand_key: Optional[str]) -> List[str]` — returns that brand's class list, falling back to the `BRAND` env / retail default.
  - `_class_key(classes: List[str]) -> tuple` — order-preserving tuple key.
  - `get_model_for_brand(brand_key: Optional[str]) -> YOLOE` — returns a cached model instance whose classes match the brand (one instance per distinct class set).
  - `_get_text_pe_direct(texts, model)` — now takes the model whose detection head builds the embeddings.

- [ ] **Step 1: Write the failing test for brand->class-key routing**

```python
# retail-vision-ui/backend/test_brand_routing.py
import importlib

main = importlib.import_module("main")


def test_retail_brands_share_one_class_key():
    blend = main._class_key(main.classes_for_brand("blend360"))
    ua = main._class_key(main.classes_for_brand("under-armour"))
    hyatt = main._class_key(main.classes_for_brand("hyatt"))
    assert blend == ua          # retail brands collapse to one model
    assert blend != hyatt       # hospitality is a distinct model
    assert len(main.classes_for_brand("hyatt")) == 9
    assert len(main.classes_for_brand("blend360")) == 14


def test_unknown_brand_falls_back_to_retail():
    assert main.classes_for_brand("does-not-exist") == main.RETAIL_CLASSES
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd retail-vision-ui/backend && source venv/bin/activate && python -m pytest test_brand_routing.py -v`
Expected: FAIL — `module 'main' has no attribute '_class_key'` / `classes_for_brand`.

- [ ] **Step 3: Add the pure routing helpers**

In `retail-vision-ui/backend/main.py`, near `get_startup_classes` (around line 39), add:

```python
def classes_for_brand(brand_key):
    """Return the YOLOE class list for a brand, falling back to the BRAND env
    default and then the retail list."""
    key = (brand_key or os.environ.get("BRAND", "")).lower()
    return BRAND_CLASSES_MAP.get(key, RETAIL_CLASSES)


def _class_key(classes):
    """Order-preserving cache key for a class list."""
    return tuple(classes)
```

- [ ] **Step 4: Run test to verify the routing helpers pass**

Run: `cd retail-vision-ui/backend && source venv/bin/activate && python -m pytest test_brand_routing.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Add the model cache and parametrize the text-PE helper**

In `main.py`:

1. Replace the single `yolo_e_model = None` global (line ~181) with a cache plus a default handle for backward-compatible references:

```python
# Cache of YOLOE instances keyed by class tuple. Each instance has its classes
# set exactly once at load, so YOLOE never reshapes its class head at runtime.
_models = {}
yolo_e_model = None  # default model handle (set at startup) for legacy references
```

2. Change `_get_text_pe_direct(texts: List[str])` to take the model:

```python
@torch.inference_mode()
def _get_text_pe_direct(texts: List[str], model):
    if not _load_mobileclip():
        return None
    tokens = _mobileclip_tokenizer(texts).to(
        next(_mobileclip_model.parameters()).device
    )
    txt_feats = _mobileclip_model.encode_text(tokens)
    txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
    txt_feats = txt_feats.reshape(1, len(texts), txt_feats.shape[-1])
    from ultralytics.nn.modules.head import YOLOEDetect
    head = model.model.model[-1]
    assert isinstance(head, YOLOEDetect)
    return F.normalize(head.reprta(txt_feats), dim=-1, p=2)
```

3. Add a model builder/cache accessor (replaces `load_yolo_e_model` + `_set_classes_cached` per-request switching):

```python
def _build_model_with_classes(classes):
    """Load a fresh YOLOE instance and set `classes` on it exactly once."""
    model_path = "yoloe-v8l-seg.pt"
    if not os.path.exists(model_path):
        if not download_yolo_e_v8l_model_direct():
            raise RuntimeError("Failed to download YOLO-E v8l model")
    model = YOLOE(model_path)
    text_pe = _get_text_pe_direct(classes, model)
    if text_pe is None:
        text_pe = model.get_text_pe(classes)
    model.set_classes(classes, text_pe)
    logger.info(f"Built YOLOE instance for classes: {classes}")
    return model


def get_model_for_brand(brand_key):
    """Return a cached YOLOE instance whose classes match the brand. Brands with
    identical class lists (blend360 + under-armour) share one instance."""
    classes = classes_for_brand(brand_key)
    key = _class_key(classes)
    model = _models.get(key)
    if model is None:
        model = _build_model_with_classes(classes)
        _models[key] = model
    return model
```

4. Delete the old `load_yolo_e_model()` body and `_set_classes_cached()` (and the now-unused `_cached_text_pe` / `_cached_prompt_key` globals). Any remaining reference to `_set_classes_cached` in `run_yolo_e_inference` / `run_yolo_e_v8l_inference` is handled in Task 7.

- [ ] **Step 6: Re-run the routing test (still passes, no model load)**

Run: `cd retail-vision-ui/backend && source venv/bin/activate && python -m pytest test_brand_routing.py -v`
Expected: PASS (the pure helpers are unaffected by the model-cache additions).

- [ ] **Step 7: Commit**

```bash
git add retail-vision-ui/backend/main.py retail-vision-ui/backend/test_brand_routing.py
git commit -m "Backend: cache YOLOE instances per class set, parametrize text-PE helper"
```

---

### Task 6: Backend — per-brand video capture cache

Replace the single global `video_cap` with a per-brand cache so frames are read from the requested brand's video.

**Files:**
- Modify: `retail-vision-ui/backend/main.py`
- Test: `retail-vision-ui/backend/test_brand_routing.py` (extend)

**Interfaces:**
- Consumes: brand keys; the `BRAND_VIDEO_MAP` currently defined inside `lifespan` (promote it to module scope).
- Produces:
  - `video_path_for_brand(brand_key: Optional[str]) -> str`
  - `get_capture_for_brand(brand_key: Optional[str]) -> cv2.VideoCapture | None` (lazily opened, cached in `_video_caps`)
  - `get_frame_at_time(target_time: float, brand_key: Optional[str] = None) -> np.ndarray` (now brand-aware)

- [ ] **Step 1: Promote the brand->video map to module scope**

In `main.py`, move the `BRAND_VIDEO_MAP` dict out of `lifespan` to module scope (near `BRAND_CLASSES_MAP`, ~line 36):

```python
BRAND_VIDEO_MAP = {
    "under-armour": "../public/Under-Armour.mp4",
    "blend360":     "../public/The BLEND360 Approach.mp4",
    "hyatt":        "../public/Hyatt.mp4",
}
```

- [ ] **Step 2: Write the failing test for video path routing**

Append to `retail-vision-ui/backend/test_brand_routing.py`:

```python
def test_video_path_for_brand():
    assert main.video_path_for_brand("hyatt").endswith("Hyatt.mp4")
    assert main.video_path_for_brand("blend360").endswith("The BLEND360 Approach.mp4")
    assert main.video_path_for_brand("under-armour").endswith("Under-Armour.mp4")
    # Unknown brand falls back to a real path string (env/default), not None
    assert isinstance(main.video_path_for_brand("nope"), str)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd retail-vision-ui/backend && source venv/bin/activate && python -m pytest test_brand_routing.py::test_video_path_for_brand -v`
Expected: FAIL — no attribute `video_path_for_brand`.

- [ ] **Step 4: Add the video cache and brand-aware frame reader**

In `main.py`:

1. Replace the `video_cap = None` global with a cache:

```python
_video_caps = {}  # brand_key -> cv2.VideoCapture
```

2. Add the helpers:

```python
def video_path_for_brand(brand_key):
    key = (brand_key or os.environ.get("BRAND", "")).lower()
    return (
        BRAND_VIDEO_MAP.get(key)
        or os.environ.get("VIDEO_PATH")
        or "../public/The BLEND360 Approach.mp4"
    )


def get_capture_for_brand(brand_key):
    key = (brand_key or os.environ.get("BRAND", "")).lower() or "blend360"
    cap = _video_caps.get(key)
    if cap is not None and cap.isOpened():
        return cap
    path = video_path_for_brand(brand_key)
    if not os.path.exists(path):
        logger.error(f"Video file not found for brand {key}: {path}")
        return None
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        logger.error(f"Failed to open video for brand {key}: {path}")
        return None
    _video_caps[key] = cap
    logger.info(f"Opened video capture for brand {key}: {path}")
    return cap
```

3. Rewrite `get_frame_at_time` to be brand-aware:

```python
def get_frame_at_time(target_time: float, brand_key=None) -> np.ndarray:
    cap = get_capture_for_brand(brand_key)
    if cap is None:
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    target_frame = int(target_time * fps)
    if target_frame >= total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        target_frame = 0
        logger.info("Video looped to beginning.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    return frame if ret else None
```

4. In `lifespan`, replace the `load_video(video_path)` startup call with pre-opening the default brand's capture (keeps the old startup-log behavior) and drop the now-unused single-video logic:

```python
default_brand = os.environ.get("BRAND", "").lower() or "blend360"
if get_capture_for_brand(default_brand) is None:
    logger.error("Failed to open default brand video on startup.")
```

5. In shutdown (after `yield`), release all captures:

```python
for cap in _video_caps.values():
    if cap is not None:
        cap.release()
logger.info("All video captures released")
```

6. Update `get_video_status` (and `/api/inference/yolo-e-v8l`, which reads `video_cap`) to use `get_capture_for_brand(None)` for the default; replace `video_cap` references accordingly.

- [ ] **Step 5: Run the video path test to verify it passes**

Run: `cd retail-vision-ui/backend && source venv/bin/activate && python -m pytest test_brand_routing.py -v`
Expected: PASS (all routing + video-path tests).

- [ ] **Step 6: Commit**

```bash
git add retail-vision-ui/backend/main.py retail-vision-ui/backend/test_brand_routing.py
git commit -m "Backend: per-brand video capture cache and brand-aware frame reader"
```

---

### Task 7: Backend — wire brand through the inference endpoint

Add `brand` to the request model and route the endpoint to the brand's model + video. Remove the per-request `_set_classes_cached` calls (classes are now fixed per model).

**Files:**
- Modify: `retail-vision-ui/backend/main.py`
- Test: `retail-vision-ui/backend/test_brand_routing.py` (extend with a request-model test)

**Interfaces:**
- Consumes: `get_model_for_brand`, `get_frame_at_time(time, brand)`.
- Produces: `ClickInferenceRequest` with an added `brand: Optional[str] = None`; `run_yolo_e_inference(frame, clicked_x, clicked_y, model, text_prompt=None)` taking an explicit model.

- [ ] **Step 1: Write the failing test for the request model field**

Append to `test_brand_routing.py`:

```python
def test_click_request_accepts_brand():
    req = main.ClickInferenceRequest(
        video_time=0.0, x=1, y=2, frame_width=10, frame_height=10, brand="hyatt"
    )
    assert req.brand == "hyatt"
    # brand is optional
    req2 = main.ClickInferenceRequest(
        video_time=0.0, x=1, y=2, frame_width=10, frame_height=10
    )
    assert req2.brand is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd retail-vision-ui/backend && source venv/bin/activate && python -m pytest test_brand_routing.py::test_click_request_accepts_brand -v`
Expected: FAIL — `ClickInferenceRequest` has no field `brand`.

- [ ] **Step 3: Add `brand` to the request model**

In `main.py`, add to `ClickInferenceRequest` (after `text_prompt`):

```python
    brand: Optional[str] = None
```

- [ ] **Step 4: Parametrize `run_yolo_e_inference` by model and drop per-request class switching**

In `run_yolo_e_inference`:
1. Change the signature to `def run_yolo_e_inference(frame, clicked_x, clicked_y, model, text_prompt=None):`
2. Replace the `if yolo_e_model is None:` guard with `if model is None:`.
3. Remove the `_set_classes_cached(...)` block (the `set_classes_ms` timing block) — the model's classes are already fixed. Keep a `timings["set_classes_ms"] = 0.0` line so the existing log statement still formats.
4. Replace every `yolo_e_model` reference in the function body with `model` (the class-name fallback that reads `yolo_e_model.names`, and the predict call `yolo_e_model.predict(...)` -> `model.predict(...)`).
5. The `class_name` resolution should prefer `model.names` (now correctly populated from the brand's set classes); keep the existing `text_prompt`-based naming as a fallback.

- [ ] **Step 5: Update the click-inference endpoint to select model + video by brand**

In `get_yolo_e_inference` (the `/api/inference/yolo-e` handler):

```python
frame = get_frame_at_time(request.video_time, request.brand)
# ...
model = get_model_for_brand(request.brand)
inference_result = await asyncio.to_thread(
    run_yolo_e_inference, frame, request.x, request.y, model, request.text_prompt
)
```

- [ ] **Step 6: Fix the v8l endpoint and helpers that referenced the global model**

In `run_yolo_e_v8l_inference`, change its signature to take `model` (`def run_yolo_e_v8l_inference(frame, text_prompt, model, confidence=0.1):`), replace `yolo_e_model` with `model`, remove the `_set_classes_cached` call, and update its fallback `return run_yolo_e_inference(frame, 0, 0, text_prompt)` to `return run_yolo_e_inference(frame, 0, 0, model, text_prompt)`. In the `/api/inference/yolo-e-v8l` and `/api/yolo-e-v8l/status` handlers, obtain the model via `get_model_for_brand(None)` (default brand) and pass it through. Update `/api/yolo-e/update-prompt` and `/api/yolo-e/current-prompt` to operate on `get_model_for_brand(None)` (or remove their `_set_classes_cached` usage) so the module imports cleanly.

- [ ] **Step 7: In `lifespan`, preload the distinct brand models with warmup**

Replace the single `load_yolo_e_model()` startup block with preloading one model per distinct class set (so first click on either brand family is fast):

```python
try:
    seen = set()
    for bkey in ("blend360", "hyatt"):  # one per distinct class set
        ckey = _class_key(classes_for_brand(bkey))
        if ckey in seen:
            continue
        seen.add(ckey)
        model = get_model_for_brand(bkey)
        global yolo_e_model
        if yolo_e_model is None:
            yolo_e_model = model  # default handle
        try:
            dummy = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
            model.predict(dummy, conf=0.1, verbose=False)
        except Exception as warmup_err:
            logger.warning(f"Warmup failed (non-critical): {warmup_err}")
    logger.info("Brand models preloaded")
except Exception as e:
    logger.error(f"Error preloading brand models: {e}")
```

- [ ] **Step 8: Run the backend routing tests and import check**

Run: `cd retail-vision-ui/backend && source venv/bin/activate && python -m pytest test_brand_routing.py -v && python -c "import main; print('import OK')"`
Expected: all tests PASS and `import OK` printed (no NameError from leftover `yolo_e_model` / `_set_classes_cached` references).

- [ ] **Step 9: Commit**

```bash
git add retail-vision-ui/backend/main.py retail-vision-ui/backend/test_brand_routing.py
git commit -m "Backend: route inference by brand to the right model and video"
```

---

### Task 8: End-to-end manual verification and regression

Verify all three brands work in one running app and the env-default Docker path still works. No code unless a check fails.

**Files:**
- None (verification only). If a check fails, fix in the relevant task's files and re-commit.

- [ ] **Step 1: Start both services**

Run: `cd /Users/josecastaneda/Documents/Projects/cv-coe/cv-coe-retail-vision && ./quick_launch.sh`
Expected: backend on :8000 (logs "Brand models preloaded"), frontend on :3000.

- [ ] **Step 2: Verify backend status**

Run: `curl -s http://localhost:8000/api/yolo-e-v8l/status`
Expected: `model_loaded: true`.

- [ ] **Step 3: Verify each brand's inference returns brand-appropriate classes**

Run:
```bash
curl -s -X POST http://localhost:8000/api/inference/yolo-e \
  -H 'Content-Type: application/json' \
  -d '{"video_time":1.0,"x":100,"y":100,"frame_width":1280,"frame_height":720,"brand":"hyatt","text_prompt":"pool, lounge chair, floats, beach, ocean, golf shorts, golfer, golf club, food"}' | python -c "import sys,json; d=json.load(sys.stdin); print(sorted({x['class_name'] for x in d['detections']}))"
```
Expected: only hospitality class names appear (subset of the 9 hospitality classes), never retail classes. Repeat with `"brand":"blend360"` and the retail prompt; expect only retail classes.

- [ ] **Step 4: Manual UI walkthrough**

In the browser at `http://localhost:3000`:
- Confirm the top tab bar shows `BLEND360 | Hyatt | Under Armour`.
- BLEND360 tab: blend logo, "AI-Powered Retail Intelligence" tagline, concept paragraph visible, video is the BLEND360 video, cart is the unified (retail) layout. Click the video -> a product is added.
- Hyatt tab: hyatt logo, "Hyatt - Hospitality" tagline, NO concept paragraph, reduced header padding, Hyatt video, split cart (experiences/booking + menu). Click the video -> experiences/menu populate.
- Under Armour tab: UA logo and tagline, UA video, unified cart.
- Switch tabs back and forth: the cart clears on every switch and the video swaps. No detections leak across brands.

- [ ] **Step 5: Regression — env-default single-brand startup still works**

Run: `cd retail-vision-ui/backend && source venv/bin/activate && BRAND=hyatt python -c "import main; print(main.video_path_for_brand(None)); print(main.classes_for_brand(None))"`
Expected: prints the Hyatt video path and the 9 hospitality classes (env default honored when no brand is passed).

- [ ] **Step 6: Final full test run**

Run:
```bash
cd retail-vision-ui && CI=true npx react-scripts test --watchAll=false
cd backend && source venv/bin/activate && python -m pytest test_brand_routing.py -v
```
Expected: all frontend and backend tests PASS.

- [ ] **Step 7: Commit any fixes**

If Steps 1-6 required changes, commit them with a descriptive message. Otherwise nothing to commit.

---

## Notes for the implementer

- The frontend already fully drives logo/tagline/header/cart-layout from the brand config; do not re-implement per-brand UI — only the tab bar and the context wiring are new.
- The backend's MobileCLIP fix (`_get_text_pe_direct`) is load-bearing; keep using it to build embeddings, just pass the target model in.
- Two YOLOE instances will be resident (retail + hospitality). This is expected and is the agreed memory trade-off for clean per-brand detection and zero reshape risk.
- If `npx tsc` is unavailable, substitute `CI=true npm run build` for type-checking steps.
