import React, { useState, useCallback } from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline, Box, Paper, Typography, Switch, FormControlLabel } from '@mui/material';
import { Tabs, Tab } from '@mui/material';
import VideoPlayer from './components/VideoPlayer';
import InferencePanel from './components/InferencePanel';
import ShoppingCart from './components/ShoppingCart';
import brands from './config/brands';
import { useBrand, useBrandKey } from './config/BrandContext';
import type { CartItem, ProductCartItem, ExperienceCartItem } from './types';
import './App.css';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: { main: '#000000' },
    secondary: { main: '#007AFF' },
    background: { default: '#FFFFFF', paper: '#FFFFFF' },
    text: { primary: '#1D1D1F', secondary: '#86868B' },
    grey: { 
      50: '#F5F5F7', 
      100: '#F2F2F7', 
      200: '#E5E5EA', 
      300: '#D1D1D6', 
      400: '#C7C7CC',
      500: '#AEAEB2',
      600: '#8E8E93',
      700: '#636366',
      800: '#48484A',
      900: '#1C1C1E'
    },
  },
  typography: {
    fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Helvetica Neue", Helvetica, Arial, sans-serif',
    h1: { 
      fontWeight: 600, 
      fontSize: '3rem', 
      letterSpacing: '-0.02em',
      lineHeight: 1.1
    },
    h2: { 
      fontWeight: 600, 
      fontSize: '2.25rem', 
      letterSpacing: '-0.01em',
      lineHeight: 1.2
    },
    h3: { 
      fontWeight: 600, 
      fontSize: '1.875rem', 
      letterSpacing: '-0.01em',
      lineHeight: 1.3
    },
    h4: { 
      fontWeight: 500, 
      fontSize: '1.5rem',
      lineHeight: 1.4
    },
    h5: { 
      fontWeight: 500, 
      fontSize: '1.25rem',
      lineHeight: 1.4
    },
    h6: { 
      fontWeight: 500, 
      fontSize: '1.125rem',
      lineHeight: 1.4
    },
    body1: { 
      fontSize: '1rem',
      lineHeight: 1.5,
      fontWeight: 400
    },
    body2: { 
      fontSize: '0.875rem',
      lineHeight: 1.5,
      fontWeight: 400
    },
  },
  shape: { borderRadius: 8 },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1), 0 1px 2px rgba(0, 0, 0, 0.06)',
          border: '1px solid rgba(0, 0, 0, 0.05)',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 500,
          borderRadius: 8,
          padding: '8px 16px',
        },
        contained: {
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1), 0 1px 2px rgba(0, 0, 0, 0.06)',
          '&:hover': {
            boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1), 0 2px 4px rgba(0, 0, 0, 0.06)',
          },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1), 0 1px 2px rgba(0, 0, 0, 0.06)',
          border: '1px solid rgba(0, 0, 0, 0.05)',
        },
      },
    },
  },
});

// Brand tab order (left to right). BLEND360 leads; unknown keys are ignored.
const TAB_ORDER = ['blend360', 'under-armour', 'hyatt'];

function App() {
  const brand = useBrand();
  const { brandKey, setBrandKey } = useBrandKey();
  const apiBase = process.env.REACT_APP_API_URL ?? 'http://localhost:8000';

  const [lastClickData, setLastClickData] = useState<{ x: number; y: number; currentTime: number; frameWidth: number; frameHeight: number } | null>(null);
  const [cartItems, setCartItems] = useState<CartItem[]>([]);
  const [showInferencePanel, setShowInferencePanel] = useState(true);

  const handleVideoTimeUpdate = (time: number) => {
    // Video time update handler - can be used for future features
  };

  const handleVideoClick = (clickData: { x: number; y: number; currentTime: number; frameWidth: number; frameHeight: number }) => {
    setLastClickData(clickData);
  };

  // Bumped each time a new inference brings in new menu items, so the cart UI can
  // auto-expand the menu section.
  const [menuRefreshVersion, setMenuRefreshVersion] = useState(0);

  const handleBrandChange = useCallback((_e: React.SyntheticEvent, newKey: string) => {
    setBrandKey(newKey);
    setCartItems([]);
    setLastClickData(null);
    setMenuRefreshVersion(0);
  }, [setBrandKey]);

  // Set-based catalog matching: a catalog entry fires only when ANY of its trigger
  // sets is fully present in the inference's detections. For hospitality brands
  // (catalog defined), every inference rebuilds the menu (clears prior unbooked /
  // unordered items). For retail brands (no catalog), single-class clicks add a
  // $99 generic item directly to the cart.
  const handleInference = useCallback((
    clickedDetection: { id: number; class_name: string; confidence: number } | null,
    detections: Array<{ id: number; class_name: string; confidence: number }>,
  ) => {
    const allowed = new Set(brand.yoloeClasses.map(c => c.toLowerCase()));
    const inBrand = (c: { class_name: string }) => allowed.has(c.class_name.toLowerCase());
    const brandDetections = detections.filter(inBrand);
    const brandClicked = clickedDetection && inBrand(clickedDetection) ? clickedDetection : null;

    const baseId = `${Date.now()}`;
    const isHospitality = (brand.catalog ?? []).length > 0;

    // Keep only items the user has confirmed (booked experiences + ordered products).
    // Everything else is "menu" content that gets rebuilt by the new inference.
    const keepConfirmed = (items: CartItem[]) => items.filter(item =>
      (item.kind === 'experience' && item.booked) ||
      (item.kind === 'product' && item.ordered)
    );

    if (isHospitality) {
      const detectedClasses = new Set(brandDetections.map(d => d.class_name.toLowerCase()));
      const matched = (brand.catalog ?? []).filter(entry =>
        entry.triggers.some(triggerSet =>
          triggerSet.every(t => detectedClasses.has(t.toLowerCase()))
        )
      );

      if (matched.length === 0) {
        // Clear menu, keep cart. Don't bump menuRefreshVersion (no new menu items).
        setCartItems(keepConfirmed);
        return;
      }

      const newExperiences: ExperienceCartItem[] = [];
      const newProducts: ProductCartItem[] = [];

      matched.forEach(entry => {
        if (entry.kind === 'experience') {
          entry.options.forEach((opt, idx) => {
            newExperiences.push({
              kind: 'experience',
              id: `${baseId}-${entry.id}-exp-${idx}`,
              name: opt.name,
              time: opt.time,
              duration: opt.duration,
              selected: idx < 2,
              booked: false,
              detectionId: brandClicked?.id ?? -1,
              confidence: brandClicked?.confidence ?? 1,
            });
          });
        } else {
          entry.items.forEach((item, idx) => {
            newProducts.push({
              kind: 'product',
              id: `${baseId}-${entry.id}-prod-${idx}`,
              name: item.name,
              price: item.price,
              quantity: 1,
              selected: true,
              ordered: false,
              detectionId: brandClicked?.id ?? -1,
              confidence: brandClicked?.confidence ?? 1,
              ...(item.size !== undefined && { size: item.size }),
              ...(item.sizes && item.sizes.length > 0 && { sizes: item.sizes }),
              ...(item.colors && item.colors.length > 0 && { color: item.colors[0] }),
            });
          });
        }
      });

      setCartItems(prev => [...keepConfirmed(prev), ...newExperiences, ...newProducts]);
      setMenuRefreshVersion(v => v + 1);
      return;
    }

    // Retail fallback (brand has no catalog).
    if (!brandClicked) return;
    const className = brandClicked.class_name.toLowerCase();
    const clothingItems = ['blazer', 'shirt', 'shorts', 'running pants', 'running shoes', 'jacket', 'gloves'];
    const isClothing = clothingItems.includes(className);
    const fallback: ProductCartItem = {
      kind: 'product',
      id: `${brandClicked.id}-${baseId}`,
      name: brandClicked.class_name,
      price: 99,
      quantity: 1,
      selected: true,
      ordered: true,
      detectionId: brandClicked.id,
      confidence: brandClicked.confidence,
      ...(isClothing && { size: 'M', color: 'Black' }),
    };
    setCartItems(prev => [...prev, fallback]);
  }, [brand]);

  const removeFromCart = (itemId: string) => {
    setCartItems(prev => prev.filter(item => item.id !== itemId));
  };

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

  const toggleSelected = (itemId: string) => {
    setCartItems(prev =>
      prev.map(item => {
        if (item.id !== itemId) return item;
        if (item.kind === 'experience' && !item.booked) return { ...item, selected: !item.selected };
        if (item.kind === 'product' && !item.ordered) return { ...item, selected: !item.selected };
        return item;
      })
    );
  };

  const confirmSelected = () => {
    setCartItems(prev =>
      prev.map(item => {
        if (item.kind === 'experience' && item.selected && !item.booked) {
          return { ...item, booked: true, selected: false };
        }
        if (item.kind === 'product' && item.selected && !item.ordered) {
          return { ...item, ordered: true, selected: false };
        }
        return item;
      })
    );
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{
        minHeight: '100vh',
        overflowY: 'auto',
        bgcolor: 'background.default',
        background: '#FFFFFF',
        display: 'flex',
        flexDirection: 'column',
        position: 'relative'
      }}>
        {/* Brand Tab Bar */}
        <Tabs
          value={brandKey}
          onChange={handleBrandChange}
          variant="fullWidth"
          aria-label="Brand selector"
          sx={{
            borderBottom: '1px solid rgba(0,0,0,0.08)',
            flexShrink: 0,
            minHeight: 44,
            bgcolor: 'background.paper',
            '& .MuiTab-root': { textTransform: 'none', fontWeight: 600, fontSize: '1rem', minHeight: 44 },
          }}
        >
          {TAB_ORDER.filter(key => brands[key]).map(key => (
            <Tab key={key} value={key} label={brands[key].name} />
          ))}
        </Tabs>

        {/* Hidden preloaders: warm the browser cache for every brand video at
            startup so switching tabs plays the video immediately (the keyed
            content subtree below remounts on switch and would otherwise refetch). */}
        <Box sx={{ display: 'none' }} aria-hidden>
          {Object.values(brands).map((cfg, i) => (
            <video key={i} src={`${apiBase}${cfg.videoUrl}`} preload="auto" muted />
          ))}
        </Box>

        {/* Wrapper below the tab bar — positioning context for the logo */}
        <Box sx={{
          position: 'relative',
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          minHeight: 0
        }}>

        {/* Logo in upper left corner */}
        <Box sx={{
          position: 'absolute',
          top: 24,
          left: 24,
          zIndex: 1000,
          display: 'flex',
          alignItems: 'center'
        }}>
          <img
            src={brand.logo}
            alt={brand.logoAlt}
            style={{
              height: brand.logoHeight,
              width: 'auto',
              objectFit: 'contain'
            }}
          />
        </Box>

        {/* Header Section */}
        <Box sx={{
          py: brand.name === 'Hyatt' ? 1.5 : 2,
          px: 6, // Match main content padding for symmetry
          textAlign: 'center',
          flexShrink: 0
        }}>
          <Typography
            variant="h1"
            component="h1"
            sx={{
              fontWeight: 600,
              color: 'text.primary',
              mb: 1,
              fontSize: { xs: '2rem', md: '2.5rem' }
            }}
          >
            Retail Vision
          </Typography>
          <Typography
            variant="h4"
            color="text.secondary"
            sx={{
              fontWeight: 500,
              fontSize: { xs: '1.25rem', md: '1.5rem' },
              mb: 1.5,
              letterSpacing: '-0.01em'
            }}
          >
            {brand.tagline}
          </Typography>
          {brand.name !== 'Hyatt' && (
            <Typography
              variant="body1"
              color="text.secondary"
              sx={{
                fontWeight: 400,
                fontSize: { xs: '0.875rem', md: '1rem' },
                maxWidth: '800px',
                mx: 'auto',
                lineHeight: 1.5,
                mb: 1
              }}
            >
              <strong>Concept:</strong> Transform any video or livestream into a shoppable experience. Users can click on products they see on screen and instantly add them to their shopping cart.
            </Typography>
          )}

          {/* Show Inference Panel Switch */}
          <Box sx={{
            display: 'flex',
            justifyContent: 'flex-end',
            maxWidth: '800px',
            mx: 'auto',
            mt: 1
          }}>
            <FormControlLabel
              control={
                <Switch
                  checked={showInferencePanel}
                  onChange={(e) => setShowInferencePanel(e.target.checked)}
                  size="medium"
                />
              }
              label="Show Inference Panel"
              labelPlacement="start"
              sx={{
                '& .MuiFormControlLabel-label': {
                  fontWeight: 500,
                  fontSize: '0.875rem',
                  color: 'text.secondary'
                }
              }}
            />
          </Box>
        </Box>

        {/* Main Content Grid - fills remaining space on tall viewports, keeps a
            usable minimum on short ones (the page scrolls rather than clipping). */}
        <Box key={brandKey} sx={{
          flex: 1,
          px: 6,
          pb: 4,
          minHeight: { xs: 480, md: 560 },
          overflow: 'visible'
        }}>
          <Box sx={{ 
            height: '100%', 
            display: 'flex', 
            gap: 3,
            alignItems: 'stretch',
            justifyContent: showInferencePanel ? 'flex-start' : 'center' // Center when inference panel is hidden
          }}>
            {/* Video Player - Expands when inference panel is hidden */}
            <Box sx={{
              flex: showInferencePanel ? '0 0 calc(50% - 6px)' : '0 0 calc(62.5% - 6px)', // Expand to take inference panel space
              height: '100%',
              transition: 'flex 0.3s ease-in-out' // Smooth transition
            }}>
              <Paper sx={{ 
                height: '100%', 
                p: 3,
                borderRadius: 3,
                bgcolor: 'background.paper',
                boxShadow: '0 4px 20px rgba(0, 0, 0, 0.08)',
                border: '1px solid rgba(0, 0, 0, 0.06)',
              }}>
                <VideoPlayer 
                  onTimeUpdate={handleVideoTimeUpdate} 
                  onVideoClick={handleVideoClick}
                />
              </Paper>
            </Box>

            {/* Right Side Panel - Expands when inference panel is hidden */}
            <Box sx={{
              flex: showInferencePanel ? '0 0 calc(50% - 6px)' : '0 0 calc(37.5% - 6px)', // Expand to take inference panel space
              height: '100%',
              display: 'flex',
              gap: 3,
              transition: 'flex 0.3s ease-in-out' // Smooth transition
            }}>
              {/* Inference Panel - Always rendered but conditionally visible */}
              <Box sx={{ 
                flex: showInferencePanel ? '0 0 calc(50% - 6px)' : '0 0 0px', // Hide width when not visible
                height: '100%',
                overflow: 'hidden', // Hide content when not visible
                opacity: showInferencePanel ? 1 : 0, // Fade out when hidden
                transition: 'opacity 0.3s ease-in-out, flex 0.3s ease-in-out' // Smooth transition
              }}>
                <Paper sx={{ 
                  height: '100%', 
                  p: 3,
                  borderRadius: 3,
                  bgcolor: 'background.paper',
                  boxShadow: '0 4px 20px rgba(0, 0, 0, 0.08)',
                  border: '1px solid rgba(0, 0, 0, 0.06)',
                  display: 'flex',
                  flexDirection: 'column',
                  overflow: 'hidden'
                }}>
                  <InferencePanel 
                    lastClickData={lastClickData}
                    onInference={handleInference}
                  />
                </Paper>
              </Box>

              {/* Shopping Cart - Expands when inference panel is hidden */}
              <Box sx={{ 
                flex: showInferencePanel ? '0 0 calc(50% - 6px)' : '0 0 calc(100% - 0px)', // Take full width when inference panel is hidden
                height: '100%',
                transition: 'flex 0.3s ease-in-out' // Smooth transition
              }}>
                <Paper sx={{ 
                  height: '100%',
                  p: 3,
                  borderRadius: 3,
                  bgcolor: 'background.paper',
                  boxShadow: '0 4px 20px rgba(0, 0, 0, 0.08)',
                  border: '1px solid rgba(0, 0, 0, 0.06)',
                  display: 'flex',
                  flexDirection: 'column',
                  overflow: 'hidden'
                }}>
                  <ShoppingCart
                    cartItems={cartItems}
                    onRemoveItem={removeFromCart}
                    onUpdateQuantity={updateQuantity}
                    onUpdateSize={updateSize}
                    onUpdateColor={updateColor}
                    onToggleSelected={toggleSelected}
                    onConfirmSelected={confirmSelected}
                    menuRefreshVersion={menuRefreshVersion}
                    layout={
                      brand.catalog?.some(e => e.kind === 'experience')
                        ? 'split'
                        : 'unified'
                    }
                  />
                </Paper>
              </Box>
            </Box>
          </Box>
        </Box>
        </Box>{/* end wrapper below tab bar */}
      </Box>
    </ThemeProvider>
  );
}

export default App;
