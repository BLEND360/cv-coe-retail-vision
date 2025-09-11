import React, { useState } from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline, Box, Paper, Typography, Switch, FormControlLabel } from '@mui/material';
import VideoPlayer from './components/VideoPlayer';
import InferencePanel from './components/InferencePanel';
import ShoppingCart from './components/ShoppingCart';
import './App.css';

interface CartItem {
  id: string;
  name: string;
  price: number;
  quantity: number;
  detectionId: number;
  confidence: number;
}

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

function App() {
  const [lastClickData, setLastClickData] = useState<{ x: number; y: number; currentTime: number; frameWidth: number; frameHeight: number } | null>(null);
  const [cartItems, setCartItems] = useState<CartItem[]>([]);
  const [showInferencePanel, setShowInferencePanel] = useState(true);

  const handleVideoTimeUpdate = (time: number) => {
    // Video time update handler - can be used for future features
  };

  const handleVideoClick = (clickData: { x: number; y: number; currentTime: number; frameWidth: number; frameHeight: number }) => {
    setLastClickData(clickData);
  };

  const addToCart = (detection: { id: number; class_name: string; confidence: number }) => {
    const cartItemId = `${detection.id}-${Date.now()}`;
    const newItem: CartItem = {
      id: cartItemId,
      name: detection.class_name,
      price: 99,
      quantity: 1,
      detectionId: detection.id,
      confidence: detection.confidence
    };
    setCartItems(prev => [...prev, newItem]);
  };

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
        item.id === itemId ? { ...item, quantity: newQuantity } : item
      )
    );
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ 
        height: '100vh', 
        bgcolor: 'background.default',
        background: '#FFFFFF',
        display: 'flex',
        flexDirection: 'column'
      }}>
        {/* Header Section */}
        <Box sx={{ 
          py: 4, 
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
              mb: 3,
              letterSpacing: '-0.01em'
            }}
          >
            See it. Click it. Own it.
          </Typography>
          <Typography 
            variant="body1" 
            color="text.secondary" 
            sx={{ 
              fontWeight: 400,
              fontSize: { xs: '0.875rem', md: '1rem' },
              maxWidth: '800px',
              mx: 'auto',
              lineHeight: 1.6,
              mb: 2
            }}
          >
            <strong>Concept:</strong> Transform any video or livestream into a shoppable experience. Users can click on products they see on screen and instantly add them to their shopping cart.
          </Typography>
          
          {/* Show Inference Panel Switch */}
          <Box sx={{ 
            display: 'flex', 
            justifyContent: 'flex-end',
            maxWidth: '800px',
            mx: 'auto',
            mt: 2
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

        {/* Main Content Grid - Takes remaining space */}
        <Box sx={{ 
          flex: 1, 
          px: 6, // Increased padding for better symmetry
          pb: 4,
          minHeight: 0, // Important for flex child to shrink
          overflow: 'hidden' // Prevent content from expanding beyond container
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
                    onAddToCart={addToCart}
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
                  />
                </Paper>
              </Box>
            </Box>
          </Box>
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App;
