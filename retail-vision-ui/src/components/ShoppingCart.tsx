import React from 'react';
import { 
  Box, 
  Typography, 
  Card, 
  CardContent, 
  IconButton, 
  Button,
  Divider,
  Chip,
  Grid,
  Paper
} from '@mui/material';
import { 
  ShoppingCart as ShoppingCartIcon, 
  Delete as DeleteIcon,
  Add as AddIcon,
  Remove as RemoveIcon
} from '@mui/icons-material';

interface CartItem {
  id: string;
  name: string;
  price: number;
  quantity: number;
  detectionId: number;
  confidence: number;
}

interface ShoppingCartProps {
  cartItems: CartItem[];
  onRemoveItem: (itemId: string) => void;
  onUpdateQuantity: (itemId: string, newQuantity: number) => void;
}

const ShoppingCart: React.FC<ShoppingCartProps> = ({ 
  cartItems, 
  onRemoveItem, 
  onUpdateQuantity 
}) => {
  const totalPrice = cartItems.reduce((sum, item) => sum + (item.price * item.quantity), 0);
  const totalItems = cartItems.reduce((sum, item) => sum + item.quantity, 0);

  const capitalizeFirstLetter = (str: string) => {
    return str.charAt(0).toUpperCase() + str.slice(1);
  };

  const handleQuantityChange = (itemId: string, currentQuantity: number, delta: number) => {
    const newQuantity = currentQuantity + delta;
    onUpdateQuantity(itemId, newQuantity);
  };

  if (cartItems.length === 0) {
    return (
      <Box sx={{ 
        height: '100%', 
        display: 'flex', 
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        p: 4
      }}>
        <ShoppingCartIcon sx={{ 
          fontSize: 64, 
          color: 'text.secondary', 
          mb: 3, 
          opacity: 0.4 
        }} />
        <Typography 
          variant="h4" 
          color="text.primary" 
          sx={{ 
            mb: 2, 
            fontWeight: 600 
          }}
        >
          Shopping Cart
        </Typography>
        <Typography 
          variant="body1" 
          color="text.secondary" 
          sx={{ 
            textAlign: 'center',
            fontWeight: 500,
            lineHeight: 1.5,
            maxWidth: '250px'
          }}
        >
          Clicked objects will be automatically added to your cart
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ 
      height: '100%', 
      display: 'flex', 
      flexDirection: 'column',
      minHeight: 0, // Important for flex child to shrink
    }}>
      {/* Header */}
      <Box sx={{ mb: 3, display: 'flex', alignItems: 'center', gap: 2 }}>
        <ShoppingCartIcon sx={{ 
          fontSize: 28, 
          color: 'text.primary' 
        }} />
        <Typography 
          variant="h5" 
          component="h2" 
          sx={{ 
            fontWeight: 600,
            color: 'text.primary'
          }}
        >
          Shopping Cart
        </Typography>
        <Chip 
          label={totalItems} 
          size="small"
          sx={{ 
            ml: 'auto',
            bgcolor: 'primary.main',
            color: 'white',
            fontWeight: 600
          }}
        />
      </Box>

      {/* Cart Items */}
      <Box sx={{ 
        flex: 1, 
        overflow: 'auto', 
        mb: 3,
        minHeight: 0, // Important for flex child to shrink
        '&::-webkit-scrollbar': {
          width: '6px',
        },
        '&::-webkit-scrollbar-track': {
          background: 'rgba(0, 0, 0, 0.02)',
          borderRadius: '3px',
        },
        '&::-webkit-scrollbar-thumb': {
          background: 'rgba(0, 0, 0, 0.15)',
          borderRadius: '3px',
          '&:hover': {
            background: 'rgba(0, 0, 0, 0.25)',
          },
        },
      }}>
        <Grid container spacing={2} sx={{ pr: 1 }}>
          {cartItems.map((item, index) => {
            return (
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
                            mb: 0.5,
                            color: 'text.primary',
                            fontSize: '1.1rem',
                            fontWeight: 600
                          }}
                        >
                          {capitalizeFirstLetter(item.name)}
                        </Typography>
                      </Box>
                      <IconButton 
                        size="small" 
                        onClick={() => onRemoveItem(item.id)}
                        sx={{ 
                          ml: 1,
                          bgcolor: 'rgba(255, 59, 48, 0.1)',
                          color: '#FF3B30',
                          '&:hover': {
                            bgcolor: 'rgba(255, 59, 48, 0.2)',
                          }
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
                            bgcolor: 'grey.100',
                            color: 'text.secondary',
                            '&:hover': {
                              bgcolor: 'grey.200',
                            },
                            '&:disabled': {
                              bgcolor: 'grey.50',
                              color: 'grey.400'
                            }
                          }}
                        >
                          <RemoveIcon fontSize="small" />
                        </IconButton>
                        <Typography 
                          variant="h6" 
                          sx={{ 
                            minWidth: '32px', 
                            textAlign: 'center',
                            color: 'text.primary',
                            fontWeight: 600
                          }}
                        >
                          {item.quantity}
                        </Typography>
                        <IconButton 
                          size="small"
                          onClick={() => handleQuantityChange(item.id, item.quantity, 1)}
                          sx={{
                            bgcolor: 'grey.100',
                            color: 'text.secondary',
                            '&:hover': {
                              bgcolor: 'grey.200',
                            }
                          }}
                        >
                          <AddIcon fontSize="small" />
                        </IconButton>
                      </Box>
                      <Typography 
                        variant="h6" 
                        sx={{ 
                          color: 'text.primary',
                          fontSize: '1.1rem',
                          fontWeight: 600
                        }}
                      >
                        ${(item.price * item.quantity).toFixed(2)}
                      </Typography>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            );
          })}
        </Grid>
      </Box>

      {/* Summary */}
      <Paper sx={{ 
        p: 3, 
        bgcolor: 'grey.50', 
        border: '1px solid', 
        borderColor: 'grey.200',
        borderRadius: 2,
        flexShrink: 0, // Prevent summary from shrinking
      }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography 
            variant="h6" 
            sx={{ 
              fontWeight: 600,
              color: 'text.primary'
            }}
          >
            Total ({totalItems} items)
          </Typography>
          <Typography 
            variant="h6" 
            sx={{ 
              fontWeight: 600, 
              color: 'text.primary'
            }}
          >
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
            '&:hover': {
              bgcolor: 'grey.800',
              transform: 'translateY(-1px)'
            }
          }}
        >
          Checkout
        </Button>
      </Paper>
    </Box>
  );
};

export default ShoppingCart;
