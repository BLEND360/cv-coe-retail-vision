import React, { useState } from 'react';
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
  AccessTime as AccessTimeIcon,
  KeyboardArrowDown as ChevronDownIcon,
  KeyboardArrowRight as ChevronRightIcon
} from '@mui/icons-material';
import type { CartItem, ProductCartItem, ExperienceCartItem } from '../types';

type CartLayout = 'unified' | 'split';

interface ShoppingCartProps {
  cartItems: CartItem[];
  onRemoveItem: (itemId: string) => void;
  onUpdateQuantity: (itemId: string, newQuantity: number) => void;
  onUpdateSize: (itemId: string, newSize: string) => void;
  onUpdateColor: (itemId: string, newColor: string) => void;
  onToggleSelected: (itemId: string) => void;
  onBookSelected: () => void;
  layout?: CartLayout;
}

const ShoppingCart: React.FC<ShoppingCartProps> = ({
  cartItems,
  onRemoveItem,
  onUpdateQuantity,
  onUpdateSize,
  onUpdateColor,
  onToggleSelected,
  onBookSelected,
  layout = 'unified'
}) => {
  const productItems = cartItems.filter((i): i is ProductCartItem => i.kind === 'product');
  const experienceItems = cartItems.filter((i): i is ExperienceCartItem => i.kind === 'experience');
  const menuExperiences = experienceItems.filter(e => !e.booked);
  const bookedExperiences = experienceItems.filter(e => e.booked);
  const totalPrice = productItems.reduce((sum, item) => sum + (item.price * item.quantity), 0);
  const totalItems = productItems.reduce((sum, item) => sum + item.quantity, 0);
  const selectedExperiences = experienceItems.filter(e => e.selected && !e.booked).length;
  const cartLineCount = totalItems + bookedExperiences.length;

  // In split layout, at most one section is expanded. Click an expanded section to collapse it.
  const [expandedSection, setExpandedSection] = useState<'experience' | 'cart' | null>('experience');
  const toggleSection = (key: 'experience' | 'cart') =>
    setExpandedSection(prev => (prev === key ? null : key));

  const capitalizeFirstLetter = (str: string) => {
    return str.charAt(0).toUpperCase() + str.slice(1);
  };

  // Color mapping for visual display
  const colorMap: { [key: string]: string } = {
    'Black': '#000000',
    'White': '#FFFFFF',
    'Gray': '#808080',
    'Navy': '#000080',
    'Red': '#DC143C',
    'Blue': '#1E90FF',
    'Green': '#228B22',
    'Brown': '#8B4513',
    'Beige': '#F5F5DC'
  };

  const ColorCircle = ({ color }: { color: string }) => (
    <Box
      sx={{
        width: 24,
        height: 24,
        borderRadius: '50%',
        bgcolor: colorMap[color],
        border: color === 'White' ? '1px solid #E0E0E0' : '1px solid transparent',
        boxShadow: '0 1px 3px rgba(0,0,0,0.2)',
      }}
    />
  );

  const handleQuantityChange = (itemId: string, currentQuantity: number, delta: number) => {
    const newQuantity = currentQuantity + delta;
    onUpdateQuantity(itemId, newQuantity);
  };

  // Check if item is clothing (needs size and color options)
  const isClothingItem = (itemName: string) => {
    const clothingItems = ['blazer', 'shirt', 'shorts', 'running pants', 'running shoes', 'jacket', 'gloves'];
    return clothingItems.includes(itemName.toLowerCase());
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
          Cart
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

  // ----- Reusable render blocks (used in both unified and split layouts) -----

  const renderExperienceCards = (items: ExperienceCartItem[]) => (
    <Grid container spacing={2} sx={{ pr: 1 }}>
      {items.map(item => (
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
  );

  const renderBookedExperienceCards = () => (
    <Grid container spacing={2} sx={{ pr: 1 }}>
      {bookedExperiences.map(item => (
        <Grid item xs={12} key={item.id}>
          <Card
            variant="outlined"
            sx={{ borderColor: 'grey.200', borderRadius: 2 }}
          >
            <CardContent sx={{ p: '16px !important' }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Box>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                    <Typography sx={{ fontWeight: 600, color: 'text.primary' }}>
                      {capitalizeFirstLetter(item.name)}
                    </Typography>
                    <Chip
                      label="Booked"
                      size="small"
                      sx={{ bgcolor: '#34C759', color: 'white', fontWeight: 600, height: 20 }}
                    />
                  </Box>
                  <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
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
  );

  const experienceFooter = (
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
          {selectedExperiences} Activities Selected
        </Typography>
      </Box>
      <Button
        variant="contained"
        fullWidth
        disabled={selectedExperiences === 0}
        onClick={() => {
          onBookSelected();
          setExpandedSection('cart');
        }}
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
  );

  const productCards = (
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
                        {(item.sizes && item.sizes.length > 0 ? item.sizes : ['Small','Medium','Large']).map(s => (
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
  );

  const productFooter = (
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
  );

  const sharedHeader = (
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
  );

  // ----- Split layout: two collapsible sections, mutually exclusive -----

  if (layout === 'split') {
    const renderSection = (
      key: 'experience' | 'cart',
      title: string,
      icon: React.ReactNode,
      countLabel: string,
      body: React.ReactNode,
      footer: React.ReactNode,
    ) => {
      const isExpanded = expandedSection === key;
      return (
        <Box
          sx={{
            display: 'flex',
            flexDirection: 'column',
            minHeight: 0,
            flex: isExpanded ? 1 : '0 0 auto',
            border: '1px solid',
            borderColor: 'grey.200',
            borderRadius: 2,
            overflow: 'hidden',
            bgcolor: 'background.paper',
          }}
        >
          {/* Section header (clickable, toggles expand/collapse) */}
          <Box
            onClick={() => toggleSection(key)}
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 1.5,
              px: 2,
              py: 1.5,
              cursor: 'pointer',
              bgcolor: isExpanded ? 'grey.50' : 'background.paper',
              borderBottom: isExpanded ? '1px solid' : 'none',
              borderColor: 'grey.200',
              '&:hover': { bgcolor: 'grey.50' },
            }}
          >
            {isExpanded ? (
              <ChevronDownIcon sx={{ fontSize: 22, color: 'text.secondary' }} />
            ) : (
              <ChevronRightIcon sx={{ fontSize: 22, color: 'text.secondary' }} />
            )}
            {icon}
            <Typography variant="h6" sx={{ fontWeight: 600, color: 'text.primary', flex: 1 }}>
              {title}
            </Typography>
            <Chip
              label={countLabel}
              size="small"
              sx={{
                bgcolor: isExpanded ? 'primary.main' : 'grey.200',
                color: isExpanded ? 'white' : 'text.primary',
                fontWeight: 600,
              }}
            />
          </Box>

          {/* Section body (only when expanded) */}
          {isExpanded && (
            <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0, p: 2 }}>
              <Box
                sx={{
                  flex: 1,
                  overflow: 'auto',
                  mb: 2,
                  minHeight: 0,
                  '&::-webkit-scrollbar': { width: '6px' },
                  '&::-webkit-scrollbar-track': { background: 'rgba(0, 0, 0, 0.02)', borderRadius: '3px' },
                  '&::-webkit-scrollbar-thumb': {
                    background: 'rgba(0, 0, 0, 0.15)',
                    borderRadius: '3px',
                    '&:hover': { background: 'rgba(0, 0, 0, 0.25)' },
                  },
                }}
              >
                {body}
              </Box>
              {footer}
            </Box>
          )}
        </Box>
      );
    };

    const cartBody =
      bookedExperiences.length === 0 && productItems.length === 0 ? (
        <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', mt: 4 }}>
          Book an activity or click a bar/coffee shop to add items here.
        </Typography>
      ) : (
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          {bookedExperiences.length > 0 && renderBookedExperienceCards()}
          {productItems.length > 0 && productCards}
        </Box>
      );

    return (
      <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column', minHeight: 0 }}>
        <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 2, minHeight: 0 }}>
          {renderSection(
            'experience',
            'Experience Menu',
            <EventIcon sx={{ fontSize: 22, color: 'text.primary' }} />,
            `${selectedExperiences} / ${menuExperiences.length}`,
            menuExperiences.length > 0 ? renderExperienceCards(menuExperiences) : (
              <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', mt: 4 }}>
                Click on a pool, sea, or beach to add experiences.
              </Typography>
            ),
            experienceFooter,
          )}
          {renderSection(
            'cart',
            'Shopping Cart',
            <ShoppingCartIcon sx={{ fontSize: 22, color: 'text.primary' }} />,
            `${cartLineCount} ${cartLineCount === 1 ? 'item' : 'items'}`,
            cartBody,
            productFooter,
          )}
        </Box>
      </Box>
    );
  }

  // ----- Unified layout (default, retail brands) -----

  return (
    <Box sx={{
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
      minHeight: 0,
    }}>
      {sharedHeader}

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
            {renderExperienceCards(experienceItems)}
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
            {productCards}
          </Box>
        )}
      </Box>

      {/* Footer summaries */}
      {experienceItems.length > 0 && (
        <Box sx={{ mb: productItems.length > 0 ? 2 : 0 }}>
          {experienceFooter}
        </Box>
      )}

      {productItems.length > 0 && productFooter}
    </Box>
  );
};

export default ShoppingCart;
