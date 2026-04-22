import underArmourLogo from '../assets/under-armour-logo.png';
import blendLogo from '../assets/blend-logo.png';
import hyattLogo from '../assets/hyatt-logo.png';
import type { CatalogEntry } from '../types';

export interface BrandConfig {
  name: string;
  logo: string;
  logoAlt: string;
  logoHeight: string;
  videoUrl: string;
  videoPath: string; // backend video file path
  tagline: string;
  yoloeClasses: string[];
  catalog?: Record<string, CatalogEntry>;
}

const brands: Record<string, BrandConfig> = {
  'under-armour': {
    name: 'Under Armour',
    logo: underArmourLogo,
    logoAlt: 'Under Armour',
    logoHeight: '112px',
    videoUrl: '/videos/Under-Armour.mp4',
    videoPath: '../public/Under-Armour.mp4',
    tagline: 'See it. Click it. Own it.',
    yoloeClasses: [
      'laptop', 'headphones', 'glasses', 'blazer', 'desk', 'watch',
      'monitor', 'trash can', 'chair', 'shirt', 'running pants',
      'running shoes', 'jacket', 'gloves',
    ],
  },
  blend360: {
    name: 'BLEND360',
    logo: blendLogo,
    logoAlt: 'BLEND360',
    logoHeight: '56px',
    videoUrl: '/videos/The BLEND360 Approach.mp4',
    videoPath: '../public/The BLEND360 Approach.mp4',
    tagline: 'AI-Powered Retail Intelligence',
    yoloeClasses: [
      'laptop', 'headphones', 'glasses', 'blazer', 'desk', 'watch',
      'monitor', 'trash can', 'chair', 'shirt', 'running pants',
      'running shoes', 'jacket', 'gloves',
    ],
  },
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
};

const brandKey = process.env.REACT_APP_BRAND || 'blend360';
export const brand: BrandConfig = brands[brandKey] || brands['under-armour'];
export default brands;
