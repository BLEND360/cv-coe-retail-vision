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
  catalog?: CatalogEntry[];
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
    logoHeight: '107px',
    videoUrl: '/videos/Hyatt.mp4',
    videoPath: '../public/Hyatt.mp4',
    tagline: 'Hyatt - Hospitality',
    yoloeClasses: [
      'pool', 'lounge chair', 'floats',
      'beach', 'ocean',
      'golf shorts', 'golfer', 'golf club',
      'food',
    ],
    catalog: [
      {
        id: 'pool',
        kind: 'experience',
        title: 'Pool Experience',
        triggers: [['pool', 'lounge chair'], ['floats']],
        options: [
          { name: 'Water aerobics',          time: '9:00 - 10:00 AM',     duration: '45 min' },
          { name: 'Aqua yoga',               time: '11:00 AM - 12:00 PM', duration: '45 min' },
          { name: 'Marco polo championship', time: '2:00 - 3:00 PM',      duration: '45 min' },
          { name: 'Zumba classes',           time: '5:00 - 6:00 PM',      duration: '45 min' },
        ],
      },
      {
        id: 'sea',
        kind: 'experience',
        title: 'Sea Experience',
        triggers: [['ocean'], ['beach']],
        options: [
          { name: 'Dolphin watching', time: '7:00 - 8:00 AM',     duration: '45 min' },
          { name: 'Snorkeling tour',  time: '9:00 - 10:00 AM',    duration: '45 min' },
          { name: 'Kayak rental',     time: '11:00 AM - 12:00 PM', duration: '45 min' },
          { name: 'Sunset cruise',    time: '5:00 - 6:00 PM',     duration: '45 min' },
        ],
      },
      {
        id: 'golf',
        kind: 'experience',
        title: 'Golf Classes',
        triggers: [['golf shorts', 'golfer', 'golf club']],
        options: [
          { name: 'Driving range',  time: '8:00 - 9:00 AM',  duration: '45 min' },
          { name: 'Group lesson',   time: '10:00 - 11:00 AM', duration: '45 min' },
          { name: 'Pro lesson',     time: '1:00 - 2:00 PM',  duration: '45 min' },
          { name: 'Putting clinic', time: '4:00 - 5:00 PM',  duration: '45 min' },
        ],
      },
      {
        id: 'restaurant',
        kind: 'product',
        title: 'Restaurant Menu',
        triggers: [['food']],
        items: [
          { name: 'Edamames', price: 8 },
          { name: 'Pad Thai', price: 18 },
          { name: 'Sushi',    price: 24 },
        ],
      },
    ],
  },
};

const brandKey = process.env.REACT_APP_BRAND || 'blend360';
export const brand: BrandConfig = brands[brandKey] || brands['under-armour'];
export default brands;
