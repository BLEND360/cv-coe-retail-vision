import underArmourLogo from '../assets/under-armour-logo.png';
import blendLogo from '../assets/blend-logo.png';

export interface BrandConfig {
  name: string;
  logo: string;
  logoAlt: string;
  videoUrl: string;
  videoPath: string; // backend video file path
  tagline: string;
  yoloeClasses: string[];
}

const brands: Record<string, BrandConfig> = {
  'under-armour': {
    name: 'Under Armour',
    logo: underArmourLogo,
    logoAlt: 'Under Armour',
    videoUrl: '/Under-Armour.mp4',
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
    videoUrl: '/The BLEND360 Approach.mp4',
    videoPath: '../public/The BLEND360 Approach.mp4',
    tagline: 'AI-Powered Retail Intelligence',
    yoloeClasses: [
      'laptop', 'headphones', 'glasses', 'blazer', 'desk', 'watch',
      'monitor', 'trash can', 'chair', 'shirt', 'running pants',
      'running shoes', 'jacket', 'gloves',
    ],
  },
};

const brandKey = process.env.REACT_APP_BRAND || 'under-armour';
export const brand: BrandConfig = brands[brandKey] || brands['under-armour'];
export default brands;
