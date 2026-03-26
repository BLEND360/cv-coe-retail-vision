import underArmourLogo from '../assets/under-armour-logo.png';

export interface BrandConfig {
  name: string;
  logo: string;
  logoAlt: string;
  logoHeight: string;
  videoUrl: string;
  videoPath: string;
  tagline: string;
  yoloeClasses: string[];
}

export const brand: BrandConfig = {
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
};

export default brand;
