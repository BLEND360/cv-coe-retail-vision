import blendLogo from '../assets/blend-logo.png';

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
};

export default brand;
