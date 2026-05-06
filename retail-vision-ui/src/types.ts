export type ProductCatalogItem = {
  name: string;
  price: number;
  size?: string;
  sizes?: string[];
  colors?: string[];
};

export type ExperienceCatalogOption = {
  name: string;
  time: string;
  duration: string;
};

export type ProductCatalogEntry = {
  id: string;
  kind: 'product';
  title: string;
  // Each inner array is a trigger set whose classes must ALL be detected together.
  // The entry fires if ANY trigger set is fully satisfied.
  triggers: string[][];
  items: ProductCatalogItem[];
};

export type ExperienceCatalogEntry = {
  id: string;
  kind: 'experience';
  title: string;
  triggers: string[][];
  options: ExperienceCatalogOption[];
};

export type CatalogEntry = ProductCatalogEntry | ExperienceCatalogEntry;

export type ProductCartItem = {
  kind: 'product';
  id: string;
  name: string;
  price: number;
  quantity: number;
  selected: boolean;  // checkbox state in the menu section (split layout)
  ordered: boolean;   // true once moved into the cart by Add to Cart
  detectionId: number;
  confidence: number;
  size?: string;
  sizes?: string[];
  color?: string;
};

export type ExperienceCartItem = {
  kind: 'experience';
  id: string;
  name: string;
  time: string;
  duration: string;
  selected: boolean;
  booked: boolean;
  detectionId: number;
  confidence: number;
};

export type CartItem = ProductCartItem | ExperienceCartItem;
