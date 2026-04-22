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
  kind: 'product';
  title: string;
  items: ProductCatalogItem[];
};

export type ExperienceCatalogEntry = {
  kind: 'experience';
  title: string;
  options: ExperienceCatalogOption[];
};

export type CatalogEntry = ProductCatalogEntry | ExperienceCatalogEntry;

export type ProductCartItem = {
  kind: 'product';
  id: string;
  name: string;
  price: number;
  quantity: number;
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
  detectionId: number;
  confidence: number;
};

export type CartItem = ProductCartItem | ExperienceCartItem;
