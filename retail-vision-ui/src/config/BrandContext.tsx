import React, { createContext, useContext, useMemo, useState } from 'react';
import brands, { BrandConfig } from './brands';

const envKey = process.env.REACT_APP_BRAND;
export const DEFAULT_BRAND_KEY: string =
  envKey && brands[envKey] ? envKey : 'blend360';

interface BrandContextValue {
  brandKey: string;
  setBrandKey: (k: string) => void;
}

const BrandContext = createContext<BrandContextValue | null>(null);

export const BrandProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [brandKey, setBrandKey] = useState<string>(DEFAULT_BRAND_KEY);
  const value = useMemo(() => ({ brandKey, setBrandKey }), [brandKey]);
  return <BrandContext.Provider value={value}>{children}</BrandContext.Provider>;
};

function useBrandContext(): BrandContextValue {
  const ctx = useContext(BrandContext);
  if (!ctx) throw new Error('useBrand/useBrandKey must be used within a BrandProvider');
  return ctx;
}

export function useBrandKey(): BrandContextValue {
  return useBrandContext();
}

export function useBrand(): BrandConfig {
  const { brandKey } = useBrandContext();
  return brands[brandKey] || brands['blend360'];
}
