import { render, screen, fireEvent } from '@testing-library/react';
import { BrandProvider, useBrand, useBrandKey } from './BrandContext';

function Probe() {
  const brand = useBrand();
  const { brandKey, setBrandKey } = useBrandKey();
  return (
    <div>
      <span data-testid="key">{brandKey}</span>
      <span data-testid="name">{brand.name}</span>
      <button onClick={() => setBrandKey('hyatt')}>to-hyatt</button>
    </div>
  );
}

test('defaults to blend360 and switches brand at runtime', () => {
  render(
    <BrandProvider>
      <Probe />
    </BrandProvider>
  );
  expect(screen.getByTestId('key').textContent).toBe('blend360');
  expect(screen.getByTestId('name').textContent).toBe('BLEND360');

  fireEvent.click(screen.getByText('to-hyatt'));
  expect(screen.getByTestId('key').textContent).toBe('hyatt');
  expect(screen.getByTestId('name').textContent).toBe('Hyatt');
});
