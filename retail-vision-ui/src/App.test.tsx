import { render, screen, fireEvent } from '@testing-library/react';
import { BrandProvider } from './config/BrandContext';
import App from './App';

function renderApp() {
  return render(
    <BrandProvider>
      <App />
    </BrandProvider>
  );
}

test('renders a tab per brand and switches the tagline on click', () => {
  renderApp();
  // Tabs labelled by brand display name
  expect(screen.getByRole('tab', { name: /BLEND360/i })).toBeInTheDocument();
  expect(screen.getByRole('tab', { name: /Hyatt/i })).toBeInTheDocument();
  expect(screen.getByRole('tab', { name: /Under Armour/i })).toBeInTheDocument();

  // Default brand tagline visible
  expect(screen.getByText(/AI-Powered Retail Intelligence/i)).toBeInTheDocument();

  // Switch to Hyatt; its tagline replaces the default
  fireEvent.click(screen.getByRole('tab', { name: /Hyatt/i }));
  expect(screen.getByText(/Hyatt - Hospitality/i)).toBeInTheDocument();
});
