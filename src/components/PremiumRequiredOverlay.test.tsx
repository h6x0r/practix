import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import React from 'react';
import { MemoryRouter } from 'react-router-dom';
import { PremiumRequiredOverlay } from './PremiumRequiredOverlay';

// Mock the LanguageContext
vi.mock('@/contexts/LanguageContext', () => ({
  useUITranslation: () => ({
    tUI: (key: string) => {
      const translations: Record<string, string> = {
        'premium.subscriptionRequired': 'Subscription Required',
        'premium.subscriptionRequiredDesc': 'This content requires a premium subscription.',
        'premium.purchaseSubscription': 'Get Premium',
      };
      return translations[key] || key;
    },
  }),
}));

describe('PremiumRequiredOverlay', () => {
  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <MemoryRouter>{children}</MemoryRouter>
  );

  it('should render children in blurred state', () => {
    render(
      <PremiumRequiredOverlay>
        <div data-testid="content">Premium Content</div>
      </PremiumRequiredOverlay>,
      { wrapper }
    );

    const content = screen.getByTestId('content');
    expect(content.parentElement).toHaveClass('blur-sm');
  });

  it('should show default title', () => {
    render(
      <PremiumRequiredOverlay>
        <div>Content</div>
      </PremiumRequiredOverlay>,
      { wrapper }
    );

    expect(screen.getByText('Subscription Required')).toBeInTheDocument();
  });

  it('should show custom title when provided', () => {
    render(
      <PremiumRequiredOverlay title="Custom Title">
        <div>Content</div>
      </PremiumRequiredOverlay>,
      { wrapper }
    );

    expect(screen.getByText('Custom Title')).toBeInTheDocument();
  });

  it('should show default description', () => {
    render(
      <PremiumRequiredOverlay>
        <div>Content</div>
      </PremiumRequiredOverlay>,
      { wrapper }
    );

    expect(screen.getByText('This content requires a premium subscription.')).toBeInTheDocument();
  });

  it('should show custom description when provided', () => {
    render(
      <PremiumRequiredOverlay description="Custom description text">
        <div>Content</div>
      </PremiumRequiredOverlay>,
      { wrapper }
    );

    expect(screen.getByText('Custom description text')).toBeInTheDocument();
  });

  it('should render purchase button link to payments', () => {
    render(
      <PremiumRequiredOverlay>
        <div>Content</div>
      </PremiumRequiredOverlay>,
      { wrapper }
    );

    const link = screen.getByRole('link', { name: /get premium/i });
    expect(link).toHaveAttribute('href', '/payments');
  });

  it('should hide children when showPreview is false', () => {
    render(
      <PremiumRequiredOverlay showPreview={false}>
        <div data-testid="content">Premium Content</div>
      </PremiumRequiredOverlay>,
      { wrapper }
    );

    expect(screen.queryByTestId('content')).not.toBeInTheDocument();
  });

  it('should show placeholder when showPreview is false', () => {
    render(
      <PremiumRequiredOverlay showPreview={false}>
        <div>Content</div>
      </PremiumRequiredOverlay>,
      { wrapper }
    );

    const placeholder = document.querySelector('.min-h-\\[400px\\]');
    expect(placeholder).toBeInTheDocument();
  });

  it('should have pointer-events disabled on blurred content', () => {
    render(
      <PremiumRequiredOverlay>
        <div data-testid="content">Premium Content</div>
      </PremiumRequiredOverlay>,
      { wrapper }
    );

    const content = screen.getByTestId('content');
    expect(content.parentElement).toHaveClass('pointer-events-none');
  });

  it('should have aria-hidden on blurred content', () => {
    render(
      <PremiumRequiredOverlay>
        <div data-testid="content">Premium Content</div>
      </PremiumRequiredOverlay>,
      { wrapper }
    );

    const content = screen.getByTestId('content');
    expect(content.parentElement).toHaveAttribute('aria-hidden', 'true');
  });
});
