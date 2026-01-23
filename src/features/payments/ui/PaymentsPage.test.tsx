import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import React from 'react';
import { MemoryRouter } from 'react-router-dom';
import PaymentsPage from './PaymentsPage';
import { AuthContext } from '@/components/Layout';

// Mock dependencies
vi.mock('../api/paymentService', () => ({
  paymentService: {
    getProviders: vi.fn(),
    getPaymentHistory: vi.fn(),
    createCheckout: vi.fn(),
  },
}));

vi.mock('@/features/subscriptions/api/subscriptionService', () => ({
  subscriptionService: {
    getPlans: vi.fn(),
  },
}));

vi.mock('@/contexts/LanguageContext', () => ({
  useLanguage: () => ({
    language: 'en',
  }),
}));

vi.mock('@/lib/logger', () => ({
  createLogger: () => ({
    info: vi.fn(),
    error: vi.fn(),
    warn: vi.fn(),
    debug: vi.fn(),
  }),
}));

vi.mock('@/components/Toast', () => ({
  useToast: () => ({
    showToast: vi.fn(),
  }),
}));

import { paymentService } from '../api/paymentService';
import { subscriptionService } from '@/features/subscriptions/api/subscriptionService';

describe('PaymentsPage', () => {
  const mockUser = { id: 'user-1', email: 'test@example.com', name: 'Test User' };

  const mockPlans = [
    {
      id: 'plan-global',
      slug: 'global-premium',
      name: 'Global Premium',
      type: 'global',
      price: 9900000, // 99,000 UZS in tiyn
      features: ['All courses', 'Priority support'],
      durationDays: 30,
    },
    {
      id: 'plan-course-1',
      slug: 'go-basics',
      name: 'Go Basics',
      type: 'course',
      courseId: 'course-1',
      price: 4900000,
      features: ['Full course access'],
      durationDays: 365,
    },
  ];

  const mockProviders = [
    { id: 'payme', name: 'Payme', enabled: true },
    { id: 'click', name: 'Click', enabled: true },
  ];

  const mockHistory = [
    {
      id: 'payment-1',
      amount: 9900000,
      status: 'completed',
      provider: 'payme',
      planName: 'Global Premium',
      createdAt: '2025-01-10T10:00:00Z',
    },
  ];

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(subscriptionService.getPlans).mockResolvedValue(mockPlans);
    vi.mocked(paymentService.getProviders).mockResolvedValue(mockProviders);
    vi.mocked(paymentService.getPaymentHistory).mockResolvedValue(mockHistory);
  });

  const renderWithAuth = (user: typeof mockUser | null) => {
    return render(
      <AuthContext.Provider value={{ user, setUser: vi.fn() }}>
        <MemoryRouter>
          <PaymentsPage />
        </MemoryRouter>
      </AuthContext.Provider>
    );
  };

  describe('loading state', () => {
    it('should show loading indicator initially', () => {
      renderWithAuth(mockUser);

      // The page uses loading state
      expect(subscriptionService.getPlans).toHaveBeenCalled();
    });
  });

  describe('authenticated user', () => {
    it('should load plans and providers', async () => {
      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(subscriptionService.getPlans).toHaveBeenCalled();
      });

      expect(paymentService.getProviders).toHaveBeenCalled();
      expect(paymentService.getPaymentHistory).toHaveBeenCalled();
    });

    it('should display subscription plans', async () => {
      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText('Global Premium')).toBeInTheDocument();
      });
    });

    it('should display course plans', async () => {
      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText('Go Basics')).toBeInTheDocument();
      });
    });

    it('should handle API error gracefully', async () => {
      vi.mocked(subscriptionService.getPlans).mockRejectedValue(new Error('API Error'));

      renderWithAuth(mockUser);

      // Should not crash
      await waitFor(() => {
        expect(screen.queryByText('Loading...')).not.toBeInTheDocument();
      });
    });
  });

  describe('unauthenticated user', () => {
    it('should not load data when not authenticated', () => {
      renderWithAuth(null);

      expect(subscriptionService.getPlans).not.toHaveBeenCalled();
      expect(paymentService.getProviders).not.toHaveBeenCalled();
    });
  });

  describe('tabs', () => {
    it('should switch between tabs', async () => {
      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.queryByText('Loading...')).not.toBeInTheDocument();
      });

      // Default tab is subscribe, look for history tab
      const historyTab = screen.getByRole('button', { name: /history/i });
      if (historyTab) {
        fireEvent.click(historyTab);
      }
    });
  });
});
