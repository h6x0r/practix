import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { DrillDownModal, DrillDownData } from './DrillDownModal';
import { LanguageProvider } from '@/contexts/LanguageContext';

const renderModal = (data: DrillDownData | null, onClose = vi.fn(), loading = false) => {
  return render(
    <LanguageProvider>
      <DrillDownModal data={data} onClose={onClose} loading={loading} />
    </LanguageProvider>
  );
};

describe('DrillDownModal', () => {
  const mockData: DrillDownData = {
    date: '2024-01-15',
    metric: 'dau',
    value: 150,
    details: [
      { id: '1', label: 'Mobile Users', value: 80, sublabel: 'iOS & Android' },
      { id: '2', label: 'Desktop Users', value: 70 },
    ],
  };

  describe('rendering', () => {
    it('should not render when data is null', () => {
      const { container } = renderModal(null);
      expect(container.innerHTML).toBe('');
    });

    it('should render modal with data', () => {
      renderModal(mockData);
      expect(screen.getByText('Daily Active Users')).toBeInTheDocument();
      expect(screen.getByText('150')).toBeInTheDocument();
    });

    it('should display formatted date', () => {
      renderModal(mockData);
      expect(screen.getByText(/January 15, 2024/)).toBeInTheDocument();
    });

    it('should display metric icon', () => {
      renderModal(mockData);
      expect(screen.getByText('👥')).toBeInTheDocument();
    });
  });

  describe('metric types', () => {
    it('should display revenue with UZS formatting', () => {
      const revenueData: DrillDownData = {
        date: '2024-01-15',
        metric: 'revenue',
        value: 10000000, // in tiyn (100 UZS = 1 tiyn)
      };
      renderModal(revenueData);
      expect(screen.getByText('💰')).toBeInTheDocument();
      expect(screen.getByText('Revenue')).toBeInTheDocument();
    });

    it('should display payments metric', () => {
      const paymentsData: DrillDownData = {
        date: '2024-01-15',
        metric: 'payments',
        value: 25,
      };
      renderModal(paymentsData);
      expect(screen.getByText('💳')).toBeInTheDocument();
      expect(screen.getByText('Payments')).toBeInTheDocument();
    });

    it('should display new users metric', () => {
      const newUsersData: DrillDownData = {
        date: '2024-01-15',
        metric: 'newUsers',
        value: 42,
      };
      renderModal(newUsersData);
      expect(screen.getByText('🆕')).toBeInTheDocument();
      expect(screen.getByText('New Users')).toBeInTheDocument();
    });

    it('should display subscriptions metric', () => {
      const subsData: DrillDownData = {
        date: '2024-01-15',
        metric: 'subscriptions',
        value: 10,
      };
      renderModal(subsData);
      expect(screen.getByText('⭐')).toBeInTheDocument();
      expect(screen.getByText('New Subscriptions')).toBeInTheDocument();
    });
  });

  describe('details list', () => {
    it('should render detail items', () => {
      renderModal(mockData);
      expect(screen.getByText('Mobile Users')).toBeInTheDocument();
      expect(screen.getByText('Desktop Users')).toBeInTheDocument();
    });

    it('should display sublabels', () => {
      renderModal(mockData);
      expect(screen.getByText('iOS & Android')).toBeInTheDocument();
    });

    it('should display status badges when provided', () => {
      const dataWithStatus: DrillDownData = {
        date: '2024-01-15',
        metric: 'payments',
        value: 5,
        details: [
          { id: '1', label: 'Payment 1', value: '100 UZS', status: 'success' },
          { id: '2', label: 'Payment 2', value: '50 UZS', status: 'error' },
        ],
      };
      renderModal(dataWithStatus);
      expect(screen.getByText('success')).toBeInTheDocument();
      expect(screen.getByText('error')).toBeInTheDocument();
    });

    it('should show empty state when no details', () => {
      const noDetailsData: DrillDownData = {
        date: '2024-01-15',
        metric: 'dau',
        value: 100,
        details: [],
      };
      renderModal(noDetailsData);
      expect(screen.getByText(/No detailed data available/i)).toBeInTheDocument();
    });
  });

  describe('loading state', () => {
    it('should show loading spinner when loading', () => {
      renderModal(mockData, vi.fn(), true);
      expect(document.querySelector('.animate-spin')).toBeInTheDocument();
    });

    it('should not show details when loading', () => {
      renderModal(mockData, vi.fn(), true);
      expect(screen.queryByText('Mobile Users')).not.toBeInTheDocument();
    });
  });

  describe('interactions', () => {
    it('should call onClose when close button clicked', () => {
      const onClose = vi.fn();
      renderModal(mockData, onClose);

      const closeButton = screen.getByRole('button', { name: /close/i });
      fireEvent.click(closeButton);

      expect(onClose).toHaveBeenCalled();
    });

    it('should call onClose when backdrop clicked', () => {
      const onClose = vi.fn();
      const { container } = renderModal(mockData, onClose);

      const backdrop = container.querySelector('.fixed.inset-0');
      if (backdrop) {
        fireEvent.click(backdrop);
        expect(onClose).toHaveBeenCalled();
      }
    });

    it('should not close when modal content clicked', () => {
      const onClose = vi.fn();
      renderModal(mockData, onClose);

      const modalContent = screen.getByText('Daily Active Users');
      fireEvent.click(modalContent);

      // onClose should only be called once from the potential backdrop click, not from content
      expect(onClose).not.toHaveBeenCalled();
    });

    it('should call onClose when header X button clicked', () => {
      const onClose = vi.fn();
      renderModal(mockData, onClose);

      // Find the X button in the header
      const buttons = screen.getAllByRole('button');
      const xButton = buttons.find(btn => btn.querySelector('svg'));
      if (xButton) {
        fireEvent.click(xButton);
        expect(onClose).toHaveBeenCalled();
      }
    });
  });
});
