import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, act, waitFor } from '@testing-library/react';
import React from 'react';
import { ToastProvider, useToast } from './Toast';

// Test component that uses the toast
const TestComponent = () => {
  const { showToast } = useToast();

  return (
    <div>
      <button onClick={() => showToast('Success message', 'success')}>
        Show Success
      </button>
      <button onClick={() => showToast('Error message', 'error')}>
        Show Error
      </button>
      <button onClick={() => showToast('Info message', 'info')}>
        Show Info
      </button>
    </div>
  );
};

describe('Toast', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  describe('useToast hook', () => {
    it('should throw error when used outside provider', () => {
      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

      expect(() => {
        render(<TestComponent />);
      }).toThrow('useToast must be used within a ToastProvider');

      consoleSpy.mockRestore();
    });
  });

  describe('ToastProvider', () => {
    it('should render children', () => {
      render(
        <ToastProvider>
          <div>Child content</div>
        </ToastProvider>
      );

      expect(screen.getByText('Child content')).toBeInTheDocument();
    });

    it('should show success toast', () => {
      render(
        <ToastProvider>
          <TestComponent />
        </ToastProvider>
      );

      fireEvent.click(screen.getByText('Show Success'));

      expect(screen.getByText('Success message')).toBeInTheDocument();
    });

    it('should show error toast', () => {
      render(
        <ToastProvider>
          <TestComponent />
        </ToastProvider>
      );

      fireEvent.click(screen.getByText('Show Error'));

      expect(screen.getByText('Error message')).toBeInTheDocument();
    });

    it('should show info toast', () => {
      render(
        <ToastProvider>
          <TestComponent />
        </ToastProvider>
      );

      fireEvent.click(screen.getByText('Show Info'));

      expect(screen.getByText('Info message')).toBeInTheDocument();
    });

    it('should auto-dismiss toast after 4 seconds', () => {
      render(
        <ToastProvider>
          <TestComponent />
        </ToastProvider>
      );

      fireEvent.click(screen.getByText('Show Success'));

      expect(screen.getByText('Success message')).toBeInTheDocument();

      act(() => {
        vi.advanceTimersByTime(4000);
      });

      expect(screen.queryByText('Success message')).not.toBeInTheDocument();
    });

    it('should allow manual dismissal', () => {
      render(
        <ToastProvider>
          <TestComponent />
        </ToastProvider>
      );

      fireEvent.click(screen.getByText('Show Success'));

      expect(screen.getByText('Success message')).toBeInTheDocument();

      // Find and click the close button (X icon)
      const closeButtons = screen.getAllByRole('button');
      // The last button added should be the close button on the toast
      fireEvent.click(closeButtons[closeButtons.length - 1]);

      expect(screen.queryByText('Success message')).not.toBeInTheDocument();
    });

    it('should show multiple toasts', () => {
      render(
        <ToastProvider>
          <TestComponent />
        </ToastProvider>
      );

      fireEvent.click(screen.getByText('Show Success'));
      fireEvent.click(screen.getByText('Show Error'));
      fireEvent.click(screen.getByText('Show Info'));

      expect(screen.getByText('Success message')).toBeInTheDocument();
      expect(screen.getByText('Error message')).toBeInTheDocument();
      expect(screen.getByText('Info message')).toBeInTheDocument();
    });

    it('should dismiss toasts independently', () => {
      render(
        <ToastProvider>
          <TestComponent />
        </ToastProvider>
      );

      // Show first toast
      fireEvent.click(screen.getByText('Show Success'));

      // Wait 2 seconds, then show second toast
      act(() => {
        vi.advanceTimersByTime(2000);
      });

      fireEvent.click(screen.getByText('Show Error'));

      // Both should be visible
      expect(screen.getByText('Success message')).toBeInTheDocument();
      expect(screen.getByText('Error message')).toBeInTheDocument();

      // After 2 more seconds, first toast should auto-dismiss
      act(() => {
        vi.advanceTimersByTime(2000);
      });

      expect(screen.queryByText('Success message')).not.toBeInTheDocument();
      expect(screen.getByText('Error message')).toBeInTheDocument();
    });

    it('should cleanup timeouts on manual dismiss', () => {
      render(
        <ToastProvider>
          <TestComponent />
        </ToastProvider>
      );

      fireEvent.click(screen.getByText('Show Success'));

      // Manually dismiss
      const closeButtons = screen.getAllByRole('button');
      fireEvent.click(closeButtons[closeButtons.length - 1]);

      // Toast should be gone immediately
      expect(screen.queryByText('Success message')).not.toBeInTheDocument();

      // Advance timers - should not cause any issues
      act(() => {
        vi.advanceTimersByTime(5000);
      });

      // Still no toast
      expect(screen.queryByText('Success message')).not.toBeInTheDocument();
    });
  });
});
