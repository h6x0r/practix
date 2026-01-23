import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import React from 'react';
import { SubscriptionProvider, useSubscription } from './SubscriptionContext';

// Mock dependencies
vi.mock('@/features/subscriptions/api/subscriptionService', () => ({
  subscriptionService: {
    getMySubscriptions: vi.fn(),
    getTaskAccess: vi.fn(),
    getCourseAccess: vi.fn(),
  },
}));

vi.mock('@/features/auth/model/useAuth', () => ({
  useAuth: vi.fn(),
}));

import { subscriptionService } from '@/features/subscriptions/api/subscriptionService';
import { useAuth } from '@/features/auth/model/useAuth';

describe('SubscriptionContext', () => {
  const mockUser = { id: 'user-1', email: 'test@example.com' };

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(useAuth).mockReturnValue({
      isAuthenticated: true,
      user: mockUser,
      login: vi.fn(),
      logout: vi.fn(),
      isLoading: false,
    });
    vi.mocked(subscriptionService.getMySubscriptions).mockResolvedValue([]);
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <SubscriptionProvider>{children}</SubscriptionProvider>
  );

  describe('useSubscription hook', () => {
    it('should throw error when used outside provider', () => {
      // Suppress console.error for this test
      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

      expect(() => {
        renderHook(() => useSubscription());
      }).toThrow('useSubscription must be used within a SubscriptionProvider');

      consoleSpy.mockRestore();
    });

    it('should provide context values', async () => {
      const { result } = renderHook(() => useSubscription(), { wrapper });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.subscriptions).toEqual([]);
      expect(result.current.hasGlobalAccess).toBe(false);
      expect(result.current.error).toBeNull();
    });
  });

  describe('initial loading', () => {
    it('should fetch subscriptions on mount for authenticated user', async () => {
      const mockSubscriptions = [
        {
          id: 'sub-1',
          status: 'active',
          plan: { id: 'plan-1', type: 'global', name: 'Premium' },
          expiresAt: '2025-12-31',
        },
      ];
      vi.mocked(subscriptionService.getMySubscriptions).mockResolvedValue(mockSubscriptions);

      const { result } = renderHook(() => useSubscription(), { wrapper });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(subscriptionService.getMySubscriptions).toHaveBeenCalled();
      expect(result.current.subscriptions).toEqual(mockSubscriptions);
    });

    it('should not fetch subscriptions for unauthenticated user', async () => {
      vi.mocked(useAuth).mockReturnValue({
        isAuthenticated: false,
        user: null,
        login: vi.fn(),
        logout: vi.fn(),
        isLoading: false,
      });

      const { result } = renderHook(() => useSubscription(), { wrapper });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(subscriptionService.getMySubscriptions).not.toHaveBeenCalled();
      expect(result.current.subscriptions).toEqual([]);
    });

    it('should handle fetch error', async () => {
      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
      vi.mocked(subscriptionService.getMySubscriptions).mockRejectedValue(new Error('Network error'));

      const { result } = renderHook(() => useSubscription(), { wrapper });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.error).toBe('Failed to load subscription information');
      consoleSpy.mockRestore();
    });
  });

  describe('hasGlobalAccess', () => {
    it('should return true when user has active global subscription', async () => {
      vi.mocked(subscriptionService.getMySubscriptions).mockResolvedValue([
        {
          id: 'sub-1',
          status: 'active',
          plan: { id: 'plan-1', type: 'global', name: 'Premium' },
          expiresAt: '2025-12-31',
        },
      ]);

      const { result } = renderHook(() => useSubscription(), { wrapper });

      await waitFor(() => {
        expect(result.current.hasGlobalAccess).toBe(true);
      });
    });

    it('should return false when user has no subscriptions', async () => {
      vi.mocked(subscriptionService.getMySubscriptions).mockResolvedValue([]);

      const { result } = renderHook(() => useSubscription(), { wrapper });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.hasGlobalAccess).toBe(false);
    });

    it('should return false when subscription is not active', async () => {
      vi.mocked(subscriptionService.getMySubscriptions).mockResolvedValue([
        {
          id: 'sub-1',
          status: 'expired',
          plan: { id: 'plan-1', type: 'global', name: 'Premium' },
          expiresAt: '2024-12-31',
        },
      ]);

      const { result } = renderHook(() => useSubscription(), { wrapper });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.hasGlobalAccess).toBe(false);
    });

    it('should return false when only course subscription exists', async () => {
      vi.mocked(subscriptionService.getMySubscriptions).mockResolvedValue([
        {
          id: 'sub-1',
          status: 'active',
          plan: { id: 'plan-1', type: 'course', name: 'Go Basics', courseId: 'course-1' },
          expiresAt: '2025-12-31',
        },
      ]);

      const { result } = renderHook(() => useSubscription(), { wrapper });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.hasGlobalAccess).toBe(false);
    });
  });

  describe('hasCourseAccess', () => {
    it('should return true for global subscription holders', async () => {
      vi.mocked(subscriptionService.getMySubscriptions).mockResolvedValue([
        {
          id: 'sub-1',
          status: 'active',
          plan: { id: 'plan-1', type: 'global', name: 'Premium' },
          expiresAt: '2025-12-31',
        },
      ]);

      const { result } = renderHook(() => useSubscription(), { wrapper });

      await waitFor(() => {
        expect(result.current.hasGlobalAccess).toBe(true);
      });

      expect(result.current.hasCourseAccess('any-course')).toBe(true);
    });

    it('should return true for specific course subscription', async () => {
      vi.mocked(subscriptionService.getMySubscriptions).mockResolvedValue([
        {
          id: 'sub-1',
          status: 'active',
          plan: { id: 'plan-1', type: 'course', name: 'Go Basics', courseId: 'course-1' },
          expiresAt: '2025-12-31',
        },
      ]);

      const { result } = renderHook(() => useSubscription(), { wrapper });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.hasCourseAccess('course-1')).toBe(true);
    });

    it('should return false for different course', async () => {
      vi.mocked(subscriptionService.getMySubscriptions).mockResolvedValue([
        {
          id: 'sub-1',
          status: 'active',
          plan: { id: 'plan-1', type: 'course', name: 'Go Basics', courseId: 'course-1' },
          expiresAt: '2025-12-31',
        },
      ]);

      const { result } = renderHook(() => useSubscription(), { wrapper });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.hasCourseAccess('course-2')).toBe(false);
    });
  });

  describe('getTaskAccess', () => {
    it('should return default access for unauthenticated user', async () => {
      vi.mocked(useAuth).mockReturnValue({
        isAuthenticated: false,
        user: null,
        login: vi.fn(),
        logout: vi.fn(),
        isLoading: false,
      });

      const { result } = renderHook(() => useSubscription(), { wrapper });

      const access = await result.current.getTaskAccess('task-1');

      expect(subscriptionService.getTaskAccess).not.toHaveBeenCalled();
      expect(access.canView).toBe(true);
      expect(access.canSeeSolution).toBe(false);
    });

    it('should fetch task access from API', async () => {
      const mockAccess = {
        canView: true,
        canRun: true,
        canSubmit: true,
        canSeeSolution: true,
        canUseAiTutor: true,
        queuePriority: 5,
      };
      vi.mocked(subscriptionService.getTaskAccess).mockResolvedValue(mockAccess);

      const { result } = renderHook(() => useSubscription(), { wrapper });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      const access = await result.current.getTaskAccess('task-1');

      expect(subscriptionService.getTaskAccess).toHaveBeenCalledWith('task-1');
      expect(access).toEqual(mockAccess);
    });

    it('should cache task access', async () => {
      const mockAccess = {
        canView: true,
        canRun: true,
        canSubmit: true,
        canSeeSolution: true,
        canUseAiTutor: true,
        queuePriority: 5,
      };
      vi.mocked(subscriptionService.getTaskAccess).mockResolvedValue(mockAccess);

      const { result } = renderHook(() => useSubscription(), { wrapper });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // First call
      await result.current.getTaskAccess('task-1');
      // Second call - should use cache
      await result.current.getTaskAccess('task-1');

      expect(subscriptionService.getTaskAccess).toHaveBeenCalledTimes(1);
    });

    it('should return default access on API error', async () => {
      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
      vi.mocked(subscriptionService.getTaskAccess).mockRejectedValue(new Error('API error'));

      const { result } = renderHook(() => useSubscription(), { wrapper });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      const access = await result.current.getTaskAccess('task-1');

      expect(access.canView).toBe(true);
      expect(access.canSeeSolution).toBe(false);
      consoleSpy.mockRestore();
    });
  });

  describe('getCourseAccess', () => {
    it('should return default access for unauthenticated user', async () => {
      vi.mocked(useAuth).mockReturnValue({
        isAuthenticated: false,
        user: null,
        login: vi.fn(),
        logout: vi.fn(),
        isLoading: false,
      });

      const { result } = renderHook(() => useSubscription(), { wrapper });

      const access = await result.current.getCourseAccess('course-1');

      expect(subscriptionService.getCourseAccess).not.toHaveBeenCalled();
      expect(access.hasAccess).toBe(false);
    });

    it('should fetch course access from API', async () => {
      const mockAccess = {
        hasAccess: true,
        queuePriority: 3,
        canUseAiTutor: true,
      };
      vi.mocked(subscriptionService.getCourseAccess).mockResolvedValue(mockAccess);

      const { result } = renderHook(() => useSubscription(), { wrapper });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      const access = await result.current.getCourseAccess('course-1');

      expect(subscriptionService.getCourseAccess).toHaveBeenCalledWith('course-1');
      expect(access).toEqual(mockAccess);
    });

    it('should cache course access', async () => {
      const mockAccess = {
        hasAccess: true,
        queuePriority: 3,
        canUseAiTutor: true,
      };
      vi.mocked(subscriptionService.getCourseAccess).mockResolvedValue(mockAccess);

      const { result } = renderHook(() => useSubscription(), { wrapper });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // First call
      await result.current.getCourseAccess('course-1');
      // Second call - should use cache
      await result.current.getCourseAccess('course-1');

      expect(subscriptionService.getCourseAccess).toHaveBeenCalledTimes(1);
    });

    it('should return default access on API error', async () => {
      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
      vi.mocked(subscriptionService.getCourseAccess).mockRejectedValue(new Error('API error'));

      const { result } = renderHook(() => useSubscription(), { wrapper });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      const access = await result.current.getCourseAccess('course-1');

      expect(access.hasAccess).toBe(false);
      consoleSpy.mockRestore();
    });
  });

  describe('refreshSubscriptions', () => {
    it('should refetch subscriptions', async () => {
      vi.mocked(subscriptionService.getMySubscriptions)
        .mockResolvedValueOnce([])
        .mockResolvedValueOnce([
          {
            id: 'sub-1',
            status: 'active',
            plan: { id: 'plan-1', type: 'global', name: 'Premium' },
            expiresAt: '2025-12-31',
          },
        ]);

      const { result } = renderHook(() => useSubscription(), { wrapper });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.subscriptions).toEqual([]);

      await act(async () => {
        await result.current.refreshSubscriptions();
      });

      expect(result.current.subscriptions).toHaveLength(1);
    });

    it('should clear cache on refresh', async () => {
      const mockAccess = {
        hasAccess: true,
        queuePriority: 3,
        canUseAiTutor: true,
      };
      vi.mocked(subscriptionService.getCourseAccess).mockResolvedValue(mockAccess);
      vi.mocked(subscriptionService.getMySubscriptions).mockResolvedValue([]);

      const { result } = renderHook(() => useSubscription(), { wrapper });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // First call - caches result
      await result.current.getCourseAccess('course-1');
      expect(subscriptionService.getCourseAccess).toHaveBeenCalledTimes(1);

      // Refresh clears cache
      await act(async () => {
        await result.current.refreshSubscriptions();
      });

      // Second call - should fetch again (cache cleared)
      await result.current.getCourseAccess('course-1');
      expect(subscriptionService.getCourseAccess).toHaveBeenCalledTimes(2);
    });
  });
});
