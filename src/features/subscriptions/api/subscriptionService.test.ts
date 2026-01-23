import { describe, it, expect, beforeEach, vi } from 'vitest';
import { subscriptionService } from './subscriptionService';

vi.mock('@/lib/api', () => ({
  api: {
    get: vi.fn(),
    post: vi.fn(),
    delete: vi.fn(),
  },
}));

import { api } from '@/lib/api';

describe('subscriptionService', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('getPlans', () => {
    it('should fetch subscription plans', async () => {
      const mockPlans = [
        { id: 'plan-1', slug: 'global-premium', name: 'Global Premium', type: 'global', price: 100000 },
        { id: 'plan-2', slug: 'go-basics', name: 'Go Basics Course', type: 'course', price: 50000 },
      ];

      vi.mocked(api.get).mockResolvedValueOnce(mockPlans);

      const result = await subscriptionService.getPlans();

      expect(api.get).toHaveBeenCalledWith('/subscriptions/plans');
      expect(result).toHaveLength(2);
    });

    it('should throw on API error', async () => {
      vi.mocked(api.get).mockRejectedValueOnce(new Error('Service unavailable'));

      await expect(subscriptionService.getPlans()).rejects.toThrow('Service unavailable');
    });
  });

  describe('getPlanBySlug', () => {
    it('should fetch specific plan', async () => {
      const mockPlan = {
        id: 'plan-1',
        slug: 'global-premium',
        name: 'Global Premium',
        type: 'global',
        price: 100000,
      };

      vi.mocked(api.get).mockResolvedValueOnce(mockPlan);

      const result = await subscriptionService.getPlanBySlug('global-premium');

      expect(api.get).toHaveBeenCalledWith('/subscriptions/plans/global-premium');
      expect(result.slug).toBe('global-premium');
    });

    it('should throw on plan not found', async () => {
      vi.mocked(api.get).mockRejectedValueOnce(new Error('Plan not found'));

      await expect(subscriptionService.getPlanBySlug('invalid')).rejects.toThrow('Plan not found');
    });
  });

  describe('getMySubscriptions', () => {
    it('should fetch user subscriptions', async () => {
      const mockSubscriptions = [
        {
          id: 'sub-1',
          planId: 'plan-1',
          status: 'active',
          startDate: '2025-01-01',
          endDate: '2025-02-01',
        },
      ];

      vi.mocked(api.get).mockResolvedValueOnce(mockSubscriptions);

      const result = await subscriptionService.getMySubscriptions();

      expect(api.get).toHaveBeenCalledWith('/subscriptions/my');
      expect(result).toHaveLength(1);
      expect(result[0].status).toBe('active');
    });

    it('should return empty array for no subscriptions', async () => {
      vi.mocked(api.get).mockResolvedValueOnce([]);

      const result = await subscriptionService.getMySubscriptions();

      expect(result).toHaveLength(0);
    });

    it('should throw on API error', async () => {
      vi.mocked(api.get).mockRejectedValueOnce(new Error('Unauthorized'));

      await expect(subscriptionService.getMySubscriptions()).rejects.toThrow('Unauthorized');
    });
  });

  describe('getCourseAccess', () => {
    it('should fetch course access for premium user', async () => {
      const mockAccess = {
        hasAccess: true,
        queuePriority: 1,
        canUseAiTutor: true,
      };

      vi.mocked(api.get).mockResolvedValueOnce(mockAccess);

      const result = await subscriptionService.getCourseAccess('course-123');

      expect(api.get).toHaveBeenCalledWith('/subscriptions/access/course/course-123');
      expect(result.hasAccess).toBe(true);
    });

    it('should fetch course access for free user', async () => {
      const mockAccess = {
        hasAccess: false,
        queuePriority: 10,
        canUseAiTutor: false,
      };

      vi.mocked(api.get).mockResolvedValueOnce(mockAccess);

      const result = await subscriptionService.getCourseAccess('course-456');

      expect(result.hasAccess).toBe(false);
      expect(result.queuePriority).toBe(10);
    });

    it('should throw on API error', async () => {
      vi.mocked(api.get).mockRejectedValueOnce(new Error('Course not found'));

      await expect(subscriptionService.getCourseAccess('invalid')).rejects.toThrow('Course not found');
    });
  });

  describe('getTaskAccess', () => {
    it('should fetch task access with full permissions', async () => {
      const mockAccess = {
        canView: true,
        canRun: true,
        canSubmit: true,
        canSeeSolution: true,
        canUseAiTutor: true,
        queuePriority: 1,
      };

      vi.mocked(api.get).mockResolvedValueOnce(mockAccess);

      const result = await subscriptionService.getTaskAccess('task-123');

      expect(api.get).toHaveBeenCalledWith('/subscriptions/access/task/task-123');
      expect(result.canView).toBe(true);
      expect(result.canSeeSolution).toBe(true);
    });

    it('should fetch task access for free user', async () => {
      const mockAccess = {
        canView: true,
        canRun: true,
        canSubmit: true,
        canSeeSolution: false,
        canUseAiTutor: false,
        queuePriority: 10,
      };

      vi.mocked(api.get).mockResolvedValueOnce(mockAccess);

      const result = await subscriptionService.getTaskAccess('task-456');

      expect(result.canSeeSolution).toBe(false);
      expect(result.canUseAiTutor).toBe(false);
    });

    it('should throw on API error', async () => {
      vi.mocked(api.get).mockRejectedValueOnce(new Error('Task not found'));

      await expect(subscriptionService.getTaskAccess('invalid')).rejects.toThrow('Task not found');
    });
  });

  describe('createSubscription', () => {
    it('should create subscription with auto-renew', async () => {
      const mockSubscription = {
        id: 'sub-new',
        planId: 'plan-1',
        status: 'active',
        autoRenew: true,
      };

      vi.mocked(api.post).mockResolvedValueOnce(mockSubscription);

      const result = await subscriptionService.createSubscription('plan-1');

      expect(api.post).toHaveBeenCalledWith('/subscriptions', { planId: 'plan-1', autoRenew: true });
      expect(result.autoRenew).toBe(true);
    });

    it('should create subscription without auto-renew', async () => {
      const mockSubscription = {
        id: 'sub-new',
        planId: 'plan-1',
        status: 'active',
        autoRenew: false,
      };

      vi.mocked(api.post).mockResolvedValueOnce(mockSubscription);

      const result = await subscriptionService.createSubscription('plan-1', false);

      expect(api.post).toHaveBeenCalledWith('/subscriptions', { planId: 'plan-1', autoRenew: false });
      expect(result.autoRenew).toBe(false);
    });

    it('should throw on payment required', async () => {
      vi.mocked(api.post).mockRejectedValueOnce(new Error('Payment required'));

      await expect(subscriptionService.createSubscription('plan-1')).rejects.toThrow('Payment required');
    });
  });

  describe('cancelSubscription', () => {
    it('should cancel subscription', async () => {
      const mockSubscription = {
        id: 'sub-1',
        status: 'cancelled',
        cancelledAt: '2025-01-16T00:00:00Z',
      };

      vi.mocked(api.delete).mockResolvedValueOnce(mockSubscription);

      const result = await subscriptionService.cancelSubscription('sub-1');

      expect(api.delete).toHaveBeenCalledWith('/subscriptions/sub-1');
      expect(result.status).toBe('cancelled');
    });

    it('should throw on subscription not found', async () => {
      vi.mocked(api.delete).mockRejectedValueOnce(new Error('Subscription not found'));

      await expect(subscriptionService.cancelSubscription('invalid')).rejects.toThrow('Subscription not found');
    });
  });
});
