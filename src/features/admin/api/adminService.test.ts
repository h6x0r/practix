import { describe, it, expect, beforeEach, vi } from 'vitest';
import { adminService } from './adminService';

// Mock the api module
vi.mock('@/lib/api', () => ({
  api: {
    get: vi.fn(),
  },
}));

import { api } from '@/lib/api';

describe('adminService', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('getDashboardStats', () => {
    it('should fetch dashboard stats', async () => {
      const mockStats = {
        totalUsers: 1500,
        newUsers: 25,
        premiumUsers: 200,
        activeUsers: {
          daily: 300,
          weekly: 800,
          monthly: 1200,
        },
      };

      vi.mocked(api.get).mockResolvedValueOnce(mockStats);

      const result = await adminService.getDashboardStats();

      expect(api.get).toHaveBeenCalledWith('/admin/analytics/dashboard');
      expect(result).toEqual(mockStats);
    });

    it('should throw on API error', async () => {
      vi.mocked(api.get).mockRejectedValueOnce(new Error('Unauthorized'));

      await expect(adminService.getDashboardStats()).rejects.toThrow('Unauthorized');
    });
  });

  describe('getCourseAnalytics', () => {
    it('should fetch course analytics', async () => {
      const mockAnalytics = {
        courses: [
          {
            courseId: 'course-1',
            courseSlug: 'go-basics',
            courseTitle: 'Go Basics',
            category: 'go',
            totalEnrolled: 500,
            completed: 100,
            completionRate: 20,
            averageProgress: 45,
            translations: { ru: { title: 'Основы Go' } },
          },
          {
            courseId: 'course-2',
            courseSlug: 'python-basics',
            courseTitle: 'Python Basics',
            category: 'python',
            totalEnrolled: 800,
            completed: 200,
            completionRate: 25,
            averageProgress: 55,
            translations: null,
          },
        ],
        totalCourses: 18,
      };

      vi.mocked(api.get).mockResolvedValueOnce(mockAnalytics);

      const result = await adminService.getCourseAnalytics();

      expect(api.get).toHaveBeenCalledWith('/admin/analytics/courses');
      expect(result).toEqual(mockAnalytics);
      expect(result.courses).toHaveLength(2);
      expect(result.totalCourses).toBe(18);
    });

    it('should handle empty courses list', async () => {
      const mockEmpty = {
        courses: [],
        totalCourses: 0,
      };

      vi.mocked(api.get).mockResolvedValueOnce(mockEmpty);

      const result = await adminService.getCourseAnalytics();

      expect(result.courses).toHaveLength(0);
    });

    it('should throw on API error', async () => {
      vi.mocked(api.get).mockRejectedValueOnce(new Error('Forbidden'));

      await expect(adminService.getCourseAnalytics()).rejects.toThrow('Forbidden');
    });
  });

  describe('getTaskAnalytics', () => {
    it('should fetch task analytics', async () => {
      const mockAnalytics = {
        hardestTasks: [
          {
            taskId: 'task-1',
            taskSlug: 'binary-search',
            taskTitle: 'Binary Search',
            difficulty: 'hard',
            isPremium: true,
            totalSubmissions: 1000,
            acceptedSubmissions: 200,
            uniqueUsers: 300,
            passRate: 20,
          },
        ],
        mostPopularTasks: [
          {
            taskId: 'task-2',
            taskSlug: 'hello-world',
            taskTitle: 'Hello World',
            difficulty: 'easy',
            isPremium: false,
            totalSubmissions: 5000,
            acceptedSubmissions: 4500,
            uniqueUsers: 1500,
            passRate: 90,
          },
        ],
        totalTasks: 921,
        tasksWithSubmissions: 500,
      };

      vi.mocked(api.get).mockResolvedValueOnce(mockAnalytics);

      const result = await adminService.getTaskAnalytics();

      expect(api.get).toHaveBeenCalledWith('/admin/analytics/tasks');
      expect(result).toEqual(mockAnalytics);
      expect(result.hardestTasks[0].passRate).toBe(20);
      expect(result.mostPopularTasks[0].passRate).toBe(90);
    });

    it('should throw on API error', async () => {
      vi.mocked(api.get).mockRejectedValueOnce(new Error('Server error'));

      await expect(adminService.getTaskAnalytics()).rejects.toThrow('Server error');
    });
  });

  describe('getSubmissionStats', () => {
    it('should fetch submission statistics', async () => {
      const mockStats = {
        totalSubmissions: 50000,
        recentSubmissions: 500,
        byStatus: [
          { status: 'passed', count: 35000, percentage: 70 },
          { status: 'failed', count: 10000, percentage: 20 },
          { status: 'error', count: 5000, percentage: 10 },
        ],
        dailySubmissions: {
          '2025-01-15': 450,
          '2025-01-14': 520,
          '2025-01-13': 480,
        },
      };

      vi.mocked(api.get).mockResolvedValueOnce(mockStats);

      const result = await adminService.getSubmissionStats();

      expect(api.get).toHaveBeenCalledWith('/admin/analytics/submissions');
      expect(result).toEqual(mockStats);
      expect(result.byStatus).toHaveLength(3);
    });

    it('should handle empty submission data', async () => {
      const mockEmpty = {
        totalSubmissions: 0,
        recentSubmissions: 0,
        byStatus: [],
        dailySubmissions: {},
      };

      vi.mocked(api.get).mockResolvedValueOnce(mockEmpty);

      const result = await adminService.getSubmissionStats();

      expect(result.totalSubmissions).toBe(0);
      expect(result.byStatus).toHaveLength(0);
    });

    it('should throw on API error', async () => {
      vi.mocked(api.get).mockRejectedValueOnce(new Error('Network error'));

      await expect(adminService.getSubmissionStats()).rejects.toThrow('Network error');
    });
  });

  describe('getSubscriptionStats', () => {
    it('should fetch subscription statistics', async () => {
      const mockStats = {
        activeSubscriptions: 200,
        newSubscriptionsThisMonth: 50,
        byPlan: [
          {
            planId: 'plan-1',
            planName: 'Global Premium',
            planSlug: 'global-premium',
            planType: 'global',
            count: 150,
            monthlyRevenue: 15000,
          },
          {
            planId: 'plan-2',
            planName: 'Go Course',
            planSlug: 'go-basics',
            planType: 'course',
            count: 50,
            monthlyRevenue: 2500,
          },
        ],
        totalMonthlyRevenue: 17500,
        completedPayments: 180,
        totalRevenue: 150000,
      };

      vi.mocked(api.get).mockResolvedValueOnce(mockStats);

      const result = await adminService.getSubscriptionStats();

      expect(api.get).toHaveBeenCalledWith('/admin/analytics/subscriptions');
      expect(result).toEqual(mockStats);
      expect(result.byPlan).toHaveLength(2);
      expect(result.totalMonthlyRevenue).toBe(17500);
    });

    it('should throw on API error', async () => {
      vi.mocked(api.get).mockRejectedValueOnce(new Error('Unauthorized'));

      await expect(adminService.getSubscriptionStats()).rejects.toThrow('Unauthorized');
    });
  });

  describe('getAiUsageStats', () => {
    it('should fetch AI usage statistics', async () => {
      const mockStats = {
        totalQueries: 5000,
        avgQueriesPerUser: 3.5,
        topicBreakdown: [
          { topic: 'debugging', count: 2000 },
          { topic: 'explanation', count: 1500 },
          { topic: 'hints', count: 1500 },
        ],
        dailyUsage: [
          { date: '2025-01-15', count: 180 },
          { date: '2025-01-14', count: 200 },
          { date: '2025-01-13', count: 150 },
        ],
      };

      vi.mocked(api.get).mockResolvedValueOnce(mockStats);

      const result = await adminService.getAiUsageStats();

      expect(api.get).toHaveBeenCalledWith('/admin/analytics/ai-usage');
      expect(result).toEqual(mockStats);
      expect(result.topicBreakdown).toHaveLength(3);
      expect(result.dailyUsage).toHaveLength(3);
    });

    it('should handle zero AI usage', async () => {
      const mockEmpty = {
        totalQueries: 0,
        avgQueriesPerUser: 0,
        topicBreakdown: [],
        dailyUsage: [],
      };

      vi.mocked(api.get).mockResolvedValueOnce(mockEmpty);

      const result = await adminService.getAiUsageStats();

      expect(result.totalQueries).toBe(0);
      expect(result.topicBreakdown).toHaveLength(0);
    });

    it('should throw on API error', async () => {
      vi.mocked(api.get).mockRejectedValueOnce(new Error('Service unavailable'));

      await expect(adminService.getAiUsageStats()).rejects.toThrow('Service unavailable');
    });
  });
});
