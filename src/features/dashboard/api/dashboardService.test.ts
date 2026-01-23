import { describe, it, expect, beforeEach, vi } from 'vitest';
import { dashboardService } from './dashboardService';

vi.mock('@/lib/api', () => ({
  api: {
    get: vi.fn(),
  },
}));

import { api } from '@/lib/api';

describe('dashboardService', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('getStats', () => {
    it('should fetch user statistics', async () => {
      const mockStats = {
        totalSolved: 150,
        totalSubmissions: 500,
        totalMinutes: 3600,
        globalRank: 42,
        topPercent: 5,
        skillPoints: 2500,
        currentStreak: 7,
        maxStreak: 30,
        weekThisWeek: 15,
      };

      vi.mocked(api.get).mockResolvedValueOnce(mockStats);

      const result = await dashboardService.getStats();

      expect(api.get).toHaveBeenCalledWith('/users/me/stats');
      expect(result).toEqual(mockStats);
    });

    it('should handle new user with zero stats', async () => {
      const mockStats = {
        totalSolved: 0,
        totalSubmissions: 0,
        totalMinutes: 0,
        globalRank: 0,
        topPercent: 100,
        skillPoints: 0,
        currentStreak: 0,
        maxStreak: 0,
        weekThisWeek: 0,
      };

      vi.mocked(api.get).mockResolvedValueOnce(mockStats);

      const result = await dashboardService.getStats();

      expect(result.totalSolved).toBe(0);
      expect(result.topPercent).toBe(100);
    });

    it('should throw on API error', async () => {
      vi.mocked(api.get).mockRejectedValueOnce(new Error('Unauthorized'));

      await expect(dashboardService.getStats()).rejects.toThrow('Unauthorized');
    });
  });

  describe('getWeeklyActivity', () => {
    it('should fetch weekly activity with default days', async () => {
      const mockActivity = [
        { name: 'Mon', date: '2025-01-13', solved: 5, submissions: 10 },
        { name: 'Tue', date: '2025-01-14', solved: 3, submissions: 8 },
        { name: 'Wed', date: '2025-01-15', solved: 7, submissions: 15 },
      ];

      vi.mocked(api.get).mockResolvedValueOnce(mockActivity);

      const result = await dashboardService.getWeeklyActivity();

      expect(api.get).toHaveBeenCalledWith('/users/me/activity?days=7');
      expect(result).toEqual(mockActivity);
      expect(result).toHaveLength(3);
    });

    it('should fetch activity with custom days parameter', async () => {
      vi.mocked(api.get).mockResolvedValueOnce([]);

      await dashboardService.getWeeklyActivity(14);

      expect(api.get).toHaveBeenCalledWith('/users/me/activity?days=14');
    });

    it('should handle empty activity', async () => {
      vi.mocked(api.get).mockResolvedValueOnce([]);

      const result = await dashboardService.getWeeklyActivity();

      expect(result).toHaveLength(0);
    });

    it('should throw on API error', async () => {
      vi.mocked(api.get).mockRejectedValueOnce(new Error('Network error'));

      await expect(dashboardService.getWeeklyActivity()).rejects.toThrow('Network error');
    });
  });

  describe('getYearlyActivity', () => {
    it('should fetch yearly activity for heatmap', async () => {
      const mockYearlyActivity = [
        { date: '2025-01-01', count: 5 },
        { date: '2025-01-02', count: 3 },
        { date: '2025-01-03', count: 0 },
        { date: '2025-01-04', count: 8 },
      ];

      vi.mocked(api.get).mockResolvedValueOnce(mockYearlyActivity);

      const result = await dashboardService.getYearlyActivity();

      expect(api.get).toHaveBeenCalledWith('/users/me/activity/yearly');
      expect(result).toEqual(mockYearlyActivity);
    });

    it('should return full year data', async () => {
      // Simulating 365 days of activity
      const mockYearlyActivity = Array.from({ length: 365 }, (_, i) => ({
        date: `2025-${String(Math.floor(i / 30) + 1).padStart(2, '0')}-${String((i % 30) + 1).padStart(2, '0')}`,
        count: Math.floor(Math.random() * 10),
      }));

      vi.mocked(api.get).mockResolvedValueOnce(mockYearlyActivity);

      const result = await dashboardService.getYearlyActivity();

      expect(result).toHaveLength(365);
    });

    it('should handle empty yearly activity', async () => {
      vi.mocked(api.get).mockResolvedValueOnce([]);

      const result = await dashboardService.getYearlyActivity();

      expect(result).toHaveLength(0);
    });

    it('should throw on API error', async () => {
      vi.mocked(api.get).mockRejectedValueOnce(new Error('Service unavailable'));

      await expect(dashboardService.getYearlyActivity()).rejects.toThrow('Service unavailable');
    });
  });
});
