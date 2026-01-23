import { describe, it, expect, beforeEach, vi } from 'vitest';
import { analyticsService } from './analyticsService';

vi.mock('@/lib/api', () => ({
  api: {
    get: vi.fn(),
  },
}));

import { api } from '@/lib/api';

describe('analyticsService', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('getWeeklyStats', () => {
    it('should fetch current week stats', async () => {
      const mockData = [
        { name: 'Mon', date: '2025-01-13', solved: 5, submissions: 10 },
        { name: 'Tue', date: '2025-01-14', solved: 3, submissions: 8 },
        { name: 'Wed', date: '2025-01-15', solved: 7, submissions: 15 },
      ];

      vi.mocked(api.get).mockResolvedValueOnce(mockData);

      const result = await analyticsService.getWeeklyStats(0);

      expect(api.get).toHaveBeenCalledWith('/users/me/activity?days=7&offset=0');
      expect(result).toHaveLength(3);
      expect(result[0].tasks).toBe(5); // transformed from 'solved'
      expect(result[0].submissions).toBe(10);
    });

    it('should fetch previous week stats with offset', async () => {
      vi.mocked(api.get).mockResolvedValueOnce([]);

      await analyticsService.getWeeklyStats(1);

      expect(api.get).toHaveBeenCalledWith('/users/me/activity?days=7&offset=7');
    });

    it('should fetch two weeks ago stats', async () => {
      vi.mocked(api.get).mockResolvedValueOnce([]);

      await analyticsService.getWeeklyStats(2);

      expect(api.get).toHaveBeenCalledWith('/users/me/activity?days=7&offset=14');
    });

    it('should transform data format correctly', async () => {
      const mockData = [
        { name: 'Thu', date: '2025-01-16', solved: 10, submissions: 25 },
      ];

      vi.mocked(api.get).mockResolvedValueOnce(mockData);

      const result = await analyticsService.getWeeklyStats(0);

      expect(result[0]).toEqual({
        name: 'Thu',
        date: '2025-01-16',
        tasks: 10,
        submissions: 25,
      });
    });

    it('should handle empty week', async () => {
      vi.mocked(api.get).mockResolvedValueOnce([]);

      const result = await analyticsService.getWeeklyStats(0);

      expect(result).toHaveLength(0);
    });

    it('should throw on API error', async () => {
      vi.mocked(api.get).mockRejectedValueOnce(new Error('Unauthorized'));

      await expect(analyticsService.getWeeklyStats(0)).rejects.toThrow('Unauthorized');
    });
  });

  describe('getYearlyContributions', () => {
    it('should fetch and transform yearly data', async () => {
      // Mock sparse data from API
      const mockData = [
        { date: new Date().toISOString().split('T')[0], count: 5 },
      ];

      vi.mocked(api.get).mockResolvedValueOnce(mockData);

      const result = await analyticsService.getYearlyContributions();

      expect(api.get).toHaveBeenCalledWith('/users/me/activity/yearly');
      expect(result).toHaveLength(365);
    });

    it('should calculate intensity levels correctly', async () => {
      const today = new Date().toISOString().split('T')[0];
      const mockData = [
        { date: today, count: 10 }, // Should have intensity 4
      ];

      vi.mocked(api.get).mockResolvedValueOnce(mockData);

      const result = await analyticsService.getYearlyContributions();

      const todayEntry = result.find(r => r.date === today);
      expect(todayEntry?.intensity).toBe(4); // count >= 8
    });

    it('should set intensity 0 for zero count', async () => {
      vi.mocked(api.get).mockResolvedValueOnce([]);

      const result = await analyticsService.getYearlyContributions();

      // All entries should have intensity 0
      expect(result.every(r => r.intensity === 0)).toBe(true);
    });

    it('should set intensity 1 for count 1-2', async () => {
      const today = new Date().toISOString().split('T')[0];
      vi.mocked(api.get).mockResolvedValueOnce([{ date: today, count: 2 }]);

      const result = await analyticsService.getYearlyContributions();

      const todayEntry = result.find(r => r.date === today);
      expect(todayEntry?.intensity).toBe(1);
    });

    it('should set intensity 2 for count 3-4', async () => {
      const today = new Date().toISOString().split('T')[0];
      vi.mocked(api.get).mockResolvedValueOnce([{ date: today, count: 4 }]);

      const result = await analyticsService.getYearlyContributions();

      const todayEntry = result.find(r => r.date === today);
      expect(todayEntry?.intensity).toBe(2);
    });

    it('should set intensity 3 for count 5-7', async () => {
      const today = new Date().toISOString().split('T')[0];
      vi.mocked(api.get).mockResolvedValueOnce([{ date: today, count: 6 }]);

      const result = await analyticsService.getYearlyContributions();

      const todayEntry = result.find(r => r.date === today);
      expect(todayEntry?.intensity).toBe(3);
    });

    it('should throw on API error', async () => {
      vi.mocked(api.get).mockRejectedValueOnce(new Error('Network error'));

      await expect(analyticsService.getYearlyContributions()).rejects.toThrow('Network error');
    });
  });

  describe('getSummary', () => {
    it('should fetch and transform summary stats', async () => {
      const mockStats = {
        totalSolved: 150,
        totalSubmissions: 500,
        hoursSpent: '60h',
        globalRank: 42,
        skillPoints: 2500,
        currentStreak: 7,
        maxStreak: 30,
        weekThisWeek: 15,
      };

      vi.mocked(api.get).mockResolvedValueOnce(mockStats);

      const result = await analyticsService.getSummary();

      expect(api.get).toHaveBeenCalledWith('/users/me/stats');
      expect(result.totalSolved).toBe(150);
      expect(result.totalSubmissions).toBe(500);
      expect(result.currentStreak).toBe(7);
      expect(result.maxStreak).toBe(30);
      expect(result.totalXP).toBe(2500);
      expect(result.weekSolvedCount).toBe(15);
    });

    it('should calculate completion rate', async () => {
      const mockStats = {
        totalSolved: 100,
        totalSubmissions: 200,
        hoursSpent: '30h',
        globalRank: 100,
        skillPoints: 1000,
        currentStreak: 5,
        maxStreak: 10,
        weekThisWeek: 5,
      };

      vi.mocked(api.get).mockResolvedValueOnce(mockStats);

      const result = await analyticsService.getSummary();

      expect(result.completionRate).toBe(50); // 100/200 = 50%
    });

    it('should handle zero submissions', async () => {
      const mockStats = {
        totalSolved: 0,
        totalSubmissions: 0,
        hoursSpent: '0h',
        globalRank: 0,
        skillPoints: 0,
        currentStreak: 0,
        maxStreak: 0,
        weekThisWeek: 0,
      };

      vi.mocked(api.get).mockResolvedValueOnce(mockStats);

      const result = await analyticsService.getSummary();

      expect(result.completionRate).toBe(0);
    });

    it('should throw on API error', async () => {
      vi.mocked(api.get).mockRejectedValueOnce(new Error('Unauthorized'));

      await expect(analyticsService.getSummary()).rejects.toThrow('Unauthorized');
    });
  });
});
