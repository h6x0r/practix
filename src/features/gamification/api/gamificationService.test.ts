import { describe, it, expect, beforeEach, vi } from 'vitest';
import { gamificationService } from './gamificationService';

vi.mock('@/lib/api', () => ({
  api: {
    get: vi.fn(),
  },
}));

import { api } from '@/lib/api';

describe('gamificationService', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('getMyStats', () => {
    it('should fetch user gamification stats', async () => {
      const mockStats = {
        xp: 2500,
        level: 15,
        currentStreak: 7,
        maxStreak: 30,
        xpProgress: 500,
        xpNeeded: 1000,
        progressPercent: 50,
        badges: [
          {
            id: 'badge-1',
            slug: 'first-solve',
            name: 'First Solve',
            description: 'Solve your first task',
            icon: '🎯',
            category: 'achievement',
            requirement: 1,
            xpReward: 50,
            earnedAt: '2025-01-01T00:00:00Z',
            translations: {
              ru: { name: 'Первое решение', description: 'Решите первую задачу' },
            },
          },
        ],
      };

      vi.mocked(api.get).mockResolvedValueOnce(mockStats);

      const result = await gamificationService.getMyStats();

      expect(api.get).toHaveBeenCalledWith('/gamification/me');
      expect(result).toEqual(mockStats);
      expect(result.badges).toHaveLength(1);
    });

    it('should handle new user with zero stats', async () => {
      const mockStats = {
        xp: 0,
        level: 1,
        currentStreak: 0,
        maxStreak: 0,
        xpProgress: 0,
        xpNeeded: 100,
        progressPercent: 0,
        badges: [],
      };

      vi.mocked(api.get).mockResolvedValueOnce(mockStats);

      const result = await gamificationService.getMyStats();

      expect(result.xp).toBe(0);
      expect(result.level).toBe(1);
      expect(result.badges).toHaveLength(0);
    });

    it('should throw on API error', async () => {
      vi.mocked(api.get).mockRejectedValueOnce(new Error('Unauthorized'));

      await expect(gamificationService.getMyStats()).rejects.toThrow('Unauthorized');
    });
  });

  describe('getMyRank', () => {
    it('should fetch user rank', async () => {
      vi.mocked(api.get).mockResolvedValueOnce({ rank: 42 });

      const result = await gamificationService.getMyRank();

      expect(api.get).toHaveBeenCalledWith('/gamification/me/rank');
      expect(result.rank).toBe(42);
    });

    it('should handle unranked user', async () => {
      vi.mocked(api.get).mockResolvedValueOnce({ rank: 0 });

      const result = await gamificationService.getMyRank();

      expect(result.rank).toBe(0);
    });

    it('should throw on API error', async () => {
      vi.mocked(api.get).mockRejectedValueOnce(new Error('Network error'));

      await expect(gamificationService.getMyRank()).rejects.toThrow('Network error');
    });
  });

  describe('getLeaderboard', () => {
    it('should fetch leaderboard with default limit', async () => {
      const mockLeaderboard = [
        {
          rank: 1,
          id: 'user-1',
          name: 'TopCoder',
          avatarUrl: 'https://example.com/avatar1.png',
          xp: 10000,
          level: 50,
          streak: 100,
          tasksSolved: 500,
        },
        {
          rank: 2,
          id: 'user-2',
          name: 'SecondBest',
          avatarUrl: null,
          xp: 9500,
          level: 48,
          streak: 50,
          tasksSolved: 450,
        },
      ];

      vi.mocked(api.get).mockResolvedValueOnce(mockLeaderboard);

      const result = await gamificationService.getLeaderboard();

      expect(api.get).toHaveBeenCalledWith('/gamification/leaderboard?limit=50');
      expect(result).toHaveLength(2);
      expect(result[0].rank).toBe(1);
    });

    it('should fetch leaderboard with custom limit', async () => {
      vi.mocked(api.get).mockResolvedValueOnce([]);

      await gamificationService.getLeaderboard(10);

      expect(api.get).toHaveBeenCalledWith('/gamification/leaderboard?limit=10');
    });

    it('should handle empty leaderboard', async () => {
      vi.mocked(api.get).mockResolvedValueOnce([]);

      const result = await gamificationService.getLeaderboard();

      expect(result).toHaveLength(0);
    });

    it('should throw on API error', async () => {
      vi.mocked(api.get).mockRejectedValueOnce(new Error('Service unavailable'));

      await expect(gamificationService.getLeaderboard()).rejects.toThrow('Service unavailable');
    });
  });
});
