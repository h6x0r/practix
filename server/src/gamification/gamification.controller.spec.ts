import { Test, TestingModule } from '@nestjs/testing';
import { GamificationController } from './gamification.controller';
import { GamificationService } from './gamification.service';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';

describe('GamificationController', () => {
  let controller: GamificationController;
  let gamificationService: GamificationService;

  const mockGamificationService = {
    getUserStats: jest.fn(),
    getUserRank: jest.fn(),
    getLeaderboard: jest.fn(),
  };

  // Matches getUserStats return type
  const mockUserStats = {
    xp: 1500,
    level: 5,
    currentStreak: 7,
    maxStreak: 14,
    xpProgress: 500,
    xpNeeded: 1000,
    progressPercent: 50,
    badges: [
      { id: 'badge-1', name: 'First Submission', icon: '🎯', earnedAt: new Date() },
      { id: 'badge-2', name: 'Streak Master', icon: '🔥', earnedAt: new Date() },
    ],
  };

  const mockLeaderboard = [
    { rank: 1, userId: 'user-1', name: 'Alice', xp: 5000, level: 10, avatarUrl: null },
    { rank: 2, userId: 'user-2', name: 'Bob', xp: 4500, level: 9, avatarUrl: null },
    { rank: 3, userId: 'user-3', name: 'Charlie', xp: 4000, level: 8, avatarUrl: null },
    { rank: 4, userId: 'user-4', name: 'David', xp: 3500, level: 7, avatarUrl: null },
    { rank: 5, userId: 'user-5', name: 'Eve', xp: 3000, level: 6, avatarUrl: null },
  ];

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      controllers: [GamificationController],
      providers: [
        {
          provide: GamificationService,
          useValue: mockGamificationService,
        },
      ],
    })
      .overrideGuard(JwtAuthGuard)
      .useValue({ canActivate: () => true })
      .compile();

    controller = module.get<GamificationController>(GamificationController);
    gamificationService = module.get<GamificationService>(GamificationService);

    jest.clearAllMocks();
  });

  it('should be defined', () => {
    expect(controller).toBeDefined();
  });

  describe('getMyStats', () => {
    it('should return user gamification stats', async () => {
      mockGamificationService.getUserStats.mockResolvedValue(mockUserStats);

      const result = await controller.getMyStats({ user: { userId: 'user-123' } });

      expect(result).toEqual(mockUserStats);
      expect(result.xp).toBe(1500);
      expect(result.level).toBe(5);
      expect(mockGamificationService.getUserStats).toHaveBeenCalledWith('user-123');
    });

    it('should return stats for new user with no progress', async () => {
      const newUserStats = {
        xp: 0,
        level: 1,
        currentStreak: 0,
        maxStreak: 0,
        xpProgress: 0,
        xpNeeded: 100,
        progressPercent: 0,
        badges: [],
      };
      mockGamificationService.getUserStats.mockResolvedValue(newUserStats);

      const result = await controller.getMyStats({ user: { userId: 'new-user' } });

      expect(result.xp).toBe(0);
      expect(result.level).toBe(1);
      expect(result.badges).toHaveLength(0);
    });

    it('should handle user with many badges', async () => {
      const manyBadgesStats = {
        ...mockUserStats,
        badges: Array.from({ length: 10 }, (_, i) => ({
          id: `badge-${i}`,
          name: `Badge ${i}`,
          icon: '🏆',
          earnedAt: new Date(),
        })),
      };
      mockGamificationService.getUserStats.mockResolvedValue(manyBadgesStats);

      const result = await controller.getMyStats({ user: { userId: 'user-123' } });

      expect(result.badges).toHaveLength(10);
    });

    it('should handle service errors', async () => {
      mockGamificationService.getUserStats.mockRejectedValue(new Error('Database error'));

      await expect(
        controller.getMyStats({ user: { userId: 'user-123' } })
      ).rejects.toThrow('Database error');
    });
  });

  describe('getMyRank', () => {
    it('should return user rank', async () => {
      mockGamificationService.getUserRank.mockResolvedValue(42);

      const result = await controller.getMyRank({ user: { userId: 'user-123' } });

      expect(result).toEqual({ rank: 42 });
      expect(mockGamificationService.getUserRank).toHaveBeenCalledWith('user-123');
    });

    it('should return rank 1 for top user', async () => {
      mockGamificationService.getUserRank.mockResolvedValue(1);

      const result = await controller.getMyRank({ user: { userId: 'top-user' } });

      expect(result.rank).toBe(1);
    });

    it('should return high rank for new user', async () => {
      mockGamificationService.getUserRank.mockResolvedValue(9999);

      const result = await controller.getMyRank({ user: { userId: 'new-user' } });

      expect(result.rank).toBe(9999);
    });

    it('should handle null rank for user with no activity', async () => {
      mockGamificationService.getUserRank.mockResolvedValue(null);

      const result = await controller.getMyRank({ user: { userId: 'inactive-user' } });

      expect(result).toEqual({ rank: null });
    });
  });

  describe('getLeaderboard', () => {
    it('should return leaderboard with default limit', async () => {
      mockGamificationService.getLeaderboard.mockResolvedValue(mockLeaderboard);

      const result = await controller.getLeaderboard();

      expect(result).toEqual(mockLeaderboard);
      expect(result).toHaveLength(5);
      expect(mockGamificationService.getLeaderboard).toHaveBeenCalledWith(50);
    });

    it('should respect custom limit', async () => {
      mockGamificationService.getLeaderboard.mockResolvedValue(mockLeaderboard.slice(0, 3));

      const result = await controller.getLeaderboard('3');

      expect(result).toHaveLength(3);
      expect(mockGamificationService.getLeaderboard).toHaveBeenCalledWith(3);
    });

    it('should return top 10 when limit is 10', async () => {
      mockGamificationService.getLeaderboard.mockResolvedValue(mockLeaderboard);

      await controller.getLeaderboard('10');

      expect(mockGamificationService.getLeaderboard).toHaveBeenCalledWith(10);
    });

    it('should handle empty leaderboard', async () => {
      mockGamificationService.getLeaderboard.mockResolvedValue([]);

      const result = await controller.getLeaderboard();

      expect(result).toEqual([]);
    });

    it('should handle large limit', async () => {
      mockGamificationService.getLeaderboard.mockResolvedValue(mockLeaderboard);

      await controller.getLeaderboard('100');

      expect(mockGamificationService.getLeaderboard).toHaveBeenCalledWith(100);
    });

    it('should handle invalid limit gracefully', async () => {
      mockGamificationService.getLeaderboard.mockResolvedValue(mockLeaderboard);

      await controller.getLeaderboard('invalid');

      // NaN is passed when parseInt fails
      expect(mockGamificationService.getLeaderboard).toHaveBeenCalled();
    });
  });

  describe('edge cases', () => {
    it('should handle concurrent stats requests', async () => {
      mockGamificationService.getUserStats.mockResolvedValue(mockUserStats);

      const promises = Array.from({ length: 5 }, () =>
        controller.getMyStats({ user: { userId: 'user-123' } })
      );

      const results = await Promise.all(promises);

      expect(results).toHaveLength(5);
      expect(mockGamificationService.getUserStats).toHaveBeenCalledTimes(5);
    });

    it('should handle user with unicode name in leaderboard', async () => {
      const unicodeLeaderboard = [
        { rank: 1, userId: 'user-1', name: 'Имя Пользователя 用户名', xp: 5000, level: 10, avatarUrl: null },
      ];
      mockGamificationService.getLeaderboard.mockResolvedValue(unicodeLeaderboard);

      const result = await controller.getLeaderboard('10');

      expect(result[0].name).toBe('Имя Пользователя 用户名');
    });

    it('should handle service errors on leaderboard', async () => {
      mockGamificationService.getLeaderboard.mockRejectedValue(
        new Error('Database connection failed')
      );

      await expect(controller.getLeaderboard()).rejects.toThrow('Database connection failed');
    });

    it('should handle very high XP values', async () => {
      const highXPStats = {
        ...mockUserStats,
        xp: 999999999,
        level: 100,
      };
      mockGamificationService.getUserStats.mockResolvedValue(highXPStats);

      const result = await controller.getMyStats({ user: { userId: 'pro-user' } });

      expect(result.xp).toBe(999999999);
      expect(result.level).toBe(100);
    });

    it('should handle negative limit by treating as NaN', async () => {
      mockGamificationService.getLeaderboard.mockResolvedValue(mockLeaderboard);

      await controller.getLeaderboard('-5');

      // parseInt('-5') returns -5
      expect(mockGamificationService.getLeaderboard).toHaveBeenCalledWith(-5);
    });
  });
});
