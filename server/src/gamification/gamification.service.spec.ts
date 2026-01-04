import { Test, TestingModule } from '@nestjs/testing';
import { GamificationService } from './gamification.service';
import { PrismaService } from '../prisma/prisma.service';
import { CacheService } from '../cache/cache.service';

describe('GamificationService', () => {
  let service: GamificationService;
  let prisma: PrismaService;
  let cache: CacheService;

  const mockUser = {
    id: 'user-123',
    xp: 0,
    level: 1,
    currentStreak: 0,
    maxStreak: 0,
    lastActivityAt: null,
  };

  const mockBadge = {
    id: 'badge-first-task',
    slug: 'first-task',
    name: 'First Task',
    description: 'Complete your first task',
    icon: '🎯',
    category: 'milestone',
    requirement: 1,
    xpReward: 10,
  };

  const mockStreakBadge = {
    id: 'badge-streak-7',
    slug: 'week-warrior',
    name: 'Week Warrior',
    description: '7 day streak',
    icon: '🔥',
    category: 'streak',
    requirement: 7,
    xpReward: 50,
  };

  const mockLevelBadge = {
    id: 'badge-level-5',
    slug: 'rising-star',
    name: 'Rising Star',
    description: 'Reach level 5',
    icon: '⭐',
    category: 'level',
    requirement: 5,
    xpReward: 100,
  };

  const mockPrismaService = {
    user: {
      findUnique: jest.fn(),
      findMany: jest.fn(),
      update: jest.fn(),
      count: jest.fn(),
    },
    badge: {
      findMany: jest.fn(),
    },
    userBadge: {
      findMany: jest.fn(),
      findUnique: jest.fn(),
      create: jest.fn(),
    },
    submission: {
      count: jest.fn(),
    },
    $transaction: jest.fn(),
  };

  const mockCacheService = {
    get: jest.fn(),
    set: jest.fn(),
    delete: jest.fn(),
    getOrSet: jest.fn(),
  };

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [
        GamificationService,
        { provide: PrismaService, useValue: mockPrismaService },
        { provide: CacheService, useValue: mockCacheService },
      ],
    }).compile();

    service = module.get<GamificationService>(GamificationService);
    prisma = module.get<PrismaService>(PrismaService);
    cache = module.get<CacheService>(CacheService);

    jest.clearAllMocks();
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });

  // ============================================
  // calculateLevel()
  // ============================================
  describe('calculateLevel()', () => {
    it('should return level 1 for 0 XP', () => {
      expect(service.calculateLevel(0)).toBe(1);
    });

    it('should return level 1 for 99 XP', () => {
      expect(service.calculateLevel(99)).toBe(1);
    });

    it('should return level 2 for 100 XP', () => {
      expect(service.calculateLevel(100)).toBe(2);
    });

    it('should return level 3 for 250 XP', () => {
      expect(service.calculateLevel(250)).toBe(3);
    });

    it('should return level 5 for 1000 XP', () => {
      expect(service.calculateLevel(1000)).toBe(5);
    });

    it('should return level 10 for 7500 XP', () => {
      expect(service.calculateLevel(7500)).toBe(10);
    });

    it('should return level 20 for 60000 XP', () => {
      expect(service.calculateLevel(60000)).toBe(20);
    });

    it('should handle XP at and beyond max threshold', () => {
      // The for loop in calculateLevel iterates from the highest threshold down
      // Any XP >= 60000 returns level 20 (since 60000 is LEVEL_THRESHOLDS[19] and i+1=20)
      // Lines 59-60 are for negative XP (unreachable in practice)
      expect(service.calculateLevel(60000)).toBe(20); // Exactly at threshold
      expect(service.calculateLevel(69999)).toBe(20); // Still level 20
      expect(service.calculateLevel(70000)).toBe(20); // Still level 20
      expect(service.calculateLevel(100000)).toBe(20); // Still level 20 (max defined level)
    });

    it('should handle edge case of very low XP', () => {
      // XP 0 should always be level 1 (LEVEL_THRESHOLDS[0] = 0)
      expect(service.calculateLevel(0)).toBe(1);
      expect(service.calculateLevel(1)).toBe(1);
    });
  });

  // ============================================
  // getXpForNextLevel()
  // ============================================
  describe('getXpForNextLevel()', () => {
    it('should return 100 for level 1', () => {
      expect(service.getXpForNextLevel(1)).toBe(100);
    });

    it('should return 250 for level 2', () => {
      expect(service.getXpForNextLevel(2)).toBe(250);
    });

    it('should return 500 for level 3', () => {
      expect(service.getXpForNextLevel(3)).toBe(500);
    });

    it('should handle levels beyond defined thresholds', () => {
      // Level 20 = 60000 XP, each level after is +10000
      expect(service.getXpForNextLevel(20)).toBe(70000);
      expect(service.getXpForNextLevel(21)).toBe(80000);
    });
  });

  // ============================================
  // awardTaskXp()
  // ============================================
  describe('awardTaskXp()', () => {
    const setupTransactionMock = (userState = mockUser) => {
      mockPrismaService.$transaction.mockImplementation(async (fn) => {
        const tx = {
          user: {
            findUnique: jest.fn().mockResolvedValue(userState),
            update: jest.fn().mockResolvedValue({ xp: (userState.xp || 0) + 10 }),
          },
        };
        return fn(tx);
      });
    };

    beforeEach(() => {
      setupTransactionMock();
      mockPrismaService.badge.findMany.mockResolvedValue([]);
      mockPrismaService.userBadge.findMany.mockResolvedValue([]);
      mockPrismaService.submission.count.mockResolvedValue(1);
    });

    it('should award 10 XP for easy difficulty', async () => {
      const result = await service.awardTaskXp('user-123', 'easy');

      expect(result.xpEarned).toBe(10);
    });

    it('should award 25 XP for medium difficulty', async () => {
      const result = await service.awardTaskXp('user-123', 'medium');

      expect(result.xpEarned).toBe(25);
    });

    it('should award 50 XP for hard difficulty', async () => {
      const result = await service.awardTaskXp('user-123', 'hard');

      expect(result.xpEarned).toBe(50);
    });

    it('should award 100 XP for expert difficulty', async () => {
      const result = await service.awardTaskXp('user-123', 'expert');

      expect(result.xpEarned).toBe(100);
    });

    it('should default to easy XP for unknown difficulty', async () => {
      const result = await service.awardTaskXp('user-123', 'unknown');

      expect(result.xpEarned).toBe(10);
    });

    it('should update user total XP', async () => {
      let txUpdateMock: jest.Mock;
      mockPrismaService.$transaction.mockImplementation(async (fn) => {
        txUpdateMock = jest.fn().mockResolvedValue({ xp: 10 });
        const tx = {
          user: {
            findUnique: jest.fn().mockResolvedValue(mockUser),
            update: txUpdateMock,
          },
        };
        return fn(tx);
      });

      await service.awardTaskXp('user-123', 'easy');

      expect(txUpdateMock).toHaveBeenCalledWith(
        expect.objectContaining({
          where: { id: 'user-123' },
          data: expect.objectContaining({
            xp: { increment: 10 },
          }),
        })
      );
    });

    it('should calculate new level correctly', async () => {
      setupTransactionMock({ ...mockUser, xp: 95 });

      const result = await service.awardTaskXp('user-123', 'easy');

      expect(result.level).toBe(2); // 95 + 10 = 105 XP = Level 2
    });

    it('should detect level up', async () => {
      setupTransactionMock({ ...mockUser, xp: 95 });

      const result = await service.awardTaskXp('user-123', 'easy');

      expect(result.leveledUp).toBe(true);
    });

    it('should not indicate level up if level unchanged', async () => {
      setupTransactionMock({ ...mockUser, xp: 50 });

      const result = await service.awardTaskXp('user-123', 'easy');

      expect(result.leveledUp).toBe(false);
    });

    it('should update streak on consecutive days', async () => {
      const yesterday = new Date();
      yesterday.setDate(yesterday.getDate() - 1);

      let txUpdateMock: jest.Mock;
      mockPrismaService.$transaction.mockImplementation(async (fn) => {
        txUpdateMock = jest.fn().mockResolvedValue({ xp: 10 });
        const tx = {
          user: {
            findUnique: jest.fn().mockResolvedValue({
              ...mockUser,
              currentStreak: 5,
              lastActivityAt: yesterday,
            }),
            update: txUpdateMock,
          },
        };
        return fn(tx);
      });

      await service.awardTaskXp('user-123', 'easy');

      expect(txUpdateMock).toHaveBeenCalledWith(
        expect.objectContaining({
          data: expect.objectContaining({
            currentStreak: 6,
          }),
        })
      );
    });

    it('should reset streak after gap', async () => {
      const twoDaysAgo = new Date();
      twoDaysAgo.setDate(twoDaysAgo.getDate() - 2);

      let txUpdateMock: jest.Mock;
      mockPrismaService.$transaction.mockImplementation(async (fn) => {
        txUpdateMock = jest.fn().mockResolvedValue({ xp: 10 });
        const tx = {
          user: {
            findUnique: jest.fn().mockResolvedValue({
              ...mockUser,
              currentStreak: 10,
              lastActivityAt: twoDaysAgo,
            }),
            update: txUpdateMock,
          },
        };
        return fn(tx);
      });

      await service.awardTaskXp('user-123', 'easy');

      expect(txUpdateMock).toHaveBeenCalledWith(
        expect.objectContaining({
          data: expect.objectContaining({
            currentStreak: 1,
          }),
        })
      );
    });

    it('should not change streak on same day', async () => {
      const today = new Date();

      let txUpdateMock: jest.Mock;
      mockPrismaService.$transaction.mockImplementation(async (fn) => {
        txUpdateMock = jest.fn().mockResolvedValue({ xp: 10 });
        const tx = {
          user: {
            findUnique: jest.fn().mockResolvedValue({
              ...mockUser,
              currentStreak: 5,
              lastActivityAt: today,
            }),
            update: txUpdateMock,
          },
        };
        return fn(tx);
      });

      await service.awardTaskXp('user-123', 'easy');

      expect(txUpdateMock).toHaveBeenCalledWith(
        expect.objectContaining({
          data: expect.objectContaining({
            currentStreak: 5,
          }),
        })
      );
    });

    it('should start streak at 1 for new user', async () => {
      let txUpdateMock: jest.Mock;
      mockPrismaService.$transaction.mockImplementation(async (fn) => {
        txUpdateMock = jest.fn().mockResolvedValue({ xp: 10 });
        const tx = {
          user: {
            findUnique: jest.fn().mockResolvedValue({
              ...mockUser,
              lastActivityAt: null,
            }),
            update: txUpdateMock,
          },
        };
        return fn(tx);
      });

      await service.awardTaskXp('user-123', 'easy');

      expect(txUpdateMock).toHaveBeenCalledWith(
        expect.objectContaining({
          data: expect.objectContaining({
            currentStreak: 1,
          }),
        })
      );
    });

    it('should update maxStreak when current exceeds max', async () => {
      const yesterday = new Date();
      yesterday.setDate(yesterday.getDate() - 1);

      let txUpdateMock: jest.Mock;
      mockPrismaService.$transaction.mockImplementation(async (fn) => {
        txUpdateMock = jest.fn().mockResolvedValue({ xp: 10 });
        const tx = {
          user: {
            findUnique: jest.fn().mockResolvedValue({
              ...mockUser,
              currentStreak: 10,
              maxStreak: 10,
              lastActivityAt: yesterday,
            }),
            update: txUpdateMock,
          },
        };
        return fn(tx);
      });

      await service.awardTaskXp('user-123', 'easy');

      expect(txUpdateMock).toHaveBeenCalledWith(
        expect.objectContaining({
          data: expect.objectContaining({
            maxStreak: 11,
          }),
        })
      );
    });
  });

  // ============================================
  // checkAndAwardBadges() (via awardTaskXp)
  // ============================================
  describe('badge awarding', () => {
    const setupBadgeTransactionMock = (userState = mockUser) => {
      let callCount = 0;
      mockPrismaService.$transaction.mockImplementation(async (fn) => {
        callCount++;
        // First transaction is for awardTaskXp (user update)
        if (callCount === 1) {
          const tx = {
            user: {
              findUnique: jest.fn().mockResolvedValue(userState),
              update: jest.fn().mockResolvedValue({ xp: (userState.xp || 0) + 10 }),
            },
          };
          return fn(tx);
        }
        // Subsequent transactions are for badge awarding
        const tx = {
          userBadge: {
            findUnique: jest.fn().mockResolvedValue(null),
            create: jest.fn().mockResolvedValue({}),
          },
          user: {
            update: jest.fn().mockResolvedValue({}),
          },
        };
        return fn(tx);
      });
    };

    beforeEach(() => {
      mockPrismaService.userBadge.findMany.mockResolvedValue([]);
      mockPrismaService.submission.count.mockResolvedValue(1);
    });

    it('should award milestone badges', async () => {
      setupBadgeTransactionMock();
      mockPrismaService.badge.findMany.mockResolvedValue([mockBadge]);

      const result = await service.awardTaskXp('user-123', 'easy');

      expect(result.newBadges).toHaveLength(1);
      expect(result.newBadges[0].slug).toBe('first-task');
    });

    it('should award streak badges', async () => {
      const yesterday = new Date();
      yesterday.setDate(yesterday.getDate() - 1);

      setupBadgeTransactionMock({
        ...mockUser,
        currentStreak: 6,
        lastActivityAt: yesterday,
      });
      mockPrismaService.badge.findMany.mockResolvedValue([mockStreakBadge]);

      const result = await service.awardTaskXp('user-123', 'easy');

      expect(result.newBadges).toHaveLength(1);
      expect(result.newBadges[0].slug).toBe('week-warrior');
    });

    it('should award level badges', async () => {
      setupBadgeTransactionMock({
        ...mockUser,
        xp: 990, // Will become 1000 = level 5
      });
      mockPrismaService.badge.findMany.mockResolvedValue([mockLevelBadge]);

      const result = await service.awardTaskXp('user-123', 'easy');

      expect(result.newBadges).toHaveLength(1);
    });

    it('should award XP badges (lines 246-247)', async () => {
      const xpBadge = {
        id: 'badge-xp-1000',
        slug: 'xp-champion',
        name: 'XP Champion',
        description: 'Reach 1000 XP',
        icon: '💰',
        category: 'xp',
        requirement: 1000,
        xpReward: 25,
      };

      setupBadgeTransactionMock({
        ...mockUser,
        xp: 990, // Will become 1000 after awarding 10 XP
      });
      mockPrismaService.badge.findMany.mockResolvedValue([xpBadge]);

      const result = await service.awardTaskXp('user-123', 'easy');

      expect(result.newBadges).toHaveLength(1);
      expect(result.newBadges[0].slug).toBe('xp-champion');
    });

    it('should not duplicate badges', async () => {
      setupBadgeTransactionMock();
      mockPrismaService.badge.findMany.mockResolvedValue([mockBadge]);
      mockPrismaService.userBadge.findMany.mockResolvedValue([
        { badgeId: mockBadge.id },
      ]);

      const result = await service.awardTaskXp('user-123', 'easy');

      expect(result.newBadges).toHaveLength(0);
    });

    it('should check for existing badge in transaction (line 261)', async () => {
      // Simulate: badge check inside transaction (findUnique returns existing badge)
      // Note: Due to current code structure, even if badge exists, it completes without error
      // and the badge gets added to newBadges. The transaction just exits early.
      let callCount = 0;
      let badgeCreateCalled = false;
      mockPrismaService.$transaction.mockImplementation(async (fn) => {
        callCount++;
        if (callCount === 1) {
          // First transaction: awardTaskXp
          const tx = {
            user: {
              findUnique: jest.fn().mockResolvedValue(mockUser),
              update: jest.fn().mockResolvedValue({ xp: 10 }),
            },
          };
          return fn(tx);
        }
        // Badge transaction: findUnique returns existing badge
        const tx = {
          userBadge: {
            findUnique: jest.fn().mockResolvedValue({ id: 'existing-badge' }),
            create: jest.fn().mockImplementation(() => {
              badgeCreateCalled = true;
              return Promise.resolve({});
            }),
          },
          user: {
            update: jest.fn(),
          },
        };
        return fn(tx);
      });
      mockPrismaService.badge.findMany.mockResolvedValue([mockBadge]);

      await service.awardTaskXp('user-123', 'easy');

      // Key assertion: create should NOT be called when badge already exists
      expect(badgeCreateCalled).toBe(false);
    });

    it('should handle Prisma P2002 race condition error (line 283)', async () => {
      // Import Prisma to create proper error
      const { Prisma } = require('@prisma/client');

      let callCount = 0;
      mockPrismaService.$transaction.mockImplementation(async (fn) => {
        callCount++;
        if (callCount === 1) {
          const tx = {
            user: {
              findUnique: jest.fn().mockResolvedValue(mockUser),
              update: jest.fn().mockResolvedValue({ xp: 10 }),
            },
          };
          return fn(tx);
        }
        // Badge transaction throws P2002 unique constraint error
        throw new Prisma.PrismaClientKnownRequestError('Unique constraint violation', {
          code: 'P2002',
          clientVersion: '5.0.0',
        });
      });
      mockPrismaService.badge.findMany.mockResolvedValue([mockBadge]);

      const result = await service.awardTaskXp('user-123', 'easy');

      // Should handle gracefully - no badges awarded due to race condition
      expect(result.newBadges).toHaveLength(0);
    });

    it('should handle race conditions gracefully', async () => {
      // First transaction succeeds, badge transaction fails
      let callCount = 0;
      mockPrismaService.$transaction.mockImplementation(async (fn) => {
        callCount++;
        if (callCount === 1) {
          const tx = {
            user: {
              findUnique: jest.fn().mockResolvedValue(mockUser),
              update: jest.fn().mockResolvedValue({ xp: 10 }),
            },
          };
          return fn(tx);
        }
        throw new Error('Duplicate key');
      });
      mockPrismaService.badge.findMany.mockResolvedValue([mockBadge]);

      const result = await service.awardTaskXp('user-123', 'easy');

      expect(result.newBadges).toHaveLength(0);
    });
  });

  // ============================================
  // getUserStats()
  // ============================================
  describe('getUserStats()', () => {
    it('should return complete user stats', async () => {
      mockPrismaService.user.findUnique.mockResolvedValue({
        xp: 150,
        level: 2,
        currentStreak: 5,
        maxStreak: 10,
        badges: [{ badge: mockBadge, earnedAt: new Date() }],
      });

      const result = await service.getUserStats('user-123');

      expect(result).toBeDefined();
      expect(result?.xp).toBe(150);
      expect(result?.level).toBe(2);
      expect(result?.currentStreak).toBe(5);
      expect(result?.maxStreak).toBe(10);
    });

    it('should calculate progress to next level', async () => {
      mockPrismaService.user.findUnique.mockResolvedValue({
        xp: 150,
        level: 2,
        currentStreak: 0,
        maxStreak: 0,
        badges: [],
      });

      const result = await service.getUserStats('user-123');

      // Level 2 starts at 100 XP, level 3 at 250 XP
      expect(result?.xpProgress).toBe(50); // 150 - 100
      expect(result?.xpNeeded).toBe(150); // 250 - 100
      expect(result?.progressPercent).toBe(33); // ~33%
    });

    it('should include badges', async () => {
      mockPrismaService.user.findUnique.mockResolvedValue({
        xp: 100,
        level: 2,
        currentStreak: 0,
        maxStreak: 0,
        badges: [
          { badge: mockBadge, earnedAt: new Date() },
          { badge: mockStreakBadge, earnedAt: new Date() },
        ],
      });

      const result = await service.getUserStats('user-123');

      expect(result?.badges).toHaveLength(2);
    });

    it('should return null if user not found', async () => {
      mockPrismaService.user.findUnique.mockResolvedValue(null);

      const result = await service.getUserStats('nonexistent');

      expect(result).toBeNull();
    });
  });

  // ============================================
  // getLeaderboard()
  // ============================================
  describe('getLeaderboard()', () => {
    it('should return top users by XP', async () => {
      mockPrismaService.user.findMany.mockResolvedValue([
        { id: 'user-1', name: 'Top User', xp: 1000, level: 5, currentStreak: 10, _count: { submissions: 50 } },
        { id: 'user-2', name: 'Second', xp: 500, level: 4, currentStreak: 5, _count: { submissions: 25 } },
      ]);

      const result = await service.getLeaderboard();

      expect(result).toHaveLength(2);
      expect(result[0].rank).toBe(1);
      expect(result[0].xp).toBe(1000);
    });

    it('should include rank', async () => {
      mockPrismaService.user.findMany.mockResolvedValue([
        { id: 'user-1', name: 'First', xp: 1000, level: 5, currentStreak: 10, _count: { submissions: 50 } },
        { id: 'user-2', name: 'Second', xp: 500, level: 4, currentStreak: 5, _count: { submissions: 25 } },
        { id: 'user-3', name: 'Third', xp: 250, level: 3, currentStreak: 3, _count: { submissions: 10 } },
      ]);

      const result = await service.getLeaderboard();

      expect(result[0].rank).toBe(1);
      expect(result[1].rank).toBe(2);
      expect(result[2].rank).toBe(3);
    });

    it('should respect limit parameter', async () => {
      mockPrismaService.user.findMany.mockResolvedValue([]);

      await service.getLeaderboard(10);

      expect(mockPrismaService.user.findMany).toHaveBeenCalledWith(
        expect.objectContaining({
          take: 10,
        })
      );
    });

    it('should use default limit of 50', async () => {
      mockPrismaService.user.findMany.mockResolvedValue([]);

      await service.getLeaderboard();

      expect(mockPrismaService.user.findMany).toHaveBeenCalledWith(
        expect.objectContaining({
          take: 50,
        })
      );
    });
  });

  // ============================================
  // getUserRank() with Caching (uses getOrSet for stampede protection)
  // ============================================
  describe('getUserRank()', () => {
    it('should return user rank via getOrSet', async () => {
      // getOrSet calls the factory function, which queries the database
      mockCacheService.getOrSet.mockImplementation(async (key, ttl, factory) => {
        return factory();
      });
      mockPrismaService.user.findUnique.mockResolvedValue({ xp: 500 });
      mockPrismaService.user.count.mockResolvedValue(10);

      const result = await service.getUserRank('user-123');

      expect(result).toBe(11); // 10 users with higher XP + 1
      expect(mockCacheService.getOrSet).toHaveBeenCalledWith(
        'rank:user-123',
        300,
        expect.any(Function)
      );
    });

    it('should return rank 1 for top user', async () => {
      mockCacheService.getOrSet.mockImplementation(async (key, ttl, factory) => {
        return factory();
      });
      mockPrismaService.user.findUnique.mockResolvedValue({ xp: 10000 });
      mockPrismaService.user.count.mockResolvedValue(0);

      const result = await service.getUserRank('user-123');

      expect(result).toBe(1);
    });

    it('should return 0 for non-existent user', async () => {
      mockCacheService.getOrSet.mockImplementation(async (key, ttl, factory) => {
        return factory();
      });
      mockPrismaService.user.findUnique.mockResolvedValue(null);

      const result = await service.getUserRank('nonexistent');

      expect(result).toBe(0);
    });

    it('should return cached rank if available (via getOrSet)', async () => {
      // getOrSet returns cached value without calling factory
      mockCacheService.getOrSet.mockResolvedValue(5);

      const result = await service.getUserRank('user-123');

      expect(result).toBe(5);
      // Should not call database since getOrSet returns cached value
      expect(mockPrismaService.user.findUnique).not.toHaveBeenCalled();
      expect(mockPrismaService.user.count).not.toHaveBeenCalled();
    });

    it('should use getOrSet for cache stampede protection', async () => {
      mockCacheService.getOrSet.mockImplementation(async (key, ttl, factory) => {
        return factory();
      });
      mockPrismaService.user.findUnique.mockResolvedValue({ xp: 500 });
      mockPrismaService.user.count.mockResolvedValue(10);

      await service.getUserRank('user-123');

      // Verify getOrSet is used with correct key and TTL
      expect(mockCacheService.getOrSet).toHaveBeenCalledWith(
        'rank:user-123',
        300, // 5 minute TTL
        expect.any(Function)
      );
    });

    it('should handle null return from getOrSet (fallback to 0)', async () => {
      mockCacheService.getOrSet.mockResolvedValue(null);

      const result = await service.getUserRank('user-123');

      expect(result).toBe(0);
    });
  });

  // ============================================
  // invalidateRankCache()
  // ============================================
  describe('invalidateRankCache()', () => {
    it('should delete rank cache for user', async () => {
      await service.invalidateRankCache('user-123');

      expect(mockCacheService.delete).toHaveBeenCalledWith('rank:user-123');
    });
  });

  // ============================================
  // Rank Cache Invalidation on XP Change
  // ============================================
  describe('Rank cache invalidation on XP change', () => {
    beforeEach(() => {
      mockPrismaService.$transaction.mockImplementation(async (fn) => {
        return fn({
          user: {
            findUnique: jest.fn().mockResolvedValue(mockUser),
            update: jest.fn().mockResolvedValue({ xp: 10 }),
          },
        });
      });
      mockPrismaService.badge.findMany.mockResolvedValue([]);
      mockPrismaService.userBadge.findMany.mockResolvedValue([]);
      mockPrismaService.submission.count.mockResolvedValue(1);
    });

    it('should invalidate rank cache after awarding XP', async () => {
      await service.awardTaskXp('user-123', 'easy');

      expect(mockCacheService.delete).toHaveBeenCalledWith('rank:user-123');
    });
  });
});
