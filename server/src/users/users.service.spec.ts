import { Test, TestingModule } from '@nestjs/testing';
import { UsersService } from './users.service';
import { PrismaService } from '../prisma/prisma.service';

describe('UsersService', () => {
  let service: UsersService;
  let prisma: PrismaService;

  const mockUser = {
    id: 'user-123',
    email: 'test@example.com',
    name: 'Test User',
    password: 'hashedPassword',
    isPremium: false,
    plan: null,
    preferences: {
      editorFontSize: 14,
      editorTheme: 'vs-dark',
    },
    avatarUrl: null,
    xp: 100,
    level: 2,
    currentStreak: 5,
    maxStreak: 10,
    createdAt: new Date(),
    updatedAt: new Date(),
  };

  const mockSubmission = {
    id: 'sub-1',
    userId: 'user-123',
    taskId: 'task-1',
    status: 'passed',
    createdAt: new Date(),
  };

  const mockPrismaService = {
    user: {
      findUnique: jest.fn(),
      create: jest.fn(),
      update: jest.fn(),
    },
    submission: {
      findMany: jest.fn(),
      groupBy: jest.fn(),
    },
    subscription: {
      findFirst: jest.fn(),
    },
  };

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [
        UsersService,
        { provide: PrismaService, useValue: mockPrismaService },
      ],
    }).compile();

    service = module.get<UsersService>(UsersService);
    prisma = module.get<PrismaService>(PrismaService);

    jest.clearAllMocks();
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });

  // ============================================
  // findOne()
  // ============================================
  describe('findOne()', () => {
    it('should find user by email', async () => {
      mockPrismaService.user.findUnique.mockResolvedValue(mockUser);

      const result = await service.findOne('test@example.com');

      expect(result).toEqual(mockUser);
      expect(mockPrismaService.user.findUnique).toHaveBeenCalledWith(
        expect.objectContaining({
          where: { email: 'test@example.com' },
        })
      );
    });

    it('should return null if user not found', async () => {
      mockPrismaService.user.findUnique.mockResolvedValue(null);

      const result = await service.findOne('nonexistent@example.com');

      expect(result).toBeNull();
    });
  });

  // ============================================
  // findById()
  // ============================================
  describe('findById()', () => {
    it('should find user by id', async () => {
      mockPrismaService.user.findUnique.mockResolvedValue(mockUser);

      const result = await service.findById('user-123');

      expect(result).toEqual(mockUser);
      expect(mockPrismaService.user.findUnique).toHaveBeenCalledWith(
        expect.objectContaining({
          where: { id: 'user-123' },
        })
      );
    });

    it('should return null if user not found', async () => {
      mockPrismaService.user.findUnique.mockResolvedValue(null);

      const result = await service.findById('nonexistent');

      expect(result).toBeNull();
    });
  });

  // ============================================
  // create()
  // ============================================
  describe('create()', () => {
    it('should create new user', async () => {
      const userData = {
        email: 'new@example.com',
        name: 'New User',
        password: 'hashedPassword',
        isPremium: false,
        plan: null,
        preferences: {},
      };
      mockPrismaService.user.create.mockResolvedValue({ ...mockUser, ...userData });

      const result = await service.create(userData);

      expect(result.email).toBe('new@example.com');
      expect(mockPrismaService.user.create).toHaveBeenCalledWith({
        data: userData,
      });
    });
  });

  // ============================================
  // updatePreferences()
  // ============================================
  describe('updatePreferences()', () => {
    it('should update user preferences', async () => {
      const newPreferences = {
        editorFontSize: 16,
        editorTheme: 'light',
      };
      mockPrismaService.user.update.mockResolvedValue({
        ...mockUser,
        preferences: newPreferences,
      });

      const result = await service.updatePreferences('user-123', newPreferences);

      expect(result.preferences).toEqual(newPreferences);
      expect(mockPrismaService.user.update).toHaveBeenCalledWith({
        where: { id: 'user-123' },
        data: { preferences: newPreferences },
      });
    });
  });

  // NOTE: updatePlan() method was removed - isPremium is now computed
  // from active subscriptions via SubscriptionsService

  // ============================================
  // updateAvatar()
  // ============================================
  describe('updateAvatar()', () => {
    it('should update user avatar URL', async () => {
      mockPrismaService.user.update.mockResolvedValue({
        ...mockUser,
        avatarUrl: 'https://example.com/avatar.jpg',
      });

      const result = await service.updateAvatar('user-123', 'https://example.com/avatar.jpg');

      expect(result.avatarUrl).toBe('https://example.com/avatar.jpg');
    });

    it('should accept base64 encoded avatar', async () => {
      const base64Avatar = 'data:image/png;base64,iVBORw0KGgoAAAANS...';
      mockPrismaService.user.update.mockResolvedValue({
        ...mockUser,
        avatarUrl: base64Avatar,
      });

      const result = await service.updateAvatar('user-123', base64Avatar);

      expect(result.avatarUrl).toBe(base64Avatar);
    });
  });

  // ============================================
  // getUserStats()
  // ============================================
  describe('getUserStats()', () => {
    it('should return user statistics', async () => {
      mockPrismaService.submission.findMany.mockResolvedValue([
        { ...mockSubmission, status: 'passed', taskId: 'task-1' },
        { ...mockSubmission, status: 'passed', taskId: 'task-2' },
        { ...mockSubmission, status: 'failed', taskId: 'task-3' },
      ]);
      mockPrismaService.submission.groupBy.mockResolvedValue([
        { userId: 'user-123', _count: { taskId: 2 } },
      ]);

      const result = await service.getUserStats('user-123');

      expect(result.totalSolved).toBe(2);
      expect(result.totalSubmissions).toBe(3);
      expect(result.globalRank).toBe(1);
    });

    it('should calculate hours spent', async () => {
      // 10 submissions * 10 min = 100 min = 1h 40m
      const submissions = Array(10).fill({ ...mockSubmission, status: 'passed' });
      mockPrismaService.submission.findMany.mockResolvedValue(submissions);
      mockPrismaService.submission.groupBy.mockResolvedValue([]);

      const result = await service.getUserStats('user-123');

      expect(result.hoursSpent).toBe('1h');
    });

    it('should handle user with no submissions', async () => {
      mockPrismaService.submission.findMany.mockResolvedValue([]);
      mockPrismaService.submission.groupBy.mockResolvedValue([]);

      const result = await service.getUserStats('user-123');

      expect(result.totalSolved).toBe(0);
      expect(result.totalSubmissions).toBe(0);
      expect(result.currentStreak).toBe(0);
    });
  });

  // ============================================
  // getWeeklyActivity()
  // ============================================
  describe('getWeeklyActivity()', () => {
    it('should return weekly activity', async () => {
      mockPrismaService.submission.findMany.mockResolvedValue([mockSubmission]);

      const result = await service.getWeeklyActivity('user-123');

      expect(result).toHaveLength(7);
      expect(result[0]).toHaveProperty('name');
      expect(result[0]).toHaveProperty('date');
      expect(result[0]).toHaveProperty('solved');
      expect(result[0]).toHaveProperty('submissions');
    });

    it('should respect days parameter', async () => {
      mockPrismaService.submission.findMany.mockResolvedValue([]);

      const result = await service.getWeeklyActivity('user-123', 14);

      expect(result).toHaveLength(14);
    });

    it('should respect offset parameter', async () => {
      mockPrismaService.submission.findMany.mockResolvedValue([]);

      await service.getWeeklyActivity('user-123', 7, 7);

      // Should query for submissions from 2 weeks ago to 1 week ago
      expect(mockPrismaService.submission.findMany).toHaveBeenCalled();
    });
  });

  // ============================================
  // getYearlyActivity()
  // ============================================
  describe('getYearlyActivity()', () => {
    it('should return yearly activity for heatmap', async () => {
      mockPrismaService.submission.findMany.mockResolvedValue([
        { createdAt: new Date('2025-01-01'), taskId: 'task-1' },
        { createdAt: new Date('2025-01-01'), taskId: 'task-2' },
        { createdAt: new Date('2025-01-02'), taskId: 'task-3' },
      ]);

      const result = await service.getYearlyActivity('user-123');

      expect(result).toHaveLength(2); // 2 unique dates
      expect(result[0]).toHaveProperty('date');
      expect(result[0]).toHaveProperty('count');
    });

    it('should count unique tasks per day', async () => {
      mockPrismaService.submission.findMany.mockResolvedValue([
        { createdAt: new Date('2025-01-01'), taskId: 'task-1' },
        { createdAt: new Date('2025-01-01'), taskId: 'task-1' }, // Duplicate
        { createdAt: new Date('2025-01-01'), taskId: 'task-2' },
      ]);

      const result = await service.getYearlyActivity('user-123');

      expect(result[0].count).toBe(2); // 2 unique tasks
    });
  });

  // ============================================
  // isPremiumUser()
  // ============================================
  describe('isPremiumUser()', () => {
    it('should return true for user with active subscription', async () => {
      mockPrismaService.subscription.findFirst.mockResolvedValue({
        id: 'sub-1',
        status: 'active',
        endDate: new Date('2025-12-31'),
      });

      const result = await service.isPremiumUser('user-123');

      expect(result).toBe(true);
    });

    it('should return false for user without subscription', async () => {
      mockPrismaService.subscription.findFirst.mockResolvedValue(null);

      const result = await service.isPremiumUser('user-123');

      expect(result).toBe(false);
    });

    it('should return false for expired subscription', async () => {
      mockPrismaService.subscription.findFirst.mockResolvedValue(null);

      const result = await service.isPremiumUser('user-123');

      expect(result).toBe(false);
    });
  });

  // ============================================
  // getActivePlan()
  // ============================================
  describe('getActivePlan()', () => {
    it('should return active plan details', async () => {
      mockPrismaService.subscription.findFirst.mockResolvedValue({
        id: 'sub-1',
        status: 'active',
        endDate: new Date('2025-12-31'),
        plan: { name: 'Premium Global' },
      });

      const result = await service.getActivePlan('user-123');

      expect(result).not.toBeNull();
      expect(result?.name).toBe('Premium Global');
      expect(result?.expiresAt).toBeDefined();
    });

    it('should return null if no active plan', async () => {
      mockPrismaService.subscription.findFirst.mockResolvedValue(null);

      const result = await service.getActivePlan('user-123');

      expect(result).toBeNull();
    });
  });
});
