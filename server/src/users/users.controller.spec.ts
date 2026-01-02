import { Test, TestingModule } from '@nestjs/testing';
import { UsersController } from './users.controller';
import { UsersService } from './users.service';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';

describe('UsersController', () => {
  let controller: UsersController;
  let usersService: UsersService;

  const mockUsersService = {
    findById: jest.fn(),
    isPremiumUser: jest.fn(),
    getActivePlan: jest.fn(),
    getUserStats: jest.fn(),
    getWeeklyActivity: jest.fn(),
    getYearlyActivity: jest.fn(),
    updatePreferences: jest.fn(),
    updateAvatar: jest.fn(),
    updatePlan: jest.fn(),
  };

  const mockUser = {
    id: 'user-123',
    email: 'test@example.com',
    name: 'Test User',
    password: 'hashedpassword',
    avatarUrl: null,
    isPremium: false,
    plan: null,
    streak: 5,
    skillPoints: 100,
    rank: 50,
    preferences: { theme: 'dark', language: 'en' },
    createdAt: new Date(),
    updatedAt: new Date(),
  };

  // Matches UserStats interface
  const mockStats = {
    totalSolved: 25,
    totalSubmissions: 40,
    hoursSpent: '15h 30m',
    globalRank: 150,
    skillPoints: 450,
    currentStreak: 7,
    maxStreak: 14,
    weekThisWeek: 5,
  };

  const mockWeeklyActivity = [
    { date: '2024-01-01', count: 5, minutes: 30 },
    { date: '2024-01-02', count: 3, minutes: 20 },
    { date: '2024-01-03', count: 0, minutes: 0 },
    { date: '2024-01-04', count: 7, minutes: 45 },
    { date: '2024-01-05', count: 2, minutes: 15 },
    { date: '2024-01-06', count: 4, minutes: 25 },
    { date: '2024-01-07', count: 6, minutes: 35 },
  ];

  const mockYearlyActivity = Array.from({ length: 365 }, (_, i) => ({
    date: new Date(2024, 0, i + 1).toISOString().split('T')[0],
    count: Math.floor(Math.random() * 10),
  }));

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      controllers: [UsersController],
      providers: [
        {
          provide: UsersService,
          useValue: mockUsersService,
        },
      ],
    })
      .overrideGuard(JwtAuthGuard)
      .useValue({ canActivate: () => true })
      .compile();

    controller = module.get<UsersController>(UsersController);
    usersService = module.get<UsersService>(UsersService);

    jest.clearAllMocks();
  });

  it('should be defined', () => {
    expect(controller).toBeDefined();
  });

  describe('getProfile', () => {
    it('should return user profile without password', async () => {
      mockUsersService.findById.mockResolvedValue(mockUser);
      mockUsersService.isPremiumUser.mockResolvedValue(false);
      mockUsersService.getActivePlan.mockResolvedValue(null);

      const result = await controller.getProfile({ user: { userId: 'user-123' } });

      expect(result).not.toHaveProperty('password');
      expect(result.email).toBe('test@example.com');
      expect(result.name).toBe('Test User');
      expect(mockUsersService.findById).toHaveBeenCalledWith('user-123');
    });

    it('should include premium status for premium user', async () => {
      const premiumUser = { ...mockUser, isPremium: true };
      mockUsersService.findById.mockResolvedValue(premiumUser);
      mockUsersService.isPremiumUser.mockResolvedValue(true);
      mockUsersService.getActivePlan.mockResolvedValue({
        name: 'Pro Annual',
        expiresAt: '2025-12-31T00:00:00.000Z',
      });

      const result = await controller.getProfile({ user: { userId: 'user-123' } });

      expect(result.isPremium).toBe(true);
      expect(result.plan).toEqual({
        name: 'Pro Annual',
        expiresAt: '2025-12-31T00:00:00.000Z',
      });
    });

    it('should handle user not found error', async () => {
      mockUsersService.findById.mockRejectedValue(new Error('User not found'));

      await expect(
        controller.getProfile({ user: { userId: 'non-existent' } })
      ).rejects.toThrow('User not found');
    });
  });

  describe('getStats', () => {
    it('should return user statistics', async () => {
      mockUsersService.getUserStats.mockResolvedValue(mockStats);

      const result = await controller.getStats({ user: { userId: 'user-123' } });

      expect(result).toEqual(mockStats);
      expect(result.totalSolved).toBe(25);
      expect(result.currentStreak).toBe(7);
      expect(mockUsersService.getUserStats).toHaveBeenCalledWith('user-123');
    });

    it('should handle new user with zero stats', async () => {
      const zeroStats = {
        totalSolved: 0,
        totalSubmissions: 0,
        hoursSpent: '0h 0m',
        globalRank: 0,
        skillPoints: 0,
        currentStreak: 0,
        maxStreak: 0,
        weekThisWeek: 0,
      };
      mockUsersService.getUserStats.mockResolvedValue(zeroStats);

      const result = await controller.getStats({ user: { userId: 'new-user' } });

      expect(result.totalSolved).toBe(0);
      expect(result.currentStreak).toBe(0);
    });
  });

  describe('getActivity', () => {
    it('should return weekly activity with default parameters', async () => {
      mockUsersService.getWeeklyActivity.mockResolvedValue(mockWeeklyActivity);

      const result = await controller.getActivity({ user: { userId: 'user-123' } });

      expect(result).toEqual(mockWeeklyActivity);
      expect(mockUsersService.getWeeklyActivity).toHaveBeenCalledWith('user-123', 7, 0);
    });

    it('should respect custom days parameter', async () => {
      mockUsersService.getWeeklyActivity.mockResolvedValue(mockWeeklyActivity.slice(0, 3));

      await controller.getActivity(
        { user: { userId: 'user-123' } },
        '3'
      );

      expect(mockUsersService.getWeeklyActivity).toHaveBeenCalledWith('user-123', 3, 0);
    });

    it('should respect offset parameter', async () => {
      mockUsersService.getWeeklyActivity.mockResolvedValue(mockWeeklyActivity);

      await controller.getActivity(
        { user: { userId: 'user-123' } },
        '7',
        '7'
      );

      expect(mockUsersService.getWeeklyActivity).toHaveBeenCalledWith('user-123', 7, 7);
    });

    it('should handle invalid parameters gracefully', async () => {
      mockUsersService.getWeeklyActivity.mockResolvedValue([]);

      await controller.getActivity(
        { user: { userId: 'user-123' } },
        'invalid',
        'invalid'
      );

      // NaN is passed when parseInt fails
      expect(mockUsersService.getWeeklyActivity).toHaveBeenCalled();
    });
  });

  describe('getYearlyActivity', () => {
    it('should return yearly activity for heatmap', async () => {
      mockUsersService.getYearlyActivity.mockResolvedValue(mockYearlyActivity);

      const result = await controller.getYearlyActivity({ user: { userId: 'user-123' } });

      expect(result).toHaveLength(365);
      expect(mockUsersService.getYearlyActivity).toHaveBeenCalledWith('user-123');
    });

    it('should handle empty activity history', async () => {
      mockUsersService.getYearlyActivity.mockResolvedValue([]);

      const result = await controller.getYearlyActivity({ user: { userId: 'new-user' } });

      expect(result).toEqual([]);
    });
  });

  describe('updatePreferences', () => {
    // Matches UpdatePreferencesDto
    const updateDto = {
      editorFontSize: 16,
      editorTheme: 'monokai',
      editorMinimap: false,
    };

    it('should update user preferences', async () => {
      const updatedUser = {
        ...mockUser,
        preferences: { editorFontSize: 16, editorTheme: 'monokai', editorMinimap: false },
      };
      mockUsersService.updatePreferences.mockResolvedValue(updatedUser);
      mockUsersService.isPremiumUser.mockResolvedValue(false);
      mockUsersService.getActivePlan.mockResolvedValue(null);

      const result = await controller.updatePreferences(
        { user: { userId: 'user-123' } },
        updateDto
      );

      expect(result).not.toHaveProperty('password');
      expect(mockUsersService.updatePreferences).toHaveBeenCalledWith('user-123', updateDto);
    });

    it('should handle partial preference updates', async () => {
      const partialUpdate = { editorFontSize: 18 };
      mockUsersService.updatePreferences.mockResolvedValue({
        ...mockUser,
        preferences: { ...mockUser.preferences, editorFontSize: 18 },
      });
      mockUsersService.isPremiumUser.mockResolvedValue(false);
      mockUsersService.getActivePlan.mockResolvedValue(null);

      await controller.updatePreferences(
        { user: { userId: 'user-123' } },
        partialUpdate
      );

      expect(mockUsersService.updatePreferences).toHaveBeenCalledWith('user-123', partialUpdate);
    });
  });

  describe('updateAvatar', () => {
    it('should update user avatar with preset URL', async () => {
      const avatarUrl = 'https://avatars.example.com/preset-1.png';
      const updatedUser = { ...mockUser, avatarUrl };
      mockUsersService.updateAvatar.mockResolvedValue(updatedUser);
      mockUsersService.isPremiumUser.mockResolvedValue(false);
      mockUsersService.getActivePlan.mockResolvedValue(null);

      const result = await controller.updateAvatar(
        { user: { userId: 'user-123' } },
        { avatarUrl }
      );

      expect(result).not.toHaveProperty('password');
      expect(mockUsersService.updateAvatar).toHaveBeenCalledWith('user-123', avatarUrl);
    });

    it('should handle base64 image data', async () => {
      const base64Avatar = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';
      mockUsersService.updateAvatar.mockResolvedValue({
        ...mockUser,
        avatarUrl: base64Avatar,
      });
      mockUsersService.isPremiumUser.mockResolvedValue(false);
      mockUsersService.getActivePlan.mockResolvedValue(null);

      await controller.updateAvatar(
        { user: { userId: 'user-123' } },
        { avatarUrl: base64Avatar }
      );

      expect(mockUsersService.updateAvatar).toHaveBeenCalledWith('user-123', base64Avatar);
    });

    it('should handle empty avatar (removal)', async () => {
      mockUsersService.updateAvatar.mockResolvedValue({
        ...mockUser,
        avatarUrl: null,
      });
      mockUsersService.isPremiumUser.mockResolvedValue(false);
      mockUsersService.getActivePlan.mockResolvedValue(null);

      await controller.updateAvatar(
        { user: { userId: 'user-123' } },
        { avatarUrl: '' }
      );

      expect(mockUsersService.updateAvatar).toHaveBeenCalledWith('user-123', '');
    });
  });

  // NOTE: upgradeToPremium tests removed - endpoint was removed for security
  // Subscription upgrades should now go through POST /subscriptions (admin only)
  // or through Stripe webhook

  describe('edge cases', () => {
    it('should handle user with unicode name', async () => {
      const unicodeUser = {
        ...mockUser,
        name: 'Имя Пользователя 用户名',
      };
      mockUsersService.findById.mockResolvedValue(unicodeUser);
      mockUsersService.isPremiumUser.mockResolvedValue(false);
      mockUsersService.getActivePlan.mockResolvedValue(null);

      const result = await controller.getProfile({ user: { userId: 'user-123' } });

      expect(result.name).toBe('Имя Пользователя 用户名');
    });

    it('should handle concurrent preference updates', async () => {
      mockUsersService.updatePreferences.mockResolvedValue(mockUser);
      mockUsersService.isPremiumUser.mockResolvedValue(false);
      mockUsersService.getActivePlan.mockResolvedValue(null);

      const promises = Array.from({ length: 3 }, () =>
        controller.updatePreferences(
          { user: { userId: 'user-123' } },
          { editorTheme: 'dracula' }
        )
      );

      const results = await Promise.all(promises);

      expect(results).toHaveLength(3);
      expect(mockUsersService.updatePreferences).toHaveBeenCalledTimes(3);
    });

    it('should handle very long avatar URL', async () => {
      const longUrl = 'https://example.com/' + 'a'.repeat(2000);
      mockUsersService.updateAvatar.mockResolvedValue({
        ...mockUser,
        avatarUrl: longUrl,
      });
      mockUsersService.isPremiumUser.mockResolvedValue(false);
      mockUsersService.getActivePlan.mockResolvedValue(null);

      await controller.updateAvatar(
        { user: { userId: 'user-123' } },
        { avatarUrl: longUrl }
      );

      expect(mockUsersService.updateAvatar).toHaveBeenCalledWith('user-123', longUrl);
    });

    it('should handle service errors gracefully', async () => {
      mockUsersService.findById.mockRejectedValue(new Error('Database connection failed'));

      await expect(
        controller.getProfile({ user: { userId: 'user-123' } })
      ).rejects.toThrow('Database connection failed');
    });
  });
});
