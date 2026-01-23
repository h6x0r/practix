import { Test, TestingModule } from '@nestjs/testing';
import { AdminController } from './admin.controller';
import { AdminService } from './admin.service';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { AdminGuard } from '../auth/guards/admin.guard';
import { ThrottlerModule } from '@nestjs/throttler';

describe('AdminController', () => {
  let controller: AdminController;

  const mockAdminService = {
    getDashboardStats: jest.fn(),
    getCourseAnalytics: jest.fn(),
    getTaskAnalytics: jest.fn(),
    getSubmissionStats: jest.fn(),
    getSubscriptionStats: jest.fn(),
    getAiUsageStats: jest.fn(),
  };

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      imports: [
        ThrottlerModule.forRoot([{ ttl: 60000, limit: 100 }]),
      ],
      controllers: [AdminController],
      providers: [
        { provide: AdminService, useValue: mockAdminService },
      ],
    })
      .overrideGuard(JwtAuthGuard)
      .useValue({ canActivate: () => true })
      .overrideGuard(AdminGuard)
      .useValue({ canActivate: () => true })
      .compile();

    controller = module.get<AdminController>(AdminController);
    jest.clearAllMocks();
  });

  it('should be defined', () => {
    expect(controller).toBeDefined();
  });

  describe('getDashboardStats()', () => {
    it('should return dashboard statistics', async () => {
      const mockStats = {
        totalUsers: 1500,
        newUsersToday: 25,
        newUsersThisWeek: 150,
        activeUsersToday: 300,
        activeUsersThisWeek: 800,
      };
      mockAdminService.getDashboardStats.mockResolvedValue(mockStats);

      const result = await controller.getDashboardStats();

      expect(result).toEqual(mockStats);
      expect(mockAdminService.getDashboardStats).toHaveBeenCalled();
    });
  });

  describe('getCourseAnalytics()', () => {
    it('should return course analytics', async () => {
      const mockAnalytics = {
        coursePopularity: [
          { courseSlug: 'go-basics', studentCount: 500 },
          { courseSlug: 'python-basics', studentCount: 400 },
        ],
        completionRates: [
          { courseSlug: 'go-basics', completionRate: 65 },
        ],
      };
      mockAdminService.getCourseAnalytics.mockResolvedValue(mockAnalytics);

      const result = await controller.getCourseAnalytics();

      expect(result).toEqual(mockAnalytics);
      expect(mockAdminService.getCourseAnalytics).toHaveBeenCalled();
    });
  });

  describe('getTaskAnalytics()', () => {
    it('should return task analytics', async () => {
      const mockAnalytics = {
        hardestTasks: [
          { taskSlug: 'task-1', failureRate: 40 },
        ],
        mostPopularTasks: [
          { taskSlug: 'task-2', submissionCount: 1000 },
        ],
      };
      mockAdminService.getTaskAnalytics.mockResolvedValue(mockAnalytics);

      const result = await controller.getTaskAnalytics();

      expect(result).toEqual(mockAnalytics);
      expect(mockAdminService.getTaskAnalytics).toHaveBeenCalled();
    });
  });

  describe('getSubmissionStats()', () => {
    it('should return submission statistics', async () => {
      const mockStats = {
        totalSubmissions: 50000,
        byStatus: {
          passed: 35000,
          failed: 15000,
        },
        dailyTrend: [
          { date: '2025-01-15', count: 500 },
        ],
      };
      mockAdminService.getSubmissionStats.mockResolvedValue(mockStats);

      const result = await controller.getSubmissionStats();

      expect(result).toEqual(mockStats);
      expect(mockAdminService.getSubmissionStats).toHaveBeenCalled();
    });
  });

  describe('getSubscriptionStats()', () => {
    it('should return subscription statistics', async () => {
      const mockStats = {
        activeSubscriptions: 200,
        newSubscriptionsThisMonth: 50,
        monthlyRevenue: 10000,
      };
      mockAdminService.getSubscriptionStats.mockResolvedValue(mockStats);

      const result = await controller.getSubscriptionStats();

      expect(result).toEqual(mockStats);
      expect(mockAdminService.getSubscriptionStats).toHaveBeenCalled();
    });
  });

  describe('getAiUsageStats()', () => {
    it('should return AI usage statistics', async () => {
      const mockStats = {
        totalRequests: 5000,
        requestsToday: 200,
        dailyTrend: [
          { date: '2025-01-15', count: 180 },
        ],
      };
      mockAdminService.getAiUsageStats.mockResolvedValue(mockStats);

      const result = await controller.getAiUsageStats();

      expect(result).toEqual(mockStats);
      expect(mockAdminService.getAiUsageStats).toHaveBeenCalled();
    });
  });
});
