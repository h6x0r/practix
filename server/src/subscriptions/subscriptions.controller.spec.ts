import { Test, TestingModule } from '@nestjs/testing';
import { SubscriptionsController } from './subscriptions.controller';
import { SubscriptionsService } from './subscriptions.service';
import { AccessControlService } from './access-control.service';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { AdminGuard } from '../auth/guards/admin.guard';
import { ConfigService } from '@nestjs/config';
import { NotFoundException, ForbiddenException } from '@nestjs/common';

describe('SubscriptionsController', () => {
  let controller: SubscriptionsController;
  let subscriptionsService: SubscriptionsService;
  let accessControlService: AccessControlService;

  const mockSubscriptionsService = {
    getPlans: jest.fn(),
    getPlanBySlug: jest.fn(),
    getUserSubscriptions: jest.fn(),
    createSubscription: jest.fn(),
    cancelSubscription: jest.fn(),
  };

  const mockAccessControlService = {
    getCourseAccess: jest.fn(),
    getTaskAccess: jest.fn(),
  };

  const mockConfigService = {
    get: jest.fn().mockReturnValue('test-webhook-secret'),
  };

  // Matches SubscriptionPlanDto
  const mockPlans = [
    {
      id: 'plan-1',
      name: 'Free',
      slug: 'free',
      nameRu: 'Бесплатный',
      type: 'global' as const,
      courseId: null,
      priceMonthly: 0,
      currency: 'USD',
      isActive: true,
    },
    {
      id: 'plan-2',
      name: 'Pro Monthly',
      slug: 'pro-monthly',
      nameRu: 'Про (месяц)',
      type: 'global' as const,
      courseId: null,
      priceMonthly: 9.99,
      currency: 'USD',
      isActive: true,
    },
    {
      id: 'plan-3',
      name: 'Pro Annual',
      slug: 'pro-annual',
      nameRu: 'Про (год)',
      type: 'global' as const,
      courseId: null,
      priceMonthly: 8.33,
      currency: 'USD',
      isActive: true,
    },
  ];

  const mockSubscription = {
    id: 'sub-123',
    userId: 'user-123',
    planId: 'plan-2',
    status: 'active',
    currentPeriodStart: new Date(),
    currentPeriodEnd: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000),
    plan: mockPlans[1],
  };

  // Matches CourseAccessDto
  const mockCourseAccess = {
    hasAccess: true,
    queuePriority: 1,
    canUseAiTutor: true,
  };

  // Matches TaskAccessDto
  const mockTaskAccess = {
    canView: true,
    canRun: true,
    canSubmit: true,
    canSeeSolution: true,
    canUseAiTutor: true,
    queuePriority: 1,
  };

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      controllers: [SubscriptionsController],
      providers: [
        {
          provide: SubscriptionsService,
          useValue: mockSubscriptionsService,
        },
        {
          provide: AccessControlService,
          useValue: mockAccessControlService,
        },
        {
          provide: ConfigService,
          useValue: mockConfigService,
        },
      ],
    })
      .overrideGuard(JwtAuthGuard)
      .useValue({ canActivate: () => true })
      .overrideGuard(AdminGuard)
      .useValue({ canActivate: () => true })
      .compile();

    controller = module.get<SubscriptionsController>(SubscriptionsController);
    subscriptionsService = module.get<SubscriptionsService>(SubscriptionsService);
    accessControlService = module.get<AccessControlService>(AccessControlService);

    jest.clearAllMocks();
  });

  it('should be defined', () => {
    expect(controller).toBeDefined();
  });

  describe('getPlans', () => {
    it('should return all subscription plans', async () => {
      mockSubscriptionsService.getPlans.mockResolvedValue(mockPlans);

      const result = await controller.getPlans();

      expect(result).toEqual(mockPlans);
      expect(result).toHaveLength(3);
      expect(mockSubscriptionsService.getPlans).toHaveBeenCalled();
    });

    it('should include free plan', async () => {
      mockSubscriptionsService.getPlans.mockResolvedValue(mockPlans);

      const result = await controller.getPlans();

      const freePlan = result.find((p) => p.slug === 'free');
      expect(freePlan).toBeDefined();
      expect(freePlan.priceMonthly).toBe(0);
    });

    it('should return empty array if no plans exist', async () => {
      mockSubscriptionsService.getPlans.mockResolvedValue([]);

      const result = await controller.getPlans();

      expect(result).toEqual([]);
    });
  });

  describe('getPlanBySlug', () => {
    it('should return plan by slug', async () => {
      mockSubscriptionsService.getPlanBySlug.mockResolvedValue(mockPlans[1]);

      const result = await controller.getPlanBySlug('pro-monthly');

      expect(result).toEqual(mockPlans[1]);
      expect(mockSubscriptionsService.getPlanBySlug).toHaveBeenCalledWith('pro-monthly');
    });

    it('should throw NotFoundException for non-existent plan', async () => {
      mockSubscriptionsService.getPlanBySlug.mockRejectedValue(
        new NotFoundException('Plan not found')
      );

      await expect(controller.getPlanBySlug('non-existent')).rejects.toThrow(
        NotFoundException
      );
    });

    it('should handle plan slug with special characters', async () => {
      mockSubscriptionsService.getPlanBySlug.mockResolvedValue(mockPlans[2]);

      await controller.getPlanBySlug('pro-annual');

      expect(mockSubscriptionsService.getPlanBySlug).toHaveBeenCalledWith('pro-annual');
    });
  });

  describe('getMySubscriptions', () => {
    it('should return user subscriptions', async () => {
      mockSubscriptionsService.getUserSubscriptions.mockResolvedValue([mockSubscription]);

      const result = await controller.getMySubscriptions({ user: { userId: 'user-123' } });

      expect(result).toHaveLength(1);
      expect(result[0].status).toBe('active');
      expect(mockSubscriptionsService.getUserSubscriptions).toHaveBeenCalledWith('user-123');
    });

    it('should return empty array for user with no subscriptions', async () => {
      mockSubscriptionsService.getUserSubscriptions.mockResolvedValue([]);

      const result = await controller.getMySubscriptions({ user: { userId: 'new-user' } });

      expect(result).toEqual([]);
    });

    it('should handle multiple subscriptions', async () => {
      const multipleSubscriptions = [
        mockSubscription,
        { ...mockSubscription, id: 'sub-456', status: 'cancelled' },
      ];
      mockSubscriptionsService.getUserSubscriptions.mockResolvedValue(multipleSubscriptions);

      const result = await controller.getMySubscriptions({ user: { userId: 'user-123' } });

      expect(result).toHaveLength(2);
    });
  });

  describe('getCourseAccess', () => {
    it('should return course access info for premium user', async () => {
      mockAccessControlService.getCourseAccess.mockResolvedValue(mockCourseAccess);

      const result = await controller.getCourseAccess(
        { user: { userId: 'user-123' } },
        'course-1'
      );

      expect(result.hasAccess).toBe(true);
      expect(result.queuePriority).toBe(1); // Premium priority
      expect(mockAccessControlService.getCourseAccess).toHaveBeenCalledWith(
        'user-123',
        'course-1'
      );
    });

    it('should return limited access for free user', async () => {
      const freeAccess = {
        hasAccess: true,
        queuePriority: 10, // Low priority for free users
        canUseAiTutor: false,
      };
      mockAccessControlService.getCourseAccess.mockResolvedValue(freeAccess);

      const result = await controller.getCourseAccess(
        { user: { userId: 'free-user' } },
        'course-1'
      );

      expect(result.hasAccess).toBe(true);
      expect(result.queuePriority).toBe(10);
      expect(result.canUseAiTutor).toBe(false);
    });

    it('should return no access for premium course without subscription', async () => {
      const noAccess = {
        hasAccess: false,
        queuePriority: 10,
        canUseAiTutor: false,
      };
      mockAccessControlService.getCourseAccess.mockResolvedValue(noAccess);

      const result = await controller.getCourseAccess(
        { user: { userId: 'free-user' } },
        'premium-course'
      );

      expect(result.hasAccess).toBe(false);
      expect(result.canUseAiTutor).toBe(false);
    });
  });

  describe('getTaskAccess', () => {
    it('should return task access info for premium user', async () => {
      mockAccessControlService.getTaskAccess.mockResolvedValue(mockTaskAccess);

      const result = await controller.getTaskAccess(
        { user: { userId: 'user-123' } },
        'task-1'
      );

      expect(result.canView).toBe(true);
      expect(result.canSubmit).toBe(true);
      expect(mockAccessControlService.getTaskAccess).toHaveBeenCalledWith(
        'user-123',
        'task-1'
      );
    });

    it('should block access to premium task for free user', async () => {
      const blockedAccess = {
        canView: true,
        canRun: false,
        canSubmit: false,
        canSeeSolution: false,
        canUseAiTutor: false,
        queuePriority: 10,
      };
      mockAccessControlService.getTaskAccess.mockResolvedValue(blockedAccess);

      const result = await controller.getTaskAccess(
        { user: { userId: 'free-user' } },
        'premium-task'
      );

      expect(result.canView).toBe(true);
      expect(result.canSubmit).toBe(false);
    });

    it('should allow access to free task for any user', async () => {
      const freeTaskAccess = {
        canView: true,
        canRun: true,
        canSubmit: true,
        canSeeSolution: false,
        canUseAiTutor: false,
        queuePriority: 10,
      };
      mockAccessControlService.getTaskAccess.mockResolvedValue(freeTaskAccess);

      const result = await controller.getTaskAccess(
        { user: { userId: 'any-user' } },
        'free-task'
      );

      expect(result.canView).toBe(true);
      expect(result.canRun).toBe(true);
    });
  });

  describe('createSubscription', () => {
    // Matches CreateSubscriptionDto
    const createDto = {
      planId: 'plan-2',
      autoRenew: true,
    };

    it('should create a new subscription', async () => {
      mockSubscriptionsService.createSubscription.mockResolvedValue(mockSubscription);

      const result = await controller.createSubscription(
        { user: { userId: 'user-123' } },
        createDto
      );

      expect(result).toEqual(mockSubscription);
      expect(result.status).toBe('active');
      expect(mockSubscriptionsService.createSubscription).toHaveBeenCalledWith(
        'user-123',
        createDto
      );
    });

    it('should handle payment failure', async () => {
      mockSubscriptionsService.createSubscription.mockRejectedValue(
        new Error('Payment declined')
      );

      await expect(
        controller.createSubscription({ user: { userId: 'user-123' } }, createDto)
      ).rejects.toThrow('Payment declined');
    });

    it('should create subscription with different plan', async () => {
      const annualDto = { planId: 'plan-3', autoRenew: false };
      const annualSubscription = { ...mockSubscription, planId: 'plan-3' };
      mockSubscriptionsService.createSubscription.mockResolvedValue(annualSubscription);

      const result = await controller.createSubscription(
        { user: { userId: 'user-123' } },
        annualDto
      );

      expect(result.planId).toBe('plan-3');
    });
  });

  describe('cancelSubscription', () => {
    it('should cancel a subscription', async () => {
      const cancelledSubscription = { ...mockSubscription, status: 'cancelled' };
      mockSubscriptionsService.cancelSubscription.mockResolvedValue(cancelledSubscription);

      const result = await controller.cancelSubscription(
        { user: { userId: 'user-123' } },
        'sub-123'
      );

      expect(result.status).toBe('cancelled');
      expect(mockSubscriptionsService.cancelSubscription).toHaveBeenCalledWith(
        'user-123',
        'sub-123'
      );
    });

    it('should throw error when cancelling non-existent subscription', async () => {
      mockSubscriptionsService.cancelSubscription.mockRejectedValue(
        new NotFoundException('Subscription not found')
      );

      await expect(
        controller.cancelSubscription({ user: { userId: 'user-123' } }, 'non-existent')
      ).rejects.toThrow(NotFoundException);
    });

    it('should throw error when cancelling another user subscription', async () => {
      mockSubscriptionsService.cancelSubscription.mockRejectedValue(
        new ForbiddenException('Not authorized')
      );

      await expect(
        controller.cancelSubscription({ user: { userId: 'other-user' } }, 'sub-123')
      ).rejects.toThrow(ForbiddenException);
    });
  });

  describe('edge cases', () => {
    it('should handle service errors gracefully', async () => {
      mockSubscriptionsService.getPlans.mockRejectedValue(
        new Error('Database connection failed')
      );

      await expect(controller.getPlans()).rejects.toThrow('Database connection failed');
    });

    it('should handle concurrent subscription requests', async () => {
      mockSubscriptionsService.createSubscription.mockResolvedValue(mockSubscription);

      const promises = Array.from({ length: 3 }, () =>
        controller.createSubscription(
          { user: { userId: 'user-123' } },
          { planId: 'plan-2', autoRenew: true }
        )
      );

      const results = await Promise.all(promises);

      expect(results).toHaveLength(3);
    });

    it('should handle task ID with special format', async () => {
      mockAccessControlService.getTaskAccess.mockResolvedValue(mockTaskAccess);

      await controller.getTaskAccess(
        { user: { userId: 'user-123' } },
        'go-basics-01-hello-world'
      );

      expect(mockAccessControlService.getTaskAccess).toHaveBeenCalledWith(
        'user-123',
        'go-basics-01-hello-world'
      );
    });

    it('should handle course ID with underscore format', async () => {
      mockAccessControlService.getCourseAccess.mockResolvedValue(mockCourseAccess);

      await controller.getCourseAccess(
        { user: { userId: 'user-123' } },
        'c_go_basics'
      );

      expect(mockAccessControlService.getCourseAccess).toHaveBeenCalledWith(
        'user-123',
        'c_go_basics'
      );
    });
  });
});
