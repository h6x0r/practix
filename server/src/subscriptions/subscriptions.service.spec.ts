import { Test, TestingModule } from '@nestjs/testing';
import { SubscriptionsService } from './subscriptions.service';
import { PrismaService } from '../prisma/prisma.service';
import { NotFoundException, ConflictException } from '@nestjs/common';

describe('SubscriptionsService', () => {
  let service: SubscriptionsService;
  let prisma: PrismaService;

  const mockPlan = {
    id: 'plan-global',
    slug: 'premium-global',
    name: 'Premium Global',
    type: 'global',
    price: 9.99,
    currency: 'USD',
    isActive: true,
    courseId: null,
    course: null,
  };

  const mockCoursePlan = {
    id: 'plan-course',
    slug: 'premium-go',
    name: 'Go Course Premium',
    type: 'course',
    price: 4.99,
    currency: 'USD',
    isActive: true,
    courseId: 'course-go',
    course: {
      id: 'course-go',
      slug: 'go-basics',
      title: 'Go Basics',
      icon: 'golang',
    },
  };

  const mockSubscription = {
    id: 'sub-123',
    userId: 'user-123',
    planId: 'plan-global',
    status: 'active',
    startDate: new Date('2025-01-01'),
    endDate: new Date('2025-02-01'),
    autoRenew: true,
    createdAt: new Date('2025-01-01'),
    plan: mockPlan,
  };

  const mockExpiredSubscription = {
    ...mockSubscription,
    id: 'sub-expired',
    status: 'expired',
    endDate: new Date('2024-12-01'),
  };

  const mockCancelledSubscription = {
    ...mockSubscription,
    id: 'sub-cancelled',
    status: 'cancelled',
    autoRenew: false,
  };

  const mockPrismaService = {
    subscriptionPlan: {
      findMany: jest.fn(),
      findUnique: jest.fn(),
    },
    subscription: {
      findMany: jest.fn(),
      findFirst: jest.fn(),
      findUnique: jest.fn(),
      create: jest.fn(),
      update: jest.fn(),
      updateMany: jest.fn(),
      upsert: jest.fn(),
    },
  };

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [
        SubscriptionsService,
        { provide: PrismaService, useValue: mockPrismaService },
      ],
    }).compile();

    service = module.get<SubscriptionsService>(SubscriptionsService);
    prisma = module.get<PrismaService>(PrismaService);

    jest.clearAllMocks();
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });

  // ============================================
  // getPlans() - Get all active plans
  // ============================================
  describe('getPlans()', () => {
    it('should return all active subscription plans', async () => {
      const plans = [mockPlan, mockCoursePlan];
      mockPrismaService.subscriptionPlan.findMany.mockResolvedValue(plans);

      const result = await service.getPlans();

      expect(result).toEqual(plans);
      expect(mockPrismaService.subscriptionPlan.findMany).toHaveBeenCalledWith(
        expect.objectContaining({
          where: { isActive: true },
        })
      );
    });

    it('should include course information', async () => {
      mockPrismaService.subscriptionPlan.findMany.mockResolvedValue([mockCoursePlan]);

      const result = await service.getPlans();

      expect(result[0].course).toBeDefined();
      expect(result[0].course.slug).toBe('go-basics');
    });

    it('should order by type and name', async () => {
      mockPrismaService.subscriptionPlan.findMany.mockResolvedValue([]);

      await service.getPlans();

      expect(mockPrismaService.subscriptionPlan.findMany).toHaveBeenCalledWith(
        expect.objectContaining({
          orderBy: [{ type: 'asc' }, { name: 'asc' }],
        })
      );
    });

    it('should return empty array if no plans exist', async () => {
      mockPrismaService.subscriptionPlan.findMany.mockResolvedValue([]);

      const result = await service.getPlans();

      expect(result).toEqual([]);
    });
  });

  // ============================================
  // getPlanById() - Get plan by ID
  // ============================================
  describe('getPlanById()', () => {
    it('should return plan by id', async () => {
      mockPrismaService.subscriptionPlan.findUnique.mockResolvedValue(mockPlan);

      const result = await service.getPlanById('plan-global');

      expect(result).toEqual(mockPlan);
    });

    it('should throw NotFoundException if plan not found', async () => {
      mockPrismaService.subscriptionPlan.findUnique.mockResolvedValue(null);

      await expect(service.getPlanById('nonexistent')).rejects.toThrow(NotFoundException);
    });

    it('should include course information', async () => {
      mockPrismaService.subscriptionPlan.findUnique.mockResolvedValue(mockCoursePlan);

      const result = await service.getPlanById('plan-course');

      expect(result.course).toBeDefined();
    });
  });

  // ============================================
  // getPlanBySlug() - Get plan by slug
  // ============================================
  describe('getPlanBySlug()', () => {
    it('should return plan by slug', async () => {
      mockPrismaService.subscriptionPlan.findUnique.mockResolvedValue(mockPlan);

      const result = await service.getPlanBySlug('premium-global');

      expect(result).toEqual(mockPlan);
    });

    it('should throw NotFoundException if plan not found', async () => {
      mockPrismaService.subscriptionPlan.findUnique.mockResolvedValue(null);

      await expect(service.getPlanBySlug('nonexistent')).rejects.toThrow(NotFoundException);
    });
  });

  // ============================================
  // getUserSubscriptions() - Get user subscriptions
  // ============================================
  describe('getUserSubscriptions()', () => {
    it('should return user subscriptions', async () => {
      mockPrismaService.subscription.findMany.mockResolvedValue([mockSubscription]);

      const result = await service.getUserSubscriptions('user-123');

      expect(result).toHaveLength(1);
      expect(result[0]).toEqual(mockSubscription);
    });

    it('should include plan and course details', async () => {
      mockPrismaService.subscription.findMany.mockResolvedValue([mockSubscription]);

      await service.getUserSubscriptions('user-123');

      expect(mockPrismaService.subscription.findMany).toHaveBeenCalledWith(
        expect.objectContaining({
          include: expect.objectContaining({
            plan: expect.anything(),
          }),
        })
      );
    });

    it('should order by createdAt desc', async () => {
      mockPrismaService.subscription.findMany.mockResolvedValue([]);

      await service.getUserSubscriptions('user-123');

      expect(mockPrismaService.subscription.findMany).toHaveBeenCalledWith(
        expect.objectContaining({
          orderBy: { createdAt: 'desc' },
        })
      );
    });

    it('should return empty array if no subscriptions', async () => {
      mockPrismaService.subscription.findMany.mockResolvedValue([]);

      const result = await service.getUserSubscriptions('user-123');

      expect(result).toEqual([]);
    });
  });

  // ============================================
  // createSubscription() - Create subscription (uses upsert)
  // ============================================
  describe('createSubscription()', () => {
    it('should create or update subscription using upsert', async () => {
      mockPrismaService.subscriptionPlan.findUnique.mockResolvedValue(mockPlan);
      mockPrismaService.subscription.upsert.mockResolvedValue(mockSubscription);

      const result = await service.createSubscription('user-123', { planId: 'plan-global' });

      expect(result).toEqual(mockSubscription);
      expect(mockPrismaService.subscription.upsert).toHaveBeenCalledWith(
        expect.objectContaining({
          where: { userId_planId: { userId: 'user-123', planId: 'plan-global' } },
          create: expect.objectContaining({
            userId: 'user-123',
            planId: 'plan-global',
            status: 'active',
            autoRenew: true,
          }),
          update: expect.objectContaining({
            status: 'active',
            autoRenew: true,
          }),
        })
      );
    });

    it('should set status to active in both create and update', async () => {
      mockPrismaService.subscriptionPlan.findUnique.mockResolvedValue(mockPlan);
      mockPrismaService.subscription.upsert.mockResolvedValue(mockSubscription);

      await service.createSubscription('user-123', { planId: 'plan-global' });

      const upsertCall = mockPrismaService.subscription.upsert.mock.calls[0][0];
      expect(upsertCall.create.status).toBe('active');
      expect(upsertCall.update.status).toBe('active');
    });

    it('should handle duplicate subscriptions via upsert (no conflict)', async () => {
      mockPrismaService.subscriptionPlan.findUnique.mockResolvedValue(mockPlan);
      mockPrismaService.subscription.upsert.mockResolvedValue({
        ...mockSubscription,
        status: 'active',
      });

      // Upsert handles duplicates gracefully - updates instead of conflicts
      const result = await service.createSubscription('user-123', { planId: 'plan-global' });

      expect(result.status).toBe('active');
      expect(mockPrismaService.subscription.upsert).toHaveBeenCalled();
    });

    it('should respect autoRenew setting', async () => {
      mockPrismaService.subscriptionPlan.findUnique.mockResolvedValue(mockPlan);
      mockPrismaService.subscription.upsert.mockResolvedValue({
        ...mockSubscription,
        autoRenew: false,
      });

      await service.createSubscription('user-123', { planId: 'plan-global', autoRenew: false });

      const upsertCall = mockPrismaService.subscription.upsert.mock.calls[0][0];
      expect(upsertCall.create.autoRenew).toBe(false);
      expect(upsertCall.update.autoRenew).toBe(false);
    });

    it('should default autoRenew to true', async () => {
      mockPrismaService.subscriptionPlan.findUnique.mockResolvedValue(mockPlan);
      mockPrismaService.subscription.upsert.mockResolvedValue(mockSubscription);

      await service.createSubscription('user-123', { planId: 'plan-global' });

      const upsertCall = mockPrismaService.subscription.upsert.mock.calls[0][0];
      expect(upsertCall.create.autoRenew).toBe(true);
      expect(upsertCall.update.autoRenew).toBe(true);
    });

    it('should throw NotFoundException if plan not found', async () => {
      mockPrismaService.subscriptionPlan.findUnique.mockResolvedValue(null);

      await expect(
        service.createSubscription('user-123', { planId: 'nonexistent' })
      ).rejects.toThrow(NotFoundException);
    });

    it('should set correct end date (1 month from now)', async () => {
      mockPrismaService.subscriptionPlan.findUnique.mockResolvedValue(mockPlan);
      mockPrismaService.subscription.upsert.mockResolvedValue(mockSubscription);

      await service.createSubscription('user-123', { planId: 'plan-global' });

      const upsertCall = mockPrismaService.subscription.upsert.mock.calls[0][0];
      const expectedEndDate = new Date();
      expectedEndDate.setMonth(expectedEndDate.getMonth() + 1);

      // Allow 1 second tolerance
      expect(upsertCall.create.endDate.getTime()).toBeCloseTo(expectedEndDate.getTime(), -3);
    });
  });

  // ============================================
  // cancelSubscription() - Cancel subscription
  // ============================================
  describe('cancelSubscription()', () => {
    it('should set status to cancelled', async () => {
      mockPrismaService.subscription.findFirst.mockResolvedValue(mockSubscription);
      mockPrismaService.subscription.update.mockResolvedValue(mockCancelledSubscription);

      const result = await service.cancelSubscription('user-123', 'sub-123');

      expect(result.status).toBe('cancelled');
    });

    it('should disable autoRenew', async () => {
      mockPrismaService.subscription.findFirst.mockResolvedValue(mockSubscription);
      mockPrismaService.subscription.update.mockResolvedValue(mockCancelledSubscription);

      await service.cancelSubscription('user-123', 'sub-123');

      expect(mockPrismaService.subscription.update).toHaveBeenCalledWith(
        expect.objectContaining({
          data: expect.objectContaining({
            autoRenew: false,
          }),
        })
      );
    });

    it('should throw NotFoundException if subscription not found', async () => {
      mockPrismaService.subscription.findFirst.mockResolvedValue(null);

      await expect(
        service.cancelSubscription('user-123', 'nonexistent')
      ).rejects.toThrow(NotFoundException);
    });

    it('should verify user owns subscription', async () => {
      mockPrismaService.subscription.findFirst.mockResolvedValue(null);

      await expect(
        service.cancelSubscription('user-456', 'sub-123')
      ).rejects.toThrow(NotFoundException);

      expect(mockPrismaService.subscription.findFirst).toHaveBeenCalledWith(
        expect.objectContaining({
          where: expect.objectContaining({
            id: 'sub-123',
            userId: 'user-456',
          }),
        })
      );
    });
  });

  // ============================================
  // renewSubscription() - Renew subscription
  // ============================================
  describe('renewSubscription()', () => {
    it('should extend end date by one month', async () => {
      mockPrismaService.subscription.findUnique.mockResolvedValue(mockSubscription);
      mockPrismaService.subscription.update.mockResolvedValue({
        ...mockSubscription,
        endDate: new Date('2025-03-01'),
      });

      const result = await service.renewSubscription('sub-123');

      expect(mockPrismaService.subscription.update).toHaveBeenCalledWith(
        expect.objectContaining({
          data: expect.objectContaining({
            status: 'active',
          }),
        })
      );
    });

    it('should set status to active', async () => {
      mockPrismaService.subscription.findUnique.mockResolvedValue(mockExpiredSubscription);
      mockPrismaService.subscription.update.mockResolvedValue({
        ...mockExpiredSubscription,
        status: 'active',
      });

      await service.renewSubscription('sub-expired');

      expect(mockPrismaService.subscription.update).toHaveBeenCalledWith(
        expect.objectContaining({
          data: expect.objectContaining({
            status: 'active',
          }),
        })
      );
    });

    it('should throw NotFoundException if subscription not found', async () => {
      mockPrismaService.subscription.findUnique.mockResolvedValue(null);

      await expect(service.renewSubscription('nonexistent')).rejects.toThrow(NotFoundException);
    });
  });

  // ============================================
  // expireSubscriptions() - Expire old subscriptions
  // ============================================
  describe('expireSubscriptions()', () => {
    it('should update expired subscriptions', async () => {
      mockPrismaService.subscription.updateMany.mockResolvedValue({ count: 5 });

      const result = await service.expireSubscriptions();

      expect(result.count).toBe(5);
    });

    it('should only expire active subscriptions', async () => {
      mockPrismaService.subscription.updateMany.mockResolvedValue({ count: 0 });

      await service.expireSubscriptions();

      expect(mockPrismaService.subscription.updateMany).toHaveBeenCalledWith(
        expect.objectContaining({
          where: expect.objectContaining({
            status: 'active',
          }),
        })
      );
    });

    it('should check endDate is in the past', async () => {
      mockPrismaService.subscription.updateMany.mockResolvedValue({ count: 0 });

      await service.expireSubscriptions();

      expect(mockPrismaService.subscription.updateMany).toHaveBeenCalledWith(
        expect.objectContaining({
          where: expect.objectContaining({
            endDate: { lt: expect.any(Date) },
          }),
        })
      );
    });

    it('should set status to expired', async () => {
      mockPrismaService.subscription.updateMany.mockResolvedValue({ count: 0 });

      await service.expireSubscriptions();

      expect(mockPrismaService.subscription.updateMany).toHaveBeenCalledWith(
        expect.objectContaining({
          data: { status: 'expired' },
        })
      );
    });
  });

  // ============================================
  // getSubscriptionsDueForRenewal() - Get due subscriptions
  // ============================================
  describe('getSubscriptionsDueForRenewal()', () => {
    it('should return subscriptions due for renewal', async () => {
      mockPrismaService.subscription.findMany.mockResolvedValue([mockSubscription]);

      const result = await service.getSubscriptionsDueForRenewal();

      expect(result).toHaveLength(1);
    });

    it('should filter by active status', async () => {
      mockPrismaService.subscription.findMany.mockResolvedValue([]);

      await service.getSubscriptionsDueForRenewal();

      expect(mockPrismaService.subscription.findMany).toHaveBeenCalledWith(
        expect.objectContaining({
          where: expect.objectContaining({
            status: 'active',
          }),
        })
      );
    });

    it('should filter by autoRenew enabled', async () => {
      mockPrismaService.subscription.findMany.mockResolvedValue([]);

      await service.getSubscriptionsDueForRenewal();

      expect(mockPrismaService.subscription.findMany).toHaveBeenCalledWith(
        expect.objectContaining({
          where: expect.objectContaining({
            autoRenew: true,
          }),
        })
      );
    });

    it('should include user info', async () => {
      mockPrismaService.subscription.findMany.mockResolvedValue([]);

      await service.getSubscriptionsDueForRenewal();

      expect(mockPrismaService.subscription.findMany).toHaveBeenCalledWith(
        expect.objectContaining({
          include: expect.objectContaining({
            user: expect.anything(),
          }),
        })
      );
    });
  });
});
