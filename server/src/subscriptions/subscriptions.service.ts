import { Injectable, NotFoundException, ConflictException } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { CreateSubscriptionDto } from './dto/subscription.dto';

@Injectable()
export class SubscriptionsService {
  constructor(private prisma: PrismaService) {}

  /**
   * Get all active subscription plans
   */
  async getPlans() {
    return this.prisma.subscriptionPlan.findMany({
      where: { isActive: true },
      include: {
        course: {
          select: {
            id: true,
            slug: true,
            title: true,
            icon: true,
            translations: true,
          },
        },
      },
      orderBy: [{ type: 'asc' }, { name: 'asc' }],
    });
  }

  /**
   * Get a single plan by ID
   */
  async getPlanById(planId: string) {
    const plan = await this.prisma.subscriptionPlan.findUnique({
      where: { id: planId },
      include: { course: true },
    });

    if (!plan) {
      throw new NotFoundException('Subscription plan not found');
    }

    return plan;
  }

  /**
   * Get a single plan by slug
   */
  async getPlanBySlug(slug: string) {
    const plan = await this.prisma.subscriptionPlan.findUnique({
      where: { slug },
      include: { course: true },
    });

    if (!plan) {
      throw new NotFoundException('Subscription plan not found');
    }

    return plan;
  }

  /**
   * Get user's subscriptions
   */
  async getUserSubscriptions(userId: string) {
    return this.prisma.subscription.findMany({
      where: { userId },
      include: {
        plan: {
          include: {
            course: {
              select: {
                id: true,
                slug: true,
                title: true,
                icon: true,
              },
            },
          },
        },
      },
      orderBy: { createdAt: 'desc' },
    });
  }

  /**
   * Create a subscription for a user
   * This is called after successful payment
   */
  async createSubscription(userId: string, dto: CreateSubscriptionDto) {
    const plan = await this.getPlanById(dto.planId);

    // Check if user already has this subscription
    const existing = await this.prisma.subscription.findUnique({
      where: {
        userId_planId: { userId, planId: dto.planId },
      },
    });

    if (existing && existing.status === 'active') {
      throw new ConflictException('User already has an active subscription to this plan');
    }

    // Calculate end date (1 month from now)
    const endDate = new Date();
    endDate.setMonth(endDate.getMonth() + 1);

    // Update existing or create new
    if (existing) {
      return this.prisma.subscription.update({
        where: { id: existing.id },
        data: {
          status: 'active',
          startDate: new Date(),
          endDate,
          autoRenew: dto.autoRenew ?? true,
        },
        include: { plan: true },
      });
    }

    return this.prisma.subscription.create({
      data: {
        userId,
        planId: dto.planId,
        status: 'active',
        endDate,
        autoRenew: dto.autoRenew ?? true,
      },
      include: { plan: true },
    });
  }

  /**
   * Cancel a subscription
   */
  async cancelSubscription(userId: string, subscriptionId: string) {
    const subscription = await this.prisma.subscription.findFirst({
      where: {
        id: subscriptionId,
        userId, // Ensure user owns this subscription
      },
    });

    if (!subscription) {
      throw new NotFoundException('Subscription not found');
    }

    return this.prisma.subscription.update({
      where: { id: subscriptionId },
      data: {
        status: 'cancelled',
        autoRenew: false,
      },
      include: { plan: true },
    });
  }

  /**
   * Renew a subscription (called by payment webhook or scheduled job)
   */
  async renewSubscription(subscriptionId: string) {
    const subscription = await this.prisma.subscription.findUnique({
      where: { id: subscriptionId },
    });

    if (!subscription) {
      throw new NotFoundException('Subscription not found');
    }

    const endDate = new Date(subscription.endDate);
    endDate.setMonth(endDate.getMonth() + 1);

    return this.prisma.subscription.update({
      where: { id: subscriptionId },
      data: {
        status: 'active',
        endDate,
      },
      include: { plan: true },
    });
  }

  /**
   * Check and expire subscriptions (called by scheduled job)
   */
  async expireSubscriptions() {
    const now = new Date();

    return this.prisma.subscription.updateMany({
      where: {
        status: 'active',
        endDate: { lt: now },
      },
      data: {
        status: 'expired',
      },
    });
  }

  /**
   * Get subscriptions due for renewal (for payment processing)
   */
  async getSubscriptionsDueForRenewal() {
    const now = new Date();
    const threeDaysFromNow = new Date();
    threeDaysFromNow.setDate(threeDaysFromNow.getDate() + 3);

    return this.prisma.subscription.findMany({
      where: {
        status: 'active',
        autoRenew: true,
        endDate: {
          gte: now,
          lte: threeDaysFromNow,
        },
      },
      include: {
        plan: true,
        user: {
          select: {
            id: true,
            email: true,
            name: true,
          },
        },
      },
    });
  }
}
