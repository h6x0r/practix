import { Injectable, NotFoundException } from "@nestjs/common";
import { PrismaService } from "../prisma/prisma.service";
import { AuditService } from "./audit/audit.service";

@Injectable()
export class AdminPaymentsService {
  constructor(
    private readonly prisma: PrismaService,
    private readonly auditService: AuditService,
  ) {}

  // ============================================
  // PAYMENTS MANAGEMENT (Admin Panel Phase 2.2)
  // ============================================

  /**
   * Get all payments with filtering and pagination
   */
  async getPayments(params: {
    status?: string;
    provider?: string;
    limit?: number;
    offset?: number;
  }) {
    const { status, provider, limit = 50, offset = 0 } = params;

    const where: {
      status?: string;
      provider?: string;
    } = {};

    if (status) where.status = status;
    if (provider) where.provider = provider;

    const [payments, total] = await Promise.all([
      this.prisma.payment.findMany({
        where,
        include: {
          subscription: {
            include: {
              user: {
                select: { id: true, email: true, name: true },
              },
              plan: {
                select: { id: true, name: true, slug: true, type: true },
              },
            },
          },
        },
        orderBy: { createdAt: "desc" },
        take: limit,
        skip: offset,
      }),
      this.prisma.payment.count({ where }),
    ]);

    return {
      payments: payments.map((p) => ({
        id: p.id,
        amount: p.amount,
        currency: p.currency,
        status: p.status,
        provider: p.provider,
        providerTxId: p.providerTxId,
        createdAt: p.createdAt,
        updatedAt: p.updatedAt,
        user: p.subscription.user,
        plan: p.subscription.plan,
        subscriptionId: p.subscriptionId,
      })),
      total,
    };
  }

  /**
   * Get all purchases (one-time payments) with filtering
   */
  async getPurchases(params: {
    status?: string;
    type?: string;
    limit?: number;
    offset?: number;
  }) {
    const { status, type, limit = 50, offset = 0 } = params;

    const where: {
      status?: string;
      type?: string;
    } = {};

    if (status) where.status = status;
    if (type) where.type = type;

    const [purchases, total] = await Promise.all([
      this.prisma.purchase.findMany({
        where,
        include: {
          user: {
            select: { id: true, email: true, name: true },
          },
        },
        orderBy: { createdAt: "desc" },
        take: limit,
        skip: offset,
      }),
      this.prisma.purchase.count({ where }),
    ]);

    return { purchases, total };
  }

  /**
   * Get payment by ID with full details
   */
  async getPaymentById(paymentId: string) {
    const payment = await this.prisma.payment.findUnique({
      where: { id: paymentId },
      include: {
        subscription: {
          include: {
            user: {
              select: {
                id: true,
                email: true,
                name: true,
                isPremium: true,
              },
            },
            plan: true,
          },
        },
      },
    });

    if (!payment) return null;

    // Get related transactions for audit trail
    const transactions = await this.prisma.paymentTransaction.findMany({
      where: {
        orderId: payment.subscriptionId,
        orderType: "subscription",
      },
      orderBy: { createdAt: "desc" },
      take: 10,
    });

    return {
      ...payment,
      transactions,
    };
  }

  /**
   * Get all subscriptions with filtering
   */
  async getSubscriptions(params: {
    status?: string;
    planId?: string;
    limit?: number;
    offset?: number;
  }) {
    const { status, planId, limit = 50, offset = 0 } = params;

    const where: {
      status?: string;
      planId?: string;
    } = {};

    if (status) where.status = status;
    if (planId) where.planId = planId;

    const [subscriptions, total] = await Promise.all([
      this.prisma.subscription.findMany({
        where,
        include: {
          user: {
            select: { id: true, email: true, name: true, isPremium: true },
          },
          plan: {
            select: {
              id: true,
              name: true,
              slug: true,
              type: true,
              priceMonthly: true,
            },
          },
          _count: {
            select: { payments: true },
          },
        },
        orderBy: { createdAt: "desc" },
        take: limit,
        skip: offset,
      }),
      this.prisma.subscription.count({ where }),
    ]);

    return {
      subscriptions: subscriptions.map((s) => ({
        id: s.id,
        status: s.status,
        startDate: s.startDate,
        endDate: s.endDate,
        autoRenew: s.autoRenew,
        createdAt: s.createdAt,
        user: s.user,
        plan: s.plan,
        paymentsCount: s._count.payments,
      })),
      total,
    };
  }

  /**
   * Get subscription by ID with payment history
   */
  async getSubscriptionById(subscriptionId: string) {
    const subscription = await this.prisma.subscription.findUnique({
      where: { id: subscriptionId },
      include: {
        user: {
          select: {
            id: true,
            email: true,
            name: true,
            isPremium: true,
            createdAt: true,
          },
        },
        plan: true,
        payments: {
          orderBy: { createdAt: "desc" },
          take: 20,
        },
      },
    });

    return subscription;
  }

  /**
   * Extend subscription manually (admin action)
   */
  async extendSubscription(subscriptionId: string, days: number) {
    const subscription = await this.prisma.subscription.findUnique({
      where: { id: subscriptionId },
    });

    if (!subscription) {
      throw new Error("Subscription not found");
    }

    // Calculate new end date
    const currentEndDate = subscription.endDate;
    const newEndDate = new Date(currentEndDate);
    newEndDate.setDate(newEndDate.getDate() + days);

    // Update subscription
    const updated = await this.prisma.subscription.update({
      where: { id: subscriptionId },
      data: {
        endDate: newEndDate,
        status: "active", // Reactivate if expired
      },
      include: {
        user: { select: { id: true, email: true, name: true } },
        plan: { select: { id: true, name: true, slug: true } },
      },
    });

    // Update user premium status
    await this.prisma.user.update({
      where: { id: subscription.userId },
      data: { isPremium: true },
    });

    return updated;
  }

  /**
   * Cancel subscription manually (admin action)
   */
  async cancelSubscription(subscriptionId: string) {
    const subscription = await this.prisma.subscription.findUnique({
      where: { id: subscriptionId },
      include: { user: true },
    });

    if (!subscription) {
      throw new Error("Subscription not found");
    }

    if (subscription.status === "cancelled") {
      throw new Error("Subscription is already cancelled");
    }

    // Cancel the subscription
    const updated = await this.prisma.subscription.update({
      where: { id: subscriptionId },
      data: {
        status: "cancelled",
        autoRenew: false,
      },
      include: {
        user: { select: { id: true, email: true, name: true } },
        plan: { select: { id: true, name: true, slug: true } },
      },
    });

    // Check if user has other active subscriptions
    const otherActiveSubscriptions = await this.prisma.subscription.count({
      where: {
        userId: subscription.userId,
        status: "active",
        id: { not: subscriptionId },
      },
    });

    // If no other active subscriptions, remove premium status
    if (otherActiveSubscriptions === 0) {
      await this.prisma.user.update({
        where: { id: subscription.userId },
        data: { isPremium: false },
      });
    }

    return updated;
  }

  /**
   * Refund a payment (admin action)
   */
  async refundPayment(paymentId: string, reason: string) {
    const payment = await this.prisma.payment.findUnique({
      where: { id: paymentId },
      include: {
        subscription: {
          include: { user: true },
        },
      },
    });

    if (!payment) {
      throw new Error("Payment not found");
    }

    if (payment.status === "refunded") {
      throw new Error("Payment is already refunded");
    }

    if (payment.status !== "completed") {
      throw new Error("Only completed payments can be refunded");
    }

    // Update payment status
    const updated = await this.prisma.payment.update({
      where: { id: paymentId },
      data: {
        status: "refunded",
        metadata: {
          ...((payment.metadata as object) || {}),
          refundReason: reason,
          refundedAt: new Date().toISOString(),
        },
      },
    });

    // Log the refund transaction
    await this.prisma.paymentTransaction.create({
      data: {
        orderId: payment.subscriptionId,
        orderType: "subscription",
        provider: payment.provider || "manual",
        amount: payment.amount,
        state: -1, // Refund state
        action: "refund",
        request: { reason },
        response: { success: true },
      },
    });

    return updated;
  }

  /**
   * Get revenue analytics
   */
}
