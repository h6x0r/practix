import { Injectable } from "@nestjs/common";
import { PrismaService } from "../prisma/prisma.service";

@Injectable()
export class AdminMetricsService {
  constructor(private readonly prisma: PrismaService) {}

  async getRevenueAnalytics() {
    const now = new Date();
    const firstDayOfMonth = new Date(now.getFullYear(), now.getMonth(), 1);
    const firstDayOfLastMonth = new Date(
      now.getFullYear(),
      now.getMonth() - 1,
      1,
    );
    const thirtyDaysAgo = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);

    // This month's revenue
    const thisMonthRevenue = await this.prisma.payment.aggregate({
      where: {
        status: "completed",
        createdAt: { gte: firstDayOfMonth },
      },
      _sum: { amount: true },
      _count: true,
    });

    // Last month's revenue
    const lastMonthRevenue = await this.prisma.payment.aggregate({
      where: {
        status: "completed",
        createdAt: {
          gte: firstDayOfLastMonth,
          lt: firstDayOfMonth,
        },
      },
      _sum: { amount: true },
      _count: true,
    });

    // Total revenue
    const totalRevenue = await this.prisma.payment.aggregate({
      where: { status: "completed" },
      _sum: { amount: true },
      _count: true,
    });

    // Revenue by provider
    const revenueByProvider = await this.prisma.payment.groupBy({
      by: ["provider"],
      where: { status: "completed" },
      _sum: { amount: true },
      _count: true,
    });

    // Refunded payments
    const refundedPayments = await this.prisma.payment.aggregate({
      where: { status: "refunded" },
      _sum: { amount: true },
      _count: true,
    });

    // Daily revenue for chart (last 30 days)
    const dailyPayments = await this.prisma.payment.findMany({
      where: {
        status: "completed",
        createdAt: { gte: thirtyDaysAgo },
      },
      select: {
        amount: true,
        createdAt: true,
      },
    });

    // Group by day
    const dailyRevenue: { [key: string]: number } = {};
    dailyPayments.forEach((p) => {
      const date = p.createdAt.toISOString().split("T")[0];
      dailyRevenue[date] = (dailyRevenue[date] || 0) + p.amount;
    });

    // Convert to sorted array
    const dailyRevenueArray = Object.entries(dailyRevenue)
      .map(([date, amount]) => ({ date, amount }))
      .sort((a, b) => a.date.localeCompare(b.date));

    // Purchase revenue (one-time purchases)
    const purchaseRevenue = await this.prisma.purchase.aggregate({
      where: { status: "completed" },
      _sum: { amount: true },
      _count: true,
    });

    return {
      thisMonth: {
        revenue: thisMonthRevenue._sum.amount || 0,
        count: thisMonthRevenue._count,
      },
      lastMonth: {
        revenue: lastMonthRevenue._sum.amount || 0,
        count: lastMonthRevenue._count,
      },
      total: {
        revenue: totalRevenue._sum.amount || 0,
        count: totalRevenue._count,
      },
      byProvider: revenueByProvider.map((p) => ({
        provider: p.provider || "unknown",
        revenue: p._sum.amount || 0,
        count: p._count,
      })),
      refunded: {
        amount: refundedPayments._sum.amount || 0,
        count: refundedPayments._count,
      },
      purchases: {
        revenue: purchaseRevenue._sum.amount || 0,
        count: purchaseRevenue._count,
      },
      dailyRevenue: dailyRevenueArray,
    };
  }

  /**
   * Get all subscription plans
   */
  async getSubscriptionPlans() {
    const plans = await this.prisma.subscriptionPlan.findMany({
      include: {
        _count: {
          select: { subscriptions: true },
        },
        course: {
          select: { id: true, title: true, slug: true },
        },
      },
      orderBy: { createdAt: "asc" },
    });

    return plans.map((p) => ({
      id: p.id,
      slug: p.slug,
      name: p.name,
      nameRu: p.nameRu,
      type: p.type,
      priceMonthly: p.priceMonthly,
      currency: p.currency,
      isActive: p.isActive,
      course: p.course,
      subscriptionsCount: p._count.subscriptions,
      createdAt: p.createdAt,
    }));
  }

  /**
   * Get analytics timeline data
   * Returns: DAU, new users, and revenue per day for the last N days
   */
  async getAnalyticsTimeline(days: number = 30) {
    const now = new Date();
    const startDate = new Date(now.getTime() - days * 24 * 60 * 60 * 1000);

    // Generate all dates in range
    const dates: string[] = [];
    for (let d = new Date(startDate); d <= now; d.setDate(d.getDate() + 1)) {
      dates.push(d.toISOString().split("T")[0]);
    }

    // Get new users per day
    const newUsersRaw = await this.prisma.user.groupBy({
      by: ["createdAt"],
      where: {
        createdAt: { gte: startDate },
      },
      _count: { id: true },
    });

    const newUsersMap = new Map<string, number>();
    newUsersRaw.forEach((u) => {
      const date = u.createdAt.toISOString().split("T")[0];
      newUsersMap.set(date, (newUsersMap.get(date) || 0) + u._count.id);
    });

    // Get active users per day (based on lastActivityAt)
    const activeUsersRaw = await this.prisma.user.groupBy({
      by: ["lastActivityAt"],
      where: {
        lastActivityAt: { gte: startDate, not: null },
      },
      _count: { id: true },
    });

    const activeUsersMap = new Map<string, number>();
    activeUsersRaw.forEach((u) => {
      if (u.lastActivityAt) {
        const date = u.lastActivityAt.toISOString().split("T")[0];
        activeUsersMap.set(date, (activeUsersMap.get(date) || 0) + u._count.id);
      }
    });

    // Get revenue per day (completed payments)
    const revenueRaw = await this.prisma.payment.groupBy({
      by: ["createdAt"],
      where: {
        status: "completed",
        createdAt: { gte: startDate },
      },
      _sum: { amount: true },
      _count: { id: true },
    });

    const revenueMap = new Map<string, { amount: number; count: number }>();
    revenueRaw.forEach((p) => {
      const date = p.createdAt.toISOString().split("T")[0];
      const existing = revenueMap.get(date) || { amount: 0, count: 0 };
      revenueMap.set(date, {
        amount: existing.amount + (p._sum.amount || 0),
        count: existing.count + p._count.id,
      });
    });

    // Get new subscriptions per day
    const newSubsRaw = await this.prisma.subscription.groupBy({
      by: ["createdAt"],
      where: {
        createdAt: { gte: startDate },
      },
      _count: { id: true },
    });

    const newSubsMap = new Map<string, number>();
    newSubsRaw.forEach((s) => {
      const date = s.createdAt.toISOString().split("T")[0];
      newSubsMap.set(date, (newSubsMap.get(date) || 0) + s._count.id);
    });

    // Build timeline data
    const timeline = dates.map((date) => {
      const revenue = revenueMap.get(date) || { amount: 0, count: 0 };
      return {
        date,
        dau: activeUsersMap.get(date) || 0,
        newUsers: newUsersMap.get(date) || 0,
        revenue: revenue.amount,
        payments: revenue.count,
        newSubscriptions: newSubsMap.get(date) || 0,
      };
    });

    // Calculate totals and averages
    const totalNewUsers = timeline.reduce((sum, d) => sum + d.newUsers, 0);
    const totalRevenue = timeline.reduce((sum, d) => sum + d.revenue, 0);
    const avgDau = Math.round(
      timeline.reduce((sum, d) => sum + d.dau, 0) / timeline.length,
    );
    const totalPayments = timeline.reduce((sum, d) => sum + d.payments, 0);
    const totalNewSubs = timeline.reduce(
      (sum, d) => sum + d.newSubscriptions,
      0,
    );

    return {
      timeline,
      summary: {
        totalNewUsers,
        totalRevenue,
        avgDau,
        totalPayments,
        totalNewSubscriptions: totalNewSubs,
        period: days,
      },
    };
  }

  /**
   * Get retention metrics (D1, D7, D30)
   * Calculates what percentage of users who registered N days ago
   * returned to the platform (had activity) after D1/D7/D30 days
   */
}
