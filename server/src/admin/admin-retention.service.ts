import { Injectable } from "@nestjs/common";
import { PrismaService } from "../prisma/prisma.service";

@Injectable()
export class AdminRetentionService {
  constructor(private readonly prisma: PrismaService) {}

  async getRetentionMetrics() {
    const now = new Date();

    // Get users who registered 1, 7, and 30 days ago (with some buffer)
    const d1Date = new Date(now.getTime() - 1 * 24 * 60 * 60 * 1000);
    const d7Date = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
    const d30Date = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);

    // D1 Retention: Users registered 2 days ago who were active yesterday
    const d1CohortStart = new Date(now.getTime() - 2 * 24 * 60 * 60 * 1000);
    const d1CohortEnd = new Date(now.getTime() - 1 * 24 * 60 * 60 * 1000);

    const d1Cohort = await this.prisma.user.count({
      where: {
        createdAt: {
          gte: new Date(d1CohortStart.toISOString().split("T")[0]),
          lt: new Date(d1CohortEnd.toISOString().split("T")[0]),
        },
      },
    });

    const d1Retained = await this.prisma.user.count({
      where: {
        createdAt: {
          gte: new Date(d1CohortStart.toISOString().split("T")[0]),
          lt: new Date(d1CohortEnd.toISOString().split("T")[0]),
        },
        lastActivityAt: {
          gte: d1Date,
        },
      },
    });

    // D7 Retention: Users registered 8 days ago who were active in last 7 days
    const d7CohortStart = new Date(now.getTime() - 8 * 24 * 60 * 60 * 1000);
    const d7CohortEnd = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);

    const d7Cohort = await this.prisma.user.count({
      where: {
        createdAt: {
          gte: new Date(d7CohortStart.toISOString().split("T")[0]),
          lt: new Date(d7CohortEnd.toISOString().split("T")[0]),
        },
      },
    });

    const d7Retained = await this.prisma.user.count({
      where: {
        createdAt: {
          gte: new Date(d7CohortStart.toISOString().split("T")[0]),
          lt: new Date(d7CohortEnd.toISOString().split("T")[0]),
        },
        lastActivityAt: {
          gte: d7Date,
        },
      },
    });

    // D30 Retention: Users registered 31 days ago who were active in last 30 days
    const d30CohortStart = new Date(now.getTime() - 31 * 24 * 60 * 60 * 1000);
    const d30CohortEnd = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);

    const d30Cohort = await this.prisma.user.count({
      where: {
        createdAt: {
          gte: new Date(d30CohortStart.toISOString().split("T")[0]),
          lt: new Date(d30CohortEnd.toISOString().split("T")[0]),
        },
      },
    });

    const d30Retained = await this.prisma.user.count({
      where: {
        createdAt: {
          gte: new Date(d30CohortStart.toISOString().split("T")[0]),
          lt: new Date(d30CohortEnd.toISOString().split("T")[0]),
        },
        lastActivityAt: {
          gte: d30Date,
        },
      },
    });

    return {
      d1: {
        cohortSize: d1Cohort,
        retained: d1Retained,
        rate: d1Cohort > 0 ? Math.round((d1Retained / d1Cohort) * 100) : 0,
      },
      d7: {
        cohortSize: d7Cohort,
        retained: d7Retained,
        rate: d7Cohort > 0 ? Math.round((d7Retained / d7Cohort) * 100) : 0,
      },
      d30: {
        cohortSize: d30Cohort,
        retained: d30Retained,
        rate: d30Cohort > 0 ? Math.round((d30Retained / d30Cohort) * 100) : 0,
      },
    };
  }

  /**
   * Get conversion metrics
   * - Free to Paid conversion rate
   * - Trial to Subscription conversion
   */
  async getConversionMetrics() {
    // Total users
    const totalUsers = await this.prisma.user.count();

    // Users with any paid subscription (ever)
    const usersWithSubscription = await this.prisma.subscription.groupBy({
      by: ["userId"],
      where: {
        OR: [
          { status: "active" },
          { status: "cancelled" },
          { status: "expired" },
        ],
      },
    });

    const paidUsers = usersWithSubscription.length;

    // Users with any purchase
    const usersWithPurchase = await this.prisma.purchase.groupBy({
      by: ["userId"],
      where: { status: "completed" },
    });

    const purchaseUsers = usersWithPurchase.length;

    // Combined paying users (subscription OR purchase)
    const allPayingUserIds = new Set([
      ...usersWithSubscription.map((u) => u.userId),
      ...usersWithPurchase.map((u) => u.userId),
    ]);

    const totalPayingUsers = allPayingUserIds.size;

    // Currently active premium users
    const activePremium = await this.prisma.user.count({
      where: { isPremium: true },
    });

    // Monthly conversion (last 30 days)
    const thirtyDaysAgo = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000);

    const newUsersLast30Days = await this.prisma.user.count({
      where: { createdAt: { gte: thirtyDaysAgo } },
    });

    const newPaidUsersLast30Days = await this.prisma.subscription.groupBy({
      by: ["userId"],
      where: {
        createdAt: { gte: thirtyDaysAgo },
        status: { in: ["active", "cancelled", "expired"] },
      },
    });

    return {
      overall: {
        totalUsers,
        totalPayingUsers,
        conversionRate:
          totalUsers > 0
            ? Math.round((totalPayingUsers / totalUsers) * 10000) / 100
            : 0,
      },
      subscriptions: {
        totalWithSubscription: paidUsers,
        conversionRate:
          totalUsers > 0
            ? Math.round((paidUsers / totalUsers) * 10000) / 100
            : 0,
      },
      purchases: {
        totalWithPurchase: purchaseUsers,
        conversionRate:
          totalUsers > 0
            ? Math.round((purchaseUsers / totalUsers) * 10000) / 100
            : 0,
      },
      currentPremium: {
        count: activePremium,
        percentage:
          totalUsers > 0
            ? Math.round((activePremium / totalUsers) * 10000) / 100
            : 0,
      },
      monthly: {
        newUsers: newUsersLast30Days,
        newPaidUsers: newPaidUsersLast30Days.length,
        conversionRate:
          newUsersLast30Days > 0
            ? Math.round(
                (newPaidUsersLast30Days.length / newUsersLast30Days) * 10000,
              ) / 100
            : 0,
      },
    };
  }

  /**
   * Get detailed breakdown for a specific day and metric (drill-down)
   */
  async getDayDetails(date: string, metric: string) {
    const startOfDay = new Date(date);
    startOfDay.setHours(0, 0, 0, 0);
    const endOfDay = new Date(date);
    endOfDay.setHours(23, 59, 59, 999);

    const details: Array<{
      id: string;
      label: string;
      value: string | number;
      sublabel?: string;
      status?: "success" | "warning" | "error" | "info";
    }> = [];

    let total = 0;

    switch (metric) {
      case "dau": {
        // Get users who had activity on this day
        const activeUsers = await this.prisma.submission.groupBy({
          by: ["userId"],
          where: {
            createdAt: { gte: startOfDay, lte: endOfDay },
          },
        });
        total = activeUsers.length;

        // Get top active users for this day
        const topUsers = await this.prisma.submission.groupBy({
          by: ["userId"],
          where: {
            createdAt: { gte: startOfDay, lte: endOfDay },
          },
          _count: { id: true },
          orderBy: { _count: { id: "desc" } },
          take: 10,
        });

        const userIds = topUsers.map((u) => u.userId);
        const users = await this.prisma.user.findMany({
          where: { id: { in: userIds } },
          select: { id: true, email: true, name: true },
        });

        const userMap = new Map(users.map((u) => [u.id, u]));

        for (const item of topUsers) {
          const user = userMap.get(item.userId);
          details.push({
            id: item.userId,
            label: user?.name || "Unknown",
            sublabel: user?.email,
            value: `${item._count.id} submissions`,
            status: "info",
          });
        }
        break;
      }

      case "newUsers": {
        const newUsers = await this.prisma.user.findMany({
          where: {
            createdAt: { gte: startOfDay, lte: endOfDay },
          },
          select: { id: true, email: true, name: true, isPremium: true },
          orderBy: { createdAt: "desc" },
          take: 20,
        });
        total = newUsers.length;

        for (const user of newUsers) {
          details.push({
            id: user.id,
            label: user.name,
            sublabel: user.email,
            value: user.isPremium ? "Premium" : "Free",
            status: user.isPremium ? "success" : "info",
          });
        }
        break;
      }

      case "revenue": {
        const payments = await this.prisma.payment.findMany({
          where: {
            createdAt: { gte: startOfDay, lte: endOfDay },
            status: "completed",
          },
          include: {
            subscription: {
              include: { plan: true, user: true },
            },
          },
          orderBy: { amount: "desc" },
        });

        total = payments.reduce((sum, p) => sum + p.amount, 0);

        for (const payment of payments) {
          details.push({
            id: payment.id,
            label: payment.subscription?.user?.name || "Unknown",
            sublabel: payment.subscription?.plan?.name || "Unknown Plan",
            value: `${(payment.amount / 100).toLocaleString()} UZS`,
            status: "success",
          });
        }
        break;
      }

      case "payments": {
        const payments = await this.prisma.payment.findMany({
          where: {
            createdAt: { gte: startOfDay, lte: endOfDay },
          },
          include: {
            subscription: {
              include: { plan: true, user: true },
            },
          },
          orderBy: { createdAt: "desc" },
        });

        total = payments.length;

        for (const payment of payments) {
          const statusMap: Record<string, "success" | "warning" | "error"> = {
            completed: "success",
            pending: "warning",
            failed: "error",
            refunded: "warning",
          };

          details.push({
            id: payment.id,
            label: payment.subscription?.user?.name || "Unknown",
            sublabel: `${(payment.amount / 100).toLocaleString()} UZS`,
            value: payment.status,
            status: statusMap[payment.status] || "info",
          });
        }
        break;
      }

      case "subscriptions": {
        const newSubs = await this.prisma.subscription.findMany({
          where: {
            createdAt: { gte: startOfDay, lte: endOfDay },
          },
          include: {
            plan: true,
            user: { select: { id: true, email: true, name: true } },
          },
          orderBy: { createdAt: "desc" },
        });

        total = newSubs.length;

        for (const sub of newSubs) {
          details.push({
            id: sub.id,
            label: sub.user.name,
            sublabel: sub.plan.name,
            value: sub.status,
            status: sub.status === "active" ? "success" : "warning",
          });
        }
        break;
      }
    }

    return {
      date,
      metric,
      total,
      details,
    };
  }
}
