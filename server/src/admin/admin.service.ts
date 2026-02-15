import { Injectable } from "@nestjs/common";
import { AdminStatsService } from "./admin-stats.service";
import { AdminMetricsService } from "./admin-metrics.service";
import { AdminRetentionService } from "./admin-retention.service";
import { AdminUsersService } from "./admin-users.service";
import { AdminPaymentsService } from "./admin-payments.service";

/**
 * Admin Service Facade
 * Delegates to specialized sub-services for maintainability.
 */
@Injectable()
export class AdminService {
  constructor(
    private readonly stats: AdminStatsService,
    private readonly metrics: AdminMetricsService,
    private readonly retention: AdminRetentionService,
    private readonly users: AdminUsersService,
    private readonly payments: AdminPaymentsService,
  ) {}

  // === Stats ===
  getDashboardStats() {
    return this.stats.getDashboardStats();
  }
  getCourseAnalytics() {
    return this.stats.getCourseAnalytics();
  }
  getTaskAnalytics() {
    return this.stats.getTaskAnalytics();
  }
  getSubmissionStats() {
    return this.stats.getSubmissionStats();
  }
  getSubscriptionStats() {
    return this.stats.getSubscriptionStats();
  }
  getAiUsageStats() {
    return this.stats.getAiUsageStats();
  }

  // === Metrics ===
  getRevenueAnalytics() {
    return this.metrics.getRevenueAnalytics();
  }
  getSubscriptionPlans() {
    return this.metrics.getSubscriptionPlans();
  }
  getAnalyticsTimeline(days?: number) {
    return this.metrics.getAnalyticsTimeline(days);
  }

  // === Retention ===
  getRetentionMetrics() {
    return this.retention.getRetentionMetrics();
  }
  getConversionMetrics() {
    return this.retention.getConversionMetrics();
  }
  getDayDetails(date: string, metric: string) {
    return this.retention.getDayDetails(date, metric);
  }

  // === Users ===
  searchUsers(query: string, limit?: number) {
    return this.users.searchUsers(query, limit);
  }
  getUserById(userId: string) {
    return this.users.getUserById(userId);
  }
  banUser(userId: string, reason: string, adminId: string) {
    return this.users.banUser(userId, reason, adminId);
  }
  unbanUser(userId: string, adminId: string) {
    return this.users.unbanUser(userId, adminId);
  }
  getBannedUsers(limit?: number, offset?: number) {
    return this.users.getBannedUsers(limit, offset);
  }

  // === Payments ===
  getPayments(params: {
    status?: string;
    provider?: string;
    limit?: number;
    offset?: number;
  }) {
    return this.payments.getPayments(params);
  }
  getPurchases(params: {
    status?: string;
    type?: string;
    limit?: number;
    offset?: number;
  }) {
    return this.payments.getPurchases(params);
  }
  getPaymentById(paymentId: string) {
    return this.payments.getPaymentById(paymentId);
  }
  getSubscriptions(params: {
    status?: string;
    planId?: string;
    limit?: number;
    offset?: number;
  }) {
    return this.payments.getSubscriptions(params);
  }
  getSubscriptionById(subscriptionId: string) {
    return this.payments.getSubscriptionById(subscriptionId);
  }
  extendSubscription(subscriptionId: string, days: number) {
    return this.payments.extendSubscription(subscriptionId, days);
  }
  cancelSubscription(subscriptionId: string) {
    return this.payments.cancelSubscription(subscriptionId);
  }
  refundPayment(paymentId: string, reason: string) {
    return this.payments.refundPayment(paymentId, reason);
  }
}
