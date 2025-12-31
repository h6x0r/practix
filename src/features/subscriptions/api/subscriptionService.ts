import { api } from '@/lib/api';
import { SubscriptionPlan, Subscription, TaskAccess, CourseAccess } from '../model/types';

export const subscriptionService = {
  /**
   * Get all available subscription plans
   */
  getPlans: async (): Promise<SubscriptionPlan[]> => {
    return api.get<SubscriptionPlan[]>('/subscriptions/plans');
  },

  /**
   * Get a specific plan by slug
   */
  getPlanBySlug: async (slug: string): Promise<SubscriptionPlan> => {
    return api.get<SubscriptionPlan>(`/subscriptions/plans/${slug}`);
  },

  /**
   * Get current user's subscriptions
   */
  getMySubscriptions: async (): Promise<Subscription[]> => {
    return api.get<Subscription[]>('/subscriptions/my');
  },

  /**
   * Get access info for a specific course
   */
  getCourseAccess: async (courseId: string): Promise<CourseAccess> => {
    return api.get<CourseAccess>(`/subscriptions/access/course/${courseId}`);
  },

  /**
   * Get access info for a specific task
   */
  getTaskAccess: async (taskId: string): Promise<TaskAccess> => {
    return api.get<TaskAccess>(`/subscriptions/access/task/${taskId}`);
  },

  /**
   * Create a subscription (after payment)
   */
  createSubscription: async (planId: string, autoRenew: boolean = true): Promise<Subscription> => {
    return api.post<Subscription>('/subscriptions', { planId, autoRenew });
  },

  /**
   * Cancel a subscription
   */
  cancelSubscription: async (subscriptionId: string): Promise<Subscription> => {
    return api.delete<Subscription>(`/subscriptions/${subscriptionId}`);
  },
};
