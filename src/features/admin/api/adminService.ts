
import { api } from '@/lib/api';

// Dashboard Stats (matches backend response)
export interface DashboardStats {
  totalUsers: number;
  newUsers: number;
  premiumUsers: number;
  activeUsers: {
    daily: number;
    weekly: number;
    monthly: number;
  };
}

// Course Analytics (matches backend response)
export interface CourseAnalyticsItem {
  courseId: string;
  courseSlug: string;
  courseTitle: string;
  category: string;
  totalEnrolled: number;
  completed: number;
  completionRate: number;
  averageProgress: number;
}

export interface CourseAnalyticsResponse {
  courses: CourseAnalyticsItem[];
  totalCourses: number;
}

// Task Analytics (matches backend response)
export interface TaskAnalyticsItem {
  taskId: string;
  taskSlug: string;
  taskTitle: string;
  difficulty: string;
  isPremium: boolean;
  totalSubmissions: number;
  acceptedSubmissions: number;
  uniqueUsers: number;
  passRate: number;
}

export interface TaskAnalyticsResponse {
  hardestTasks: TaskAnalyticsItem[];
  mostPopularTasks: TaskAnalyticsItem[];
  totalTasks: number;
  tasksWithSubmissions: number;
}

// Submission Stats (matches backend response)
export interface SubmissionStatusStat {
  status: string;
  count: number;
  percentage: number;
}

export interface SubmissionStatsResponse {
  totalSubmissions: number;
  recentSubmissions: number;
  byStatus: SubmissionStatusStat[];
  dailySubmissions: Record<string, number>;
}

// Subscription Stats
export interface SubscriptionStats {
  totalPremium: number;
  totalFree: number;
  monthlyRevenue: number;
  conversionRate: number;
  churnRate: number;
}

// AI Usage Stats
export interface AiUsageStats {
  totalQueries: number;
  avgQueriesPerUser: number;
  topicBreakdown: Array<{
    topic: string;
    count: number;
  }>;
  dailyUsage: Array<{
    date: string;
    count: number;
  }>;
}

/**
 * Admin Analytics Service - Connected to Backend Admin API
 *
 * All endpoints require admin role authentication
 */
export const adminService = {
  /**
   * Get dashboard overview stats
   * GET /admin/analytics/dashboard
   */
  getDashboardStats: async (): Promise<DashboardStats> => {
    return await api.get<DashboardStats>('/admin/analytics/dashboard');
  },

  /**
   * Get course analytics
   * GET /admin/analytics/courses
   */
  getCourseAnalytics: async (): Promise<CourseAnalyticsResponse> => {
    return await api.get<CourseAnalyticsResponse>('/admin/analytics/courses');
  },

  /**
   * Get task analytics
   * GET /admin/analytics/tasks
   */
  getTaskAnalytics: async (): Promise<TaskAnalyticsResponse> => {
    return await api.get<TaskAnalyticsResponse>('/admin/analytics/tasks');
  },

  /**
   * Get submission statistics
   * GET /admin/analytics/submissions
   */
  getSubmissionStats: async (): Promise<SubmissionStatsResponse> => {
    return await api.get<SubmissionStatsResponse>('/admin/analytics/submissions');
  },

  /**
   * Get subscription statistics
   * GET /admin/analytics/subscriptions
   */
  getSubscriptionStats: async (): Promise<SubscriptionStats> => {
    return await api.get<SubscriptionStats>('/admin/analytics/subscriptions');
  },

  /**
   * Get AI usage statistics
   * GET /admin/analytics/ai-usage
   */
  getAiUsageStats: async (): Promise<AiUsageStats> => {
    return await api.get<AiUsageStats>('/admin/analytics/ai-usage');
  },
};
