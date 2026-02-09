// Dashboard Stats
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

// Course Analytics
export interface CourseAnalyticsItem {
  courseId: string;
  courseSlug: string;
  courseTitle: string;
  category: string;
  totalEnrolled: number;
  completed: number;
  completionRate: number;
  averageProgress: number;
  translations?: Record<
    string,
    { title?: string; description?: string }
  > | null;
}

export interface CourseAnalyticsResponse {
  courses: CourseAnalyticsItem[];
  totalCourses: number;
}

// Task Analytics
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

// Submission Stats
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
export interface PlanStat {
  planId: string;
  planName: string;
  planSlug: string;
  planType: string;
  count: number;
  monthlyRevenue: number;
}

export interface SubscriptionStats {
  activeSubscriptions: number;
  newSubscriptionsThisMonth: number;
  byPlan: PlanStat[];
  totalMonthlyRevenue: number;
  completedPayments: number;
  totalRevenue: number;
}

// Analytics Timeline
export interface TimelineDataPoint {
  date: string;
  dau: number;
  newUsers: number;
  revenue: number;
  payments: number;
  newSubscriptions: number;
}

export interface AnalyticsTimelineResponse {
  timeline: TimelineDataPoint[];
  summary: {
    totalNewUsers: number;
    totalRevenue: number;
    avgDau: number;
    totalPayments: number;
    totalNewSubscriptions: number;
    period: number;
  };
}

// AI Usage Stats
export interface AiUsageStats {
  totalQueries: number;
  avgQueriesPerUser: number;
  topicBreakdown: Array<{ topic: string; count: number }>;
  dailyUsage: Array<{ date: string; count: number }>;
}

// Retention Metrics
export interface RetentionCohort {
  cohortSize: number;
  retained: number;
  rate: number;
}

export interface RetentionMetrics {
  d1: RetentionCohort;
  d7: RetentionCohort;
  d30: RetentionCohort;
}

// Conversion Metrics
export interface ConversionMetrics {
  overall: {
    totalUsers: number;
    totalPayingUsers: number;
    conversionRate: number;
  };
  subscriptions: {
    totalWithSubscription: number;
    conversionRate: number;
  };
  purchases: {
    totalWithPurchase: number;
    conversionRate: number;
  };
  currentPremium: {
    count: number;
    percentage: number;
  };
  monthly: {
    newUsers: number;
    newPaidUsers: number;
    conversionRate: number;
  };
}

// Day Details (Drill-down)
export interface DayDetailItem {
  id: string;
  label: string;
  value: string | number;
  sublabel?: string;
  status?: "success" | "warning" | "error" | "info";
}

export interface DayDetailsResponse {
  date: string;
  metric: string;
  total: number;
  details: DayDetailItem[];
}
