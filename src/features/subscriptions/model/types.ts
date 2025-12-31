export interface SubscriptionPlan {
  id: string;
  slug: string;
  name: string;
  nameRu?: string;
  type: 'global' | 'course';
  courseId?: string;
  course?: {
    id: string;
    slug: string;
    title: string;
    icon: string;
  };
  priceMonthly: number;
  currency: string;
  isActive: boolean;
}

export interface Subscription {
  id: string;
  userId: string;
  planId: string;
  plan: SubscriptionPlan;
  status: 'active' | 'cancelled' | 'expired' | 'pending';
  startDate: string;
  endDate: string;
  autoRenew: boolean;
  createdAt: string;
  updatedAt: string;
}

export interface TaskAccess {
  canView: boolean;
  canRun: boolean;
  canSubmit: boolean;
  canSeeSolution: boolean;
  canUseAiTutor: boolean;
  queuePriority: number; // 1 = high (premium), 10 = low (free)
}

export interface CourseAccess {
  hasAccess: boolean;
  queuePriority: number;
  canUseAiTutor: boolean;
}
