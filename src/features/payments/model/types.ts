/**
 * Payment feature types
 */

export type PaymentStatus = 'pending' | 'completed' | 'failed' | 'refunded';

export interface PaymentHistoryItem {
  id: string;
  type: 'subscription' | 'purchase';
  description: string;
  amount: number;
  currency: string;
  status: string;
  provider?: string;
  createdAt: string;
}

export interface PaymentProvider {
  id: string;
  name: string;
  configured: boolean;
}

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
