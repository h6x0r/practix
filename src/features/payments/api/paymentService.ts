import { api } from '@/lib/api';

export interface PaymentProvider {
  id: string;
  name: string;
  configured: boolean;
}

export interface PurchasePricing {
  type: string;
  price: number;
  name: string;
  nameRu: string;
  priceFormatted: string;
}

export interface RoadmapCredits {
  used: number;
  available: number;
  canGenerate: boolean;
}

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

export interface CheckoutRequest {
  orderType: 'subscription' | 'purchase';
  planId?: string;
  purchaseType?: 'roadmap_generation' | 'ai_credits';
  quantity?: number;
  provider: 'payme' | 'click';
  returnUrl?: string;
}

export interface CheckoutResponse {
  orderId: string;
  paymentUrl: string;
  amount: number;
  currency: string;
  provider: string;
}

export interface PaymentStatus {
  status: string;
  orderType: 'subscription' | 'purchase' | null;
  amount?: number;
}

export const paymentService = {
  /**
   * Get available payment providers
   */
  getProviders: async (): Promise<PaymentProvider[]> => {
    return api.get<PaymentProvider[]>('/payments/providers');
  },

  /**
   * Get pricing for one-time purchases
   */
  getPricing: async (): Promise<PurchasePricing[]> => {
    return api.get<PurchasePricing[]>('/payments/pricing');
  },

  /**
   * Get user's roadmap credits
   */
  getRoadmapCredits: async (): Promise<RoadmapCredits> => {
    return api.get<RoadmapCredits>('/payments/roadmap-credits');
  },

  /**
   * Get payment history
   */
  getPaymentHistory: async (): Promise<PaymentHistoryItem[]> => {
    return api.get<PaymentHistoryItem[]>('/payments/history');
  },

  /**
   * Check payment status
   */
  getPaymentStatus: async (orderId: string): Promise<PaymentStatus> => {
    return api.get<PaymentStatus>(`/payments/status/${orderId}`);
  },

  /**
   * Create checkout session
   * Returns payment URL for redirect
   */
  createCheckout: async (request: CheckoutRequest): Promise<CheckoutResponse> => {
    return api.post<CheckoutResponse>('/payments/checkout', request);
  },
};
