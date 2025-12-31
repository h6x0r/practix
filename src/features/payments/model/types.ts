/**
 * Payment feature types
 */

export type PaymentStatus = 'paid' | 'failed' | 'pending' | 'refunded';

export interface PaymentHistoryItem {
  id: string;
  date: string;
  amount: number; // Amount in cents
  status: PaymentStatus;
  description: string;
}
