import { PaymentHistoryItem } from '../model/types';

export const PAYMENT_HISTORY: PaymentHistoryItem[] = [
  { id: 'inv_001', date: '2023-10-01', amount: 1900, status: 'paid', description: 'Pro Plan (Monthly)' },
  { id: 'inv_002', date: '2023-09-01', amount: 1900, status: 'paid', description: 'Pro Plan (Monthly)' },
  { id: 'inv_003', date: '2023-08-01', amount: 1900, status: 'failed', description: 'Pro Plan (Monthly)' },
];

export const paymentRepository = {
  getHistory: async (): Promise<PaymentHistoryItem[]> => {
    return PAYMENT_HISTORY;
  }
};
