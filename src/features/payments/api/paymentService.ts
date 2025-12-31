import { paymentRepository } from '../data/repository';
import { PaymentHistoryItem } from '../model/types';

export const paymentService = {
  getPaymentHistory: async (): Promise<PaymentHistoryItem[]> => {
    return new Promise<PaymentHistoryItem[]>(async (resolve) => {
      const history = await paymentRepository.getHistory();
      setTimeout(() => resolve(history), 300);
    });
  }
};
