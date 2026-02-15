import React from 'react';
import { PaymentHistoryItem } from '../../api/paymentService';
import { IconCheck, IconX } from '@/components/Icons';

interface PaymentHistoryTabProps {
  history: PaymentHistoryItem[];
  language: string;
}

const formatPrice = (amountInTiyn: number): string => {
  const uzs = amountInTiyn / 100;
  return new Intl.NumberFormat('uz-UZ').format(uzs);
};

const formatDate = (isoDate: string | undefined): string => {
  if (!isoDate) return 'N/A';
  try {
    return new Date(isoDate).toLocaleDateString('ru-RU', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    });
  } catch {
    return isoDate;
  }
};

const PaymentHistoryTab: React.FC<PaymentHistoryTabProps> = ({ history, language }) => {
  return (
    <div className="bg-white dark:bg-dark-surface rounded-2xl border border-gray-200 dark:border-dark-border overflow-hidden">
      <div className="p-6 border-b border-gray-100 dark:border-dark-border">
        <h3 className="text-lg font-bold text-gray-900 dark:text-white">
          {language === 'ru' ? 'История платежей' : 'Payment History'}
        </h3>
      </div>
      <div className="overflow-x-auto">
        <table data-testid="payment-history-table" className="w-full text-left text-sm">
          <thead className="bg-gray-50 dark:bg-dark-bg text-gray-500 uppercase font-bold text-xs">
            <tr>
              <th className="px-6 py-4">{language === 'ru' ? 'Дата' : 'Date'}</th>
              <th className="px-6 py-4">{language === 'ru' ? 'Описание' : 'Description'}</th>
              <th className="px-6 py-4">{language === 'ru' ? 'Сумма' : 'Amount'}</th>
              <th className="px-6 py-4">{language === 'ru' ? 'Статус' : 'Status'}</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100 dark:divide-dark-border">
            {history.length === 0 ? (
              <tr data-testid="history-empty">
                <td colSpan={4} className="px-6 py-8 text-center text-gray-500">
                  {language === 'ru' ? 'История платежей пуста' : 'No payment history'}
                </td>
              </tr>
            ) : (
              history.map((item) => (
                <tr
                  key={item.id}
                  data-testid="payment-history-row"
                  className="hover:bg-gray-50 dark:hover:bg-dark-bg/50 transition-colors"
                >
                  <td className="px-6 py-4 text-gray-900 dark:text-white">
                    {formatDate(item.createdAt)}
                  </td>
                  <td className="px-6 py-4 text-gray-500">{item.description}</td>
                  <td className="px-6 py-4 font-bold text-gray-900 dark:text-white">
                    {formatPrice(item.amount)} {item.currency}
                  </td>
                  <td className="px-6 py-4">
                    <span
                      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-bold uppercase tracking-wide ${
                        item.status === 'completed'
                          ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                          : item.status === 'pending'
                            ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400'
                            : 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
                      }`}
                    >
                      {item.status === 'completed' ? (
                        <IconCheck className="w-3 h-3" />
                      ) : (
                        <IconX className="w-3 h-3" />
                      )}
                      {item.status}
                    </span>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default PaymentHistoryTab;
