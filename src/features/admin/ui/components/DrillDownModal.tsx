import React from 'react';
import { useUITranslation } from '@/contexts/LanguageContext';

export interface DrillDownData {
  date: string;
  metric: 'dau' | 'revenue' | 'payments' | 'newUsers' | 'subscriptions';
  value: number;
  details?: DrillDownDetail[];
}

export interface DrillDownDetail {
  id: string;
  label: string;
  value: string | number;
  sublabel?: string;
  status?: 'success' | 'warning' | 'error' | 'info';
}

interface DrillDownModalProps {
  data: DrillDownData | null;
  onClose: () => void;
  loading?: boolean;
}

const METRIC_LABELS: Record<string, string> = {
  dau: 'Daily Active Users',
  revenue: 'Revenue',
  payments: 'Payments',
  newUsers: 'New Users',
  subscriptions: 'New Subscriptions',
};

const METRIC_ICONS: Record<string, string> = {
  dau: '👥',
  revenue: '💰',
  payments: '💳',
  newUsers: '🆕',
  subscriptions: '⭐',
};

const STATUS_COLORS: Record<string, string> = {
  success: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400',
  warning: 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400',
  error: 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400',
  info: 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400',
};

export const DrillDownModal: React.FC<DrillDownModalProps> = ({
  data,
  onClose,
  loading = false,
}) => {
  const { tUI } = useUITranslation();

  if (!data) return null;

  const formattedDate = new Date(data.date).toLocaleDateString('en-US', {
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });

  const formatValue = (value: number, metric: string) => {
    if (metric === 'revenue') {
      const uzs = value / 100;
      return `${uzs.toLocaleString()} UZS`;
    }
    return value.toLocaleString();
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm animate-in fade-in duration-200"
      onClick={onClose}
    >
      <div
        className="bg-white dark:bg-[#161b22] rounded-2xl shadow-2xl border border-gray-200 dark:border-gray-700 w-full max-w-lg mx-4 max-h-[80vh] overflow-hidden transform animate-in zoom-in-95 duration-200"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-5 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3">
            <span className="text-2xl">{METRIC_ICONS[data.metric]}</span>
            <div>
              <h3 className="text-lg font-bold text-gray-900 dark:text-white">
                {METRIC_LABELS[data.metric]}
              </h3>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                {formattedDate}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Main Value */}
        <div className="p-5 bg-gradient-to-r from-brand-500/10 to-purple-500/10 border-b border-gray-200 dark:border-gray-700">
          <div className="text-center">
            <div className="text-4xl font-display font-bold text-brand-600 dark:text-brand-400">
              {formatValue(data.value, data.metric)}
            </div>
            <div className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              {tUI('admin.totalForDay') || 'Total for this day'}
            </div>
          </div>
        </div>

        {/* Details List */}
        <div className="p-5 overflow-y-auto max-h-[40vh]">
          {loading ? (
            <div className="flex items-center justify-center py-8">
              <div className="w-8 h-8 border-2 border-brand-500 border-t-transparent rounded-full animate-spin" />
            </div>
          ) : data.details && data.details.length > 0 ? (
            <div className="space-y-2">
              <h4 className="text-xs font-bold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-3">
                {tUI('admin.breakdown') || 'Breakdown'}
              </h4>
              {data.details.map((detail) => (
                <div
                  key={detail.id}
                  className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800/50 rounded-xl hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
                >
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium text-gray-900 dark:text-white truncate">
                      {detail.label}
                    </div>
                    {detail.sublabel && (
                      <div className="text-xs text-gray-500 dark:text-gray-400 truncate">
                        {detail.sublabel}
                      </div>
                    )}
                  </div>
                  <div className="flex items-center gap-2 ml-3">
                    {detail.status && (
                      <span className={`px-2 py-0.5 text-xs font-medium rounded-full ${STATUS_COLORS[detail.status]}`}>
                        {detail.status}
                      </span>
                    )}
                    <span className="text-sm font-bold text-gray-900 dark:text-white">
                      {typeof detail.value === 'number' ? detail.value.toLocaleString() : detail.value}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500 dark:text-gray-400">
              <svg className="w-12 h-12 mx-auto mb-3 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <p>{tUI('admin.noDetailsAvailable') || 'No detailed data available'}</p>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50">
          <button
            onClick={onClose}
            className="w-full px-4 py-2.5 bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-300 font-medium rounded-xl transition-colors"
          >
            {tUI('common.close') || 'Close'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default DrillDownModal;
