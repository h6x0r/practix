import React from 'react';
import { IconSparkles } from '@/components/Icons';

interface StatusCardProps {
  isPremium: boolean;
  expiresAt?: string;
  language: string;
}

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

const StatusCard: React.FC<StatusCardProps> = ({ isPremium, expiresAt, language }) => {
  return (
    <div
      className={`rounded-3xl p-8 text-white shadow-xl relative overflow-hidden transition-all duration-500 ${
        isPremium
          ? 'bg-gradient-to-r from-brand-600 to-purple-600'
          : 'bg-gradient-to-r from-gray-900 to-gray-800 dark:from-dark-surface dark:to-black'
      }`}
    >
      <div className="relative z-10 flex flex-col md:flex-row justify-between items-start md:items-center gap-6">
        {isPremium ? (
          <div data-testid="subscription-active">
            <div className="flex items-center gap-2 text-white/90 font-bold uppercase tracking-wider mb-2 text-xs">
              <IconSparkles className="w-4 h-4" />{' '}
              {language === 'ru' ? 'Текущий план' : 'Current Plan'}
            </div>
            <h2 className="text-3xl font-display font-bold mb-2" data-testid="current-plan-name">
              Premium
            </h2>
            <p className="text-white/90 max-w-md">
              {language === 'ru'
                ? `Активная подписка до ${formatDate(expiresAt)}. Полный доступ ко всем функциям.`
                : `Active subscription until ${formatDate(expiresAt)}. Full access to all features.`}
            </p>
          </div>
        ) : (
          <div>
            <div className="text-brand-400 font-bold uppercase tracking-wider mb-2 text-xs">
              {language === 'ru' ? 'Текущий план' : 'Current Plan'}
            </div>
            <h2 className="text-3xl font-display font-bold mb-2">Free</h2>
            <p className="text-gray-400 max-w-md">
              {language === 'ru'
                ? 'Базовый доступ. Обновитесь для полного доступа к курсам и AI-тьютору.'
                : 'Basic access. Upgrade for full course access and AI Tutor.'}
            </p>
          </div>
        )}
      </div>
      <div className="absolute top-0 right-0 w-64 h-64 bg-white opacity-10 rounded-full blur-3xl transform translate-x-1/2 -translate-y-1/2 pointer-events-none"></div>
    </div>
  );
};

export default StatusCard;
