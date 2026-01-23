import React from 'react';
import { Link } from 'react-router-dom';
import { useUITranslation } from '@/contexts/LanguageContext';
import { IconLock } from './Icons';

interface PremiumRequiredOverlayProps {
  children: React.ReactNode;
  title?: string;
  description?: string;
  showPreview?: boolean;
}

/**
 * PremiumRequiredOverlay - Wraps content that requires premium subscription.
 *
 * Shows blurred children with subscription prompt overlay
 * when user doesn't have access to premium content.
 */
export const PremiumRequiredOverlay: React.FC<PremiumRequiredOverlayProps> = ({
  children,
  title,
  description,
  showPreview = true
}) => {
  const { tUI } = useUITranslation();

  const defaultTitle = tUI('premium.subscriptionRequired');
  const defaultDescription = tUI('premium.subscriptionRequiredDesc');

  return (
    <div className="relative overflow-hidden h-full">
      {/* Blurred content preview or placeholder */}
      {showPreview ? (
        <div className="blur-sm pointer-events-none select-none h-full" aria-hidden="true">
          {children}
        </div>
      ) : (
        <div className="min-h-[400px] bg-gray-50 dark:bg-dark-bg h-full" />
      )}

      {/* Overlay */}
      <div className="absolute inset-0 flex items-center justify-center bg-white/80 dark:bg-dark-bg/80 backdrop-blur-sm z-40">
        <div className="text-center p-8 max-w-md">
          {/* Lock Icon */}
          <div className="w-16 h-16 bg-amber-100 dark:bg-amber-900/30 rounded-2xl flex items-center justify-center mx-auto mb-6">
            <IconLock className="w-8 h-8 text-amber-600 dark:text-amber-400" />
          </div>

          {/* Title */}
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-3">
            {title || defaultTitle}
          </h2>

          {/* Description */}
          <p className="text-gray-500 dark:text-gray-400 mb-6 leading-relaxed">
            {description || defaultDescription}
          </p>

          {/* Purchase Button */}
          <Link
            to="/payments"
            className="inline-block px-8 py-3 bg-gradient-to-r from-amber-500 to-orange-500 hover:from-amber-400 hover:to-orange-400 text-white font-bold rounded-xl shadow-lg shadow-amber-500/25 transition-all transform hover:-translate-y-0.5"
          >
            {tUI('premium.purchaseSubscription')}
          </Link>
        </div>
      </div>
    </div>
  );
};

export default PremiumRequiredOverlay;
