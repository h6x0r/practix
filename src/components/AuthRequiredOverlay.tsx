import React, { useContext, useState } from 'react';
import { AuthContext } from './Layout';
import { AuthModal } from './AuthModal';
import { useUITranslation } from '@/contexts/LanguageContext';
import { IconLock } from './Icons';

interface AuthRequiredOverlayProps {
  children: React.ReactNode;
  title?: string;
  description?: string;
  showPreview?: boolean;
}

/**
 * AuthRequiredOverlay - Wraps content that requires authentication.
 *
 * For authenticated users: renders children normally
 * For unauthenticated users:
 *   - If showPreview=true: shows blurred children with login overlay
 *   - If showPreview=false: shows placeholder with login prompt
 */
export const AuthRequiredOverlay: React.FC<AuthRequiredOverlayProps> = ({
  children,
  title,
  description,
  showPreview = true
}) => {
  const { user } = useContext(AuthContext);
  const { tUI } = useUITranslation();
  const [isModalOpen, setIsModalOpen] = useState(false);

  // If user is authenticated, just render children
  if (user) {
    return <>{children}</>;
  }

  const defaultTitle = tUI('auth.loginRequired');
  const defaultDescription = tUI('auth.loginRequiredDesc');

  return (
    <>
      <div className="relative overflow-hidden rounded-2xl border border-gray-200 dark:border-dark-border">
        {/* Blurred content preview or placeholder */}
        {showPreview ? (
          <div className="blur-sm pointer-events-none select-none" aria-hidden="true">
            {children}
          </div>
        ) : (
          <div className="min-h-[400px] bg-gray-50 dark:bg-dark-bg" />
        )}

        {/* Overlay */}
        <div className="absolute inset-0 flex items-center justify-center bg-white/80 dark:bg-dark-bg/80 backdrop-blur-sm">
          <div className="text-center p-8 max-w-md">
            {/* Lock Icon */}
            <div className="w-16 h-16 bg-brand-100 dark:bg-brand-900/30 rounded-2xl flex items-center justify-center mx-auto mb-6">
              <IconLock className="w-8 h-8 text-brand-600 dark:text-brand-400" />
            </div>

            {/* Title */}
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-3">
              {title || defaultTitle}
            </h2>

            {/* Description */}
            <p className="text-gray-500 dark:text-gray-400 mb-6 leading-relaxed">
              {description || defaultDescription}
            </p>

            {/* Login Button */}
            <button
              onClick={() => setIsModalOpen(true)}
              className="px-8 py-3 bg-gradient-to-r from-brand-600 to-purple-600 hover:from-brand-500 hover:to-purple-500 text-white font-bold rounded-xl shadow-lg shadow-brand-500/25 transition-all transform hover:-translate-y-0.5"
            >
              {tUI('auth.signInToContinue')}
            </button>
          </div>
        </div>
      </div>

      {/* Auth Modal */}
      <AuthModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        message={description || defaultDescription}
      />
    </>
  );
};

/**
 * Hook to require auth for an action.
 * Returns a function that either executes the callback (if authenticated)
 * or shows the auth modal (if not authenticated).
 */
export const useRequireAuth = () => {
  const { user } = useContext(AuthContext);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [pendingAction, setPendingAction] = useState<(() => void) | null>(null);

  const requireAuth = (callback: () => void, message?: string) => {
    if (user) {
      callback();
    } else {
      setPendingAction(() => callback);
      setIsModalOpen(true);
    }
  };

  const handleSuccess = () => {
    if (pendingAction) {
      pendingAction();
      setPendingAction(null);
    }
  };

  const AuthModalComponent = (
    <AuthModal
      isOpen={isModalOpen}
      onClose={() => {
        setIsModalOpen(false);
        setPendingAction(null);
      }}
      onSuccess={handleSuccess}
    />
  );

  return { requireAuth, AuthModal: AuthModalComponent, isAuthenticated: !!user };
};

export default AuthRequiredOverlay;
