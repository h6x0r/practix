import React, { useState, useEffect, useContext } from 'react';
import { AuthContext } from './Layout';
import { authService } from '@/features/auth/api/authService';
import { ApiError } from '@/lib/api';
import { useToast } from './Toast';
import { useUITranslation } from '@/contexts/LanguageContext';

interface AuthModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess?: () => void;
  message?: string;
}

export const AuthModal: React.FC<AuthModalProps> = ({
  isOpen,
  onClose,
  onSuccess,
  message
}) => {
  const { login } = useContext(AuthContext);
  const { showToast } = useToast();
  const { tUI } = useUITranslation();

  const [mode, setMode] = useState<'login' | 'register'>('login');
  const [isLoading, setIsLoading] = useState(false);
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    name: ''
  });

  // Reset form when modal opens/closes
  useEffect(() => {
    if (!isOpen) {
      setFormData({ email: '', password: '', name: '' });
      setMode('login');
    }
  }, [isOpen]);

  // Close on Escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    if (isOpen) {
      window.addEventListener('keydown', handleEscape);
      return () => window.removeEventListener('keydown', handleEscape);
    }
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);

    try {
      if (mode === 'login') {
        const resp = await authService.login({
          email: formData.email,
          password: formData.password
        });
        await login(resp.user);
        showToast(`Welcome back, ${resp.user.name.split(' ')[0]}!`, 'success');
      } else {
        const resp = await authService.register({
          name: formData.name,
          email: formData.email,
          password: formData.password
        });
        await login(resp.user);
        showToast('Account created successfully!', 'success');
      }
      onClose();
      onSuccess?.();
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        showToast(err.message, 'error');
      } else {
        showToast('Connection failed. Please check your internet.', 'error');
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-fade-in"
      onClick={onClose}
    >
      <div
        className="relative w-full max-w-md bg-white dark:bg-dark-surface rounded-3xl border border-gray-200 dark:border-dark-border shadow-2xl transform animate-scale-in"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 transition-colors"
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>

        <div className="p-8">
          {/* Header */}
          <div className="text-center mb-6">
            <div className="w-12 h-12 bg-gradient-to-br from-brand-500 to-purple-600 rounded-xl flex items-center justify-center text-white font-display font-black text-xl mx-auto mb-4 shadow-lg shadow-brand-500/25">
              P
            </div>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
              {mode === 'login' ? tUI('auth.signIn') : tUI('auth.createAccount')}
            </h2>
            {message && (
              <p className="text-sm text-gray-500 dark:text-gray-400">
                {message}
              </p>
            )}
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            {mode === 'register' && (
              <div>
                <label className="block text-xs font-bold text-gray-500 uppercase mb-2">
                  {tUI('auth.fullName')}
                </label>
                <input
                  type="text"
                  required
                  placeholder="Alex Developer"
                  className="w-full bg-gray-50 dark:bg-dark-bg border border-gray-200 dark:border-dark-border rounded-xl px-4 py-3 text-sm focus:ring-2 focus:ring-brand-500 outline-none dark:text-white transition-all"
                  value={formData.name}
                  onChange={e => setFormData({...formData, name: e.target.value})}
                />
              </div>
            )}

            <div>
              <label className="block text-xs font-bold text-gray-500 uppercase mb-2">
                {tUI('auth.email')}
              </label>
              <input
                type="email"
                required
                placeholder="alex@example.com"
                className="w-full bg-gray-50 dark:bg-dark-bg border border-gray-200 dark:border-dark-border rounded-xl px-4 py-3 text-sm focus:ring-2 focus:ring-brand-500 outline-none dark:text-white transition-all"
                value={formData.email}
                onChange={e => setFormData({...formData, email: e.target.value})}
              />
            </div>

            <div>
              <label className="block text-xs font-bold text-gray-500 uppercase mb-2">
                {tUI('auth.password')}
              </label>
              <input
                type="password"
                required
                placeholder="••••••••"
                className="w-full bg-gray-50 dark:bg-dark-bg border border-gray-200 dark:border-dark-border rounded-xl px-4 py-3 text-sm focus:ring-2 focus:ring-brand-500 outline-none dark:text-white transition-all"
                value={formData.password}
                onChange={e => setFormData({...formData, password: e.target.value})}
              />
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="w-full py-3 bg-gradient-to-r from-brand-600 to-purple-600 hover:from-brand-500 hover:to-purple-500 text-white font-bold rounded-xl shadow-lg shadow-brand-500/25 transition-all flex items-center justify-center gap-2"
            >
              {isLoading ? (
                <span className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></span>
              ) : (
                mode === 'login' ? tUI('auth.signIn') : tUI('auth.createAccount')
              )}
            </button>
          </form>

          {/* Toggle Mode */}
          <p className="text-center mt-6 text-sm text-gray-500 dark:text-gray-400">
            {mode === 'login' ? (
              <>
                {tUI('auth.noAccount')}{' '}
                <button
                  onClick={() => setMode('register')}
                  className="font-bold text-brand-600 hover:text-brand-500 transition-colors"
                >
                  {tUI('auth.signUp')}
                </button>
              </>
            ) : (
              <>
                {tUI('auth.hasAccount')}{' '}
                <button
                  onClick={() => setMode('login')}
                  className="font-bold text-brand-600 hover:text-brand-500 transition-colors"
                >
                  {tUI('auth.logIn')}
                </button>
              </>
            )}
          </p>
        </div>
      </div>
    </div>
  );
};

export default AuthModal;
