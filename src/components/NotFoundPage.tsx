
import React from 'react';
import { Link } from 'react-router-dom';
import { useUITranslation } from '@/contexts/LanguageContext';

export const NotFoundPage = () => {
  const { tUI } = useUITranslation();

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-dark-bg flex items-center justify-center p-4">
      <div className="text-center max-w-md">
        {/* 404 Display */}
        <div className="relative mb-8">
          <div className="text-[150px] font-bold text-gray-200 dark:text-dark-border leading-none select-none">
            404
          </div>
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="w-24 h-24 bg-gradient-to-br from-brand-500 to-purple-600 rounded-full flex items-center justify-center shadow-xl">
              <svg className="w-12 h-12 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
          </div>
        </div>

        {/* Message */}
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-3">
          {tUI('notFound.title') || 'Page Not Found'}
        </h1>
        <p className="text-gray-500 dark:text-gray-400 mb-8">
          {tUI('notFound.description') || "The page you're looking for doesn't exist or has been moved."}
        </p>

        {/* Actions */}
        <div className="flex flex-col sm:flex-row gap-3 justify-center">
          <Link
            to="/"
            className="px-6 py-3 bg-gradient-to-r from-brand-600 to-purple-600 hover:from-brand-500 hover:to-purple-500 text-white font-bold rounded-xl shadow-lg shadow-brand-500/25 transition-all transform hover:-translate-y-0.5"
          >
            {tUI('notFound.goHome') || 'Go to Dashboard'}
          </Link>
          <Link
            to="/courses"
            className="px-6 py-3 bg-white dark:bg-dark-surface border border-gray-200 dark:border-dark-border text-gray-700 dark:text-gray-300 font-bold rounded-xl hover:bg-gray-50 dark:hover:bg-dark-border transition-all"
          >
            {tUI('notFound.browseCourses') || 'Browse Courses'}
          </Link>
        </div>
      </div>
    </div>
  );
};
