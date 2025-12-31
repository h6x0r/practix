import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';
import { initSentry, SentryErrorBoundary } from './lib/sentry';

// Initialize Sentry error tracking
initSentry();

const rootElement = document.getElementById('root');
if (!rootElement) {
  throw new Error("Could not find root element to mount to");
}

const root = ReactDOM.createRoot(rootElement);
root.render(
  <React.StrictMode>
    <SentryErrorBoundary fallback={<ErrorFallback />}>
      <App />
    </SentryErrorBoundary>
  </React.StrictMode>
);

// Simple error fallback UI
function ErrorFallback() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900">
      <div className="text-center p-8">
        <div className="text-6xl mb-4">😵</div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
          Something went wrong
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mb-4">
          We've been notified and are working on a fix.
        </p>
        <button
          onClick={() => window.location.reload()}
          className="px-4 py-2 bg-gradient-to-r from-brand-600 to-purple-600 hover:from-brand-500 hover:to-purple-500 text-white font-bold rounded-xl shadow-lg shadow-brand-500/25 transition-all"
        >
          Reload Page
        </button>
      </div>
    </div>
  );
}