import React from 'react';

interface ErrorBoundaryProps {
  children: React.ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = {
    hasError: false,
    error: null
  };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('Uncaught error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-gray-50 dark:bg-black flex items-center justify-center p-4">
          <div className="bg-white dark:bg-dark-surface p-8 rounded-3xl border border-gray-200 dark:border-dark-border shadow-xl max-w-md text-center">
             <div className="w-16 h-16 bg-red-100 dark:bg-red-900/20 rounded-2xl flex items-center justify-center mx-auto mb-6 text-3xl">
                💥
             </div>
             <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">Something went wrong</h1>
             <p className="text-gray-500 dark:text-gray-400 mb-6 text-sm">
               Our engineering team has been notified. Please try reloading the page.
             </p>
             <button
               onClick={() => window.location.reload()}
               className="px-6 py-2.5 bg-gradient-to-r from-brand-600 to-purple-600 hover:from-brand-500 hover:to-purple-500 text-white font-bold rounded-xl shadow-lg shadow-brand-500/25 transition-all transform hover:-translate-y-0.5"
             >
               Reload Application
             </button>
             {process.env.NODE_ENV === 'development' && this.state.error && (
                <div className="mt-8 text-left p-4 bg-gray-100 dark:bg-black rounded-lg overflow-auto max-h-40 text-xs font-mono text-red-600">
                    {this.state.error.toString()}
                </div>
             )}
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
