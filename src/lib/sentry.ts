import * as Sentry from '@sentry/react';

const SENTRY_DSN = import.meta.env.VITE_SENTRY_DSN;

export function initSentry() {
  if (!SENTRY_DSN) {
    console.warn('[Sentry] DSN not configured. Error tracking disabled.');
    return;
  }

  Sentry.init({
    dsn: SENTRY_DSN,
    environment: import.meta.env.MODE,

    // Performance monitoring
    tracesSampleRate: import.meta.env.PROD ? 0.1 : 1.0,

    // Session replay (optional)
    replaysSessionSampleRate: 0.1,
    replaysOnErrorSampleRate: 1.0,

    // Integrations
    integrations: [
      Sentry.browserTracingIntegration(),
      Sentry.replayIntegration({
        maskAllText: false,
        blockAllMedia: false,
      }),
    ],

    // Filter out common noise
    ignoreErrors: [
      'ResizeObserver loop limit exceeded',
      'ResizeObserver loop completed with undelivered notifications',
      'Non-Error exception captured',
      'Network request failed',
      'Load failed',
      'ChunkLoadError',
    ],

    // Scrub sensitive data
    beforeSend(event) {
      // Remove sensitive data from errors
      if (event.request?.headers) {
        delete event.request.headers['Authorization'];
        delete event.request.headers['Cookie'];
      }
      return event;
    },
  });

  console.log('[Sentry] Initialized for environment:', import.meta.env.MODE);
}

// Helper to capture exceptions with context
export function captureError(error: Error, context?: Record<string, unknown>) {
  if (context) {
    Sentry.setContext('additional', context);
  }
  Sentry.captureException(error);
}

// Helper to set user context
export function setUserContext(user: { id: string; email?: string; username?: string } | null) {
  if (user) {
    Sentry.setUser({
      id: user.id,
      email: user.email,
      username: user.username,
    });
  } else {
    Sentry.setUser(null);
  }
}

// Error boundary component
export const SentryErrorBoundary = Sentry.ErrorBoundary;

// Re-export for convenience
export { Sentry };
