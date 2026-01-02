/**
 * Frontend Logger Utility
 *
 * Centralized logging with environment-aware behavior:
 * - Development: Full console output
 * - Production: Only errors and warnings, can be extended to send to Sentry
 */

type LogLevel = 'debug' | 'info' | 'warn' | 'error';

interface LoggerConfig {
  enabledInProduction: boolean;
  minLevel: LogLevel;
}

const LOG_LEVELS: Record<LogLevel, number> = {
  debug: 0,
  info: 1,
  warn: 2,
  error: 3,
};

const isDevelopment = import.meta.env.DEV;

const config: LoggerConfig = {
  enabledInProduction: true,
  minLevel: isDevelopment ? 'debug' : 'warn',
};

function shouldLog(level: LogLevel): boolean {
  // In production mode (when enabledInProduction is true), we only log at minLevel or above
  // Note: enabledInProduction is always true in current config, so all logging goes through minLevel check
  return LOG_LEVELS[level] >= LOG_LEVELS[config.minLevel];
}

function formatMessage(level: LogLevel, context: string, message: string): string {
  const timestamp = new Date().toISOString();
  return `[${timestamp}] [${level.toUpperCase()}] [${context}] ${message}`;
}

export const logger = {
  debug(context: string, message: string, ...args: unknown[]): void {
    if (shouldLog('debug')) {
      console.debug(formatMessage('debug', context, message), ...args);
    }
  },

  info(context: string, message: string, ...args: unknown[]): void {
    if (shouldLog('info')) {
      console.info(formatMessage('info', context, message), ...args);
    }
  },

  warn(context: string, message: string, ...args: unknown[]): void {
    if (shouldLog('warn')) {
      console.warn(formatMessage('warn', context, message), ...args);
    }
  },

  error(context: string, message: string, error?: unknown): void {
    if (shouldLog('error')) {
      const errorDetails = error instanceof Error ? error : undefined;
      console.error(formatMessage('error', context, message), errorDetails);

      // In production, could send to Sentry here
      // if (!isDevelopment && errorDetails) {
      //   Sentry.captureException(errorDetails);
      // }
    }
  },
};

// Convenience function for creating a scoped logger
export function createLogger(context: string) {
  return {
    debug: (message: string, ...args: unknown[]) => logger.debug(context, message, ...args),
    info: (message: string, ...args: unknown[]) => logger.info(context, message, ...args),
    warn: (message: string, ...args: unknown[]) => logger.warn(context, message, ...args),
    error: (message: string, error?: unknown) => logger.error(context, message, error),
  };
}

export default logger;
