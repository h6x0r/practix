import { Module, Global, OnModuleInit } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import * as Sentry from '@sentry/node';

@Global()
@Module({})
export class SentryModule implements OnModuleInit {
  constructor(private configService: ConfigService) {}

  onModuleInit() {
    const dsn = this.configService.get<string>('SENTRY_DSN');

    if (!dsn) {
      console.warn('[Sentry] DSN not configured. Error tracking disabled.');
      return;
    }

    Sentry.init({
      dsn,
      environment: this.configService.get<string>('NODE_ENV') || 'development',

      // Performance monitoring
      tracesSampleRate: this.configService.get<string>('NODE_ENV') === 'production' ? 0.1 : 1.0,

      // Integrations
      integrations: [
        Sentry.httpIntegration(),
        Sentry.expressIntegration(),
      ],

      // Filter out common noise
      ignoreErrors: [
        'ECONNRESET',
        'ECONNREFUSED',
        'ETIMEDOUT',
        'NotFoundException',
        'UnauthorizedException',
      ],

      // Scrub sensitive data
      beforeSend(event) {
        // Remove sensitive headers
        if (event.request?.headers) {
          delete event.request.headers['authorization'];
          delete event.request.headers['cookie'];
        }
        return event;
      },
    });

    console.log('[Sentry] Initialized for environment:', this.configService.get<string>('NODE_ENV') || 'development');
  }
}

// Re-export Sentry for use in other modules
export { Sentry };
