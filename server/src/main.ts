import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import { ValidationPipe } from '@nestjs/common';
import { WINSTON_MODULE_NEST_PROVIDER } from 'nest-winston';
import { GlobalExceptionFilter } from './common/filters/http-exception.filter';
import { SentryInterceptor } from './common/sentry/sentry.interceptor';
import helmet from 'helmet';

async function bootstrap() {
  const app = await NestFactory.create(AppModule, {
    bufferLogs: true,
  });

  // Use Winston logger
  const logger = app.get(WINSTON_MODULE_NEST_PROVIDER);
  app.useLogger(logger);

  // Security headers with Helmet
  const isProduction = process.env.NODE_ENV === 'production';
  app.use(
    helmet({
      // Content Security Policy
      contentSecurityPolicy: isProduction
        ? {
            directives: {
              defaultSrc: ["'self'"],
              // Remove 'unsafe-inline' - use strict CSP
              scriptSrc: ["'self'", 'cdn.jsdelivr.net'],
              styleSrc: ["'self'", 'fonts.googleapis.com'],
              fontSrc: ["'self'", 'fonts.gstatic.com'],
              imgSrc: ["'self'", 'data:', 'blob:'],
              connectSrc: ["'self'"],
              frameSrc: ["'none'"],
              objectSrc: ["'none'"],
              upgradeInsecureRequests: [],
            },
          }
        : false, // Disable in development for easier debugging
      // HTTP Strict Transport Security
      hsts: isProduction
        ? {
            maxAge: 31536000, // 1 year
            includeSubDomains: true,
            preload: true,
          }
        : false,
      // Prevent clickjacking
      frameguard: { action: 'deny' },
      // Hide X-Powered-By header
      hidePoweredBy: true,
      // Prevent MIME type sniffing
      noSniff: true,
      // Enable XSS filter
      xssFilter: true,
      // Referrer policy
      referrerPolicy: { policy: 'strict-origin-when-cross-origin' },
    }),
  );

  // Enable validation for DTOs
  app.useGlobalPipes(new ValidationPipe({ whitelist: true, transform: true }));

  // Enable global exception filter
  app.useGlobalFilters(new GlobalExceptionFilter());

  // Enable Sentry error tracking
  app.useGlobalInterceptors(new SentryInterceptor());

  // CORS configuration
  const allowedOrigins = process.env.CORS_ORIGINS
    ? process.env.CORS_ORIGINS.split(',').map(o => o.trim()).filter(o => o !== '*') // Never allow wildcard with credentials
    : ['http://localhost:5173', 'http://localhost:3000', 'http://127.0.0.1:5173'];

  app.enableCors({
    origin: (origin, callback) => {
      // In production: require Origin header for security
      // In development: allow requests without Origin (curl, Postman)
      if (!origin) {
        if (isProduction) {
          logger.warn('CORS blocked request without Origin header', 'CORS');
          return callback(new Error('Origin header required'));
        }
        // Development: allow no-origin requests
        return callback(null, true);
      }

      if (allowedOrigins.includes(origin)) {
        callback(null, true);
      } else {
        logger.warn(`CORS blocked request from origin: ${origin}`, 'CORS');
        callback(new Error('Not allowed by CORS'));
      }
    },
    methods: 'GET,HEAD,PUT,PATCH,POST,DELETE',
    credentials: true,
  });

  const port = process.env.PORT ? Number(process.env.PORT) : 8080;
  await app.listen(port, '0.0.0.0');
  logger.log(`🚀 Server running on http://0.0.0.0:${port}`, 'Bootstrap');
}
bootstrap();
