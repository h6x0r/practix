import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import { ValidationPipe } from '@nestjs/common';
import { WINSTON_MODULE_NEST_PROVIDER } from 'nest-winston';
import { GlobalExceptionFilter } from './common/filters/http-exception.filter';
import { SentryInterceptor } from './common/sentry/sentry.interceptor';

async function bootstrap() {
  const app = await NestFactory.create(AppModule, {
    bufferLogs: true,
  });

  // Use Winston logger
  const logger = app.get(WINSTON_MODULE_NEST_PROVIDER);
  app.useLogger(logger);

  // Enable validation for DTOs
  app.useGlobalPipes(new ValidationPipe({ whitelist: true, transform: true }));

  // Enable global exception filter
  app.useGlobalFilters(new GlobalExceptionFilter());

  // Enable Sentry error tracking
  app.useGlobalInterceptors(new SentryInterceptor());

  // CORS configuration
  const allowedOrigins = process.env.CORS_ORIGINS
    ? process.env.CORS_ORIGINS.split(',').map(o => o.trim())
    : ['http://localhost:5173', 'http://localhost:3000', 'http://127.0.0.1:5173'];

  app.enableCors({
    origin: (origin, callback) => {
      // Allow requests with no origin (mobile apps, curl, etc.)
      if (!origin) return callback(null, true);

      if (allowedOrigins.includes(origin) || allowedOrigins.includes('*')) {
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
