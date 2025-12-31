/**
 * Go Modules Index
 * Exports all Go language modules
 */

// Core Fundamentals
export { fundamentalsModule } from './fundamentals';
export { errorHandlingModule } from './error-handling';
export { module as pointersModule } from './pointersx/module';
export { module as dataStructuresModule } from './datastructsx/module';
export { module as jsonEncodingModule } from './encodingx/module';
export { module as genericsModule } from './generics/module';
export { module as ioInterfacesModule } from './io-interfaces/module';
export { constructorPatternsModule } from './constructor-patterns';
export { panicRecoveryModule } from './panic-recovery';

// Concurrency
export { goroutinesModule } from './goroutines';
export { channelsModule } from './channels';
export { synchronizationModule } from './synchronization';
export { concurrencyPatternsModule } from './concurrency-patterns';

// Web & APIs
export { httpMiddlewareModule } from './http-middleware';
export { grpcInterceptorsModule } from './grpc-interceptors';
export { rateLimitingModule } from './rate-limiting';
export { circuitBreakerModule } from './circuit-breaker';
export { cachingModule } from './caching';
export { retryPatternsModule } from './retry-patterns';

// Production
export { testingModule } from './testing';
export { loggingModule } from './logging';
export { metricsModule } from './metrics';
export { profilingModule } from './profiling';
export { module as databaseModule } from './database/module';
export { configManagementModule } from './config-management';
