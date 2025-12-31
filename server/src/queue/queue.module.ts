import { Module } from '@nestjs/common';
import { BullModule } from '@nestjs/bullmq';
import { ConfigModule, ConfigService } from '@nestjs/config';
import { CodeExecutionProcessor } from './code-execution.processor';
import { CodeExecutionService } from './code-execution.service';
import { PistonModule } from '../piston/piston.module';
import { CODE_EXECUTION_QUEUE } from './constants';

@Module({
  imports: [
    ConfigModule,
    PistonModule,
    BullModule.forRootAsync({
      imports: [ConfigModule],
      useFactory: (configService: ConfigService) => ({
        connection: {
          host: configService.get('REDIS_HOST') || 'redis',
          port: parseInt(configService.get('REDIS_PORT') || '6379', 10),
        },
      }),
      inject: [ConfigService],
    }),
    BullModule.registerQueue({
      name: CODE_EXECUTION_QUEUE,
      defaultJobOptions: {
        attempts: 3,
        backoff: {
          type: 'exponential',
          delay: 5000, // 5s → 10s → 20s (covers ~15s Piston restart)
        },
        removeOnComplete: {
          count: 100, // Keep last 100 completed jobs
          age: 3600,  // Remove jobs older than 1 hour
        },
        removeOnFail: {
          count: 50,  // Keep last 50 failed jobs
          age: 86400, // Remove failed jobs older than 24 hours
        },
      },
    }),
  ],
  providers: [CodeExecutionProcessor, CodeExecutionService],
  exports: [CodeExecutionService],
})
export class QueueModule {}
