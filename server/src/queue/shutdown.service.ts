import { Injectable, OnModuleDestroy, Logger, BeforeApplicationShutdown } from '@nestjs/common';
import { InjectQueue } from '@nestjs/bullmq';
import { Queue } from 'bullmq';
import { CODE_EXECUTION_QUEUE, DEAD_LETTER_QUEUE } from './constants';

/**
 * Service to handle graceful shutdown of BullMQ queues
 * Ensures all active jobs complete before the application terminates
 */
@Injectable()
export class ShutdownService implements OnModuleDestroy, BeforeApplicationShutdown {
  private readonly logger = new Logger(ShutdownService.name);
  private isShuttingDown = false;

  constructor(
    @InjectQueue(CODE_EXECUTION_QUEUE)
    private readonly codeExecutionQueue: Queue,
    @InjectQueue(DEAD_LETTER_QUEUE)
    private readonly deadLetterQueue: Queue,
  ) {}

  /**
   * Called before the application shutdown begins
   * Pauses queues to prevent new jobs from being processed
   */
  async beforeApplicationShutdown(signal?: string) {
    if (this.isShuttingDown) return;
    this.isShuttingDown = true;

    this.logger.log(`Received shutdown signal: ${signal || 'unknown'}`);
    this.logger.log('Pausing job queues...');

    try {
      // Pause both queues to stop accepting new jobs
      await Promise.all([
        this.codeExecutionQueue.pause(),
        this.deadLetterQueue.pause(),
      ]);
      this.logger.log('Queues paused successfully');
    } catch (error) {
      this.logger.error('Error pausing queues:', error);
    }
  }

  /**
   * Called when the module is being destroyed
   * Waits for active jobs to complete and closes connections
   */
  async onModuleDestroy() {
    this.logger.log('Initiating graceful shutdown of BullMQ...');

    try {
      // Get active job counts
      const activeJobs = await this.codeExecutionQueue.getActiveCount();

      if (activeJobs > 0) {
        this.logger.log(`Waiting for ${activeJobs} active job(s) to complete...`);

        // Wait up to 30 seconds for active jobs to complete
        const maxWaitTime = 30000; // 30 seconds
        const checkInterval = 1000; // 1 second
        let elapsed = 0;

        while (elapsed < maxWaitTime) {
          const remaining = await this.codeExecutionQueue.getActiveCount();
          if (remaining === 0) {
            this.logger.log('All active jobs completed');
            break;
          }

          await this.delay(checkInterval);
          elapsed += checkInterval;

          if (elapsed % 5000 === 0) {
            this.logger.log(`Still waiting for ${remaining} job(s)... (${elapsed / 1000}s)`);
          }
        }

        const finalActive = await this.codeExecutionQueue.getActiveCount();
        if (finalActive > 0) {
          this.logger.warn(
            `Timeout reached with ${finalActive} job(s) still active. These will be retried on restart.`
          );
        }
      }

      // Close queue connections
      this.logger.log('Closing queue connections...');
      await Promise.allSettled([
        this.codeExecutionQueue.close(),
        this.deadLetterQueue.close(),
      ]);

      this.logger.log('BullMQ graceful shutdown complete');
    } catch (error) {
      this.logger.error('Error during graceful shutdown:', error);
    }
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Get current queue status
   */
  async getQueueStatus() {
    const [waiting, active, completed, failed] = await Promise.all([
      this.codeExecutionQueue.getWaitingCount(),
      this.codeExecutionQueue.getActiveCount(),
      this.codeExecutionQueue.getCompletedCount(),
      this.codeExecutionQueue.getFailedCount(),
    ]);

    return {
      queue: CODE_EXECUTION_QUEUE,
      waiting,
      active,
      completed,
      failed,
      isShuttingDown: this.isShuttingDown,
    };
  }
}
