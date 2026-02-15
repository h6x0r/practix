import { Injectable, Logger } from "@nestjs/common";
import { InjectQueue } from "@nestjs/bullmq";
import { Queue, Job } from "bullmq";
import { CODE_EXECUTION_QUEUE, DEAD_LETTER_QUEUE } from "./constants";
import { CodeExecutionJob } from "./code-execution.processor";

/**
 * Dead Letter Job - includes original job data plus failure info
 */
export interface DeadLetterJob {
  originalJobId: string;
  originalQueue: string;
  originalData: CodeExecutionJob;
  failedReason: string;
  failedAt: string;
  attemptsMade: number;
  stacktrace?: string[];
}

/**
 * Dead Letter Queue Service
 * Handles permanently failed jobs for analysis and potential retry
 */
@Injectable()
export class DeadLetterService {
  private readonly logger = new Logger(DeadLetterService.name);

  constructor(
    @InjectQueue(CODE_EXECUTION_QUEUE)
    private readonly executionQueue: Queue<CodeExecutionJob>,
    @InjectQueue(DEAD_LETTER_QUEUE)
    private readonly deadLetterQueue: Queue<DeadLetterJob>,
  ) {
    this.logger.log("Dead Letter Service initialized");
  }

  /**
   * Move a permanently failed job to DLQ
   */
  async moveToDeadLetter(job: Job<CodeExecutionJob>): Promise<void> {
    const deadLetterJob: DeadLetterJob = {
      originalJobId: job.id || "unknown",
      originalQueue: CODE_EXECUTION_QUEUE,
      originalData: job.data,
      failedReason: job.failedReason || "Unknown error",
      failedAt: new Date().toISOString(),
      attemptsMade: job.attemptsMade,
      stacktrace: job.stacktrace,
    };

    await this.deadLetterQueue.add("failed-job", deadLetterJob, {
      jobId: `dlq-${job.id}`,
    });

    this.logger.warn(`Job ${job.id} moved to DLQ: ${job.failedReason}`, {
      taskId: job.data.taskId,
      userId: job.data.userId,
    });
  }

  /**
   * Get DLQ statistics
   */
  async getStats(): Promise<{
    waiting: number;
    total: number;
    oldestJobAge?: number;
  }> {
    const waiting = await this.deadLetterQueue.getWaitingCount();
    const jobs = await this.deadLetterQueue.getJobs(["waiting", "delayed"]);

    let oldestJobAge: number | undefined;
    if (jobs.length > 0) {
      const timestamps = jobs.map((j) => j.timestamp);
      const oldest = Math.min(...timestamps);
      oldestJobAge = Date.now() - oldest;
    }

    return {
      waiting,
      total: jobs.length,
      oldestJobAge,
    };
  }

  /**
   * Get DLQ jobs with pagination
   */
  async getJobs(
    start = 0,
    end = 20,
  ): Promise<
    Array<{
      id: string;
      data: DeadLetterJob;
      timestamp: number;
    }>
  > {
    const jobs = await this.deadLetterQueue.getJobs(
      ["waiting", "delayed"],
      start,
      end,
    );

    return jobs.map((job) => ({
      id: job.id || "unknown",
      data: job.data,
      timestamp: job.timestamp,
    }));
  }

  /**
   * Retry a job from DLQ
   * Creates a new job in the original queue
   */
  async retryJob(
    dlqJobId: string,
  ): Promise<{ success: boolean; newJobId?: string; error?: string }> {
    const dlqJob = await this.deadLetterQueue.getJob(dlqJobId);

    if (!dlqJob) {
      return { success: false, error: "DLQ job not found" };
    }

    try {
      // Add back to original queue
      const newJob = await this.executionQueue.add(
        "execute",
        dlqJob.data.originalData,
        {
          jobId: `retry-${dlqJob.data.originalJobId}-${Date.now()}`,
        },
      );

      // Remove from DLQ
      await dlqJob.remove();

      this.logger.log(`Retried DLQ job ${dlqJobId} as ${newJob.id}`);

      return { success: true, newJobId: newJob.id };
    } catch (error: any) {
      this.logger.error(`Failed to retry DLQ job ${dlqJobId}`, error);
      return { success: false, error: error.message };
    }
  }

  /**
   * Remove a job from DLQ (acknowledge as unrecoverable)
   */
  async removeJob(dlqJobId: string): Promise<boolean> {
    const dlqJob = await this.deadLetterQueue.getJob(dlqJobId);

    if (!dlqJob) {
      return false;
    }

    await dlqJob.remove();
    this.logger.log(`Removed DLQ job ${dlqJobId}`);
    return true;
  }

  /**
   * Clear all DLQ jobs (use with caution)
   * Uses batch processing to avoid blocking event loop
   */
  async clearAll(): Promise<number> {
    const jobs = await this.deadLetterQueue.getJobs(["waiting", "delayed"]);
    const count = jobs.length;

    if (count === 0) {
      return 0;
    }

    // Process in batches of 50 to avoid blocking
    const BATCH_SIZE = 50;
    for (let i = 0; i < jobs.length; i += BATCH_SIZE) {
      const batch = jobs.slice(i, i + BATCH_SIZE);
      // Remove jobs in parallel within each batch
      await Promise.all(batch.map((job) => job.remove()));

      // Yield to event loop between batches
      if (i + BATCH_SIZE < jobs.length) {
        await new Promise((resolve) => setImmediate(resolve));
      }
    }

    this.logger.warn(`Cleared ${count} jobs from DLQ`);
    return count;
  }

  /**
   * Process failed jobs from main queue and move to DLQ
   * Should be called periodically or on 'failed' event
   */
  async processFailedJobs(): Promise<number> {
    const failedJobs = await this.executionQueue.getFailed();
    let movedCount = 0;

    for (const job of failedJobs) {
      // Only move jobs that have exhausted all retries
      if (job.attemptsMade >= (job.opts.attempts || 3)) {
        await this.moveToDeadLetter(job);
        await job.remove();
        movedCount++;
      }
    }

    if (movedCount > 0) {
      this.logger.log(`Moved ${movedCount} permanently failed jobs to DLQ`);
    }

    return movedCount;
  }
}
