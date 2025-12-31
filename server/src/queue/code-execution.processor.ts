import { Processor, WorkerHost, OnWorkerEvent } from '@nestjs/bullmq';
import { Logger } from '@nestjs/common';
import { Job } from 'bullmq';
import { PistonService, ExecutionResult } from '../piston/piston.service';
import { CODE_EXECUTION_QUEUE } from './constants';

/**
 * Job payload for code execution
 */
export interface CodeExecutionJob {
  code: string;
  language: string;
  stdin?: string;
  expectedOutput?: string;
  taskId?: string;
  userId?: string;
}

/**
 * Job result after execution
 */
export interface CodeExecutionResult extends ExecutionResult {
  jobId: string;
  taskId?: string;
  userId?: string;
}

@Processor(CODE_EXECUTION_QUEUE, {
  concurrency: 4, // Process 4 jobs in parallel
})
export class CodeExecutionProcessor extends WorkerHost {
  private readonly logger = new Logger(CodeExecutionProcessor.name);

  constructor(private readonly pistonService: PistonService) {
    super();
    this.logger.log('Code execution processor initialized');
  }

  /**
   * Process code execution job
   */
  async process(job: Job<CodeExecutionJob>): Promise<CodeExecutionResult> {
    const { code, language, stdin, taskId, userId, expectedOutput } = job.data;

    this.logger.debug(
      `Processing job ${job.id}: ${language} code (task: ${taskId || 'playground'})`,
    );

    // Execute code via Piston
    const result = await this.pistonService.execute(code, language, stdin);

    // Check expected output if provided
    if (expectedOutput && result.status === 'passed') {
      const normalizedOutput = result.stdout.trim();
      const normalizedExpected = expectedOutput.trim();

      if (normalizedOutput !== normalizedExpected) {
        result.status = 'failed';
        result.statusId = 4;
        result.description = 'Wrong Answer';
        result.message = 'Output does not match expected result';
      }
    }

    return {
      ...result,
      jobId: job.id!,
      taskId,
      userId,
    };
  }

  @OnWorkerEvent('completed')
  onCompleted(job: Job<CodeExecutionJob>) {
    this.logger.debug(`Job ${job.id} completed`);
  }

  @OnWorkerEvent('failed')
  onFailed(job: Job<CodeExecutionJob>, error: Error) {
    this.logger.error(`Job ${job.id} failed: ${error.message}`);
  }

  @OnWorkerEvent('active')
  onActive(job: Job<CodeExecutionJob>) {
    this.logger.debug(`Job ${job.id} started processing`);
  }

  @OnWorkerEvent('stalled')
  onStalled(jobId: string) {
    this.logger.warn(`Job ${jobId} stalled`);
  }
}
