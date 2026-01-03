import { Test, TestingModule } from '@nestjs/testing';
import { getQueueToken } from '@nestjs/bullmq';
import { Job, Queue } from 'bullmq';
import { DeadLetterService, DeadLetterJob } from './dead-letter.service';
import { CodeExecutionJob } from './code-execution.processor';
import { CODE_EXECUTION_QUEUE, DEAD_LETTER_QUEUE } from './constants';

describe('DeadLetterService', () => {
  let service: DeadLetterService;

  const mockExecutionQueue = {
    add: jest.fn(),
    getFailed: jest.fn(),
  };

  const mockDeadLetterQueue = {
    add: jest.fn(),
    getWaitingCount: jest.fn(),
    getJobs: jest.fn(),
    getJob: jest.fn(),
  };

  const createMockJob = (overrides: Partial<Job<CodeExecutionJob>> = {}): Job<CodeExecutionJob> => ({
    id: 'job-123',
    data: {
      code: 'print("hello")',
      language: 'python',
      taskId: 'task-456',
      userId: 'user-789',
    },
    failedReason: 'Execution timeout',
    attemptsMade: 3,
    stacktrace: ['Error at line 1', 'at execute()'],
    opts: { attempts: 3 },
    remove: jest.fn().mockResolvedValue(undefined),
    ...overrides,
  } as unknown as Job<CodeExecutionJob>);

  const createMockDlqJob = (overrides: Partial<Job<DeadLetterJob>> = {}): Job<DeadLetterJob> => ({
    id: 'dlq-job-123',
    data: {
      originalJobId: 'job-123',
      originalQueue: CODE_EXECUTION_QUEUE,
      originalData: {
        code: 'print("hello")',
        language: 'python',
        taskId: 'task-456',
        userId: 'user-789',
      },
      failedReason: 'Execution timeout',
      failedAt: '2024-01-01T00:00:00.000Z',
      attemptsMade: 3,
      stacktrace: ['Error at line 1'],
    },
    timestamp: Date.now(),
    remove: jest.fn().mockResolvedValue(undefined),
    ...overrides,
  } as unknown as Job<DeadLetterJob>);

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [
        DeadLetterService,
        { provide: getQueueToken(CODE_EXECUTION_QUEUE), useValue: mockExecutionQueue },
        { provide: getQueueToken(DEAD_LETTER_QUEUE), useValue: mockDeadLetterQueue },
      ],
    }).compile();

    service = module.get<DeadLetterService>(DeadLetterService);

    jest.clearAllMocks();
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });

  // ============================================
  // moveToDeadLetter()
  // ============================================
  describe('moveToDeadLetter()', () => {
    it('should move a failed job to DLQ', async () => {
      const job = createMockJob();

      await service.moveToDeadLetter(job);

      expect(mockDeadLetterQueue.add).toHaveBeenCalledWith(
        'failed-job',
        expect.objectContaining({
          originalJobId: 'job-123',
          originalQueue: CODE_EXECUTION_QUEUE,
          originalData: job.data,
          failedReason: 'Execution timeout',
          attemptsMade: 3,
          stacktrace: ['Error at line 1', 'at execute()'],
        }),
        { jobId: 'dlq-job-123' },
      );
    });

    it('should handle job with unknown id', async () => {
      const job = createMockJob({ id: undefined });

      await service.moveToDeadLetter(job);

      expect(mockDeadLetterQueue.add).toHaveBeenCalledWith(
        'failed-job',
        expect.objectContaining({
          originalJobId: 'unknown',
        }),
        { jobId: 'dlq-undefined' },
      );
    });

    it('should handle job with no failed reason', async () => {
      const job = createMockJob({ failedReason: undefined });

      await service.moveToDeadLetter(job);

      expect(mockDeadLetterQueue.add).toHaveBeenCalledWith(
        'failed-job',
        expect.objectContaining({
          failedReason: 'Unknown error',
        }),
        expect.any(Object),
      );
    });
  });

  // ============================================
  // getStats()
  // ============================================
  describe('getStats()', () => {
    it('should return queue statistics', async () => {
      mockDeadLetterQueue.getWaitingCount.mockResolvedValue(5);
      mockDeadLetterQueue.getJobs.mockResolvedValue([
        { timestamp: Date.now() - 60000 },
        { timestamp: Date.now() - 30000 },
        { timestamp: Date.now() - 120000 },
      ]);

      const stats = await service.getStats();

      expect(stats.waiting).toBe(5);
      expect(stats.total).toBe(3);
      expect(stats.oldestJobAge).toBeGreaterThanOrEqual(120000);
      expect(stats.oldestJobAge).toBeLessThan(130000);
    });

    it('should handle empty queue', async () => {
      mockDeadLetterQueue.getWaitingCount.mockResolvedValue(0);
      mockDeadLetterQueue.getJobs.mockResolvedValue([]);

      const stats = await service.getStats();

      expect(stats.waiting).toBe(0);
      expect(stats.total).toBe(0);
      expect(stats.oldestJobAge).toBeUndefined();
    });
  });

  // ============================================
  // getJobs()
  // ============================================
  describe('getJobs()', () => {
    it('should return paginated jobs', async () => {
      const mockJobs = [
        createMockDlqJob({ id: 'dlq-1' }),
        createMockDlqJob({ id: 'dlq-2' }),
      ];
      mockDeadLetterQueue.getJobs.mockResolvedValue(mockJobs);

      const jobs = await service.getJobs(0, 10);

      expect(mockDeadLetterQueue.getJobs).toHaveBeenCalledWith(['waiting', 'delayed'], 0, 10);
      expect(jobs).toHaveLength(2);
      expect(jobs[0].id).toBe('dlq-1');
      expect(jobs[1].id).toBe('dlq-2');
    });

    it('should use default pagination', async () => {
      mockDeadLetterQueue.getJobs.mockResolvedValue([]);

      await service.getJobs();

      expect(mockDeadLetterQueue.getJobs).toHaveBeenCalledWith(['waiting', 'delayed'], 0, 20);
    });

    it('should handle job with unknown id', async () => {
      const mockJobs = [createMockDlqJob({ id: undefined })];
      mockDeadLetterQueue.getJobs.mockResolvedValue(mockJobs);

      const jobs = await service.getJobs();

      expect(jobs[0].id).toBe('unknown');
    });
  });

  // ============================================
  // retryJob()
  // ============================================
  describe('retryJob()', () => {
    it('should retry a DLQ job successfully', async () => {
      const dlqJob = createMockDlqJob();
      mockDeadLetterQueue.getJob.mockResolvedValue(dlqJob);
      mockExecutionQueue.add.mockResolvedValue({ id: 'new-job-456' });

      const result = await service.retryJob('dlq-job-123');

      expect(result.success).toBe(true);
      expect(result.newJobId).toBe('new-job-456');
      expect(mockExecutionQueue.add).toHaveBeenCalledWith(
        'execute',
        dlqJob.data.originalData,
        expect.objectContaining({
          jobId: expect.stringContaining('retry-job-123-'),
        }),
      );
      expect(dlqJob.remove).toHaveBeenCalled();
    });

    it('should return error when DLQ job not found', async () => {
      mockDeadLetterQueue.getJob.mockResolvedValue(null);

      const result = await service.retryJob('non-existent');

      expect(result.success).toBe(false);
      expect(result.error).toBe('DLQ job not found');
    });

    it('should handle retry failure', async () => {
      const dlqJob = createMockDlqJob();
      mockDeadLetterQueue.getJob.mockResolvedValue(dlqJob);
      mockExecutionQueue.add.mockRejectedValue(new Error('Queue full'));

      const result = await service.retryJob('dlq-job-123');

      expect(result.success).toBe(false);
      expect(result.error).toBe('Queue full');
    });
  });

  // ============================================
  // removeJob()
  // ============================================
  describe('removeJob()', () => {
    it('should remove a DLQ job', async () => {
      const dlqJob = createMockDlqJob();
      mockDeadLetterQueue.getJob.mockResolvedValue(dlqJob);

      const result = await service.removeJob('dlq-job-123');

      expect(result).toBe(true);
      expect(dlqJob.remove).toHaveBeenCalled();
    });

    it('should return false when job not found', async () => {
      mockDeadLetterQueue.getJob.mockResolvedValue(null);

      const result = await service.removeJob('non-existent');

      expect(result).toBe(false);
    });
  });

  // ============================================
  // clearAll()
  // ============================================
  describe('clearAll()', () => {
    it('should clear all DLQ jobs', async () => {
      const mockJobs = [
        createMockDlqJob({ id: 'dlq-1' }),
        createMockDlqJob({ id: 'dlq-2' }),
        createMockDlqJob({ id: 'dlq-3' }),
      ];
      mockDeadLetterQueue.getJobs.mockResolvedValue(mockJobs);

      const count = await service.clearAll();

      expect(count).toBe(3);
      mockJobs.forEach(job => {
        expect(job.remove).toHaveBeenCalled();
      });
    });

    it('should return 0 when queue is empty', async () => {
      mockDeadLetterQueue.getJobs.mockResolvedValue([]);

      const count = await service.clearAll();

      expect(count).toBe(0);
    });

    it('should process in batches for large queues', async () => {
      // Create 75 mock jobs to test batching (batch size is 50)
      const mockJobs = Array.from({ length: 75 }, (_, i) =>
        createMockDlqJob({ id: `dlq-${i}` }),
      );
      mockDeadLetterQueue.getJobs.mockResolvedValue(mockJobs);

      const count = await service.clearAll();

      expect(count).toBe(75);
      mockJobs.forEach(job => {
        expect(job.remove).toHaveBeenCalled();
      });
    });
  });

  // ============================================
  // processFailedJobs()
  // ============================================
  describe('processFailedJobs()', () => {
    it('should move failed jobs to DLQ', async () => {
      const failedJobs = [
        createMockJob({ id: 'job-1', attemptsMade: 3, opts: { attempts: 3 } }),
        createMockJob({ id: 'job-2', attemptsMade: 3, opts: { attempts: 3 } }),
      ];
      mockExecutionQueue.getFailed.mockResolvedValue(failedJobs);

      const count = await service.processFailedJobs();

      expect(count).toBe(2);
      expect(mockDeadLetterQueue.add).toHaveBeenCalledTimes(2);
      failedJobs.forEach(job => {
        expect(job.remove).toHaveBeenCalled();
      });
    });

    it('should skip jobs with remaining retries', async () => {
      const failedJobs = [
        createMockJob({ id: 'job-1', attemptsMade: 1, opts: { attempts: 3 } }),
        createMockJob({ id: 'job-2', attemptsMade: 3, opts: { attempts: 3 } }),
      ];
      mockExecutionQueue.getFailed.mockResolvedValue(failedJobs);

      const count = await service.processFailedJobs();

      expect(count).toBe(1);
      expect(mockDeadLetterQueue.add).toHaveBeenCalledTimes(1);
    });

    it('should use default attempts when not specified', async () => {
      const failedJobs = [
        createMockJob({ id: 'job-1', attemptsMade: 3, opts: {} }),
      ];
      mockExecutionQueue.getFailed.mockResolvedValue(failedJobs);

      const count = await service.processFailedJobs();

      expect(count).toBe(1);
    });

    it('should return 0 when no failed jobs', async () => {
      mockExecutionQueue.getFailed.mockResolvedValue([]);

      const count = await service.processFailedJobs();

      expect(count).toBe(0);
      expect(mockDeadLetterQueue.add).not.toHaveBeenCalled();
    });
  });
});
