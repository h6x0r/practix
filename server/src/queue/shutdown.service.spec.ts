import { Test, TestingModule } from '@nestjs/testing';
import { getQueueToken } from '@nestjs/bullmq';
import { ShutdownService } from './shutdown.service';
import { CODE_EXECUTION_QUEUE, DEAD_LETTER_QUEUE } from './constants';

describe('ShutdownService', () => {
  let service: ShutdownService;
  let codeExecutionQueue: jest.Mocked<any>;
  let deadLetterQueue: jest.Mocked<any>;

  const createMockQueue = () => ({
    pause: jest.fn().mockResolvedValue(undefined),
    close: jest.fn().mockResolvedValue(undefined),
    getActiveCount: jest.fn().mockResolvedValue(0),
    getWaitingCount: jest.fn().mockResolvedValue(0),
    getCompletedCount: jest.fn().mockResolvedValue(0),
    getFailedCount: jest.fn().mockResolvedValue(0),
  });

  beforeEach(async () => {
    codeExecutionQueue = createMockQueue();
    deadLetterQueue = createMockQueue();

    const module: TestingModule = await Test.createTestingModule({
      providers: [
        ShutdownService,
        {
          provide: getQueueToken(CODE_EXECUTION_QUEUE),
          useValue: codeExecutionQueue,
        },
        {
          provide: getQueueToken(DEAD_LETTER_QUEUE),
          useValue: deadLetterQueue,
        },
      ],
    }).compile();

    service = module.get<ShutdownService>(ShutdownService);
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });

  // ============================================
  // beforeApplicationShutdown() - Pause queues
  // ============================================
  describe('beforeApplicationShutdown()', () => {
    it('should pause both queues', async () => {
      await service.beforeApplicationShutdown('SIGTERM');

      expect(codeExecutionQueue.pause).toHaveBeenCalled();
      expect(deadLetterQueue.pause).toHaveBeenCalled();
    });

    it('should handle different shutdown signals', async () => {
      await service.beforeApplicationShutdown('SIGINT');

      expect(codeExecutionQueue.pause).toHaveBeenCalled();
    });

    it('should handle undefined signal', async () => {
      await service.beforeApplicationShutdown();

      expect(codeExecutionQueue.pause).toHaveBeenCalled();
    });

    it('should only execute once (idempotent)', async () => {
      await service.beforeApplicationShutdown('SIGTERM');
      await service.beforeApplicationShutdown('SIGTERM');
      await service.beforeApplicationShutdown('SIGTERM');

      // Should only be called once
      expect(codeExecutionQueue.pause).toHaveBeenCalledTimes(1);
      expect(deadLetterQueue.pause).toHaveBeenCalledTimes(1);
    });

    it('should handle queue pause errors gracefully', async () => {
      codeExecutionQueue.pause.mockRejectedValue(new Error('Pause failed'));

      // Should not throw
      await expect(service.beforeApplicationShutdown('SIGTERM')).resolves.not.toThrow();
    });
  });

  // ============================================
  // onModuleDestroy() - Graceful shutdown
  // ============================================
  describe('onModuleDestroy()', () => {
    it('should close both queues when no active jobs', async () => {
      codeExecutionQueue.getActiveCount.mockResolvedValue(0);

      await service.onModuleDestroy();

      expect(codeExecutionQueue.close).toHaveBeenCalled();
      expect(deadLetterQueue.close).toHaveBeenCalled();
    });

    it('should wait for active jobs to complete', async () => {
      // First call returns 1 active, subsequent calls return 0
      codeExecutionQueue.getActiveCount
        .mockResolvedValueOnce(1)
        .mockResolvedValue(0);

      await service.onModuleDestroy();

      // Should check active count at least twice (initial check + at least one loop iteration)
      expect(codeExecutionQueue.getActiveCount.mock.calls.length).toBeGreaterThanOrEqual(2);
      expect(codeExecutionQueue.close).toHaveBeenCalled();
    });

    it('should check active count multiple times', async () => {
      codeExecutionQueue.getActiveCount
        .mockResolvedValueOnce(2) // Initial check
        .mockResolvedValueOnce(1) // First loop iteration
        .mockResolvedValueOnce(0) // Second loop iteration - done!
        .mockResolvedValueOnce(0); // Final check (if any)

      await service.onModuleDestroy();

      // Should check active count at least twice (initial + loop)
      expect(codeExecutionQueue.getActiveCount.mock.calls.length).toBeGreaterThanOrEqual(2);
    });

    it('should close queues even when jobs remain', async () => {
      // Return active jobs initially, then 0 after first check
      codeExecutionQueue.getActiveCount
        .mockResolvedValueOnce(1) // Initial check shows jobs
        .mockResolvedValue(0); // All subsequent checks show 0

      await service.onModuleDestroy();

      // Should still close queues
      expect(codeExecutionQueue.close).toHaveBeenCalled();
      expect(deadLetterQueue.close).toHaveBeenCalled();
    });

    it('should handle close errors gracefully', async () => {
      codeExecutionQueue.getActiveCount.mockResolvedValue(0);
      codeExecutionQueue.close.mockRejectedValue(new Error('Close failed'));
      deadLetterQueue.close.mockRejectedValue(new Error('Close failed'));

      // Should not throw
      await expect(service.onModuleDestroy()).resolves.not.toThrow();
    });
  });

  // ============================================
  // getQueueStatus() - Queue status
  // ============================================
  describe('getQueueStatus()', () => {
    it('should return queue status', async () => {
      codeExecutionQueue.getWaitingCount.mockResolvedValue(5);
      codeExecutionQueue.getActiveCount.mockResolvedValue(2);
      codeExecutionQueue.getCompletedCount.mockResolvedValue(100);
      codeExecutionQueue.getFailedCount.mockResolvedValue(3);

      const status = await service.getQueueStatus();

      expect(status).toEqual({
        queue: CODE_EXECUTION_QUEUE,
        waiting: 5,
        active: 2,
        completed: 100,
        failed: 3,
        isShuttingDown: false,
      });
    });

    it('should reflect shutdown state', async () => {
      await service.beforeApplicationShutdown('SIGTERM');

      const status = await service.getQueueStatus();

      expect(status.isShuttingDown).toBe(true);
    });

    it('should handle zero values', async () => {
      codeExecutionQueue.getWaitingCount.mockResolvedValue(0);
      codeExecutionQueue.getActiveCount.mockResolvedValue(0);
      codeExecutionQueue.getCompletedCount.mockResolvedValue(0);
      codeExecutionQueue.getFailedCount.mockResolvedValue(0);

      const status = await service.getQueueStatus();

      expect(status.waiting).toBe(0);
      expect(status.active).toBe(0);
      expect(status.completed).toBe(0);
      expect(status.failed).toBe(0);
    });

    it('should handle large job counts', async () => {
      codeExecutionQueue.getWaitingCount.mockResolvedValue(10000);
      codeExecutionQueue.getActiveCount.mockResolvedValue(50);
      codeExecutionQueue.getCompletedCount.mockResolvedValue(1000000);
      codeExecutionQueue.getFailedCount.mockResolvedValue(500);

      const status = await service.getQueueStatus();

      expect(status.waiting).toBe(10000);
      expect(status.completed).toBe(1000000);
    });
  });

  // ============================================
  // Edge cases
  // ============================================
  describe('Edge cases', () => {
    it('should handle multiple shutdown signals', async () => {
      // Call shutdown with different signals
      await service.beforeApplicationShutdown('SIGTERM');
      await service.beforeApplicationShutdown('SIGINT');

      // Should only pause once
      expect(codeExecutionQueue.pause).toHaveBeenCalledTimes(1);
    });

    it('should handle queue.pause returning undefined', async () => {
      codeExecutionQueue.pause.mockResolvedValue(undefined);
      deadLetterQueue.pause.mockResolvedValue(undefined);

      await expect(service.beforeApplicationShutdown('SIGTERM')).resolves.not.toThrow();
    });
  });
});
