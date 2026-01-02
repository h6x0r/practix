import { describe, it, expect, beforeEach, vi } from 'vitest';
import { taskService } from './taskService';

// Mock the api module
vi.mock('@/lib/api', () => ({
  api: {
    get: vi.fn(),
    post: vi.fn(),
  },
  isAbortError: vi.fn((error) => error instanceof DOMException && error.name === 'AbortError'),
}));

// Mock storage
vi.mock('@/lib/storage', () => ({
  storage: {
    getCompletedTasks: vi.fn(() => []),
    addCompletedTask: vi.fn(),
  },
}));

// Mock logger
vi.mock('@/lib/logger', () => ({
  createLogger: () => ({
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
    debug: vi.fn(),
  }),
}));

import { api, isAbortError } from '@/lib/api';
import { storage } from '@/lib/storage';

describe('taskService', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('fetchTask', () => {
    it('should fetch a task by slug', async () => {
      const mockTask = {
        id: 'task-1',
        slug: 'hello-world',
        title: 'Hello World',
        description: 'Write hello world',
        initialCode: 'package main',
      };

      vi.mocked(api.get).mockResolvedValueOnce(mockTask);

      const result = await taskService.fetchTask('hello-world');

      expect(api.get).toHaveBeenCalledWith('/tasks/hello-world', undefined);
      expect(result).toEqual(mockTask);
    });

    it('should pass request options', async () => {
      const controller = new AbortController();
      vi.mocked(api.get).mockResolvedValueOnce({});

      await taskService.fetchTask('test', { signal: controller.signal });

      expect(api.get).toHaveBeenCalledWith('/tasks/test', { signal: controller.signal });
    });
  });

  describe('getRecentTasks', () => {
    it('should fetch recent tasks', async () => {
      const mockTasks = [
        { id: 'task-1', slug: 'task-1', title: 'Task 1' },
        { id: 'task-2', slug: 'task-2', title: 'Task 2' },
      ];

      vi.mocked(api.get).mockResolvedValueOnce(mockTasks);

      const result = await taskService.getRecentTasks();

      expect(api.get).toHaveBeenCalledWith('/tasks', undefined);
      expect(result).toEqual(mockTasks);
    });
  });

  describe('runTests', () => {
    it('should run quick tests', async () => {
      const mockResult = {
        status: 'passed',
        testsPassed: 5,
        testsTotal: 5,
        testCases: [],
        runtime: '100ms',
        message: '',
      };

      vi.mocked(api.post).mockResolvedValueOnce(mockResult);

      const result = await taskService.runTests(
        'package main',
        'task-123',
        'go'
      );

      expect(api.post).toHaveBeenCalledWith(
        '/submissions/run-tests',
        { code: 'package main', taskId: 'task-123', language: 'go' },
        undefined
      );
      expect(result).toEqual(mockResult);
    });

    it('should pass abort signal', async () => {
      const controller = new AbortController();
      vi.mocked(api.post).mockResolvedValueOnce({});

      await taskService.runTests('code', 'task-1', 'go', { signal: controller.signal });

      expect(api.post).toHaveBeenCalledWith(
        '/submissions/run-tests',
        expect.any(Object),
        { signal: controller.signal }
      );
    });
  });

  describe('submitCode', () => {
    it('should submit code for full evaluation', async () => {
      const mockSubmission = {
        id: 'sub-1',
        status: 'passed',
        score: 100,
        testsPassed: 10,
        testsTotal: 10,
        runtime: '150ms',
        createdAt: '2024-01-01T00:00:00Z',
        code: 'package main',
        message: '',
      };

      vi.mocked(api.post).mockResolvedValueOnce(mockSubmission);

      const result = await taskService.submitCode(
        'package main',
        'task-123',
        'go'
      );

      expect(api.post).toHaveBeenCalledWith(
        '/submissions',
        { code: 'package main', taskId: 'task-123', language: 'go' },
        undefined
      );
      expect(result).toEqual(mockSubmission);
    });
  });

  describe('getTaskSubmissions', () => {
    it('should fetch submissions for a task', async () => {
      const mockSubmissions = [
        { id: 'sub-1', status: 'passed', score: 100 },
        { id: 'sub-2', status: 'failed', score: 50 },
      ];

      vi.mocked(api.get).mockResolvedValueOnce(mockSubmissions);

      const result = await taskService.getTaskSubmissions('task-123');

      expect(api.get).toHaveBeenCalledWith('/submissions/task/task-123', undefined);
      expect(result).toEqual(mockSubmissions);
    });

    it('should return empty array on error', async () => {
      vi.mocked(api.get).mockRejectedValueOnce(new Error('Network error'));

      const result = await taskService.getTaskSubmissions('task-123');

      expect(result).toEqual([]);
    });

    it('should re-throw abort errors', async () => {
      const abortError = new DOMException('Aborted', 'AbortError');
      vi.mocked(api.get).mockRejectedValueOnce(abortError);
      vi.mocked(isAbortError).mockReturnValueOnce(true);

      await expect(taskService.getTaskSubmissions('task-123')).rejects.toThrow();
    });
  });

  describe('getRecentSubmissions', () => {
    it('should fetch recent submissions with default limit', async () => {
      const mockSubmissions = [{ id: 'sub-1' }];
      vi.mocked(api.get).mockResolvedValueOnce(mockSubmissions);

      const result = await taskService.getRecentSubmissions();

      expect(api.get).toHaveBeenCalledWith('/submissions/user/recent?limit=10');
      expect(result).toEqual(mockSubmissions);
    });

    it('should fetch recent submissions with custom limit', async () => {
      vi.mocked(api.get).mockResolvedValueOnce([]);

      await taskService.getRecentSubmissions(5);

      expect(api.get).toHaveBeenCalledWith('/submissions/user/recent?limit=5');
    });

    it('should return empty array on error', async () => {
      vi.mocked(api.get).mockRejectedValueOnce(new Error('Error'));

      const result = await taskService.getRecentSubmissions();

      expect(result).toEqual([]);
    });
  });

  describe('getCompletedTaskIds', () => {
    it('should return completed task IDs from storage', () => {
      vi.mocked(storage.getCompletedTasks).mockReturnValueOnce(['task-1', 'task-2']);

      const result = taskService.getCompletedTaskIds();

      expect(storage.getCompletedTasks).toHaveBeenCalled();
      expect(result).toEqual(['task-1', 'task-2']);
    });
  });

  describe('markTaskAsCompleted', () => {
    it('should mark task as completed in storage', () => {
      taskService.markTaskAsCompleted('task-123');

      expect(storage.addCompletedTask).toHaveBeenCalledWith('task-123');
    });
  });

  describe('isResourceCompleted', () => {
    it('should return true if task is completed', () => {
      vi.mocked(storage.getCompletedTasks).mockReturnValueOnce(['task-1', 'task-2']);

      const result = taskService.isResourceCompleted('task-1', 'task');

      expect(result).toBe(true);
    });

    it('should return false if task is not completed', () => {
      vi.mocked(storage.getCompletedTasks).mockReturnValueOnce(['task-1']);

      const result = taskService.isResourceCompleted('task-3', 'task');

      expect(result).toBe(false);
    });

    it('should return false for topic type', () => {
      vi.mocked(storage.getCompletedTasks).mockReturnValueOnce(['topic-1']);

      const result = taskService.isResourceCompleted('topic-1', 'topic');

      expect(result).toBe(false);
    });
  });
});
