import { describe, it, expect, beforeEach, vi } from 'vitest';
import { renderHook, waitFor, act } from '@testing-library/react';
import { useTaskState } from './useTaskState';

vi.mock('../api/taskService', () => ({
  taskService: {
    fetchTask: vi.fn(),
  },
}));

vi.mock('@/lib/logger', () => ({
  createLogger: () => ({
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
    debug: vi.fn(),
  }),
}));

import { taskService } from '../api/taskService';

describe('useTaskState', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  const mockTask = {
    id: 'task-123',
    slug: 'hello-world',
    title: 'Hello World',
    description: 'Print hello world',
    initialCode: 'console.log("hello")',
    difficulty: 'easy' as const,
    tags: ['javascript'],
  };

  describe('initial state', () => {
    it('should start with loading state when slug provided', () => {
      vi.mocked(taskService.fetchTask).mockResolvedValue(mockTask);

      const { result } = renderHook(() => useTaskState('hello-world'));

      expect(result.current.isLoading).toBe(true);
      expect(result.current.task).toBeNull();
      expect(result.current.error).toBeNull();
    });

    it('should not be loading when no slug provided', async () => {
      const { result } = renderHook(() => useTaskState(undefined));

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.task).toBeNull();
      expect(taskService.fetchTask).not.toHaveBeenCalled();
    });
  });

  describe('successful fetch', () => {
    it('should fetch and set task', async () => {
      vi.mocked(taskService.fetchTask).mockResolvedValue(mockTask);

      const { result } = renderHook(() => useTaskState('hello-world'));

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(taskService.fetchTask).toHaveBeenCalledWith('hello-world');
      expect(result.current.task).toEqual(mockTask);
      expect(result.current.error).toBeNull();
    });

    it('should refetch when slug changes', async () => {
      vi.mocked(taskService.fetchTask).mockResolvedValue(mockTask);

      const { result, rerender } = renderHook(
        ({ slug }) => useTaskState(slug),
        { initialProps: { slug: 'task-1' } }
      );

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(taskService.fetchTask).toHaveBeenCalledWith('task-1');

      const newTask = { ...mockTask, slug: 'task-2', title: 'Task 2' };
      vi.mocked(taskService.fetchTask).mockResolvedValue(newTask);

      rerender({ slug: 'task-2' });

      await waitFor(() => {
        expect(result.current.task?.title).toBe('Task 2');
      });

      expect(taskService.fetchTask).toHaveBeenCalledWith('task-2');
    });
  });

  describe('error handling', () => {
    it('should handle 404 error', async () => {
      const error = { status: 404, message: 'Not found' };
      vi.mocked(taskService.fetchTask).mockRejectedValue(error);

      const { result } = renderHook(() => useTaskState('non-existent'));

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.task).toBeNull();
      expect(result.current.error).toEqual({
        message: 'Task not found',
        status: 404,
        isNetworkError: false,
      });
    });

    it('should handle network error with "Failed to fetch"', async () => {
      const error = new Error('Failed to fetch');
      vi.mocked(taskService.fetchTask).mockRejectedValue(error);

      const { result } = renderHook(() => useTaskState('hello-world'));

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.error?.isNetworkError).toBe(true);
      expect(result.current.error?.message).toBe('Network error. Please check your connection.');
    });

    it('should handle network error with "network" in message', async () => {
      const error = new Error('network connection lost');
      vi.mocked(taskService.fetchTask).mockRejectedValue(error);

      const { result } = renderHook(() => useTaskState('hello-world'));

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.error?.isNetworkError).toBe(true);
    });

    it('should handle generic error', async () => {
      const error = new Error('Server error');
      vi.mocked(taskService.fetchTask).mockRejectedValue(error);

      const { result } = renderHook(() => useTaskState('hello-world'));

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.error?.isNetworkError).toBe(false);
      expect(result.current.error?.message).toBe('Failed to load task. Please try again.');
    });

    it('should handle error with status code', async () => {
      const error = { status: 500, message: 'Internal server error' };
      vi.mocked(taskService.fetchTask).mockRejectedValue(error);

      const { result } = renderHook(() => useTaskState('hello-world'));

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.error?.status).toBe(500);
    });
  });

  describe('retry functionality', () => {
    it('should retry fetch when retry is called', async () => {
      vi.mocked(taskService.fetchTask).mockRejectedValueOnce(new Error('Network error'));

      const { result } = renderHook(() => useTaskState('hello-world'));

      await waitFor(() => {
        expect(result.current.error).not.toBeNull();
      });

      // Now succeed on retry
      vi.mocked(taskService.fetchTask).mockResolvedValue(mockTask);

      await act(async () => {
        result.current.retry();
      });

      await waitFor(() => {
        expect(result.current.task).toEqual(mockTask);
      });

      expect(taskService.fetchTask).toHaveBeenCalledTimes(2);
    });

    it('should not retry when no slug', async () => {
      const { result } = renderHook(() => useTaskState(undefined));

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      await act(async () => {
        result.current.retry();
      });

      expect(taskService.fetchTask).not.toHaveBeenCalled();
    });

    it('should clear error on successful retry', async () => {
      vi.mocked(taskService.fetchTask).mockRejectedValueOnce({ status: 500 });

      const { result } = renderHook(() => useTaskState('hello-world'));

      await waitFor(() => {
        expect(result.current.error).not.toBeNull();
      });

      vi.mocked(taskService.fetchTask).mockResolvedValue(mockTask);

      await act(async () => {
        result.current.retry();
      });

      await waitFor(() => {
        expect(result.current.error).toBeNull();
      });

      expect(result.current.task).toEqual(mockTask);
    });
  });
});
