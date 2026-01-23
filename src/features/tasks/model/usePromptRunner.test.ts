import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import { usePromptRunner } from './usePromptRunner';

// Use vi.hoisted for mocks
const { isAbortErrorMock } = vi.hoisted(() => {
  return {
    isAbortErrorMock: vi.fn((error: unknown) => {
      return error instanceof DOMException && error.name === 'AbortError';
    }),
  };
});

vi.mock('../api/taskService', () => ({
  taskService: {
    getTaskSubmissions: vi.fn(),
    submitPrompt: vi.fn(),
  },
}));

vi.mock('@/lib/storage', () => ({
  storage: {
    getTaskCode: vi.fn(),
    setTaskCode: vi.fn(),
  },
}));

vi.mock('@/lib/api', () => ({
  isAbortError: isAbortErrorMock,
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
import { storage } from '@/lib/storage';

describe('usePromptRunner', () => {
  const mockTask = {
    id: 'task-123',
    slug: 'prompt-task',
    title: 'Prompt Task',
    description: 'Create a prompt',
    initialCode: 'Initial prompt template',
    difficulty: 'easy' as const,
    tags: ['prompt'],
  };

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(storage.getTaskCode).mockReturnValue(null);
    vi.mocked(taskService.getTaskSubmissions).mockResolvedValue([]);
    isAbortErrorMock.mockImplementation((error: unknown) => {
      return error instanceof DOMException && error.name === 'AbortError';
    });
  });

  describe('initial state', () => {
    it('should use initialCode when no saved prompt', async () => {
      vi.mocked(storage.getTaskCode).mockReturnValue(null);

      const { result } = renderHook(() => usePromptRunner(mockTask));

      expect(result.current.prompt).toBe('Initial prompt template');
    });

    it('should use saved prompt from storage', async () => {
      vi.mocked(storage.getTaskCode).mockReturnValue('Saved prompt');

      const { result } = renderHook(() => usePromptRunner(mockTask));

      expect(result.current.prompt).toBe('Saved prompt');
    });

    it('should start on editor tab', () => {
      const { result } = renderHook(() => usePromptRunner(mockTask));

      expect(result.current.activeTab).toBe('editor');
    });

    it('should have empty submissions initially', () => {
      const { result } = renderHook(() => usePromptRunner(mockTask));

      expect(result.current.submissions).toEqual([]);
    });

    it('should not be submitting initially', () => {
      const { result } = renderHook(() => usePromptRunner(mockTask));

      expect(result.current.isSubmitting).toBe(false);
    });
  });

  describe('prompt management', () => {
    it('should update prompt', () => {
      const { result } = renderHook(() => usePromptRunner(mockTask));

      act(() => {
        result.current.setPrompt('New prompt');
      });

      expect(result.current.prompt).toBe('New prompt');
    });

    describe('debounce behavior', () => {
      beforeEach(() => {
        vi.useFakeTimers();
      });

      afterEach(() => {
        vi.useRealTimers();
      });

      it('should debounce saving to storage', async () => {
        const { result } = renderHook(() => usePromptRunner(mockTask));

        act(() => {
          result.current.setPrompt('Prompt 1');
        });

        // Storage should not be called immediately
        expect(storage.setTaskCode).not.toHaveBeenCalled();

        // Fast forward 400ms - still not saved
        act(() => {
          vi.advanceTimersByTime(400);
        });
        expect(storage.setTaskCode).not.toHaveBeenCalled();

        // Fast forward past 500ms - now saved
        act(() => {
          vi.advanceTimersByTime(200);
        });
        expect(storage.setTaskCode).toHaveBeenCalledWith('prompt-task', 'Prompt 1');
      });

      it('should reset debounce on rapid changes', async () => {
        const { result } = renderHook(() => usePromptRunner(mockTask));

        act(() => {
          result.current.setPrompt('Prompt 1');
        });

        act(() => {
          vi.advanceTimersByTime(300);
        });

        act(() => {
          result.current.setPrompt('Prompt 2');
        });

        act(() => {
          vi.advanceTimersByTime(300);
        });

        // Not yet saved (reset timer)
        expect(storage.setTaskCode).not.toHaveBeenCalled();

        act(() => {
          vi.advanceTimersByTime(300);
        });

        // Now saved with final value
        expect(storage.setTaskCode).toHaveBeenCalledWith('prompt-task', 'Prompt 2');
        expect(storage.setTaskCode).toHaveBeenCalledTimes(1);
      });
    });
  });

  describe('tab management', () => {
    it('should switch tabs', () => {
      const { result } = renderHook(() => usePromptRunner(mockTask));

      act(() => {
        result.current.setActiveTab('history');
      });

      expect(result.current.activeTab).toBe('history');

      act(() => {
        result.current.setActiveTab('editor');
      });

      expect(result.current.activeTab).toBe('editor');
    });
  });

  describe('loading submissions', () => {
    it('should load submissions on mount', async () => {
      const mockSubmissions = [
        {
          id: 'sub-1',
          status: 'passed',
          score: 100,
          message: 'Great!',
          createdAt: '2025-01-16T00:00:00Z',
          testCases: [
            { passed: true, input: 'test', actualOutput: 'output' },
          ],
        },
      ];
      vi.mocked(taskService.getTaskSubmissions).mockResolvedValue(mockSubmissions);

      const { result } = renderHook(() => usePromptRunner(mockTask));

      // Initially loading
      expect(result.current.isLoadingSubmissions).toBe(true);

      await waitFor(() => {
        expect(result.current.isLoadingSubmissions).toBe(false);
      });

      expect(taskService.getTaskSubmissions).toHaveBeenCalledWith(
        'task-123',
        { signal: expect.any(AbortSignal) }
      );
      expect(result.current.submissions).toHaveLength(1);
    });

    it('should convert submissions to prompt format', async () => {
      const mockSubmissions = [
        {
          id: 'sub-1',
          status: 'passed',
          score: 80,
          message: 'Good job',
          createdAt: '2025-01-16T00:00:00Z',
          testCases: [
            { passed: true, input: 'in1', actualOutput: 'out1', error: null, expectedOutput: '' },
            { passed: false, input: 'in2', actualOutput: 'out2', error: 'Wrong', expectedOutput: 'expected' },
          ],
        },
      ];
      vi.mocked(taskService.getTaskSubmissions).mockResolvedValue(mockSubmissions);

      const { result } = renderHook(() => usePromptRunner(mockTask));

      await waitFor(() => {
        expect(result.current.isLoadingSubmissions).toBe(false);
      });

      expect(result.current.submissions[0].scenarioResults).toHaveLength(2);
      expect(result.current.submissions[0].scenarioResults[0].passed).toBe(true);
      expect(result.current.submissions[0].scenarioResults[1].passed).toBe(false);
    });

    it('should handle submission load error', async () => {
      vi.mocked(taskService.getTaskSubmissions).mockRejectedValue(new Error('Network error'));

      const { result } = renderHook(() => usePromptRunner(mockTask));

      await waitFor(() => {
        expect(result.current.isLoadingSubmissions).toBe(false);
      });

      expect(result.current.submissions).toEqual([]);
    });

    it('should not load submissions when task is null', () => {
      const { result } = renderHook(() => usePromptRunner(null));

      expect(taskService.getTaskSubmissions).not.toHaveBeenCalled();
      expect(result.current.submissions).toEqual([]);
    });
  });

  describe('submitPrompt', () => {
    it('should not submit when task is null', async () => {
      const { result } = renderHook(() => usePromptRunner(null));

      await act(async () => {
        await result.current.submitPrompt();
      });

      expect(taskService.submitPrompt).not.toHaveBeenCalled();
    });

    it('should submit prompt and add to submissions', async () => {
      const mockResult = {
        id: 'new-sub',
        status: 'passed',
        score: 100,
        message: 'Perfect!',
        createdAt: '2025-01-16T12:00:00Z',
        scenarioResults: [
          { scenarioIndex: 0, input: 'test', output: 'result', score: 10, feedback: 'Good', passed: true },
        ],
        summary: 'All tests passed',
        xpEarned: 50,
        totalXp: 500,
        level: 5,
        leveledUp: false,
      };
      vi.mocked(taskService.submitPrompt).mockResolvedValue(mockResult);

      const { result } = renderHook(() => usePromptRunner(mockTask));

      act(() => {
        result.current.setPrompt('My prompt');
      });

      await act(async () => {
        await result.current.submitPrompt();
      });

      expect(taskService.submitPrompt).toHaveBeenCalledWith(
        'My prompt',
        'task-123',
        { signal: expect.any(AbortSignal) }
      );
      expect(result.current.submissions[0].id).toBe('new-sub');
      expect(result.current.submissions[0].score).toBe(100);
    });

    it('should switch to history tab on submit', async () => {
      vi.mocked(taskService.submitPrompt).mockResolvedValue({
        id: 'sub-1',
        status: 'passed',
        score: 100,
        message: '',
        createdAt: '2025-01-16T00:00:00Z',
        scenarioResults: [],
        summary: '',
      });

      const { result } = renderHook(() => usePromptRunner(mockTask));

      expect(result.current.activeTab).toBe('editor');

      await act(async () => {
        await result.current.submitPrompt();
      });

      expect(result.current.activeTab).toBe('history');
    });

    it('should set isSubmitting during submission', async () => {
      let resolvePromise: (value: any) => void;
      const promise = new Promise((resolve) => {
        resolvePromise = resolve;
      });
      vi.mocked(taskService.submitPrompt).mockReturnValue(promise);

      const { result } = renderHook(() => usePromptRunner(mockTask));

      act(() => {
        result.current.submitPrompt();
      });

      expect(result.current.isSubmitting).toBe(true);

      await act(async () => {
        resolvePromise!({
          id: 'sub-1',
          status: 'passed',
          score: 100,
          message: '',
          createdAt: '2025-01-16T00:00:00Z',
          scenarioResults: [],
          summary: '',
        });
      });

      expect(result.current.isSubmitting).toBe(false);
    });

    it('should handle submit error', async () => {
      vi.mocked(taskService.submitPrompt).mockRejectedValue(new Error('API Error'));

      const { result } = renderHook(() => usePromptRunner(mockTask));

      await act(async () => {
        await result.current.submitPrompt();
      });

      expect(result.current.isSubmitting).toBe(false);
      expect(result.current.submissions[0].status).toBe('error');
      expect(result.current.submissions[0].message).toContain('Failed to evaluate');
    });

    it('should handle abort error silently', async () => {
      const abortError = new DOMException('Aborted', 'AbortError');
      vi.mocked(taskService.submitPrompt).mockRejectedValue(abortError);
      isAbortErrorMock.mockReturnValue(true);

      const { result } = renderHook(() => usePromptRunner(mockTask));

      await act(async () => {
        await result.current.submitPrompt();
      });

      // No error submission added
      expect(result.current.submissions).toHaveLength(0);
    });

    it('should prepend new submission to list', async () => {
      const existingSubs = [
        {
          id: 'old-sub',
          status: 'passed' as const,
          score: 80,
          message: 'Old',
          createdAt: '2025-01-15T00:00:00Z',
          scenarioResults: [],
          summary: '',
        },
      ];
      vi.mocked(taskService.getTaskSubmissions).mockResolvedValue([
        { id: 'old-sub', status: 'passed', score: 80, message: 'Old', createdAt: '2025-01-15T00:00:00Z', testCases: [] },
      ]);
      vi.mocked(taskService.submitPrompt).mockResolvedValue({
        id: 'new-sub',
        status: 'passed',
        score: 100,
        message: 'New',
        createdAt: '2025-01-16T00:00:00Z',
        scenarioResults: [],
        summary: '',
      });

      const { result } = renderHook(() => usePromptRunner(mockTask));

      await waitFor(() => {
        expect(result.current.submissions).toHaveLength(1);
      });

      await act(async () => {
        await result.current.submitPrompt();
      });

      expect(result.current.submissions).toHaveLength(2);
      expect(result.current.submissions[0].id).toBe('new-sub'); // New first
      expect(result.current.submissions[1].id).toBe('old-sub'); // Old second
    });
  });

  describe('task change', () => {
    it('should reload prompt when task changes', async () => {
      vi.mocked(storage.getTaskCode).mockReturnValue(null);

      const { result, rerender } = renderHook(
        ({ task }) => usePromptRunner(task),
        { initialProps: { task: mockTask } }
      );

      expect(result.current.prompt).toBe('Initial prompt template');

      const newTask = {
        ...mockTask,
        id: 'task-456',
        slug: 'new-task',
        initialCode: 'New initial code',
      };

      rerender({ task: newTask });

      expect(result.current.prompt).toBe('New initial code');
    });

    it('should load saved code for new task', async () => {
      vi.mocked(storage.getTaskCode)
        .mockReturnValueOnce(null)
        .mockReturnValueOnce('Saved for new task');

      const { result, rerender } = renderHook(
        ({ task }) => usePromptRunner(task),
        { initialProps: { task: mockTask } }
      );

      const newTask = { ...mockTask, id: 'task-456', slug: 'new-task' };
      rerender({ task: newTask });

      expect(result.current.prompt).toBe('Saved for new task');
    });
  });
});
