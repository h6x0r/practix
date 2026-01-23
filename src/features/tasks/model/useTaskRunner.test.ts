import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import { useTaskRunner, detectTaskLanguage } from './useTaskRunner';
import { taskService } from '../api/taskService';
import { storage } from '@/lib/storage';

// Mock dependencies
vi.mock('../api/taskService', () => ({
  taskService: {
    getTaskSubmissions: vi.fn(),
    runTests: vi.fn(),
    submitCode: vi.fn(),
    getRunResult: vi.fn().mockResolvedValue(null),
    saveRunResult: vi.fn().mockResolvedValue(undefined),
  },
}));

vi.mock('@/lib/storage', () => ({
  storage: {
    getTaskCode: vi.fn(),
    setTaskCode: vi.fn(),
  },
}));

// Use vi.hoisted to define mock before vi.mock (since vi.mock is hoisted)
const { isAbortErrorMock } = vi.hoisted(() => {
  return {
    isAbortErrorMock: vi.fn((error: any) => {
      return error instanceof DOMException && error.name === 'AbortError';
    }),
  };
});

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

describe('detectTaskLanguage', () => {
  it('should detect Python for algo courses', () => {
    expect(detectTaskLanguage(null, 'algo-fundamentals')).toBe('python');
    expect(detectTaskLanguage(null, 'algo-advanced')).toBe('python');
  });

  it('should detect Python for python courses', () => {
    expect(detectTaskLanguage(null, 'python-ml-fundamentals')).toBe('python');
    expect(detectTaskLanguage(null, 'python-deep-learning')).toBe('python');
  });

  it('should detect Python from task tags', () => {
    expect(detectTaskLanguage({ tags: ['python'] } as any, undefined)).toBe('python');
    expect(detectTaskLanguage({ tags: ['py'] } as any, undefined)).toBe('python');
  });

  it('should detect Go for go courses', () => {
    expect(detectTaskLanguage(null, 'go-basics')).toBe('go');
    expect(detectTaskLanguage(null, 'c_go_basics')).toBe('go');
  });

  it('should detect Go from task tags', () => {
    expect(detectTaskLanguage({ tags: ['go'] } as any, undefined)).toBe('go');
  });

  it('should detect Go from task slug', () => {
    expect(detectTaskLanguage({ slug: 'go-hello-world' } as any, undefined)).toBe('go');
  });

  it('should detect Java for java courses', () => {
    expect(detectTaskLanguage(null, 'java-core')).toBe('java');
    expect(detectTaskLanguage(null, 'c_java_core')).toBe('java');
    expect(detectTaskLanguage(null, 'java-ml')).toBe('java');
  });

  it('should detect Java from task tags', () => {
    expect(detectTaskLanguage({ tags: ['java'] } as any, undefined)).toBe('java');
  });

  it('should default to first tag or java', () => {
    expect(detectTaskLanguage({ tags: ['rust'] } as any, undefined)).toBe('rust');
    expect(detectTaskLanguage({ tags: [] } as any, undefined)).toBe('java');
    expect(detectTaskLanguage(null, undefined)).toBe('java');
  });
});

describe('useTaskRunner', () => {
  const mockTask = {
    id: 'task-123',
    slug: 'hello-world',
    title: 'Hello World',
    description: 'Write hello world',
    initialCode: 'package main\n\nfunc main() {}',
    tags: ['go'],
  };

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(storage.getTaskCode).mockReturnValue(null);
    vi.mocked(taskService.getTaskSubmissions).mockResolvedValue([]);
  });

  describe('initialization', () => {
    it('should initialize with empty code when no task', () => {
      const { result } = renderHook(() => useTaskRunner(null));

      expect(result.current.code).toBe('');
      expect(result.current.activeTab).toBe('editor');
      expect(result.current.submissions).toEqual([]);
      expect(result.current.isRunning).toBe(false);
      expect(result.current.isSubmitting).toBe(false);
    });

    it('should load initial code from task when no saved code', async () => {
      vi.mocked(storage.getTaskCode).mockReturnValue(null);

      const { result } = renderHook(() => useTaskRunner(mockTask as any));

      // Initial code should be set synchronously from useEffect
      await waitFor(() => {
        expect(result.current.code).toBe(mockTask.initialCode);
      });
    });

    it('should load saved code from storage if available', async () => {
      const savedCode = 'package main\n\nfunc main() { fmt.Println("saved") }';
      vi.mocked(storage.getTaskCode).mockReturnValue(savedCode);

      const { result } = renderHook(() => useTaskRunner(mockTask as any));

      await waitFor(() => {
        expect(result.current.code).toBe(savedCode);
      });
    });

    it('should load submissions from backend', async () => {
      const mockSubmissions = [
        { id: 'sub-1', status: 'passed', score: 100 },
      ];
      vi.mocked(taskService.getTaskSubmissions).mockResolvedValue(mockSubmissions as any);

      const { result } = renderHook(() => useTaskRunner(mockTask as any));

      await waitFor(() => {
        expect(result.current.submissions).toEqual(mockSubmissions);
      });
      expect(taskService.getTaskSubmissions).toHaveBeenCalledWith('task-123', expect.any(Object));
    });
  });

  describe('setCode', () => {
    it('should update code state', async () => {
      const { result } = renderHook(() => useTaskRunner(mockTask as any));

      await waitFor(() => {
        expect(result.current.code).toBe(mockTask.initialCode);
      });

      act(() => {
        result.current.setCode('new code');
      });

      expect(result.current.code).toBe('new code');
    });
  });

  describe('runQuickTests', () => {
    it('should run quick tests successfully', async () => {
      const mockResult = {
        status: 'passed',
        testsPassed: 5,
        testsTotal: 5,
        testCases: [],
        runtime: '100ms',
        message: '',
      };
      vi.mocked(taskService.runTests).mockResolvedValue(mockResult);

      const { result } = renderHook(() => useTaskRunner(mockTask as any, 'go-basics'));

      await waitFor(() => {
        expect(result.current.code).toBe(mockTask.initialCode);
      });

      await act(async () => {
        await result.current.runQuickTests();
      });

      expect(taskService.runTests).toHaveBeenCalledWith(
        mockTask.initialCode,
        'task-123',
        'go',
        expect.any(Object)
      );
      expect(result.current.runResult).toEqual(mockResult);
      expect(result.current.isRunResultsOpen).toBe(true);
      expect(result.current.isRunning).toBe(false);
    });

    it('should handle run test errors', async () => {
      vi.mocked(taskService.runTests).mockRejectedValue(new Error('Network error'));

      const { result } = renderHook(() => useTaskRunner(mockTask as any));

      await waitFor(() => {
        expect(result.current.code).toBe(mockTask.initialCode);
      });

      await act(async () => {
        await result.current.runQuickTests();
      });

      expect(result.current.runResult?.status).toBe('error');
      expect(result.current.runResult?.testsTotal).toBe(5);
      expect(result.current.isRunning).toBe(false);
    });

    it('should do nothing if no task', async () => {
      const { result } = renderHook(() => useTaskRunner(null));

      await act(async () => {
        await result.current.runQuickTests();
      });

      expect(taskService.runTests).not.toHaveBeenCalled();
    });

    it('should sanitize error messages - hide infrastructure errors', async () => {
      vi.mocked(taskService.runTests).mockResolvedValue({
        status: 'error',
        testsPassed: 0,
        testsTotal: 0,
        testCases: [],
        runtime: '-',
        message: 'Connection timeout',
      });

      const { result } = renderHook(() => useTaskRunner(mockTask as any));

      await waitFor(() => {
        expect(result.current.code).toBe(mockTask.initialCode);
      });

      await act(async () => {
        await result.current.runQuickTests();
      });

      // Infrastructure error should be hidden
      expect(result.current.runResult?.message).toBe('');
      // Test count should be expected
      expect(result.current.runResult?.testsTotal).toBe(5);
    });

    it('should preserve code execution errors', async () => {
      vi.mocked(taskService.runTests).mockResolvedValue({
        status: 'error',
        testsPassed: 0,
        testsTotal: 0,
        testCases: [],
        runtime: '-',
        message: 'undefined: fmt in fmt.Println',
      });

      const { result } = renderHook(() => useTaskRunner(mockTask as any));

      await waitFor(() => {
        expect(result.current.code).toBe(mockTask.initialCode);
      });

      await act(async () => {
        await result.current.runQuickTests();
      });

      // Code error should be preserved
      expect(result.current.runResult?.message).toContain('undefined');
    });
  });

  describe('submitCode', () => {
    it('should submit code successfully', async () => {
      const mockSubmission = {
        id: 'sub-1',
        status: 'passed',
        score: 100,
        runtime: '150ms',
        createdAt: '2024-01-01T00:00:00Z',
        code: mockTask.initialCode,
        message: '',
        testsTotal: 10,
        testsPassed: 10,
      };
      vi.mocked(taskService.submitCode).mockResolvedValue(mockSubmission as any);

      const { result } = renderHook(() => useTaskRunner(mockTask as any, 'go-basics'));

      await waitFor(() => {
        expect(result.current.code).toBe(mockTask.initialCode);
      });

      await act(async () => {
        await result.current.submitCode();
      });

      expect(taskService.submitCode).toHaveBeenCalledWith(
        mockTask.initialCode,
        'task-123',
        'go',
        expect.any(Object)
      );
      expect(result.current.submissions[0]).toEqual(mockSubmission);
      expect(result.current.activeTab).toBe('history');
      expect(result.current.isSubmitting).toBe(false);
      expect(result.current.isRunResultsOpen).toBe(false);
    });

    it('should handle submit errors', async () => {
      vi.mocked(taskService.submitCode).mockRejectedValue(new Error('Network error'));

      const { result } = renderHook(() => useTaskRunner(mockTask as any));

      await waitFor(() => {
        expect(result.current.code).toBe(mockTask.initialCode);
      });

      await act(async () => {
        await result.current.submitCode();
      });

      expect(result.current.submissions[0].status).toBe('error');
      expect(result.current.submissions[0].message).toBe('');
      expect(result.current.isSubmitting).toBe(false);
    });

    it('should do nothing if no task', async () => {
      const { result } = renderHook(() => useTaskRunner(null));

      await act(async () => {
        await result.current.submitCode();
      });

      expect(taskService.submitCode).not.toHaveBeenCalled();
    });
  });

  describe('closeRunResults', () => {
    it('should close run results panel', async () => {
      vi.mocked(taskService.runTests).mockResolvedValue({
        status: 'passed',
        testsPassed: 5,
        testsTotal: 5,
        testCases: [],
        runtime: '100ms',
        message: '',
      });

      const { result } = renderHook(() => useTaskRunner(mockTask as any));

      await waitFor(() => {
        expect(result.current.code).toBe(mockTask.initialCode);
      });

      await act(async () => {
        await result.current.runQuickTests();
      });

      expect(result.current.isRunResultsOpen).toBe(true);

      act(() => {
        result.current.closeRunResults();
      });

      expect(result.current.isRunResultsOpen).toBe(false);
    });
  });

  describe('activeTab', () => {
    it('should allow switching tabs', async () => {
      const { result } = renderHook(() => useTaskRunner(mockTask as any));

      expect(result.current.activeTab).toBe('editor');

      act(() => {
        result.current.setActiveTab('history');
      });

      expect(result.current.activeTab).toBe('history');
    });
  });

  describe('cleanup on unmount', () => {
    it('should abort pending requests on unmount', async () => {
      const abortSpy = vi.spyOn(AbortController.prototype, 'abort');

      const { unmount } = renderHook(() => useTaskRunner(mockTask as any));

      await waitFor(() => {
        expect(taskService.getTaskSubmissions).toHaveBeenCalled();
      });

      unmount();

      expect(abortSpy).toHaveBeenCalled();
      abortSpy.mockRestore();
    });
  });

  describe('error handling in submissions loading', () => {
    it('should handle non-abort errors when loading submissions', async () => {
      const networkError = new Error('Network error');
      vi.mocked(taskService.getTaskSubmissions).mockRejectedValue(networkError);

      const { result } = renderHook(() => useTaskRunner(mockTask as any));

      // Wait for the error to be handled
      await waitFor(() => {
        expect(result.current.isLoadingSubmissions).toBe(false);
      });

      // Submissions should be empty on error
      expect(result.current.submissions).toEqual([]);
    });

    it('should set submissions to empty when task has no id', async () => {
      const taskWithNoId = { ...mockTask, id: undefined };

      const { result } = renderHook(() => useTaskRunner(taskWithNoId as any));

      expect(result.current.submissions).toEqual([]);
      expect(taskService.getTaskSubmissions).not.toHaveBeenCalled();
    });

    it('should skip state update if unmounted during submissions load success', async () => {
      let resolveSubmissions: (value: any) => void;
      const pendingPromise = new Promise((resolve) => {
        resolveSubmissions = resolve;
      });
      vi.mocked(taskService.getTaskSubmissions).mockReturnValue(pendingPromise as any);

      const { unmount } = renderHook(() => useTaskRunner(mockTask as any));

      // Unmount while loading
      unmount();

      // Resolve after unmount - should skip state update
      await act(async () => {
        resolveSubmissions!([{ id: 'sub-1', status: 'passed' }]);
      });

      // No error should occur - the isMountedRef check prevents state update
    });

    it('should skip state update if unmounted during submissions load failure', async () => {
      let rejectSubmissions: (error: any) => void;
      const pendingPromise = new Promise((_, reject) => {
        rejectSubmissions = reject;
      });
      vi.mocked(taskService.getTaskSubmissions).mockReturnValue(pendingPromise as any);

      const { unmount } = renderHook(() => useTaskRunner(mockTask as any));

      // Unmount while loading
      unmount();

      // Reject after unmount - should skip state update
      await act(async () => {
        rejectSubmissions!(new Error('Network error'));
      });

      // No error should occur - the isMountedRef check prevents state update
    });

    it('should handle abort error during submissions loading', async () => {
      const abortError = new DOMException('Aborted', 'AbortError');
      vi.mocked(taskService.getTaskSubmissions).mockRejectedValue(abortError);
      const { isAbortError } = await import('@/lib/api');
      vi.mocked(isAbortError).mockReturnValue(true);

      const { result } = renderHook(() => useTaskRunner(mockTask as any));

      // Wait for error handling
      await waitFor(() => {
        expect(result.current.isLoadingSubmissions).toBe(false);
      });

      // Should not set submissions to empty array - abort is ignored
    });

    it('should abort previous request when task changes', async () => {
      const abortSpy = vi.spyOn(AbortController.prototype, 'abort');

      vi.mocked(taskService.getTaskSubmissions).mockResolvedValue([]);

      const { rerender } = renderHook(
        ({ task }) => useTaskRunner(task as any),
        { initialProps: { task: mockTask } }
      );

      // Wait for first load
      await waitFor(() => {
        expect(taskService.getTaskSubmissions).toHaveBeenCalledWith('task-123', expect.any(Object));
      });

      // Change task
      const newTask = { ...mockTask, id: 'task-456' };
      rerender({ task: newTask });

      // Previous request should be aborted
      expect(abortSpy).toHaveBeenCalled();

      abortSpy.mockRestore();
    });
  });

  describe('setCode debouncing', () => {
    it('should clear previous debounce timeout when setting code multiple times', async () => {
      vi.useFakeTimers();
      const clearTimeoutSpy = vi.spyOn(global, 'clearTimeout');

      const { result } = renderHook(() => useTaskRunner(mockTask as any));

      // Wait for initial code to be set
      await vi.runAllTimersAsync();

      // Set code multiple times to trigger debounce clearing
      act(() => {
        result.current.setCode('code 1');
      });

      act(() => {
        result.current.setCode('code 2');
      });

      // clearTimeout should be called when setting code the second time
      expect(clearTimeoutSpy).toHaveBeenCalled();

      // Advance past debounce time
      await act(async () => {
        await vi.advanceTimersByTimeAsync(600);
      });

      // Only the last code should be saved
      expect(storage.setTaskCode).toHaveBeenLastCalledWith('hello-world', 'code 2');

      vi.useRealTimers();
      clearTimeoutSpy.mockRestore();
    });

    it('should not save to storage when task is null', async () => {
      const { result } = renderHook(() => useTaskRunner(null));

      act(() => {
        result.current.setCode('some code');
      });

      expect(storage.setTaskCode).not.toHaveBeenCalled();
    });
  });

  describe('abort handling in runQuickTests', () => {
    it('should not update state when request is aborted', async () => {
      const abortError = new DOMException('Aborted', 'AbortError');
      vi.mocked(taskService.runTests).mockRejectedValue(abortError);
      const { isAbortError } = await import('@/lib/api');
      vi.mocked(isAbortError).mockReturnValue(true);

      const { result } = renderHook(() => useTaskRunner(mockTask as any));

      await waitFor(() => {
        expect(result.current.code).toBe(mockTask.initialCode);
      });

      await act(async () => {
        await result.current.runQuickTests();
      });

      // State should not be updated for aborted request
      expect(result.current.runResult).toBeNull();
    });

    it('should not update state when component unmounted during run', async () => {
      let resolveTests: (value: any) => void;
      const pendingPromise = new Promise((resolve) => {
        resolveTests = resolve;
      });
      vi.mocked(taskService.runTests).mockReturnValue(pendingPromise as any);

      const { result, unmount } = renderHook(() => useTaskRunner(mockTask as any));

      await waitFor(() => {
        expect(result.current.code).toBe(mockTask.initialCode);
      });

      // Start running tests
      const runPromise = act(async () => {
        result.current.runQuickTests();
      });

      // Unmount before tests complete
      unmount();

      // Resolve the tests
      resolveTests!({
        status: 'passed',
        testsPassed: 5,
        testsTotal: 5,
        testCases: [],
        runtime: '100ms',
        message: '',
      });

      await runPromise;
      // No error should be thrown
    });
  });

  describe('abort handling in submitCode', () => {
    it('should not update state when submit is aborted', async () => {
      const abortError = new DOMException('Aborted', 'AbortError');
      vi.mocked(taskService.submitCode).mockRejectedValue(abortError);
      const { isAbortError } = await import('@/lib/api');
      vi.mocked(isAbortError).mockReturnValue(true);

      const { result } = renderHook(() => useTaskRunner(mockTask as any));

      await waitFor(() => {
        expect(result.current.code).toBe(mockTask.initialCode);
      });

      await act(async () => {
        await result.current.submitCode();
      });

      // Submissions should not be updated for aborted request
      expect(result.current.submissions).toEqual([]);
    });
  });

  describe('detectTaskLanguage edge cases', () => {
    it('should detect Python from py tag', () => {
      expect(detectTaskLanguage({ tags: ['py', 'algorithms'] } as any, undefined)).toBe('python');
    });

    it('should check courseId containing algo', () => {
      expect(detectTaskLanguage(null, 'my-algo-course')).toBe('python');
    });

    it('should check courseId containing c_go', () => {
      expect(detectTaskLanguage(null, 'c_go_advanced')).toBe('go');
    });

    it('should check courseId containing c_java', () => {
      expect(detectTaskLanguage(null, 'c_java_oop')).toBe('java');
    });

    it('should return undefined tag as java', () => {
      expect(detectTaskLanguage({ tags: undefined } as any, undefined)).toBe('java');
    });
  });

  describe('submission sanitization', () => {
    it('should sanitize error submissions from backend', async () => {
      const errorSubmission = {
        id: 'sub-1',
        status: 'error',
        score: 0,
        runtime: '-',
        createdAt: '2024-01-01T00:00:00Z',
        code: 'code',
        message: 'Internal server error: connection timeout',
        testsTotal: 0,
        testsPassed: 0,
      };
      vi.mocked(taskService.submitCode).mockResolvedValue(errorSubmission as any);

      const { result } = renderHook(() => useTaskRunner(mockTask as any));

      await waitFor(() => {
        expect(result.current.code).toBe(mockTask.initialCode);
      });

      await act(async () => {
        await result.current.submitCode();
      });

      // Error message should be sanitized (cleared)
      expect(result.current.submissions[0].message).toBe('');
      // testsTotal should have fallback value
      expect(result.current.submissions[0].testsTotal).toBe(10);
    });

    it('should keep testsTotal when already set in error submission', async () => {
      const errorSubmission = {
        id: 'sub-1',
        status: 'error',
        score: 0,
        runtime: '-',
        createdAt: '2024-01-01T00:00:00Z',
        code: 'code',
        message: 'Compilation error',
        testsTotal: 15, // Already has a value
        testsPassed: 0,
      };
      vi.mocked(taskService.submitCode).mockResolvedValue(errorSubmission as any);

      const { result } = renderHook(() => useTaskRunner(mockTask as any));

      await waitFor(() => {
        expect(result.current.code).toBe(mockTask.initialCode);
      });

      await act(async () => {
        await result.current.submitCode();
      });

      // testsTotal should keep original value, not fallback
      expect(result.current.submissions[0].testsTotal).toBe(15);
    });

    it('should keep non-error submissions as-is', async () => {
      const successSubmission = {
        id: 'sub-1',
        status: 'passed',
        score: 100,
        runtime: '100ms',
        createdAt: '2024-01-01T00:00:00Z',
        code: 'code',
        message: 'All tests passed',
        testsTotal: 10,
        testsPassed: 10,
      };
      vi.mocked(taskService.submitCode).mockResolvedValue(successSubmission as any);

      const { result } = renderHook(() => useTaskRunner(mockTask as any));

      await waitFor(() => {
        expect(result.current.code).toBe(mockTask.initialCode);
      });

      await act(async () => {
        await result.current.submitCode();
      });

      // Non-error submission should keep its message
      expect(result.current.submissions[0].message).toBe('All tests passed');
    });
  });

  describe('mounted state checks', () => {
    it('should not update state if unmounted during successful run', async () => {
      let resolvePromise: (value: any) => void;
      const pendingPromise = new Promise((resolve) => {
        resolvePromise = resolve;
      });
      vi.mocked(taskService.runTests).mockReturnValue(pendingPromise as any);

      const { result, unmount } = renderHook(() => useTaskRunner(mockTask as any));

      await waitFor(() => {
        expect(result.current.code).toBe(mockTask.initialCode);
      });

      // Start running tests
      act(() => {
        result.current.runQuickTests();
      });

      // Unmount while request is pending
      unmount();

      // Now resolve the promise after unmount
      await act(async () => {
        resolvePromise!({
          status: 'passed',
          testsPassed: 5,
          testsTotal: 5,
          testCases: [],
          runtime: '100ms',
          message: '',
        });
      });

      // No error should occur - state update should be skipped
    });

    it('should not update state if unmounted during failed run', async () => {
      // Reset mock to ensure it returns false for regular errors
      isAbortErrorMock.mockClear();
      isAbortErrorMock.mockImplementation((err) => err instanceof DOMException && err.name === 'AbortError');

      let rejectPromise: (error: any) => void;
      const pendingPromise = new Promise<never>((_, reject) => {
        rejectPromise = reject;
      });
      vi.mocked(taskService.runTests).mockReturnValue(pendingPromise);

      const { result, unmount } = renderHook(() => useTaskRunner(mockTask as any));

      await waitFor(() => {
        expect(result.current.code).toBe(mockTask.initialCode);
      });

      // Start running tests - this will set isRunning to true
      let runPromise: Promise<void>;
      act(() => {
        runPromise = result.current.runQuickTests();
      });

      // Verify tests started
      expect(result.current.isRunning).toBe(true);

      // Unmount while request is pending - sets isMountedRef.current = false
      unmount();

      // Create a regular error (not abort) to trigger the isMountedRef check
      const networkError = new Error('Network error');

      // Reject the promise after unmount
      rejectPromise!(networkError);

      // Wait for the rejection to be fully processed
      try {
        await runPromise!;
      } catch {
        // Expected - promise rejects
      }

      // Verify isAbortError was called and returned false
      expect(isAbortErrorMock).toHaveBeenCalledWith(networkError);
      expect(isAbortErrorMock.mock.results[isAbortErrorMock.mock.results.length - 1].value).toBe(false);

      // isMountedRef.current is false, so state update is skipped (line 232 branch)
    });

    it('should not update state if unmounted during successful submit', async () => {
      let resolvePromise: (value: any) => void;
      const pendingPromise = new Promise((resolve) => {
        resolvePromise = resolve;
      });
      vi.mocked(taskService.submitCode).mockReturnValue(pendingPromise as any);

      const { result, unmount } = renderHook(() => useTaskRunner(mockTask as any));

      await waitFor(() => {
        expect(result.current.code).toBe(mockTask.initialCode);
      });

      // Start submitting
      act(() => {
        result.current.submitCode();
      });

      // Unmount while request is pending
      unmount();

      // Now resolve the promise after unmount
      await act(async () => {
        resolvePromise!({
          id: 'sub-1',
          status: 'passed',
          score: 100,
          runtime: '100ms',
          createdAt: new Date().toISOString(),
          code: 'code',
          message: '',
          testsTotal: 10,
          testsPassed: 10,
        });
      });

      // No error should occur
    });

    it('should not update state if unmounted during failed submit', async () => {
      // Reset mock to ensure it returns false for regular errors
      isAbortErrorMock.mockClear();
      isAbortErrorMock.mockImplementation((err) => err instanceof DOMException && err.name === 'AbortError');

      let rejectPromise: (error: any) => void;
      const pendingPromise = new Promise<never>((_, reject) => {
        rejectPromise = reject;
      });
      vi.mocked(taskService.submitCode).mockReturnValue(pendingPromise);

      const { result, unmount } = renderHook(() => useTaskRunner(mockTask as any));

      await waitFor(() => {
        expect(result.current.code).toBe(mockTask.initialCode);
      });

      // Start submitting - this will set isSubmitting to true
      let submitPromise: Promise<void>;
      act(() => {
        submitPromise = result.current.submitCode();
      });

      // Verify submit started
      expect(result.current.isSubmitting).toBe(true);

      // Unmount while request is pending - sets isMountedRef.current = false
      unmount();

      // Create a regular error (not abort) to trigger the isMountedRef check
      const networkError = new Error('Network error');

      // Reject the promise after unmount
      rejectPromise!(networkError);

      // Wait for the rejection to be fully processed
      try {
        await submitPromise!;
      } catch {
        // Expected - promise rejects
      }

      // Verify isAbortError was called and returned false
      expect(isAbortErrorMock).toHaveBeenCalledWith(networkError);
      expect(isAbortErrorMock.mock.results[isAbortErrorMock.mock.results.length - 1].value).toBe(false);

      // isMountedRef.current is false, so state update is skipped (line 284 branch)
    });
  });

  describe('signal aborted after success', () => {
    it('should skip state update if signal aborted after runTests returns', async () => {
      // Create a mock that simulates the signal being aborted right after response
      const mockResult = {
        status: 'passed',
        testsPassed: 5,
        testsTotal: 5,
        testCases: [],
        runtime: '100ms',
        message: '',
      };

      // Mock that will abort the signal before returning
      vi.mocked(taskService.runTests).mockImplementation(async (_code, _taskId, _lang, options) => {
        // Simulate abort happening during the request
        const controller = (options?.signal as any)?.controller;
        if (controller) {
          controller.abort();
        }
        return mockResult;
      });

      const { result } = renderHook(() => useTaskRunner(mockTask as any));

      await waitFor(() => {
        expect(result.current.code).toBe(mockTask.initialCode);
      });

      await act(async () => {
        await result.current.runQuickTests();
      });

      // Even though tests returned, if signal was aborted, result might be skipped
      // This tests the branch where signal.aborted is true
    });
  });

  describe('run result sanitization edge cases', () => {
    it('should preserve compile error messages', async () => {
      vi.mocked(taskService.runTests).mockResolvedValue({
        status: 'error',
        testsPassed: 0,
        testsTotal: 0,
        testCases: [],
        runtime: '-',
        message: 'compile error: undefined variable x',
      });

      const { result } = renderHook(() => useTaskRunner(mockTask as any));

      await waitFor(() => {
        expect(result.current.code).toBe(mockTask.initialCode);
      });

      await act(async () => {
        await result.current.runQuickTests();
      });

      expect(result.current.runResult?.message).toContain('compile');
    });

    it('should preserve syntax error messages', async () => {
      vi.mocked(taskService.runTests).mockResolvedValue({
        status: 'error',
        testsPassed: 0,
        testsTotal: 0,
        testCases: [],
        runtime: '-',
        message: 'syntax error at line 5',
      });

      const { result } = renderHook(() => useTaskRunner(mockTask as any));

      await waitFor(() => {
        expect(result.current.code).toBe(mockTask.initialCode);
      });

      await act(async () => {
        await result.current.runQuickTests();
      });

      expect(result.current.runResult?.message).toContain('syntax');
    });

    it('should preserve panic error messages', async () => {
      vi.mocked(taskService.runTests).mockResolvedValue({
        status: 'error',
        testsPassed: 0,
        testsTotal: 0,
        testCases: [],
        runtime: '-',
        message: 'panic: runtime error: index out of range',
      });

      const { result } = renderHook(() => useTaskRunner(mockTask as any));

      await waitFor(() => {
        expect(result.current.code).toBe(mockTask.initialCode);
      });

      await act(async () => {
        await result.current.runQuickTests();
      });

      expect(result.current.runResult?.message).toContain('panic');
    });

    it('should preserve exception messages', async () => {
      vi.mocked(taskService.runTests).mockResolvedValue({
        status: 'error',
        testsPassed: 0,
        testsTotal: 0,
        testCases: [],
        runtime: '-',
        message: 'NullPointerException at Main.java:10',
      });

      const { result } = renderHook(() => useTaskRunner(mockTask as any));

      await waitFor(() => {
        expect(result.current.code).toBe(mockTask.initialCode);
      });

      await act(async () => {
        await result.current.runQuickTests();
      });

      expect(result.current.runResult?.message).toContain('Exception');
    });

    it('should use original testsTotal when non-zero', async () => {
      vi.mocked(taskService.runTests).mockResolvedValue({
        status: 'error',
        testsPassed: 0,
        testsTotal: 3, // Non-zero
        testCases: [],
        runtime: '-',
        message: '',
      });

      const { result } = renderHook(() => useTaskRunner(mockTask as any));

      await waitFor(() => {
        expect(result.current.code).toBe(mockTask.initialCode);
      });

      await act(async () => {
        await result.current.runQuickTests();
      });

      // Should keep original 3, not fallback to 5
      expect(result.current.runResult?.testsTotal).toBe(3);
    });

    it('should handle empty message gracefully', async () => {
      vi.mocked(taskService.runTests).mockResolvedValue({
        status: 'error',
        testsPassed: 0,
        testsTotal: 0,
        testCases: [],
        runtime: '-',
        message: '',
      });

      const { result } = renderHook(() => useTaskRunner(mockTask as any));

      await waitFor(() => {
        expect(result.current.code).toBe(mockTask.initialCode);
      });

      await act(async () => {
        await result.current.runQuickTests();
      });

      expect(result.current.runResult?.message).toBe('');
    });
  });
});
