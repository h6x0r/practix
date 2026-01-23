import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import { useAiChat } from './useAiChat';

// Use vi.hoisted for mocks that need to reference each other
const { isAbortErrorMock } = vi.hoisted(() => {
  return {
    isAbortErrorMock: vi.fn((error: unknown) => {
      return error instanceof DOMException && error.name === 'AbortError';
    }),
  };
});

vi.mock('../../ai/api/geminiService', () => ({
  askAiTutor: vi.fn(),
}));

vi.mock('@/lib/api', () => ({
  isAbortError: isAbortErrorMock,
}));

import { askAiTutor } from '../../ai/api/geminiService';

describe('useAiChat', () => {
  const mockTask = {
    id: 'task-123',
    slug: 'hello-world',
    title: 'Hello World',
    description: 'Print hello world',
    initialCode: 'console.log("hello")',
    difficulty: 'easy' as const,
    tags: ['javascript'],
  };

  beforeEach(() => {
    vi.clearAllMocks();
    isAbortErrorMock.mockImplementation((error: unknown) => {
      return error instanceof DOMException && error.name === 'AbortError';
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('initial state', () => {
    it('should have empty question', () => {
      const { result } = renderHook(() =>
        useAiChat(mockTask, 'console.log("test")', 'javascript', 'en')
      );

      expect(result.current.question).toBe('');
    });

    it('should have empty chat', () => {
      const { result } = renderHook(() =>
        useAiChat(mockTask, 'code', 'javascript', 'en')
      );

      expect(result.current.chat).toEqual([]);
    });

    it('should not be loading', () => {
      const { result } = renderHook(() =>
        useAiChat(mockTask, 'code', 'javascript', 'en')
      );

      expect(result.current.isLoading).toBe(false);
    });

    it('should be open by default', () => {
      const { result } = renderHook(() =>
        useAiChat(mockTask, 'code', 'javascript', 'en')
      );

      expect(result.current.isOpen).toBe(true);
    });
  });

  describe('setQuestion', () => {
    it('should update question', () => {
      const { result } = renderHook(() =>
        useAiChat(mockTask, 'code', 'javascript', 'en')
      );

      act(() => {
        result.current.setQuestion('How do I fix this?');
      });

      expect(result.current.question).toBe('How do I fix this?');
    });
  });

  describe('setIsOpen', () => {
    it('should toggle isOpen state', () => {
      const { result } = renderHook(() =>
        useAiChat(mockTask, 'code', 'javascript', 'en')
      );

      expect(result.current.isOpen).toBe(true);

      act(() => {
        result.current.setIsOpen(false);
      });

      expect(result.current.isOpen).toBe(false);
    });
  });

  describe('askAi', () => {
    it('should not call API when question is empty', async () => {
      const { result } = renderHook(() =>
        useAiChat(mockTask, 'code', 'javascript', 'en')
      );

      await act(async () => {
        await result.current.askAi();
      });

      expect(askAiTutor).not.toHaveBeenCalled();
    });

    it('should not call API when question is only whitespace', async () => {
      const { result } = renderHook(() =>
        useAiChat(mockTask, 'code', 'javascript', 'en')
      );

      act(() => {
        result.current.setQuestion('   ');
      });

      await act(async () => {
        await result.current.askAi();
      });

      expect(askAiTutor).not.toHaveBeenCalled();
    });

    it('should not call API when task is null', async () => {
      const { result } = renderHook(() =>
        useAiChat(null, 'code', 'javascript', 'en')
      );

      act(() => {
        result.current.setQuestion('How do I fix this?');
      });

      await act(async () => {
        await result.current.askAi();
      });

      expect(askAiTutor).not.toHaveBeenCalled();
    });

    it('should call API with correct parameters', async () => {
      vi.mocked(askAiTutor).mockResolvedValue('Here is a hint...');

      const { result } = renderHook(() =>
        useAiChat(mockTask, 'my code', 'python', 'ru')
      );

      act(() => {
        result.current.setQuestion('Help me please');
      });

      await act(async () => {
        await result.current.askAi();
      });

      expect(askAiTutor).toHaveBeenCalledWith(
        'task-123',
        'Hello World',
        'my code',
        'Help me please',
        'python',
        'ru',
        { signal: expect.any(AbortSignal) }
      );
    });

    it('should add user message to chat immediately', async () => {
      vi.mocked(askAiTutor).mockResolvedValue('Response');

      const { result } = renderHook(() =>
        useAiChat(mockTask, 'code', 'javascript', 'en')
      );

      act(() => {
        result.current.setQuestion('My question');
      });

      await act(async () => {
        await result.current.askAi();
      });

      expect(result.current.chat[0]).toEqual({
        role: 'user',
        text: 'My question',
      });
    });

    it('should clear question after asking', async () => {
      vi.mocked(askAiTutor).mockResolvedValue('Response');

      const { result } = renderHook(() =>
        useAiChat(mockTask, 'code', 'javascript', 'en')
      );

      act(() => {
        result.current.setQuestion('My question');
      });

      await act(async () => {
        await result.current.askAi();
      });

      expect(result.current.question).toBe('');
    });

    it('should set loading state during request', async () => {
      let resolvePromise: (value: string) => void;
      const promise = new Promise<string>((resolve) => {
        resolvePromise = resolve;
      });
      vi.mocked(askAiTutor).mockReturnValue(promise);

      const { result } = renderHook(() =>
        useAiChat(mockTask, 'code', 'javascript', 'en')
      );

      act(() => {
        result.current.setQuestion('Question');
      });

      act(() => {
        result.current.askAi();
      });

      expect(result.current.isLoading).toBe(true);

      await act(async () => {
        resolvePromise!('Answer');
      });

      expect(result.current.isLoading).toBe(false);
    });

    it('should add AI response to chat', async () => {
      vi.mocked(askAiTutor).mockResolvedValue('Here is the answer');

      const { result } = renderHook(() =>
        useAiChat(mockTask, 'code', 'javascript', 'en')
      );

      act(() => {
        result.current.setQuestion('Question');
      });

      await act(async () => {
        await result.current.askAi();
      });

      expect(result.current.chat).toHaveLength(2);
      expect(result.current.chat[1]).toEqual({
        role: 'model',
        text: 'Here is the answer',
      });
    });

    it('should handle connection error', async () => {
      vi.mocked(askAiTutor).mockRejectedValue(new Error('Network error'));

      const { result } = renderHook(() =>
        useAiChat(mockTask, 'code', 'javascript', 'en')
      );

      act(() => {
        result.current.setQuestion('Question');
      });

      await act(async () => {
        await result.current.askAi();
      });

      expect(result.current.chat).toHaveLength(2);
      expect(result.current.chat[1]).toEqual({
        role: 'model',
        text: 'Connection error. Please try again.',
      });
    });

    it('should handle abort error silently', async () => {
      const abortError = new DOMException('Aborted', 'AbortError');
      vi.mocked(askAiTutor).mockRejectedValue(abortError);
      isAbortErrorMock.mockReturnValue(true);

      const { result } = renderHook(() =>
        useAiChat(mockTask, 'code', 'javascript', 'en')
      );

      act(() => {
        result.current.setQuestion('Question');
      });

      await act(async () => {
        await result.current.askAi();
      });

      // Only user message, no error message
      expect(result.current.chat).toHaveLength(1);
      expect(result.current.chat[0].role).toBe('user');
    });

    it('should maintain chat history across multiple questions', async () => {
      vi.mocked(askAiTutor)
        .mockResolvedValueOnce('Answer 1')
        .mockResolvedValueOnce('Answer 2');

      const { result } = renderHook(() =>
        useAiChat(mockTask, 'code', 'javascript', 'en')
      );

      // First question
      act(() => {
        result.current.setQuestion('Question 1');
      });

      await act(async () => {
        await result.current.askAi();
      });

      // Second question
      act(() => {
        result.current.setQuestion('Question 2');
      });

      await act(async () => {
        await result.current.askAi();
      });

      expect(result.current.chat).toHaveLength(4);
      expect(result.current.chat[0].text).toBe('Question 1');
      expect(result.current.chat[1].text).toBe('Answer 1');
      expect(result.current.chat[2].text).toBe('Question 2');
      expect(result.current.chat[3].text).toBe('Answer 2');
    });

    it('should use default uiLanguage when not provided', async () => {
      vi.mocked(askAiTutor).mockResolvedValue('Response');

      const { result } = renderHook(() =>
        useAiChat(mockTask, 'code', 'javascript')
      );

      act(() => {
        result.current.setQuestion('Question');
      });

      await act(async () => {
        await result.current.askAi();
      });

      expect(askAiTutor).toHaveBeenCalledWith(
        expect.any(String),
        expect.any(String),
        expect.any(String),
        expect.any(String),
        expect.any(String),
        'en',
        expect.any(Object)
      );
    });
  });
});
