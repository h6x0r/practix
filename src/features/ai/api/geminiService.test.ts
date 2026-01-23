import { describe, it, expect, beforeEach, vi } from 'vitest';
import { askAiTutor } from './geminiService';

vi.mock('@/lib/api', () => ({
  api: {
    post: vi.fn(),
  },
  isAbortError: (error: unknown) => error instanceof Error && error.name === 'AbortError',
}));

vi.mock('@/lib/logger', () => ({
  createLogger: () => ({
    info: vi.fn(),
    error: vi.fn(),
    warn: vi.fn(),
    debug: vi.fn(),
  }),
}));

import { api } from '@/lib/api';

describe('askAiTutor', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should return AI answer on success', async () => {
    vi.mocked(api.post).mockResolvedValueOnce({
      answer: 'Here is a hint: try using a loop.',
      remaining: 95,
    });

    const result = await askAiTutor(
      'task-123',
      'Hello World',
      'print("hello")',
      'How do I fix this?',
      'python',
      'en'
    );

    expect(api.post).toHaveBeenCalledWith('/ai/tutor', {
      taskId: 'task-123',
      taskTitle: 'Hello World',
      userCode: 'print("hello")',
      question: 'How do I fix this?',
      language: 'python',
      uiLanguage: 'en',
    }, { signal: undefined });
    expect(result).toBe('Here is a hint: try using a loop.');
  });

  it('should use default uiLanguage when not provided', async () => {
    vi.mocked(api.post).mockResolvedValueOnce({
      answer: 'Response',
      remaining: 90,
    });

    await askAiTutor(
      'task-123',
      'Task',
      'code',
      'question',
      'python'
    );

    expect(api.post).toHaveBeenCalledWith('/ai/tutor', expect.objectContaining({
      uiLanguage: 'en',
    }), expect.any(Object));
  });

  it('should pass abort signal when provided', async () => {
    const controller = new AbortController();
    vi.mocked(api.post).mockResolvedValueOnce({ answer: 'Response' });

    await askAiTutor(
      'task-123',
      'Task',
      'code',
      'question',
      'python',
      'en',
      { signal: controller.signal }
    );

    expect(api.post).toHaveBeenCalledWith('/ai/tutor', expect.any(Object), {
      signal: controller.signal,
    });
  });

  it('should re-throw abort errors', async () => {
    const abortError = new Error('Aborted');
    abortError.name = 'AbortError';
    vi.mocked(api.post).mockRejectedValueOnce(abortError);

    await expect(askAiTutor(
      'task-123',
      'Task',
      'code',
      'question',
      'python'
    )).rejects.toThrow('Aborted');
  });

  it('should return limit message on 403 error', async () => {
    const error = { status: 403, message: 'Forbidden' };
    vi.mocked(api.post).mockRejectedValueOnce(error);

    const result = await askAiTutor(
      'task-123',
      'Task',
      'code',
      'question',
      'python'
    );

    expect(result).toContain('daily AI limit');
    expect(result).toContain('Premium');
  });

  it('should return generic error message on other errors', async () => {
    vi.mocked(api.post).mockRejectedValueOnce(new Error('Network error'));

    const result = await askAiTutor(
      'task-123',
      'Task',
      'code',
      'question',
      'python'
    );

    expect(result).toContain('trouble connecting');
    expect(result).toContain('try again');
  });

  it('should handle 500 errors gracefully', async () => {
    const error = { status: 500, message: 'Internal Server Error' };
    vi.mocked(api.post).mockRejectedValueOnce(error);

    const result = await askAiTutor(
      'task-123',
      'Task',
      'code',
      'question',
      'python'
    );

    expect(result).toContain('trouble connecting');
  });
});
