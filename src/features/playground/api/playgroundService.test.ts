import { describe, it, expect, beforeEach, vi } from 'vitest';
import { playgroundService, isRateLimitError, extractRateLimitInfo } from './playgroundService';
import { ApiError } from '@/lib/api';

vi.mock('@/lib/api', () => ({
  api: {
    get: vi.fn(),
    post: vi.fn(),
  },
  ApiError: class ApiError extends Error {
    status: number;
    data: unknown;
    constructor(message: string, status: number, data?: unknown) {
      super(message);
      this.status = status;
      this.data = data;
    }
  },
}));

import { api } from '@/lib/api';

describe('playgroundService', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('runCode', () => {
    it('should execute code and return result', async () => {
      const mockResult = {
        status: 'passed' as const,
        statusId: 3,
        description: 'Accepted',
        stdout: 'Hello, World!\n',
        stderr: '',
        compileOutput: '',
        time: '0.01',
        memory: 3456,
        exitCode: 0,
      };

      vi.mocked(api.post).mockResolvedValueOnce(mockResult);

      const result = await playgroundService.runCode(
        'print("Hello, World!")',
        'python'
      );

      expect(api.post).toHaveBeenCalledWith('/submissions/run', {
        code: 'print("Hello, World!")',
        language: 'python',
        stdin: undefined,
      });
      expect(result.status).toBe('passed');
      expect(result.stdout).toBe('Hello, World!\n');
    });

    it('should send stdin when provided', async () => {
      const mockResult = {
        status: 'passed' as const,
        statusId: 3,
        description: 'Accepted',
        stdout: '5\n',
        stderr: '',
        compileOutput: '',
        time: '0.02',
        memory: 4000,
        exitCode: 0,
      };

      vi.mocked(api.post).mockResolvedValueOnce(mockResult);

      await playgroundService.runCode('n = int(input())\nprint(n)', 'python', '5');

      expect(api.post).toHaveBeenCalledWith('/submissions/run', {
        code: 'n = int(input())\nprint(n)',
        language: 'python',
        stdin: '5',
      });
    });

    it('should handle compilation error', async () => {
      const mockResult = {
        status: 'compileError' as const,
        statusId: 6,
        description: 'Compilation Error',
        stdout: '',
        stderr: '',
        compileOutput: 'syntax error: unexpected EOF',
        time: '0.00',
        memory: 0,
        exitCode: null,
      };

      vi.mocked(api.post).mockResolvedValueOnce(mockResult);

      const result = await playgroundService.runCode('invalid code', 'go');

      expect(result.status).toBe('compileError');
      expect(result.compileOutput).toContain('syntax error');
    });

    it('should handle runtime error', async () => {
      const mockResult = {
        status: 'error' as const,
        statusId: 11,
        description: 'Runtime Error',
        stdout: '',
        stderr: 'ZeroDivisionError: division by zero',
        compileOutput: '',
        time: '0.01',
        memory: 3000,
        exitCode: 1,
      };

      vi.mocked(api.post).mockResolvedValueOnce(mockResult);

      const result = await playgroundService.runCode('print(1/0)', 'python');

      expect(result.status).toBe('error');
      expect(result.stderr).toContain('ZeroDivisionError');
    });

    it('should handle timeout', async () => {
      const mockResult = {
        status: 'timeout' as const,
        statusId: 5,
        description: 'Time Limit Exceeded',
        stdout: '',
        stderr: '',
        compileOutput: '',
        time: '5.00',
        memory: 10000,
        exitCode: null,
        message: 'Execution timed out after 5 seconds',
      };

      vi.mocked(api.post).mockResolvedValueOnce(mockResult);

      const result = await playgroundService.runCode('while True: pass', 'python');

      expect(result.status).toBe('timeout');
    });

    it('should throw on API error', async () => {
      vi.mocked(api.post).mockRejectedValueOnce(new Error('Server error'));

      await expect(
        playgroundService.runCode('code', 'python')
      ).rejects.toThrow('Server error');
    });
  });

  describe('getLanguages', () => {
    it('should fetch available languages', async () => {
      const mockLanguages = {
        languages: [
          { id: 71, name: 'python', extension: 'py', monacoId: 'python', timeLimit: 5, memoryLimit: 262144 },
          { id: 60, name: 'go', extension: 'go', monacoId: 'go', timeLimit: 5, memoryLimit: 262144 },
          { id: 62, name: 'java', extension: 'java', monacoId: 'java', timeLimit: 10, memoryLimit: 524288 },
        ],
        default: 'python',
      };

      vi.mocked(api.get).mockResolvedValueOnce(mockLanguages);

      const result = await playgroundService.getLanguages();

      expect(api.get).toHaveBeenCalledWith('/submissions/languages');
      expect(result.languages).toHaveLength(3);
      expect(result.default).toBe('python');
    });

    it('should throw on API error', async () => {
      vi.mocked(api.get).mockRejectedValueOnce(new Error('Service unavailable'));

      await expect(playgroundService.getLanguages()).rejects.toThrow('Service unavailable');
    });
  });

  describe('getJudgeStatus', () => {
    it('should fetch judge status', async () => {
      const mockStatus = {
        available: true,
        queue: { queued: 5, running: 2 },
        languages: ['python', 'go', 'java', 'javascript'],
      };

      vi.mocked(api.get).mockResolvedValueOnce(mockStatus);

      const result = await playgroundService.getJudgeStatus();

      expect(api.get).toHaveBeenCalledWith('/submissions/judge/status');
      expect(result.available).toBe(true);
      expect(result.queue.queued).toBe(5);
    });

    it('should handle unavailable judge', async () => {
      const mockStatus = {
        available: false,
        queue: { queued: 0, running: 0 },
        languages: [],
      };

      vi.mocked(api.get).mockResolvedValueOnce(mockStatus);

      const result = await playgroundService.getJudgeStatus();

      expect(result.available).toBe(false);
    });

    it('should throw on API error', async () => {
      vi.mocked(api.get).mockRejectedValueOnce(new Error('Network error'));

      await expect(playgroundService.getJudgeStatus()).rejects.toThrow('Network error');
    });
  });

  describe('getRateLimitInfo', () => {
    it('should fetch rate limit info', async () => {
      const mockInfo = {
        rateLimitSeconds: 5,
        isPremium: false,
      };

      vi.mocked(api.get).mockResolvedValueOnce(mockInfo);

      const result = await playgroundService.getRateLimitInfo();

      expect(api.get).toHaveBeenCalledWith('/submissions/rate-limit-info');
      expect(result.rateLimitSeconds).toBe(5);
      expect(result.isPremium).toBe(false);
    });

    it('should return premium rate limit', async () => {
      const mockInfo = {
        rateLimitSeconds: 2,
        isPremium: true,
      };

      vi.mocked(api.get).mockResolvedValueOnce(mockInfo);

      const result = await playgroundService.getRateLimitInfo();

      expect(result.rateLimitSeconds).toBe(2);
      expect(result.isPremium).toBe(true);
    });

    it('should throw on API error', async () => {
      vi.mocked(api.get).mockRejectedValueOnce(new Error('Unauthorized'));

      await expect(playgroundService.getRateLimitInfo()).rejects.toThrow('Unauthorized');
    });
  });
});

describe('isRateLimitError', () => {
  it('should return true for rate limit error', () => {
    const error = new ApiError('Too Many Requests', 429, { retryAfter: 10 });

    expect(isRateLimitError(error)).toBe(true);
  });

  it('should return false for non-429 error', () => {
    const error = new ApiError('Server Error', 500, {});

    expect(isRateLimitError(error)).toBe(false);
  });

  it('should return false for error without retryAfter', () => {
    const error = new ApiError('Too Many Requests', 429, {});

    expect(isRateLimitError(error)).toBe(false);
  });

  it('should return false for non-ApiError', () => {
    const error = new Error('Regular error');

    expect(isRateLimitError(error)).toBe(false);
  });
});

describe('extractRateLimitInfo', () => {
  it('should extract rate limit info from error', () => {
    const error = new ApiError('Too Many Requests', 429, {
      retryAfter: 15,
      rateLimitSeconds: 10,
      isPremium: true,
      message: 'Custom rate limit message',
    });

    const info = extractRateLimitInfo(error);

    expect(info.isRateLimited).toBe(true);
    expect(info.retryAfter).toBe(15);
    expect(info.rateLimitSeconds).toBe(10);
    expect(info.isPremium).toBe(true);
    expect(info.message).toBe('Custom rate limit message');
  });

  it('should use defaults for missing data', () => {
    const error = new ApiError('Too Many Requests', 429, {});

    const info = extractRateLimitInfo(error);

    expect(info.retryAfter).toBe(10);
    expect(info.rateLimitSeconds).toBe(10);
    expect(info.isPremium).toBe(false);
    expect(info.message).toContain('Rate limit exceeded');
  });
});
