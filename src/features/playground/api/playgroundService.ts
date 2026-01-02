import { api, ApiError } from '@/lib/api';

export interface ExecutionResult {
  status: 'passed' | 'failed' | 'error' | 'timeout' | 'compileError';
  statusId: number;
  description: string;
  stdout: string;
  stderr: string;
  compileOutput: string;
  time: string;
  memory: number;
  exitCode: number | null;
  message?: string;
}

export interface LanguageInfo {
  id: number;
  name: string;
  extension: string;
  monacoId: string;
  timeLimit: number;
  memoryLimit: number;
}

export interface JudgeStatus {
  available: boolean;
  queue: { queued: number; running: number };
  languages: string[];
}

export interface RateLimitInfo {
  rateLimitSeconds: number;
  isPremium: boolean;
}

export interface RateLimitError {
  isRateLimited: true;
  retryAfter: number;
  rateLimitSeconds: number;
  isPremium: boolean;
  message: string;
}

/**
 * Check if error is a rate limit error
 */
export function isRateLimitError(error: unknown): error is RateLimitError {
  return (
    error instanceof ApiError &&
    error.status === 429 &&
    typeof (error.data as any)?.retryAfter === 'number'
  );
}

/**
 * Extract rate limit info from ApiError
 */
export function extractRateLimitInfo(error: ApiError): RateLimitError {
  const data = error.data as any;
  return {
    isRateLimited: true,
    retryAfter: data?.retryAfter || 10,
    rateLimitSeconds: data?.rateLimitSeconds || 10,
    isPremium: data?.isPremium || false,
    message: data?.message || 'Rate limit exceeded. Please wait before running code again.',
  };
}

export const playgroundService = {
  runCode: async (
    code: string,
    language: string,
    stdin?: string
  ): Promise<ExecutionResult> => {
    return api.post<ExecutionResult>('/submissions/run', {
      code,
      language,
      stdin,
    });
  },

  getLanguages: async (): Promise<{ languages: LanguageInfo[]; default: string }> => {
    return api.get('/submissions/languages');
  },

  getJudgeStatus: async (): Promise<JudgeStatus> => {
    return api.get('/submissions/judge/status');
  },

  getRateLimitInfo: async (): Promise<RateLimitInfo> => {
    return api.get('/submissions/rate-limit-info');
  },
};
