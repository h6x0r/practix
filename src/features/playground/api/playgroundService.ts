import { api } from '@/lib/api';

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
};
