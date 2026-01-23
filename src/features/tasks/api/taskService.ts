import { Task, Submission, TestCaseResult, PromptSubmission, PromptScenarioResult } from '@/types';
import { api, isAbortError } from '@/lib/api';
import { storage } from '@/lib/storage';
import { createLogger } from '@/lib/logger';

const log = createLogger('TaskService');

// Run tests result (not saved to DB)
export interface RunTestsResult {
  status: string;
  testsPassed: number;
  testsTotal: number;
  testCases: TestCaseResult[];
  runtime: string;
  message: string;
}

// Prompt submission result
export interface PromptSubmissionResult {
  id: string;
  status: string;
  score: number;
  message: string;
  createdAt: string;
  scenarioResults: PromptScenarioResult[];
  summary: string;
  xpEarned?: number;
  totalXp?: number;
  level?: number;
  leveledUp?: boolean;
}

interface RequestOptions {
  signal?: AbortSignal;
}

export const taskService = {

  fetchTask: async (slug: string, options?: RequestOptions): Promise<Task> => {
    return api.get<Task>(`/tasks/${slug}`, options);
  },

  getRecentTasks: async (options?: RequestOptions): Promise<Task[]> => {
    return api.get<Task[]>('/tasks', options);
  },

  /**
   * Run quick tests (5 tests) without saving to database
   * Used for "Run Code" button - fast feedback
   */
  runTests: async (code: string, taskId: string, language: string, options?: RequestOptions): Promise<RunTestsResult> => {
    return api.post<RunTestsResult>('/submissions/run-tests', { code, taskId, language }, options);
  },

  /**
   * Submit code for full evaluation (all tests) and save to database
   * Used for "Submit" button
   */
  submitCode: async (code: string, taskId: string, language: string, options?: RequestOptions): Promise<Submission> => {
    return api.post<Submission>('/submissions', { code, taskId, language }, options);
  },

  /**
   * Submit a prompt for AI evaluation (prompt engineering tasks)
   * Used for "Submit" button on PROMPT type tasks
   */
  submitPrompt: async (prompt: string, taskId: string, options?: RequestOptions): Promise<PromptSubmissionResult> => {
    return api.post<PromptSubmissionResult>('/submissions/prompt', { prompt, taskId }, options);
  },

  /**
   * Get user's submissions for a specific task
   * Requires authentication
   */
  getTaskSubmissions: async (taskId: string, options?: RequestOptions): Promise<Submission[]> => {
    try {
      return await api.get<Submission[]>(`/submissions/task/${taskId}`, options);
    } catch (error) {
      // Don't log abort errors as warnings
      if (isAbortError(error)) {
        throw error;
      }
      // Return empty array if not authenticated or error
      log.warn('Failed to fetch submissions', error);
      return [];
    }
  },

  /**
   * Get the latest run result for a specific task
   * Returns null if no run result exists
   * Requires authentication
   */
  getRunResult: async (taskId: string, options?: RequestOptions): Promise<RunTestsResult | null> => {
    try {
      // Backend wraps result in { data: ... } to avoid NestJS empty body issue with null
      const response = await api.get<{ data: RunTestsResult | null }>(`/submissions/run-result/${taskId}`, options);
      return response.data;
    } catch (error) {
      if (isAbortError(error)) {
        throw error;
      }
      log.warn('Failed to fetch run result', error);
      return null;
    }
  },

  /**
   * Get user's recent submissions across all tasks
   */
  getRecentSubmissions: async (limit: number = 10): Promise<Submission[]> => {
    try {
      return await api.get<Submission[]>(`/submissions/user/recent?limit=${limit}`);
    } catch (error) {
      log.warn('Failed to fetch recent submissions', error);
      return [];
    }
  },

  /**
   * Get list of all completed task IDs from local storage
   * @deprecated Use backend submission status instead
   */
  getCompletedTaskIds: (): string[] => {
    return storage.getCompletedTasks();
  },

  /**
   * Mark a task as completed in local storage
   * @deprecated Backend tracks completion via passed submissions
   */
  markTaskAsCompleted: (taskId: string) => {
    storage.addCompletedTask(taskId);
  },

  isResourceCompleted: (resourceId: string, type: 'task' | 'topic'): boolean => {
    const completedIds = taskService.getCompletedTaskIds();

    if (type === 'task') {
      return completedIds.includes(resourceId);
    }

    return false;
  }
};
