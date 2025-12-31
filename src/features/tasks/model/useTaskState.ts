import { useState, useEffect, useCallback } from 'react';
import { Task } from '@/types';
import { taskService } from '../api/taskService';
import { createLogger } from '@/lib/logger';

const log = createLogger('useTaskState');

export interface TaskStateError {
  message: string;
  status?: number;
  isNetworkError: boolean;
}

export const useTaskState = (slug?: string) => {
  const [task, setTask] = useState<Task | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<TaskStateError | null>(null);

  const fetchTask = useCallback(async (taskSlug: string) => {
    setIsLoading(true);
    setTask(null);
    setError(null);

    try {
      const t = await taskService.fetchTask(taskSlug);
      setTask(t);
      setError(null);
    } catch (err: unknown) {
      log.error('Failed to fetch task', err);

      const isNetworkError = err instanceof Error &&
        (err.message.includes('fetch') || err.message.includes('network') || err.message === 'Failed to fetch');

      const status = (err as { status?: number })?.status;
      const message = status === 404
        ? 'Task not found'
        : isNetworkError
          ? 'Network error. Please check your connection.'
          : 'Failed to load task. Please try again.';

      setError({ message, status, isNetworkError });
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!slug) {
      setIsLoading(false);
      return;
    }

    fetchTask(slug);
  }, [slug, fetchTask]);

  const retry = useCallback(() => {
    if (slug) {
      fetchTask(slug);
    }
  }, [slug, fetchTask]);

  return { task, isLoading, error, retry };
};
