import { useState, useEffect, useRef, useCallback } from 'react';
import { Task, PromptSubmission } from '@/types';
import { taskService, PromptSubmissionResult } from '../api/taskService';
import { storage } from '@/lib/storage';
import { createLogger } from '@/lib/logger';
import { isAbortError } from '@/lib/api';

const log = createLogger('PromptRunner');

/**
 * Hook for managing prompt engineering task state and submission
 */
export const usePromptRunner = (task: Task | null) => {
  const [prompt, setPromptState] = useState('');
  const [activeTab, setActiveTab] = useState<'editor' | 'history'>('editor');
  const [submissions, setSubmissions] = useState<PromptSubmission[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isLoadingSubmissions, setIsLoadingSubmissions] = useState(false);
  const debounceRef = useRef<NodeJS.Timeout | null>(null);

  // Track mounted state and abort controllers
  const isMountedRef = useRef(true);
  const submissionsAbortRef = useRef<AbortController | null>(null);
  const submitAbortRef = useRef<AbortController | null>(null);

  // Initialize prompt when task loads - check localStorage first
  useEffect(() => {
    if (task) {
      const savedPrompt = storage.getTaskCode(task.slug);

      if (savedPrompt !== null) {
        setPromptState(savedPrompt);
      } else {
        setPromptState(task.initialCode || '');
      }
    }
  }, [task]);

  // Load submissions for the task from backend
  useEffect(() => {
    if (task?.id) {
      // Abort previous request if task changed
      submissionsAbortRef.current?.abort();
      const controller = new AbortController();
      submissionsAbortRef.current = controller;

      setIsLoadingSubmissions(true);

      // For prompt tasks, we fetch regular submissions and convert them
      taskService.getTaskSubmissions(task.id, { signal: controller.signal })
        .then((subs) => {
          if (isMountedRef.current && !controller.signal.aborted) {
            // Convert regular submissions to PromptSubmission format
            const promptSubs: PromptSubmission[] = subs.map(sub => ({
              id: sub.id,
              status: sub.status as any,
              score: sub.score,
              message: sub.message || '',
              createdAt: sub.createdAt,
              scenarioResults: (sub.testCases || []).map((tc, idx) => ({
                scenarioIndex: idx,
                input: tc.input || '',
                output: tc.actualOutput || '',
                score: tc.passed ? 10 : 0,
                feedback: tc.passed ? 'Passed' : (tc.error || tc.expectedOutput || 'Failed'),
                passed: tc.passed,
              })),
              summary: sub.message || '',
            }));
            setSubmissions(promptSubs);
          }
        })
        .catch((error) => {
          if (isAbortError(error)) return;
          log.warn('Failed to load submissions', error);
          if (isMountedRef.current) {
            setSubmissions([]);
          }
        })
        .finally(() => {
          if (isMountedRef.current && !controller.signal.aborted) {
            setIsLoadingSubmissions(false);
          }
        });
    } else {
      setSubmissions([]);
    }
  }, [task?.id]);

  // Wrapper to save prompt to localStorage with debounce
  const setPrompt = useCallback((newPrompt: string) => {
    setPromptState(newPrompt);

    // Debounce localStorage save (500ms)
    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
    }

    if (task) {
      debounceRef.current = setTimeout(() => {
        storage.setTaskCode(task.slug, newPrompt);
      }, 500);
    }
  }, [task]);

  // Cleanup on unmount
  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
      submissionsAbortRef.current?.abort();
      submitAbortRef.current?.abort();
    };
  }, []);

  /**
   * Submit prompt for AI evaluation
   */
  const submitPrompt = useCallback(async () => {
    if (!task) return;

    // Abort previous submission if still pending
    submitAbortRef.current?.abort();
    const controller = new AbortController();
    submitAbortRef.current = controller;

    setIsSubmitting(true);

    // Auto-switch to history tab to show result
    setActiveTab('history');

    try {
      const result = await taskService.submitPrompt(prompt, task.id, { signal: controller.signal });

      if (isMountedRef.current && !controller.signal.aborted) {
        // Convert to PromptSubmission format
        const newSubmission: PromptSubmission = {
          id: result.id,
          status: result.status as any,
          score: result.score,
          message: result.message,
          createdAt: result.createdAt,
          scenarioResults: result.scenarioResults || [],
          summary: result.summary,
          xpEarned: result.xpEarned,
          totalXp: result.totalXp,
          level: result.level,
          leveledUp: result.leveledUp,
        };

        setSubmissions(prev => [newSubmission, ...prev]);
        setIsSubmitting(false);

        // Return the result for toast notifications
        return newSubmission;
      }
    } catch (e) {
      if (isAbortError(e)) return;

      if (isMountedRef.current) {
        setIsSubmitting(false);
        log.error('submitPrompt failed', e);

        // Show error submission
        const errorSubmission: PromptSubmission = {
          id: `error-${Date.now()}`,
          status: 'error',
          score: 0,
          message: 'Failed to evaluate prompt. Please try again.',
          createdAt: new Date().toISOString(),
          scenarioResults: [],
          summary: 'Evaluation failed',
        };
        setSubmissions(prev => [errorSubmission, ...prev]);
      }
    }
  }, [task, prompt]);

  return {
    prompt,
    setPrompt,
    activeTab,
    setActiveTab,
    submissions,
    isSubmitting,
    isLoadingSubmissions,
    submitPrompt,
  };
};
