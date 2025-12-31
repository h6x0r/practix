import { useState, useEffect, useRef, useCallback } from 'react';
import { Task, Submission } from '@/types';
import { taskService, RunTestsResult } from '../api/taskService';
import { storage } from '@/lib/storage';
import { createLogger } from '@/lib/logger';
import { isAbortError } from '@/lib/api';

const log = createLogger('TaskRunner');

/**
 * Check if an error message is a code execution error (useful for user)
 * vs. an infrastructure error (should be hidden)
 */
const isCodeExecutionError = (message: string): boolean => {
  if (!message) return false;
  const lowerMsg = message.toLowerCase();
  // Code execution errors that help users debug their code
  const codeErrors = [
    'compile', 'syntax', 'undefined', 'error:', 'exception',
    'panic:', 'cannot find', 'not found', 'invalid', 'expected',
    'unexpected', 'missing', 'redeclared', 'import', 'package',
    'type mismatch', 'cannot convert', 'unresolved', 'nil pointer',
    'index out of', 'runtime error', 'assertionerror', 'nullpointer'
  ];
  return codeErrors.some(err => lowerMsg.includes(err));
};

/**
 * Sanitize run/submit results
 * - Preserves code execution errors (compilation, runtime errors) - these help users debug
 * - Hides infrastructure errors (network, timeout, internal) - these confuse users
 */
const sanitizeRunResult = (result: RunTestsResult, expectedTests: number): RunTestsResult => {
  // If status is error and we got 0/0 tests, show expected count
  const testsTotal = result.testsTotal === 0 ? expectedTests : result.testsTotal;

  if (result.status === 'error') {
    // Check if this is a code execution error (useful for user) or infrastructure error
    const hasUsefulError = isCodeExecutionError(result.message);

    return {
      ...result,
      testsTotal,
      // Keep the message if it's a code execution error, otherwise clear it
      message: hasUsefulError ? result.message : '',
    };
  }

  return {
    ...result,
    testsTotal,
  };
};

/**
 * Sanitize submission result to hide technical error details
 */
const sanitizeSubmission = (submission: Submission): Submission => {
  // If it's an error, clear the technical message
  if (submission.status === 'error') {
    return {
      ...submission,
      message: '', // Clear technical message - UI shows generic error
      testsTotal: submission.testsTotal || 10, // Ensure we have expected count
    };
  }
  return submission;
};

/**
 * Detect programming language for a task based on courseId, tags, and slug
 */
export const detectTaskLanguage = (task: Task | null, courseId?: string): string => {
  // Python courses: algo-*, python-ml-fundamentals, python-deep-learning, python-llm
  if (
    courseId?.includes('algo') ||
    courseId?.startsWith('python-') ||
    task?.tags?.includes('python') ||
    task?.tags?.includes('py')
  ) {
    return 'python';
  }
  // Go courses: go-*, including go-ml-inference
  if (
    courseId?.startsWith('go-') ||
    courseId?.includes('c_go') ||
    task?.tags?.includes('go') ||
    task?.slug?.includes('go')
  ) {
    return 'go';
  }
  // Java courses: java-*, including java-ml, java-nlp
  if (
    courseId?.startsWith('java-') ||
    courseId?.includes('c_java') ||
    task?.tags?.includes('java')
  ) {
    return 'java';
  }
  // Default to the first tag or java
  return task?.tags?.[0] || 'java';
};

export const useTaskRunner = (task: Task | null, courseId?: string) => {
  const [code, setCodeState] = useState('');
  const [activeTab, setActiveTab] = useState<'editor' | 'history'>('editor');
  const [submissions, setSubmissions] = useState<Submission[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isLoadingSubmissions, setIsLoadingSubmissions] = useState(false);
  const debounceRef = useRef<NodeJS.Timeout | null>(null);

  // Run results state (for quick tests panel)
  const [runResult, setRunResult] = useState<RunTestsResult | null>(null);
  const [isRunResultsOpen, setIsRunResultsOpen] = useState(false);

  // Track mounted state and abort controllers for cleanup
  const isMountedRef = useRef(true);
  const submissionsAbortRef = useRef<AbortController | null>(null);
  const runAbortRef = useRef<AbortController | null>(null);
  const submitAbortRef = useRef<AbortController | null>(null);

  // Initialize code when task loads - check localStorage first
  useEffect(() => {
    if (task) {
      const savedCode = storage.getTaskCode(task.slug);

      if (savedCode !== null) {
        // Use saved code if available
        setCodeState(savedCode);
      } else {
        // Use initial code from task
        setCodeState(task.initialCode);
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
      taskService.getTaskSubmissions(task.id, { signal: controller.signal })
        .then((subs) => {
          if (isMountedRef.current && !controller.signal.aborted) {
            setSubmissions(subs);
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

  // Wrapper to save code to localStorage with debounce
  const setCode = useCallback((newCode: string) => {
    setCodeState(newCode);

    // Debounce localStorage save (500ms)
    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
    }

    if (task) {
      debounceRef.current = setTimeout(() => {
        storage.setTaskCode(task.slug, newCode);
      }, 500);
    }
  }, [task]);

  // Cleanup on unmount - abort all pending requests and clear debounce
  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
      // Abort all pending requests
      submissionsAbortRef.current?.abort();
      runAbortRef.current?.abort();
      submitAbortRef.current?.abort();
    };
  }, []);

  /**
   * Run quick tests (5 tests) - opens bottom panel, doesn't save to DB
   */
  const runQuickTests = useCallback(async () => {
    if (!task) return;

    // Abort previous run if still pending
    runAbortRef.current?.abort();
    const controller = new AbortController();
    runAbortRef.current = controller;

    setIsRunning(true);
    setRunResult(null);
    setIsRunResultsOpen(true);

    const language = detectTaskLanguage(task, courseId);

    try {
      const result = await taskService.runTests(code, task.id, language, { signal: controller.signal });

      // Only update state if mounted and not aborted
      if (isMountedRef.current && !controller.signal.aborted) {
        // Sanitize error messages - don't expose technical details to users
        const sanitizedResult = sanitizeRunResult(result, 5);
        setRunResult(sanitizedResult);
        setIsRunning(false);
      }
    } catch (e) {
      // Don't update state for aborted requests
      if (isAbortError(e)) return;

      if (isMountedRef.current) {
        setIsRunning(false);
        log.error('runQuickTests failed', e);

        // Always show user-friendly error
        setRunResult({
          status: 'error',
          testsPassed: 0,
          testsTotal: 5, // Show expected test count even on error
          testCases: [],
          runtime: '-',
          message: '', // Empty message - UI will show generic error
        });
      }
    }
  }, [task, code, courseId]);

  /**
   * Submit code for full evaluation (all tests) - saves to DB
   */
  const submitCode = useCallback(async () => {
    if (!task) return;

    // Abort previous submission if still pending
    submitAbortRef.current?.abort();
    const controller = new AbortController();
    submitAbortRef.current = controller;

    setIsSubmitting(true);

    // Close run results panel if open
    setIsRunResultsOpen(false);

    // Auto-switch to history tab to show progress/result
    setActiveTab('history');

    const language = detectTaskLanguage(task, courseId);

    try {
      const newSub = await taskService.submitCode(code, task.id, language, { signal: controller.signal });

      // Only update state if mounted and not aborted
      if (isMountedRef.current && !controller.signal.aborted) {
        // Sanitize submission result - hide technical errors
        const sanitizedSub = sanitizeSubmission(newSub);
        setSubmissions(prev => [sanitizedSub, ...prev]);
        setIsSubmitting(false);
      }
    } catch (e) {
      // Don't update state for aborted requests
      if (isAbortError(e)) return;

      if (isMountedRef.current) {
        setIsSubmitting(false);
        log.error('submitCode failed', e);

        // Show user-friendly error submission
        const errorSubmission: Submission = {
          id: `error-${Date.now()}`,
          status: 'error',
          score: 0,
          runtime: '-',
          createdAt: new Date().toISOString(),
          code: code,
          message: '', // Empty - UI will show generic error
          testsTotal: 10, // Expected test count for submit
          testsPassed: 0,
        };
        setSubmissions(prev => [errorSubmission, ...prev]);
      }
    }
  }, [task, code, courseId]);

  const closeRunResults = () => {
    setIsRunResultsOpen(false);
  };

  return {
    code,
    setCode,
    activeTab,
    setActiveTab,
    submissions,
    isRunning,
    isSubmitting,
    isLoadingSubmissions,
    runQuickTests,
    submitCode,
    // Run results panel state
    runResult,
    isRunResultsOpen,
    closeRunResults,
  };
};