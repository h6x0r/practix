import { Task, Submission } from '@/types';
import { RunTestsResult } from '../api/taskService';

/** Rate limiting cooldown in milliseconds (5 seconds) */
export const RATE_LIMIT_COOLDOWN_MS = 5000;

/**
 * Check if an error message is a code execution error (useful for user)
 * vs. an infrastructure error (should be hidden)
 */
export const isCodeExecutionError = (message: string): boolean => {
  if (!message) return false;
  const lowerMsg = message.toLowerCase();
  const codeErrors = [
    'compile', 'syntax', 'undefined', 'error:', 'exception',
    'panic:', 'cannot find', 'not found', 'invalid', 'expected',
    'unexpected', 'missing', 'redeclared', 'import', 'package',
    'type mismatch', 'cannot convert', 'unresolved', 'nil pointer',
    'index out of', 'runtime error', 'assertionerror', 'nullpointer',
  ];
  return codeErrors.some(err => lowerMsg.includes(err));
};

/**
 * Sanitize run/submit results
 * - Preserves code execution errors (compilation, runtime errors)
 * - Hides infrastructure errors (network, timeout, internal)
 */
export const sanitizeRunResult = (
  result: RunTestsResult,
  expectedTests: number,
): RunTestsResult => {
  const testsTotal = result.testsTotal === 0 ? expectedTests : result.testsTotal;

  if (result.status === 'error') {
    const hasUsefulError = isCodeExecutionError(result.message);
    return {
      ...result,
      testsTotal,
      message: hasUsefulError ? result.message : '',
    };
  }

  return { ...result, testsTotal };
};

/**
 * Sanitize submission result to hide technical error details
 */
export const sanitizeSubmission = (submission: Submission): Submission => {
  if (submission.status === 'error') {
    return {
      ...submission,
      message: '',
      testsTotal: submission.testsTotal || 10,
    };
  }
  return submission;
};

/**
 * Detect programming language for a task based on courseId, tags, and slug
 */
export const detectTaskLanguage = (
  task: Task | null,
  courseId?: string,
): string => {
  if (
    courseId?.includes('algo') ||
    courseId?.startsWith('python-') ||
    task?.tags?.includes('python') ||
    task?.tags?.includes('py')
  ) {
    return 'python';
  }

  if (
    courseId?.startsWith('go-') ||
    courseId?.includes('c_go') ||
    task?.tags?.includes('go') ||
    task?.slug?.includes('go')
  ) {
    return 'go';
  }

  if (
    courseId?.startsWith('java-') ||
    courseId?.includes('c_java') ||
    task?.tags?.includes('java')
  ) {
    return 'java';
  }

  return task?.tags?.[0] || 'java';
};
