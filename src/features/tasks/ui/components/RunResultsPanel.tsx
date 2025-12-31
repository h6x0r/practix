import React, { memo } from 'react';
import { IconCheck, IconX, IconClock } from '@/components/Icons';
import { useUITranslation } from '@/contexts/LanguageContext';
import { TestCaseResult } from '@/types';

interface RunResultsProps {
  isOpen: boolean;
  onClose: () => void;
  isLoading: boolean;
  result: {
    status: string;
    testsPassed: number;
    testsTotal: number;
    testCases: TestCaseResult[];
    runtime: string;
    message: string;
  } | null;
}

export const RunResultsPanel: React.FC<RunResultsProps> = memo(({
  isOpen,
  onClose,
  isLoading,
  result,
}) => {
  const { tUI } = useUITranslation();

  if (!isOpen) return null;

  const isPassed = result?.status === 'passed';
  const isError = result?.status === 'error';
  const isFailed = result?.status === 'failed';

  // Get first failed test for details
  const failedTest = result?.testCases?.find(tc => !tc.passed);

  return (
    <div className="absolute bottom-0 left-0 right-0 bg-white dark:bg-[#161b22] border-t border-gray-200 dark:border-gray-700 shadow-lg z-20 max-h-[50%] flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-gray-200 dark:border-gray-700 flex-shrink-0">
        <div className="flex items-center gap-3">
          <span className="text-sm font-medium text-gray-900 dark:text-white">
            {tUI('task.runResults')}
          </span>

          {isLoading && (
            <span className="text-xs text-gray-500 animate-pulse flex items-center gap-1">
              <IconClock className="w-3 h-3 animate-spin" />
              {tUI('task.running')}
            </span>
          )}

          {!isLoading && result && (
            <>
              {/* Status Badge */}
              <span className={`text-xs font-semibold px-2 py-0.5 rounded ${
                isPassed
                  ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400'
                  : isError
                  ? 'bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-400'
                  : 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400'
              }`}>
                {isPassed ? tUI('task.passed') : isError ? tUI('task.error') : tUI('task.failed')}
              </span>

              {/* Test Count */}
              <span className="text-xs text-gray-500 dark:text-gray-400">
                {result.testsPassed}/{result.testsTotal} {tUI('task.testsPassed')}
              </span>

              {/* Runtime */}
              {result.runtime && result.runtime !== '-' && (
                <span className="text-xs text-gray-400 dark:text-gray-500 font-mono">
                  {result.runtime}
                </span>
              )}
            </>
          )}
        </div>

        <button
          onClick={onClose}
          className="p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-500"
        >
          <IconX className="w-4 h-4" />
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-4">
        {isLoading && (
          <div className="flex items-center justify-center py-8">
            <div className="animate-pulse text-gray-500 text-sm">
              {tUI('task.executingCode')}
            </div>
          </div>
        )}

        {!isLoading && result && (
          <div className="space-y-4">
            {/* Success Message */}
            {isPassed && (
              <div className="flex items-center gap-2 text-green-600 dark:text-green-400">
                <IconCheck className="w-5 h-5" />
                <span className="text-sm font-medium">{tUI('task.allTestsPassed')}</span>
              </div>
            )}

            {/* Error Message - Show code errors with details, infrastructure errors with generic message */}
            {isError && (
              <div className="space-y-3">
                {/* If we have a useful error message (compilation/runtime error), show it */}
                {result.message ? (
                  <div>
                    <div className="flex items-center gap-2 mb-2">
                      <span className="text-lg">⚠️</span>
                      <span className="text-sm font-medium text-orange-700 dark:text-orange-400">
                        {tUI('task.executionError')}
                      </span>
                    </div>
                    <div className="bg-gray-50 dark:bg-[#0d1117] rounded border border-orange-200 dark:border-orange-900/50 p-3">
                      <pre className="text-[12px] font-mono text-red-600 dark:text-red-400 whitespace-pre-wrap break-all">
                        {result.message}
                      </pre>
                    </div>
                  </div>
                ) : (
                  // Generic error message for infrastructure errors
                  <div className="flex flex-col items-center justify-center py-4 text-center">
                    <div className="w-12 h-12 rounded-full bg-orange-100 dark:bg-orange-900/30 flex items-center justify-center mb-3">
                      <span className="text-2xl">⚠️</span>
                    </div>
                    <p className="text-sm text-gray-700 dark:text-gray-300 font-medium mb-1">
                      {tUI('task.executionError')}
                    </p>
                    <p className="text-xs text-gray-500 dark:text-gray-400">
                      {tUI('task.executionErrorHint')}
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* Failed Test Details */}
            {isFailed && failedTest && (
              <div className="space-y-3">
                {/* Output section */}
                {failedTest.actualOutput && (
                  <div>
                    <div className="text-xs uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-1.5 font-medium">
                      {tUI('task.output')}
                    </div>
                    <div className="bg-gray-50 dark:bg-[#0d1117] rounded border border-gray-200 dark:border-gray-700 p-2.5">
                      <code className="text-[12px] font-mono text-red-600 dark:text-red-400">
                        {failedTest.actualOutput}
                      </code>
                    </div>
                  </div>
                )}

                {/* Expected section */}
                {failedTest.expectedOutput && (
                  <div>
                    <div className="text-xs uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-1.5 font-medium">
                      {tUI('task.expected')}
                    </div>
                    <div className="bg-gray-50 dark:bg-[#0d1117] rounded border border-gray-200 dark:border-gray-700 p-2.5">
                      <code className="text-[12px] font-mono text-green-600 dark:text-green-400">
                        {failedTest.expectedOutput}
                      </code>
                    </div>
                  </div>
                )}

                {/* Error message (only if no output/expected) */}
                {!failedTest.actualOutput && !failedTest.expectedOutput && failedTest.error && (
                  <div>
                    <div className="text-xs uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-1.5 font-medium">
                      {tUI('task.errorDetails')}
                    </div>
                    <div className="bg-gray-50 dark:bg-[#0d1117] rounded border border-red-200 dark:border-red-900/50 p-2.5">
                      <code className="text-[12px] font-mono text-red-600 dark:text-red-400 break-all">
                        {failedTest.error}
                      </code>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Test Case List */}
            {result.testCases && result.testCases.length > 0 && (
              <div className="space-y-1">
                <div className="text-xs uppercase tracking-wider text-gray-500 dark:text-gray-400 font-medium mb-2">
                  {tUI('task.testResults')}
                </div>
                <div className="flex flex-wrap gap-2">
                  {result.testCases.map((tc, idx) => (
                    <div
                      key={idx}
                      className={`flex items-center gap-1.5 px-2 py-1 rounded text-xs ${
                        tc.passed
                          ? 'bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-400'
                          : 'bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-400'
                      }`}
                    >
                      {tc.passed ? (
                        <IconCheck className="w-3 h-3" />
                      ) : (
                        <IconX className="w-3 h-3" />
                      )}
                      <span className="font-mono text-[11px]">
                        {tc.name.replace(/^test_/, '').replace(/^Test/, '')}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
});
