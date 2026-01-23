import React, { memo, useState } from 'react';
import { IconCheck, IconX, IconClock, IconChevronDown } from '@/components/Icons';
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
  const [expandedTest, setExpandedTest] = useState<number | null>(null);

  if (!isOpen) return null;

  const isPassed = result?.status === 'passed';
  const isError = result?.status === 'error';
  const isFailed = result?.status === 'failed';

  // Auto-expand first failed test
  const firstFailedIdx = result?.testCases?.findIndex(tc => !tc.passed) ?? -1;

  const toggleTest = (idx: number) => {
    setExpandedTest(expandedTest === idx ? null : idx);
  };

  return (
    <div className="h-full bg-white dark:bg-[#161b22] border-t border-gray-200 dark:border-gray-700 shadow-lg flex flex-col overflow-hidden">
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
      <div className="flex-1 overflow-auto p-3">
        {isLoading && (
          <div className="flex items-center justify-center py-6">
            <div className="animate-pulse text-gray-500 text-sm">
              {tUI('task.executingCode')}
            </div>
          </div>
        )}

        {!isLoading && result && (
          <div className="space-y-2">
            {/* Success Message */}
            {isPassed && (
              <div className="flex items-center gap-2 text-green-600 dark:text-green-400 mb-3">
                <IconCheck className="w-5 h-5" />
                <span className="text-sm font-medium">{tUI('task.allTestsPassed')}</span>
              </div>
            )}

            {/* Error Message */}
            {isError && result.message && (
              <div className="mb-3">
                <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg border border-orange-200 dark:border-orange-800 p-3">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-sm">⚠️</span>
                    <span className="text-xs font-medium text-orange-700 dark:text-orange-400">
                      {tUI('task.executionError')}
                    </span>
                  </div>
                  <pre className="text-[11px] font-mono text-red-600 dark:text-red-400 whitespace-pre-wrap break-all">
                    {result.message}
                  </pre>
                </div>
              </div>
            )}

            {/* Test Cases - Expandable Cards */}
            {result.testCases && result.testCases.length > 0 && (
              <div className="space-y-1.5">
                {result.testCases.map((tc, idx) => {
                  const isExpanded = expandedTest === idx || (expandedTest === null && idx === firstFailedIdx && !tc.passed);
                  const hasDetails = tc.input || tc.expectedOutput || tc.actualOutput || tc.error;

                  return (
                    <div
                      key={idx}
                      className={`rounded-lg border overflow-hidden transition-all ${
                        tc.passed
                          ? 'bg-green-50 dark:bg-green-900/10 border-green-200 dark:border-green-800/50'
                          : 'bg-red-50 dark:bg-red-900/10 border-red-200 dark:border-red-800/50'
                      }`}
                    >
                      {/* Test Header - Clickable */}
                      <button
                        onClick={() => hasDetails && toggleTest(idx)}
                        className={`w-full flex items-center gap-2 px-3 py-2 text-left ${hasDetails ? 'cursor-pointer hover:bg-black/5 dark:hover:bg-white/5' : ''}`}
                      >
                        {tc.passed ? (
                          <IconCheck className="w-4 h-4 text-green-600 dark:text-green-400 flex-shrink-0" />
                        ) : (
                          <IconX className="w-4 h-4 text-red-600 dark:text-red-400 flex-shrink-0" />
                        )}
                        <span className={`text-xs font-bold ${tc.passed ? 'text-green-700 dark:text-green-400' : 'text-red-700 dark:text-red-400'}`}>
                          Test {idx + 1}
                        </span>
                        {hasDetails && (
                          <IconChevronDown className={`w-3.5 h-3.5 ml-auto text-gray-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`} />
                        )}
                      </button>

                      {/* Test Details - Expanded */}
                      {isExpanded && hasDetails && (
                        <div className="px-3 pb-3 pt-1 border-t border-gray-200 dark:border-gray-700/50 space-y-2">
                          {/* Input Arguments */}
                          {tc.input && (
                            <div>
                              <div className="text-[10px] uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-1 font-semibold">
                                Input
                              </div>
                              <div className="bg-white dark:bg-[#0d1117] rounded border border-gray-200 dark:border-gray-700 p-2">
                                <code className="text-[11px] font-mono text-gray-800 dark:text-gray-200 break-all whitespace-pre-wrap">
                                  {tc.input}
                                </code>
                              </div>
                            </div>
                          )}

                          {/* Expected Output */}
                          {tc.expectedOutput && (
                            <div>
                              <div className="text-[10px] uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-1 font-semibold">
                                Expected
                              </div>
                              <div className="bg-white dark:bg-[#0d1117] rounded border border-green-200 dark:border-green-800/50 p-2">
                                <code className="text-[11px] font-mono text-green-700 dark:text-green-400 break-all">
                                  {tc.expectedOutput}
                                </code>
                              </div>
                            </div>
                          )}

                          {/* Actual Output */}
                          {tc.actualOutput && (
                            <div>
                              <div className="text-[10px] uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-1 font-semibold">
                                Output
                              </div>
                              <div className="bg-white dark:bg-[#0d1117] rounded border border-red-200 dark:border-red-800/50 p-2">
                                <code className="text-[11px] font-mono text-red-600 dark:text-red-400 break-all">
                                  {tc.actualOutput}
                                </code>
                              </div>
                            </div>
                          )}

                          {/* Error */}
                          {tc.error && !tc.actualOutput && (
                            <div>
                              <div className="text-[10px] uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-1 font-semibold">
                                Error
                              </div>
                              <div className="bg-white dark:bg-[#0d1117] rounded border border-red-200 dark:border-red-800/50 p-2">
                                <code className="text-[11px] font-mono text-red-600 dark:text-red-400 break-all">
                                  {tc.error}
                                </code>
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
});
