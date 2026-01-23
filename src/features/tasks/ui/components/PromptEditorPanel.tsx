import React, { useState, useCallback, memo, useContext } from 'react';
import { IconClock, IconChevronDown, IconRefresh, IconSparkles, IconCheck, IconX } from '@/components/Icons';
import { Task, PromptSubmission, PromptScenarioResult } from '@/types';
import { AuthContext, ThemeContext } from '@/components/Layout';
import { useUITranslation } from '@/contexts/LanguageContext';
import { TimerStopwatch } from './TimerStopwatch';
import { storage } from '@/lib/storage';

// Safe date formatting helper
const formatDate = (dateStr: string) => {
  try {
    const date = new Date(dateStr);
    if (isNaN(date.getTime())) return '-';
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  } catch {
    return '-';
  }
};

interface PromptEditorPanelProps {
  activeTab: 'editor' | 'history';
  setActiveTab: (tab: 'editor' | 'history') => void;
  prompt: string;
  setPrompt: (prompt: string) => void;
  isPremium: boolean;
  submissions: PromptSubmission[];
  isLoadingSubmissions?: boolean;
  task: Task;
  isSubmitting?: boolean;
}

export const PromptEditorPanel = memo(({
  activeTab, setActiveTab,
  prompt, setPrompt,
  isPremium,
  submissions, isLoadingSubmissions = false, task,
  isSubmitting = false,
}: PromptEditorPanelProps) => {
  const { user } = useContext(AuthContext);
  const { isDark } = useContext(ThemeContext);
  const { tUI } = useUITranslation();
  const [expandedSubmissionId, setExpandedSubmissionId] = useState<string | null>(null);
  const [showResetConfirm, setShowResetConfirm] = useState(false);

  // Reset prompt to initial state
  const handleResetPrompt = useCallback(() => {
    if (task?.initialCode) {
      setPrompt(task.initialCode);
      storage.removeTaskCode(task.slug);
      setShowResetConfirm(false);
    }
  }, [task, setPrompt]);

  const toggleSubmission = (id: string) => {
    setExpandedSubmissionId(expandedSubmissionId === id ? null : id);
  };

  // Count {{INPUT}} placeholders in the prompt
  const inputPlaceholderCount = (prompt.match(/\{\{INPUT\}\}/gi) || []).length;

  return (
    <div className="flex-1 flex flex-col bg-white dark:bg-[#0d1117] border-l border-gray-200 dark:border-gray-800 h-full">
      {/* Tab Header */}
      <div className="flex bg-gray-50 dark:bg-[#161b22] border-b border-gray-200 dark:border-[#21262d] select-none flex-shrink-0">
        <button
          onClick={() => setActiveTab('editor')}
          className={`px-4 py-2.5 text-xs flex items-center gap-2 border-t-2 transition-colors ${
            activeTab === 'editor'
              ? 'bg-white dark:bg-[#0d1117] text-gray-900 dark:text-white border-purple-500'
              : 'bg-gray-100 dark:bg-[#21262d] text-gray-500 border-transparent hover:bg-gray-200 dark:hover:bg-[#30363d]'
          }`}
        >
          <IconSparkles className="w-3.5 h-3.5 text-purple-400" />
          prompt.txt
        </button>
        <button
          onClick={() => setActiveTab('history')}
          className={`px-4 py-2.5 text-xs flex items-center gap-2 border-t-2 transition-colors ${
            activeTab === 'history'
              ? 'bg-white dark:bg-[#0d1117] text-gray-900 dark:text-white border-purple-500'
              : 'bg-gray-100 dark:bg-[#21262d] text-gray-500 border-transparent hover:bg-gray-200 dark:hover:bg-[#30363d]'
          }`}
        >
          <IconClock className="w-3 h-3" />
          {tUI('task.submissions') || 'Submissions'}
        </button>
        <div className="flex-1" />
        <div className="flex items-center gap-1 px-2">
          <TimerStopwatch />
          {/* Reset Prompt Button */}
          <div className="relative">
            <button
              onClick={() => setShowResetConfirm(true)}
              className="w-8 h-8 flex items-center justify-center rounded-md transition-colors text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-[#333]"
              title="Reset to initial prompt"
            >
              <IconRefresh className="w-4 h-4" />
            </button>
            {/* Reset Confirmation Popup */}
            {showResetConfirm && (
              <div className="absolute right-0 top-full mt-1 z-50 bg-white dark:bg-[#21262d] border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg p-3 w-56">
                <p className="text-xs text-gray-600 dark:text-gray-300 mb-3">
                  Reset to initial prompt? Your current prompt will be lost.
                </p>
                <div className="flex justify-end gap-2">
                  <button
                    onClick={() => setShowResetConfirm(false)}
                    className="px-2 py-1 text-xs text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleResetPrompt}
                    className="px-2 py-1 text-xs bg-red-500 hover:bg-red-600 text-white rounded transition-colors"
                  >
                    Reset
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="flex-1 relative flex flex-col min-h-0">
        {/* Prompt Editor Tab */}
        {activeTab === 'editor' && (
          <div className="flex-1 flex flex-col bg-gray-50 dark:bg-[#0d1117] overflow-hidden">
            {/* Prompt Info Bar */}
            <div className="flex items-center gap-3 px-4 py-2 bg-purple-50 dark:bg-purple-900/20 border-b border-purple-100 dark:border-purple-800/30">
              <span className="text-xs text-purple-700 dark:text-purple-300">
                <span className="font-medium">Prompt Engineering</span> - Write your prompt below
              </span>
              <div className="flex-1" />
              {inputPlaceholderCount > 0 ? (
                <span className="text-xs text-green-600 dark:text-green-400 flex items-center gap-1">
                  <IconCheck className="w-3 h-3" />
                  {'{{INPUT}}'} placeholder found
                </span>
              ) : (
                <span className="text-xs text-amber-600 dark:text-amber-400 flex items-center gap-1">
                  <IconX className="w-3 h-3" />
                  Add {'{{INPUT}}'} placeholder for test data
                </span>
              )}
            </div>

            {/* Textarea */}
            <div className="flex-1 p-4">
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Write your prompt here...&#10;&#10;Use {{INPUT}} where the test scenario data will be inserted."
                className="w-full h-full resize-none rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-[#161b22] text-gray-900 dark:text-gray-100 p-4 text-sm font-mono leading-relaxed focus:outline-none focus:ring-2 focus:ring-purple-500/50 focus:border-purple-500"
                spellCheck={false}
                disabled={isSubmitting}
              />
            </div>

            {/* Character/Word Count */}
            <div className="flex items-center justify-between px-4 py-2 bg-gray-100 dark:bg-[#161b22] border-t border-gray-200 dark:border-gray-700 text-xs text-gray-500">
              <span>{prompt.length} characters</span>
              <span>{prompt.split(/\s+/).filter(w => w).length} words</span>
            </div>
          </div>
        )}

        {/* Submissions History Tab */}
        {activeTab === 'history' && (
          <div className="flex-1 bg-white dark:bg-[#0d1117] p-4 overflow-auto">
            <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-3">
              Prompt Submissions
            </h3>
            <div className="space-y-2">
              {isLoadingSubmissions ? (
                <div className="space-y-2">
                  {[1, 2, 3].map((i) => (
                    <div key={i} className="rounded-lg border border-gray-200 dark:border-gray-700 p-3 animate-pulse">
                      <div className="flex items-center gap-3">
                        <div className="w-2 h-2 rounded-full bg-gray-300 dark:bg-gray-600" />
                        <div className="h-3 bg-gray-300 dark:bg-gray-600 rounded w-14" />
                        <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-20 ml-auto" />
                      </div>
                    </div>
                  ))}
                </div>
              ) : submissions.length === 0 ? (
                <div className="text-center text-gray-400 py-8 text-sm">
                  No submissions yet. Write your prompt and click Submit.
                </div>
              ) : (
                submissions.map((sub) => {
                  const isExpanded = expandedSubmissionId === sub.id;
                  const isPassed = sub.status === 'passed';
                  const isFailed = sub.status === 'failed';
                  const passedScenarios = sub.scenarioResults?.filter(r => r.passed).length || 0;
                  const totalScenarios = sub.scenarioResults?.length || 0;

                  const getStatusInfo = () => {
                    if (isPassed) return { label: 'Passed', color: 'text-green-600 dark:text-green-400', dot: 'bg-green-500' };
                    if (isFailed) return { label: 'Failed', color: 'text-red-600 dark:text-red-400', dot: 'bg-red-500' };
                    return { label: 'Error', color: 'text-orange-600 dark:text-orange-400', dot: 'bg-orange-500' };
                  };
                  const statusInfo = getStatusInfo();

                  return (
                    <div key={sub.id} className="rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden">
                      {/* Submission Header */}
                      <div
                        onClick={() => toggleSubmission(sub.id)}
                        className="flex items-center gap-2.5 p-2.5 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors"
                      >
                        {/* Status Dot */}
                        <div className={`w-2 h-2 rounded-full flex-shrink-0 ${statusInfo.dot}`} />

                        {/* Status Label */}
                        <span className={`text-xs font-semibold uppercase ${statusInfo.color}`}>
                          {statusInfo.label}
                        </span>

                        {/* Scenario Count Badge */}
                        {totalScenarios > 0 && (
                          <span className={`text-[11px] font-mono px-1.5 py-0.5 rounded ${
                            isPassed
                              ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400'
                              : 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400'
                          }`}>
                            {passedScenarios}/{totalScenarios}
                          </span>
                        )}

                        {/* Score */}
                        <span className="text-xs text-gray-500 dark:text-gray-400">
                          Score: {sub.score}/10
                        </span>

                        <div className="flex-1" />

                        {/* Date */}
                        <span className="text-[10px] text-gray-400 dark:text-gray-500">
                          {formatDate(sub.createdAt)}
                        </span>

                        {/* Expand Icon */}
                        <IconChevronDown className={`w-3 h-3 text-gray-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`} />
                      </div>

                      {/* Expanded Details */}
                      {isExpanded && (
                        <div className="border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-[#161b22] p-3 space-y-3">
                          {/* Summary */}
                          <div className="text-xs text-gray-600 dark:text-gray-300">
                            {sub.summary}
                          </div>

                          {/* Scenario Results */}
                          {sub.scenarioResults?.map((scenario, idx) => (
                            <div
                              key={idx}
                              className={`rounded-md border p-3 ${
                                scenario.passed
                                  ? 'border-green-200 dark:border-green-800/50 bg-green-50 dark:bg-green-900/20'
                                  : 'border-red-200 dark:border-red-800/50 bg-red-50 dark:bg-red-900/20'
                              }`}
                            >
                              <div className="flex items-center gap-2 mb-2">
                                <span className={`text-xs font-medium ${
                                  scenario.passed ? 'text-green-700 dark:text-green-400' : 'text-red-700 dark:text-red-400'
                                }`}>
                                  Scenario {scenario.scenarioIndex + 1}
                                </span>
                                <span className="text-xs text-gray-500">
                                  Score: {scenario.score}/10
                                </span>
                              </div>
                              <p className="text-xs text-gray-600 dark:text-gray-300 mb-2">
                                {scenario.feedback}
                              </p>
                              {scenario.output && (
                                <div className="mt-2">
                                  <div className="text-[10px] uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-1">
                                    Output Preview
                                  </div>
                                  <div className="bg-white dark:bg-[#0d1117] rounded border border-gray-200 dark:border-gray-700 p-2 text-xs font-mono text-gray-600 dark:text-gray-300 max-h-24 overflow-auto">
                                    {scenario.output}
                                  </div>
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  );
                })
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
});
