
import React, { useContext, useState, useRef, useCallback, memo } from 'react';
import Editor, { loader, Monaco } from '@monaco-editor/react';
import type { editor } from 'monaco-editor';
import { IconClock, IconChevronDown, IconRefresh } from '@/components/Icons';
import { Submission, Task } from '@/types';
import { AuthContext, ThemeContext } from '@/components/Layout';
import { useUITranslation } from '@/contexts/LanguageContext';
import { EditorSettingsDropdown } from './EditorSettingsDropdown';
import { TimerStopwatch } from './TimerStopwatch';
import { storage } from '@/lib/storage';

// Configure Monaco Loader to use a stable CDN version to avoid worker loading issues
loader.config({
  paths: { vs: 'https://cdn.jsdelivr.net/npm/monaco-editor@0.46.0/min/vs' }
});

// Define custom dark theme with darker background matching the description panel
const defineCustomTheme = (monaco: Monaco) => {
  monaco.editor.defineTheme('practix-dark', {
    base: 'vs-dark',
    inherit: true,
    rules: [],
    colors: {
      'editor.background': '#0d1117',
      'editor.lineHighlightBackground': '#161b22',
      'editorLineNumber.foreground': '#484f58',
      'editorLineNumber.activeForeground': '#8b949e',
      'editor.selectionBackground': '#264f78',
      'editorCursor.foreground': '#c9d1d9',
    }
  });
};

interface CodeEditorPanelProps {
  activeTab: 'editor' | 'history';
  setActiveTab: (tab: 'editor' | 'history') => void;
  code: string;
  setCode: (code: string) => void;
  isGo: boolean;
  fileExt: string;
  isPremium: boolean;
  canSeeSolution?: boolean;
  submissions: Submission[];
  isLoadingSubmissions?: boolean;
  task: Task;
  language?: string;
}

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

export const CodeEditorPanel = memo(({
  activeTab, setActiveTab,
  code, setCode,
  isGo, fileExt, isPremium,
  canSeeSolution = false,
  submissions, isLoadingSubmissions = false, task,
  language
}: CodeEditorPanelProps) => {

  const { user } = useContext(AuthContext);
  const { isDark } = useContext(ThemeContext);
  const { tUI } = useUITranslation();
  const [expandedSubmissionId, setExpandedSubmissionId] = useState<string | null>(null);
  const [showResetConfirm, setShowResetConfirm] = useState(false);
  const editorRef = useRef<editor.IStandaloneCodeEditor | null>(null);

  // Reset code to initial state
  const handleResetCode = useCallback(() => {
    if (task?.initialCode) {
      setCode(task.initialCode);
      storage.removeTaskCode(task.slug);
      setShowResetConfirm(false);
    }
  }, [task, setCode]);

  // Determine language for Monaco editor
  const editorLanguage = language || (isGo ? 'go' : 'java');

  // Handle editor mount
  const handleEditorMount = useCallback((editorInstance: editor.IStandaloneCodeEditor, monaco: Monaco) => {
    editorRef.current = editorInstance;
    editorInstance.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, () => {});
  }, []);

  // Editor options
  const editorOptions = {
    minimap: { enabled: user?.preferences?.editorMinimap ?? false },
    fontSize: user?.preferences?.editorFontSize || 14,
    lineNumbers: (user?.preferences?.editorLineNumbers ? 'on' : 'off') as 'on' | 'off',
    scrollBeyondLastLine: false,
    automaticLayout: true,
    padding: { top: 16, bottom: 16 },
    fontFamily: user?.preferences?.editorFontFamily || "'JetBrains Mono', 'Courier New', monospace",
    lineHeight: 24,
    renderLineHighlight: 'all' as const,
    hideCursorInOverviewRuler: true,
    overviewRulerBorder: false,
  };

  const toggleSubmission = (id: string) => {
    setExpandedSubmissionId(expandedSubmissionId === id ? null : id);
  };

  // Get language display info
  const getLangDisplay = () => {
    if (language === 'python' || language === 'py') return { label: 'PY', color: 'text-yellow-400' };
    if (isGo) return { label: 'GO', color: 'text-cyan-400' };
    return { label: 'J', color: 'text-orange-400' };
  };
  const langDisplay = getLangDisplay();

  return (
    <div className="flex-1 flex flex-col bg-white dark:bg-[#0d1117] border-l border-gray-200 dark:border-gray-800 h-full">
      {/* File Tabs */}
      <div className="flex bg-gray-50 dark:bg-[#161b22] border-b border-gray-200 dark:border-[#21262d] select-none flex-shrink-0">
        <button
          onClick={() => setActiveTab('editor')}
          className={`px-4 py-2.5 text-xs flex items-center gap-2 border-t-2 transition-colors ${
            activeTab === 'editor'
              ? 'bg-white dark:bg-[#0d1117] text-gray-900 dark:text-white border-brand-500'
              : 'bg-gray-100 dark:bg-[#21262d] text-gray-500 border-transparent hover:bg-gray-200 dark:hover:bg-[#30363d]'
          }`}
        >
          <span className={langDisplay.color}>{langDisplay.label}</span>
          main{fileExt}
        </button>
        <button
          onClick={() => setActiveTab('history')}
          className={`px-4 py-2.5 text-xs flex items-center gap-2 border-t-2 transition-colors ${
            activeTab === 'history'
              ? 'bg-white dark:bg-[#0d1117] text-gray-900 dark:text-white border-brand-500'
              : 'bg-gray-100 dark:bg-[#21262d] text-gray-500 border-transparent hover:bg-gray-200 dark:hover:bg-[#30363d]'
          }`}
        >
          <IconClock className="w-3 h-3" />
          {tUI('task.submissions')}
        </button>
        <div className="flex-1" />
        <div className="flex items-center gap-1 px-2">
          <TimerStopwatch />
          {/* Reset Code Button */}
          <div className="relative">
            <button
              onClick={() => setShowResetConfirm(true)}
              className="w-8 h-8 flex items-center justify-center rounded-md transition-colors text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-[#333]"
              title={tUI('task.resetCode') || 'Reset to initial code'}
            >
              <IconRefresh className="w-4 h-4" />
            </button>
            {/* Reset Confirmation Popup */}
            {showResetConfirm && (
              <div className="absolute right-0 top-full mt-1 z-50 bg-white dark:bg-[#21262d] border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg p-3 w-56">
                <p className="text-xs text-gray-600 dark:text-gray-300 mb-3">
                  {tUI('task.resetConfirm') || 'Reset to initial code? Your current code will be lost.'}
                </p>
                <div className="flex justify-end gap-2">
                  <button
                    onClick={() => setShowResetConfirm(false)}
                    className="px-2 py-1 text-xs text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                  >
                    {tUI('common.cancel') || 'Cancel'}
                  </button>
                  <button
                    onClick={handleResetCode}
                    className="px-2 py-1 text-xs bg-red-500 hover:bg-red-600 text-white rounded transition-colors"
                  >
                    {tUI('task.reset') || 'Reset'}
                  </button>
                </div>
              </div>
            )}
          </div>
          <EditorSettingsDropdown />
        </div>
      </div>

      <div className="flex-1 relative flex flex-col min-h-0">
        {/* Code Editor Tab */}
        {activeTab === 'editor' && (
          <div className="flex-1 relative bg-gray-50 dark:bg-[#0d1117]">
            <Editor
              height="100%"
              language={editorLanguage}
              theme={isDark ? "practix-dark" : "light"}
              beforeMount={defineCustomTheme}
              onMount={handleEditorMount}
              value={code}
              onChange={(value) => setCode(value || '')}
              options={editorOptions}
              loading={
                <div className="flex h-full items-center justify-center text-gray-500 text-sm bg-gray-50 dark:bg-[#0d1117]">
                  <span className="animate-pulse">{tUI('task.loadingEditor')}</span>
                </div>
              }
            />
          </div>
        )}

        {/* Submissions History Tab */}
        {activeTab === 'history' && (
          <div className="flex-1 bg-white dark:bg-[#0d1117] p-4 overflow-auto">
            <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-3">{tUI('task.submissionHistory')}</h3>
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
                <div className="text-center text-gray-400 py-8 text-sm">{tUI('task.noSubmissions')}</div>
              ) : (
                submissions.map((sub) => {
                  const isExpanded = expandedSubmissionId === sub.id;
                  const hasTests = sub.testsTotal !== undefined && sub.testsTotal > 0;
                  const testsPassed = sub.testsPassed ?? 0;
                  const testsTotal = sub.testsTotal ?? 0;

                  // Determine display status: Error (compile/runtime), Failed (tests), Passed (success)
                  const isError = sub.status === 'error';
                  const isPassed = sub.status === 'passed';
                  const isFailed = sub.status === 'failed';

                  // Get status label and colors
                  const getStatusInfo = () => {
                    if (isPassed) return { label: 'Passed', color: 'text-green-600 dark:text-green-400', dot: 'bg-green-500' };
                    if (isError) return { label: 'Error', color: 'text-orange-600 dark:text-orange-400', dot: 'bg-orange-500' };
                    return { label: 'Failed', color: 'text-red-600 dark:text-red-400', dot: 'bg-red-500' };
                  };
                  const statusInfo = getStatusInfo();

                  // Check if there's anything to show in expanded view
                  const failedTests = sub.testCases?.filter(tc => !tc.passed) || [];
                  const hasExpandableContent = !isPassed && (sub.message || failedTests.length > 0);

                  return (
                    <div key={sub.id} className="rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden">
                      {/* Submission Header */}
                      <div
                        onClick={() => hasExpandableContent && toggleSubmission(sub.id)}
                        className={`flex items-center gap-2.5 p-2.5 transition-colors ${hasExpandableContent ? 'cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800/50' : ''}`}
                      >
                        {/* Status Dot */}
                        <div className={`w-2 h-2 rounded-full flex-shrink-0 ${statusInfo.dot}`} />

                        {/* Status Label */}
                        <span className={`text-xs font-semibold uppercase ${statusInfo.color}`}>
                          {statusInfo.label}
                        </span>

                        {/* Test Count Badge - only show if there are tests */}
                        {hasTests && (
                          <span className={`text-[11px] font-mono px-1.5 py-0.5 rounded ${
                            isPassed
                              ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400'
                              : 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400'
                          }`}>
                            {testsPassed}/{testsTotal}
                          </span>
                        )}

                        {/* Spacer */}
                        <div className="flex-1" />

                        {/* Runtime & Memory */}
                        <div className="flex items-center gap-2 text-[10px] text-gray-500 dark:text-gray-400 font-mono">
                          {sub.runtime && sub.runtime !== '-' && <span>{sub.runtime}</span>}
                          {sub.memory && sub.memory !== '-' && <span className="hidden sm:inline">{sub.memory}</span>}
                        </div>

                        {/* Date */}
                        <span className="text-[10px] text-gray-400 dark:text-gray-500">{formatDate(sub.createdAt)}</span>

                        {/* Expand Icon - only show if expandable */}
                        {hasExpandableContent && (
                          <IconChevronDown className={`w-3 h-3 text-gray-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`} />
                        )}
                      </div>

                      {/* Expanded Details - LeetCode style */}
                      {isExpanded && hasExpandableContent && (
                        <div className="border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-[#161b22]">
                          {/* Compilation/Runtime Error */}
                          {isError && sub.message && (
                            <div className="p-3 border-b border-gray-200 dark:border-gray-700">
                              <div className="text-xs font-medium text-orange-600 dark:text-orange-400 mb-1">
                                {sub.message.toLowerCase().includes('compile') ? 'Compilation Error' :
                                 sub.message.toLowerCase().includes('timeout') ? 'Time Limit Exceeded' :
                                 'Runtime Error'}
                              </div>
                            </div>
                          )}

                          {/* Test Results - LeetCode style */}
                          {failedTests.length > 0 && (
                            <div className="p-3">
                              {/* First failed test details */}
                              {(() => {
                                const tc = failedTests[0];
                                const hasOutput = tc.actualOutput || tc.expectedOutput;

                                return (
                                  <div className="space-y-3">
                                    {/* Output section */}
                                    {tc.actualOutput && (
                                      <div>
                                        <div className="text-[10px] uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-1.5 font-medium">
                                          Output
                                        </div>
                                        <div className="bg-white dark:bg-[#0d1117] rounded border border-gray-200 dark:border-gray-700 p-2.5">
                                          <code className="text-[12px] font-mono text-red-600 dark:text-red-400">
                                            {tc.actualOutput}
                                          </code>
                                        </div>
                                      </div>
                                    )}

                                    {/* Expected section */}
                                    {tc.expectedOutput && (
                                      <div>
                                        <div className="text-[10px] uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-1.5 font-medium">
                                          Expected
                                        </div>
                                        <div className="bg-white dark:bg-[#0d1117] rounded border border-gray-200 dark:border-gray-700 p-2.5">
                                          <code className="text-[12px] font-mono text-green-600 dark:text-green-400">
                                            {tc.expectedOutput}
                                          </code>
                                        </div>
                                      </div>
                                    )}

                                    {/* Error message (only if no output/expected) */}
                                    {!hasOutput && tc.error && (
                                      <div>
                                        <div className="text-[10px] uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-1.5 font-medium">
                                          Error
                                        </div>
                                        <div className="bg-white dark:bg-[#0d1117] rounded border border-red-200 dark:border-red-900/50 p-2.5">
                                          <code className="text-[12px] font-mono text-red-600 dark:text-red-400 break-all">
                                            {tc.error}
                                          </code>
                                        </div>
                                      </div>
                                    )}
                                  </div>
                                );
                              })()}
                            </div>
                          )}

                          {/* Fallback for failed without test details */}
                          {isFailed && failedTests.length === 0 && sub.message && (
                            <div className="p-3">
                              <div className="text-[10px] uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-1.5 font-medium">
                                Error
                              </div>
                              <div className="bg-white dark:bg-[#0d1117] rounded border border-gray-200 dark:border-gray-700 p-2.5">
                                <code className="text-[12px] font-mono text-red-600 dark:text-red-400">
                                  {sub.message}
                                </code>
                              </div>
                            </div>
                          )}
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
