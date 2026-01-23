
import React, { useState, useContext, useMemo, memo, useRef, useEffect, useCallback } from 'react';
import { Task, AiChatMessage } from '@/types';
import { DescriptionRenderer } from './DescriptionRenderer';
import { HintsPanel } from './HintsPanel';
import { SolutionExplanationTab } from './SolutionExplanationTab';
import { ResultsTab } from './ResultsTab';
import { IconBook, IconSparkles, IconLock, IconMessageCircle, IconPlayCircle } from '@/components/Icons';
import { AuthContext } from '@/components/Layout';
import { Link } from 'react-router-dom';
import { useUITranslation, useLanguage } from '@/contexts/LanguageContext';
import { Submission } from '../../model/types';
import { RunTestsResult } from '../../api/taskService';

export type DescriptionPanelTab = 'description' | 'results' | 'solution' | 'ai';

interface TaskDescriptionPanelProps {
  task: Task;
  // AI Tutor props
  aiChat?: AiChatMessage[];
  aiQuestion?: string;
  onAiQuestionChange?: (val: string) => void;
  onAiSend?: () => void;
  aiLoading?: boolean;
  // Access control props
  canSeeSolution?: boolean;
  canUseAiTutor?: boolean;
  // Results tab props
  runResult?: RunTestsResult | null;
  isRunLoading?: boolean;
  submissions?: Submission[];
  isLoadingSubmissions?: boolean;
  onLoadSubmissionCode?: (code: string) => void;
  // Controlled tab state
  activeTab?: DescriptionPanelTab;
  onTabChange?: (tab: DescriptionPanelTab) => void;
}

export const TaskDescriptionPanel = memo(({
  task,
  aiChat = [],
  aiQuestion = '',
  onAiQuestionChange,
  onAiSend,
  aiLoading = false,
  canSeeSolution = false,
  canUseAiTutor = false,
  // Results tab props
  runResult,
  isRunLoading = false,
  submissions = [],
  isLoadingSubmissions = false,
  onLoadSubmissionCode,
  // Controlled tab state
  activeTab: controlledActiveTab,
  onTabChange,
}: TaskDescriptionPanelProps) => {
  // Use controlled state if provided, otherwise use internal state
  const [internalTab, setInternalTab] = useState<DescriptionPanelTab>('description');
  const activeTab = controlledActiveTab ?? internalTab;
  const setActiveTab = onTabChange ?? setInternalTab;
  const { user } = useContext(AuthContext);
  const { tUI } = useUITranslation();
  const { language } = useLanguage();

  // Compact mode for tabs when they don't fit in one line
  const tabsContainerRef = useRef<HTMLDivElement>(null);
  const [isCompactTabs, setIsCompactTabs] = useState(false);

  // Check if tabs overflow their container
  // We use a threshold based on container width - if container is too narrow, switch to compact
  const checkTabsOverflow = useCallback(() => {
    const container = tabsContainerRef.current;
    if (!container) return;

    // Minimum width needed to show all tabs with text (approximately)
    // Description (~100px) + Results (~80px) + Solution (~80px) + AI Tutor (~90px) + padding (~80px) ≈ 430px
    const MIN_WIDTH_FOR_TEXT = 420;

    const containerWidth = container.clientWidth;
    const shouldBeCompact = containerWidth < MIN_WIDTH_FOR_TEXT;

    setIsCompactTabs(shouldBeCompact);
  }, []);

  // Use ResizeObserver to detect container size changes
  useEffect(() => {
    const container = tabsContainerRef.current;
    if (!container) return;

    // Initial check
    checkTabsOverflow();

    const resizeObserver = new ResizeObserver(() => {
      checkTabsOverflow();
    });

    resizeObserver.observe(container);

    return () => {
      resizeObserver.disconnect();
    };
  }, [checkTabsOverflow]);

  // Apply translations based on current language
  const localizedTask = useMemo(() => {
    const translations = task.translations as Record<string, {
      title?: string;
      description?: string;
      hint1?: string;
      hint2?: string;
      solutionCode?: string;
      whyItMatters?: string;
    }> | undefined;

    if (language === 'en' || !translations?.[language]) {
      return task;
    }

    const langTranslations = translations[language];
    return {
      ...task,
      title: langTranslations.title || task.title,
      description: langTranslations.description || task.description,
      hint1: langTranslations.hint1 || task.hint1,
      hint2: langTranslations.hint2 || task.hint2,
      solutionCode: langTranslations.solutionCode || task.solutionCode,
      whyItMatters: langTranslations.whyItMatters || task.whyItMatters,
    };
  }, [task, language]);

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Tab Header */}
      <div
        ref={tabsContainerRef}
        className="flex border-b border-gray-200 dark:border-dark-border bg-gray-50 dark:bg-dark-bg/50 px-4"
      >
        <button
          onClick={() => setActiveTab('description')}
          data-testid="description-tab"
          title={isCompactTabs ? tUI('task.description') : undefined}
          className={`flex items-center gap-2 px-4 py-3 text-sm font-bold border-b-2 transition-colors whitespace-nowrap ${
            activeTab === 'description'
              ? 'border-brand-500 text-brand-600 dark:text-brand-400'
              : 'border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
          }`}
        >
          <IconBook className="w-4 h-4 flex-shrink-0" />
          {!isCompactTabs && <span>{tUI('task.description')}</span>}
        </button>
        <button
          onClick={() => setActiveTab('results')}
          data-testid="results-tab"
          title={isCompactTabs ? tUI('task.results') : undefined}
          className={`flex items-center gap-2 px-4 py-3 text-sm font-bold border-b-2 transition-colors whitespace-nowrap ${
            activeTab === 'results'
              ? 'border-brand-500 text-brand-600 dark:text-brand-400'
              : 'border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
          }`}
        >
          <IconPlayCircle className="w-4 h-4 flex-shrink-0" />
          {!isCompactTabs && <span>{tUI('task.results')}</span>}
          {/* Status indicator dot */}
          {runResult && (
            <span className={`w-2 h-2 rounded-full flex-shrink-0 ${
              runResult.status === 'passed' ? 'bg-green-500' : 'bg-red-500'
            }`} />
          )}
        </button>
        <button
          onClick={() => setActiveTab('solution')}
          title={isCompactTabs ? tUI('task.solution') : undefined}
          className={`flex items-center gap-2 px-4 py-3 text-sm font-bold border-b-2 transition-colors whitespace-nowrap ${
            activeTab === 'solution'
              ? 'border-brand-500 text-brand-600 dark:text-brand-400'
              : 'border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
          }`}
        >
          <IconSparkles className={`w-4 h-4 flex-shrink-0 ${canSeeSolution ? 'text-green-500' : 'text-amber-500'}`} />
          {!isCompactTabs && <span>{tUI('task.solution')}</span>}
          {!canSeeSolution && <IconLock className="w-3 h-3 text-amber-500 flex-shrink-0" />}
        </button>
        <button
          data-testid="ai-tutor-toggle"
          onClick={() => setActiveTab('ai')}
          title={isCompactTabs ? tUI('task.aiTutor') : undefined}
          className={`flex items-center gap-2 px-4 py-3 text-sm font-bold border-b-2 transition-colors whitespace-nowrap ${
            activeTab === 'ai'
              ? 'border-brand-500 text-brand-600 dark:text-brand-400'
              : 'border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
          }`}
        >
          <IconMessageCircle className={`w-4 h-4 flex-shrink-0 ${canUseAiTutor ? 'text-purple-500' : 'text-amber-500'}`} />
          {!isCompactTabs && <span>{tUI('task.aiTutor')}</span>}
          {!canUseAiTutor && <IconLock className="w-3 h-3 text-amber-500 flex-shrink-0" />}
        </button>
      </div>

      {/* Tab Content */}
      <div className="flex-1 overflow-y-auto custom-scrollbar">
        {activeTab === 'description' && (
          <div className="p-5" data-testid="task-description">
            <DescriptionRenderer text={localizedTask.description} />

            {/* Hints Section */}
            <HintsPanel hint1={localizedTask.hint1} hint2={localizedTask.hint2} />

            <div className="mt-10 pt-6 border-t border-gray-100 dark:border-dark-border">
              <h4 className="text-xs uppercase font-bold text-gray-400 mb-3 tracking-wider">
                {tUI('task.relatedTopics')}
              </h4>
              <div className="flex flex-wrap gap-2">
                {task.tags.map(tag => (
                  <span key={tag} className="px-2.5 py-1 bg-gray-100 dark:bg-dark-bg text-gray-600 dark:text-gray-400 text-xs font-medium rounded-full border border-gray-200 dark:border-dark-border">
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'results' && (
          <ResultsTab
            runResult={runResult ?? null}
            isRunLoading={isRunLoading}
            submissions={submissions}
            isLoadingSubmissions={isLoadingSubmissions}
            onLoadSubmissionCode={onLoadSubmissionCode}
          />
        )}

        {activeTab === 'solution' && (
          canSeeSolution ? (
            <SolutionExplanationTab task={localizedTask} />
          ) : (
            <div className="flex-1 flex items-center justify-center p-8">
              <div className="text-center max-w-sm p-8 bg-white dark:bg-dark-surface rounded-2xl shadow-xl border border-gray-100 dark:border-dark-border">
                <div className="w-16 h-16 bg-amber-100 dark:bg-amber-900/20 rounded-full flex items-center justify-center mx-auto mb-6 text-amber-500">
                  <IconSparkles className="w-8 h-8" />
                </div>
                <h3 className="text-xl font-bold text-gray-900 dark:text-white">{tUI('task.solutionLocked')}</h3>
                <p className="text-gray-500 dark:text-gray-400 text-sm mt-3 leading-relaxed">
                  {tUI('task.solutionLockedDesc')}
                </p>
                <Link to="/premium" className="inline-block mt-8 px-8 py-3 bg-gradient-to-r from-brand-600 to-purple-600 hover:from-brand-500 hover:to-purple-500 text-white font-bold rounded-xl shadow-lg shadow-brand-500/25 transition-all transform hover:-translate-y-0.5">
                  {tUI('task.unlockNow')}
                </Link>
              </div>
            </div>
          )
        )}

        {activeTab === 'ai' && (
          canUseAiTutor ? (
            <div data-testid="ai-tutor-chat" className="flex flex-col h-full">
              {/* Chat Messages */}
              <div data-testid="ai-tutor-messages" className="flex-1 overflow-y-auto p-6 space-y-4">
                {aiChat.length === 0 && (
                  <div data-testid="ai-tutor-empty" className="h-full flex flex-col items-center justify-center text-center py-16">
                    <div className="w-20 h-20 bg-purple-100 dark:bg-purple-900/20 rounded-full flex items-center justify-center mb-4">
                      <IconMessageCircle className="w-10 h-10 text-purple-500" />
                    </div>
                    <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-2">{tUI('task.aiTutor')}</h3>
                    <p className="text-sm text-gray-500 dark:text-gray-400 max-w-xs">
                      {tUI('task.aiTutorEmpty')}
                    </p>
                  </div>
                )}
                {aiChat.map((msg, i) => (
                  <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed shadow-sm ${
                      msg.role === 'user'
                        ? 'bg-brand-600 text-white rounded-br-none'
                        : 'bg-gray-100 dark:bg-dark-bg text-gray-800 dark:text-gray-200 rounded-bl-none border border-gray-200 dark:border-dark-border'
                    }`}>
                      {msg.text}
                    </div>
                  </div>
                ))}
                {aiLoading && (
                  <div data-testid="ai-tutor-loading" className="flex justify-start">
                    <div className="bg-gray-100 dark:bg-dark-bg rounded-2xl rounded-bl-none px-4 py-3 border border-gray-200 dark:border-dark-border">
                      <div className="flex gap-1">
                        <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
                        <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
                        <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Chat Input */}
              <div className="p-4 bg-gray-50 dark:bg-black border-t border-gray-200 dark:border-dark-border">
                <div className="relative">
                  <input
                    data-testid="ai-tutor-input"
                    type="text"
                    value={aiQuestion}
                    onChange={(e) => onAiQuestionChange?.(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && onAiSend?.()}
                    disabled={aiLoading}
                    placeholder={tUI('task.aiTutorPlaceholder')}
                    className="w-full bg-white dark:bg-dark-surface border border-gray-200 dark:border-dark-border rounded-xl pl-4 pr-12 py-3 text-sm focus:ring-2 focus:ring-purple-500 outline-none dark:text-white shadow-sm transition-all"
                  />
                  <button
                    data-testid="ai-tutor-send"
                    onClick={onAiSend}
                    disabled={aiLoading || !aiQuestion.trim()}
                    className="absolute right-2 top-1/2 -translate-y-1/2 p-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" />
                    </svg>
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div data-testid="ai-tutor-locked" className="flex-1 flex items-center justify-center p-8">
              <div className="text-center max-w-sm p-8 bg-white dark:bg-dark-surface rounded-2xl shadow-xl border border-gray-100 dark:border-dark-border">
                <div className="w-16 h-16 bg-purple-100 dark:bg-purple-900/20 rounded-full flex items-center justify-center mx-auto mb-6 text-purple-500">
                  <IconMessageCircle className="w-8 h-8" />
                </div>
                <h3 className="text-xl font-bold text-gray-900 dark:text-white">{tUI('task.aiTutorLockedTitle')}</h3>
                <p className="text-gray-500 dark:text-gray-400 text-sm mt-3 leading-relaxed">
                  {tUI('task.aiTutorLocked')}
                </p>
                <Link to="/premium" className="inline-block mt-8 px-8 py-3 bg-gradient-to-r from-purple-500 to-purple-700 text-white font-bold rounded-xl hover:shadow-lg hover:shadow-purple-500/30 transition-all transform hover:-translate-y-0.5">
                  {tUI('task.unlockNow')}
                </Link>
              </div>
            </div>
          )
        )}
      </div>
    </div>
  );
});
