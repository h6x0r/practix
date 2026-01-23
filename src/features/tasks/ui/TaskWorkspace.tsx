
import React, { useContext, useState, useEffect, useMemo, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { AuthContext } from '@/components/Layout';
import { useToast } from '@/components/Toast';
import { useLanguage, useUITranslation } from '@/contexts/LanguageContext';
import { useSubscription } from '@/contexts/SubscriptionContext';
import { TaskAccess } from '@/features/subscriptions/model/types';

// UI Components
import { WorkspaceHeader } from './components/WorkspaceHeader';
import { TaskDescriptionPanel, DescriptionPanelTab } from './components/TaskDescriptionPanel';
import { CodeEditorPanel } from './components/CodeEditorPanel';
import { PromptEditorPanel } from './components/PromptEditorPanel';
import { BugReportModal } from './components/BugReportModal';
import { PremiumRequiredOverlay } from '@/components/PremiumRequiredOverlay';
import { ResizeHandle } from '@/components/ResizeHandle';
import { IconCode, IconBook, IconSparkles } from '@/components/Icons';

// Hooks
import { useResizablePanel } from '@/hooks/useResizablePanel';

// Business Logic Hooks
import { useTaskState } from '../model/useTaskState';
import { useTaskRunner, detectTaskLanguage } from '../model/useTaskRunner';
import { usePromptRunner } from '../model/usePromptRunner';
import { useAiChat } from '../model/useAiChat';
import { useTaskNavigation } from '../model/useTaskNavigation';
import { userCoursesService } from '@/features/courses/api/userCoursesService';

// Default access for free users
const defaultAccess: TaskAccess = {
  canView: true,
  canRun: true,
  canSubmit: true,
  canSeeSolution: false,
  canUseAiTutor: false,
  queuePriority: 10,
};

const TaskWorkspace = () => {
  const { slug, courseId } = useParams();
  const navigate = useNavigate();
  const { user } = useContext(AuthContext);
  const { showToast } = useToast();
  const { t, language: uiLanguage } = useLanguage();
  const { tUI } = useUITranslation();
  const { getTaskAccess } = useSubscription();

  // Mobile State
  const [mobileTab, setMobileTab] = useState<'task' | 'code' | 'ai'>('task');

  // Resizable Panel State (dynamic maxWidth based on viewport)
  const { width: leftPanelWidth, isResizing, startResize } = useResizablePanel({
    storageKey: 'task-workspace-left-panel-width',
    defaultWidth: 500,
    minWidth: 350,
    maxWidth: 800,
    maxWidthRatio: 0.55, // Max 55% of viewport width
  });

  // Track desktop mode for resizable panels (md breakpoint = 768px)
  const [isDesktop, setIsDesktop] = useState(() => window.innerWidth >= 768);
  useEffect(() => {
    const mediaQuery = window.matchMedia('(min-width: 768px)');
    const handleChange = (e: MediaQueryListEvent) => setIsDesktop(e.matches);
    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, []);

  // Bug Report Modal State
  const [showBugReport, setShowBugReport] = useState(false);

  // Access Control State
  const [taskAccess, setTaskAccess] = useState<TaskAccess>(defaultAccess);

  // Left Panel Tab State (lifted from TaskDescriptionPanel for auto-switching)
  const [descriptionPanelTab, setDescriptionPanelTab] = useState<DescriptionPanelTab>('description');

  // 1. Data Layer
  const { task: rawTask, isLoading } = useTaskState(slug);

  // Apply translations to task based on current language
  const task = useMemo(() => t(rawTask), [rawTask, t]);

  // Fetch task access when task loads
  useEffect(() => {
    const fetchAccess = async () => {
      if (rawTask?.id && user) {
        try {
          const access = await getTaskAccess(rawTask.id);
          setTaskAccess(access);
        } catch (err) {
          console.error('Failed to fetch task access:', err);
        }
      } else {
        setTaskAccess(defaultAccess);
      }
    };
    fetchAccess();
  }, [rawTask?.id, user, getTaskAccess]);

  // Update course last accessed time (moves course to top of My Tasks)
  useEffect(() => {
    if (courseId && user) {
      userCoursesService.updateLastAccessed(courseId).catch(() => {
        // Silently fail - this is a non-critical operation
      });
    }
  }, [courseId, user]);

  // 1.5 Navigation Layer
  const {
    modules,
    flatTasks,
    currentIndex,
    goToPrev,
    goToNext,
  } = useTaskNavigation(courseId, slug);

  // Keyboard shortcuts for task navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      const isTyping = target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable;

      // Don't trigger navigation when user is typing
      if (isTyping) return;

      // Alt+ArrowLeft/Right for prev/next task
      if (e.altKey && e.key === 'ArrowLeft') {
        const prevUrl = goToPrev();
        if (prevUrl) {
          e.preventDefault();
          navigate(prevUrl);
        }
      } else if (e.altKey && e.key === 'ArrowRight') {
        const nextUrl = goToNext();
        if (nextUrl) {
          e.preventDefault();
          navigate(nextUrl);
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [goToPrev, goToNext, navigate]);

  // Detect if this is a Prompt Engineering task
  const isPromptTask = rawTask?.taskType === 'PROMPT';

  // 2. Logic Layer - use rawTask to avoid resetting code on language change
  // Code tasks
  const codeRunner = useTaskRunner(isPromptTask ? null : rawTask, courseId);
  // Prompt tasks
  const promptRunner = usePromptRunner(isPromptTask ? rawTask : null);

  // Unified state accessors
  const code = isPromptTask ? promptRunner.prompt : codeRunner.code;
  const setCode = isPromptTask ? promptRunner.setPrompt : codeRunner.setCode;
  const activeTab = isPromptTask ? promptRunner.activeTab : codeRunner.activeTab;
  const setActiveTab = isPromptTask ? promptRunner.setActiveTab : codeRunner.setActiveTab;
  const isSubmitting = isPromptTask ? promptRunner.isSubmitting : codeRunner.isSubmitting;
  const isLoadingSubmissions = isPromptTask ? promptRunner.isLoadingSubmissions : codeRunner.isLoadingSubmissions;
  const isRunning = isPromptTask ? false : codeRunner.isRunning;
  const runResult = isPromptTask ? null : codeRunner.runResult;
  const isRunResultsOpen = isPromptTask ? false : codeRunner.isRunResultsOpen;
  const closeRunResults = isPromptTask ? () => {} : codeRunner.closeRunResults;
  const cooldownRemaining = isPromptTask ? 0 : codeRunner.cooldownRemaining;

  // Derived State - Language Detection (using shared function)
  const language = detectTaskLanguage(task, courseId);
  const isGo = language === 'go';
  const isPython = language === 'python';
  const langLabel = isPromptTask ? 'Prompt' : (isPython ? 'Python' : isGo ? 'Go' : 'Java');
  const fileExt = isPython ? '.py' : isGo ? '.go' : '.java';
  const backLink = courseId ? `/course/${courseId}` : '/courses';

  // 3. AI Layer (with UI language for localized responses)
  const {
    question, setQuestion,
    chat, isLoading: aiLoading,
    askAi
  } = useAiChat(task, code, langLabel, uiLanguage);

  // Handler for loading code from a previous submission
  const handleLoadSubmissionCode = useCallback((submissionCode: string) => {
    codeRunner.setCode(submissionCode);
  }, [codeRunner]);

  // Handler for Run Code button - quick tests (5 tests, no save)
  const handleRunCode = async () => {
    if (!isPromptTask) {
      setDescriptionPanelTab('results'); // Auto-switch to Results tab
      await codeRunner.runQuickTests();
    }
  };

  // Handler for Submit button - full evaluation
  const handleSubmit = async () => {
    if (isPromptTask) {
      await promptRunner.submitPrompt();
    } else {
      setDescriptionPanelTab('results'); // Auto-switch to Results tab
      await codeRunner.submitCode();
    }
  };


  // Update Page Title
  useEffect(() => {
    if (task) {
      document.title = `${task.title} — Practix`;
    }
    // Cleanup to default on unmount is handled by Layout, but we can reset manually if needed.
  }, [task]);


  if (isLoading || !task) {
    return <div className="h-full flex items-center justify-center text-gray-500 animate-pulse">{tUI('task.loadingEnv')}</div>;
  }

  // Show premium overlay for premium tasks when user doesn't have access
  const showPremiumOverlay = rawTask?.isPremium && !taskAccess.canView;

  return (
    <div className="flex flex-col h-[calc(100vh-80px)] -m-6 bg-gray-100 dark:bg-black relative">

      <WorkspaceHeader
        task={task}
        backLink={backLink}
        langLabel={langLabel}
        isRunning={isRunning}
        isSubmitting={isSubmitting}
        onRun={handleRunCode}
        onSubmit={handleSubmit}
        onBugReport={() => setShowBugReport(true)}
        courseId={courseId}
        modules={modules}
        flatTasks={flatTasks}
        currentIndex={currentIndex}
        prevTaskUrl={goToPrev()}
        nextTaskUrl={goToNext()}
        isPromptTask={isPromptTask}
        cooldownRemaining={cooldownRemaining}
      />

      {/* Main Content Area - wrapped with premium overlay when access is restricted */}
      {showPremiumOverlay ? (
        <PremiumRequiredOverlay>
          <div className="flex-1 flex overflow-hidden pb-16 md:pb-0 relative">
            {/* LEFT PANEL (Description & Video & AI) */}
            <div className="flex flex-col bg-white dark:bg-dark-surface border-r border-gray-200 dark:border-dark-border w-full md:w-[40%] md:min-w-[350px] md:max-w-[50%]">
              <TaskDescriptionPanel
                task={task}
                aiChat={[]}
                aiQuestion=""
                onAiQuestionChange={() => {}}
                onAiSend={() => {}}
                aiLoading={false}
                canSeeSolution={false}
                canUseAiTutor={false}
              />
            </div>
            {/* RIGHT PANEL (Editor) */}
            <div className="flex-1 flex-col bg-white dark:bg-[#1e1e1e] border-l border-gray-200 dark:border-gray-800 flex">
              <CodeEditorPanel
                code={code}
                setCode={() => {}}
                isGo={!!isGo}
                fileExt={fileExt}
                isPremium={false}
                canSeeSolution={false}
                task={task}
                language={language}
              />
            </div>
          </div>
        </PremiumRequiredOverlay>
      ) : (
        <div className={`flex-1 flex overflow-hidden pb-16 md:pb-0 relative ${isResizing ? 'select-none' : ''}`}>

          {/* LEFT PANEL (Description & Video & AI) */}
          {/* Hidden on mobile unless tab is task or ai */}
          <div
            className={`
              flex-col bg-white dark:bg-dark-surface
              w-full md:flex flex-shrink-0
              ${mobileTab === 'code' ? 'hidden' : 'flex'}
            `}
            style={{ width: isDesktop ? leftPanelWidth : '100%' }}
          >

            {/* Task Description Panel with AI Tutor and Results tabs */}
            <div className={`flex-1 overflow-hidden flex flex-col ${mobileTab === 'ai' ? 'hidden md:flex' : 'flex'}`}>
              <TaskDescriptionPanel
                task={task}
                aiChat={chat}
                aiQuestion={question}
                onAiQuestionChange={setQuestion}
                onAiSend={askAi}
                aiLoading={aiLoading}
                canSeeSolution={taskAccess.canSeeSolution}
                canUseAiTutor={taskAccess.canUseAiTutor}
                // Results tab props
                runResult={codeRunner.runResult}
                isRunLoading={codeRunner.isRunning}
                submissions={codeRunner.submissions}
                isLoadingSubmissions={codeRunner.isLoadingSubmissions}
                onLoadSubmissionCode={handleLoadSubmissionCode}
                // Controlled tab state
                activeTab={descriptionPanelTab}
                onTabChange={setDescriptionPanelTab}
              />
            </div>
          </div>

          {/* Resize Handle */}
          <ResizeHandle onMouseDown={startResize} isResizing={isResizing} />

          {/* RIGHT PANEL (Editor) */}
          {/* Hidden on mobile unless tab is code */}
          <div className={`
              flex-1 flex-col bg-white dark:bg-[#1e1e1e]
              w-full md:w-auto md:flex relative
              ${mobileTab !== 'code' ? 'hidden' : 'flex'}
          `}>
            {isPromptTask ? (
              <PromptEditorPanel
                activeTab={activeTab}
                setActiveTab={setActiveTab}
                prompt={promptRunner.prompt}
                setPrompt={promptRunner.setPrompt}
                isPremium={!!user?.isPremium}
                submissions={promptRunner.submissions}
                isLoadingSubmissions={promptRunner.isLoadingSubmissions}
                task={task}
                isSubmitting={promptRunner.isSubmitting}
              />
            ) : (
              <CodeEditorPanel
                code={code}
                setCode={setCode}
                isGo={!!isGo}
                fileExt={fileExt}
                isPremium={!!user?.isPremium}
                canSeeSolution={taskAccess.canSeeSolution}
                task={task}
                language={language}
              />
            )}
          </div>
        </div>
      )}

      {/* MOBILE BOTTOM NAVIGATION - hidden when premium overlay is shown */}
      {!showPremiumOverlay && (
      <div className="md:hidden fixed bottom-0 left-0 right-0 h-16 bg-white dark:bg-dark-surface border-t border-gray-200 dark:border-dark-border flex items-center justify-around z-50 pb-safe">
        <button
          onClick={() => setMobileTab('task')}
          className={`flex flex-col items-center gap-1 p-2 ${mobileTab === 'task' ? 'text-brand-600 dark:text-brand-400' : 'text-gray-400'}`}
        >
          <IconBook className="w-5 h-5" />
          <span className="text-[10px] font-bold uppercase">{tUI('task.mobileTask')}</span>
        </button>

        <button
          onClick={() => setMobileTab('code')}
          className={`flex flex-col items-center gap-1 p-2 ${mobileTab === 'code' ? 'text-brand-600 dark:text-brand-400' : 'text-gray-400'}`}
        >
          <IconCode className="w-5 h-5" />
          <span className="text-[10px] font-bold uppercase">{tUI('task.mobileEditor')}</span>
        </button>

        <button
          onClick={() => setMobileTab('ai')}
          className={`flex flex-col items-center gap-1 p-2 ${mobileTab === 'ai' ? 'text-brand-600 dark:text-brand-400' : 'text-gray-400'}`}
        >
          <IconSparkles className="w-5 h-5" />
          <span className="text-[10px] font-bold uppercase">{tUI('task.mobileTutor')}</span>
        </button>
      </div>
      )}

      {/* Bug Report Modal */}
      <BugReportModal
        isOpen={showBugReport}
        onClose={() => setShowBugReport(false)}
        taskId={task.id}
        taskSlug={task.slug}
        isPremium={!!user?.isPremium}
        userCode={code}
      />

    </div>
  );
};

export default TaskWorkspace;