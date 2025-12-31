
import React, { useContext, useState, useEffect, useMemo, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { AuthContext } from '@/components/Layout';
import { useToast } from '@/components/Toast';
import { useLanguage, useUITranslation } from '@/contexts/LanguageContext';
import { useSubscription } from '@/contexts/SubscriptionContext';
import { TaskAccess } from '@/features/subscriptions/model/types';

// UI Components
import { WorkspaceHeader } from './components/WorkspaceHeader';
import { TaskDescriptionPanel } from './components/TaskDescriptionPanel';
import { CodeEditorPanel } from './components/CodeEditorPanel';
import { BugReportModal } from './components/BugReportModal';
import { RunResultsPanel } from './components/RunResultsPanel';
import { IconCode, IconBook, IconSparkles } from '@/components/Icons';

// Business Logic Hooks
import { useTaskState } from '../model/useTaskState';
import { useTaskRunner, detectTaskLanguage } from '../model/useTaskRunner';
import { useAiChat } from '../model/useAiChat';
import { useTaskNavigation } from '../model/useTaskNavigation';

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

  // Bug Report Modal State
  const [showBugReport, setShowBugReport] = useState(false);

  // Access Control State
  const [taskAccess, setTaskAccess] = useState<TaskAccess>(defaultAccess);

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

  // 2. Logic Layer - use rawTask to avoid resetting code on language change
  const {
    code, setCode,
    activeTab, setActiveTab,
    submissions, isRunning, isSubmitting, isLoadingSubmissions,
    runQuickTests, submitCode,
    runResult, isRunResultsOpen, closeRunResults
  } = useTaskRunner(rawTask, courseId);

  // Derived State - Language Detection (using shared function)
  const language = detectTaskLanguage(task, courseId);
  const isGo = language === 'go';
  const isPython = language === 'python';
  const langLabel = isPython ? 'Python' : isGo ? 'Go' : 'Java';
  const fileExt = isPython ? '.py' : isGo ? '.go' : '.java';
  const backLink = courseId ? `/course/${courseId}` : '/courses';

  // 3. AI Layer (with UI language for localized responses)
  const {
    question, setQuestion,
    chat, isLoading: aiLoading,
    askAi
  } = useAiChat(task, code, langLabel, uiLanguage);

  // Handler for Run Code button - quick tests (5 tests, no save)
  const handleRunCode = async () => {
    await runQuickTests();
  };

  // Handler for Submit button - full evaluation (10 tests, saved to DB)
  const handleSubmit = async () => {
    await submitCode();
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
      />

      {/* Main Content Area */}
      <div className="flex-1 flex overflow-hidden pb-16 md:pb-0 relative">
        
        {/* LEFT PANEL (Description & Video & AI) */}
        {/* Hidden on mobile unless tab is task or ai */}
        <div className={`
            flex-col bg-white dark:bg-dark-surface border-r border-gray-200 dark:border-dark-border
            w-full md:w-[40%] md:min-w-[350px] md:max-w-[50%] md:flex
            ${mobileTab === 'code' ? 'hidden' : 'flex'}
        `}>
          
          {/* Task Description Panel with AI Tutor integrated as tab */}
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
            />
          </div>
        </div>

        {/* RIGHT PANEL (Editor) */}
        {/* Hidden on mobile unless tab is code */}
        <div className={`
            flex-1 flex-col bg-white dark:bg-[#1e1e1e] border-l border-gray-200 dark:border-gray-800
            w-full md:w-auto md:flex relative
            ${mobileTab !== 'code' ? 'hidden' : 'flex'}
        `}>
          <CodeEditorPanel
              activeTab={activeTab}
              setActiveTab={setActiveTab}
              code={code}
              setCode={setCode}
              isGo={!!isGo}
              fileExt={fileExt}
              isPremium={!!user?.isPremium}
              canSeeSolution={taskAccess.canSeeSolution}
              submissions={submissions}
              isLoadingSubmissions={isLoadingSubmissions}
              task={task}
              language={language}
          />

          {/* Run Results Panel - shows quick test results */}
          <RunResultsPanel
            isOpen={isRunResultsOpen}
            onClose={closeRunResults}
            isLoading={isRunning}
            result={runResult}
          />
        </div>
      </div>

      {/* MOBILE BOTTOM NAVIGATION */}
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