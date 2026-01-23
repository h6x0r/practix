
import React, { memo } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { IconX, IconPlay, IconChevronLeft, IconChevronRight, IconBug } from '@/components/Icons';
import { Task, CourseModule } from '@/types';
import { useUITranslation } from '@/contexts/LanguageContext';
import { TaskNavigationDropdown } from './TaskNavigationDropdown';
import { FlatTask } from '../../model/useTaskNavigation';

interface WorkspaceHeaderProps {
  task: Task;
  backLink: string;
  langLabel: string;
  isRunning: boolean;
  isSubmitting?: boolean;
  onRun: () => void;
  onSubmit: () => void;
  onBugReport: () => void;
  // Navigation props
  courseId?: string;
  modules?: CourseModule[];
  flatTasks?: FlatTask[];
  currentIndex?: number;
  prevTaskUrl?: string | null;
  nextTaskUrl?: string | null;
  // Prompt task indicator
  isPromptTask?: boolean;
  // Rate limiting - cooldown remaining in ms (0 = ready)
  cooldownRemaining?: number;
}

export const WorkspaceHeader = memo(({
  task,
  backLink,
  langLabel,
  isRunning,
  isSubmitting = false,
  onRun,
  onSubmit,
  onBugReport,
  courseId,
  modules = [],
  flatTasks = [],
  currentIndex = -1,
  prevTaskUrl,
  nextTaskUrl,
  isPromptTask = false,
  cooldownRemaining = 0,
}: WorkspaceHeaderProps) => {
  const { tUI } = useUITranslation();
  const navigate = useNavigate();

  const hasNavigation = courseId && flatTasks.length > 0 && currentIndex >= 0;
  const isOnCooldown = cooldownRemaining > 0;
  const cooldownSeconds = Math.ceil(cooldownRemaining / 1000);

  return (
    <div className="h-14 flex items-center justify-between px-4 bg-white dark:bg-dark-surface border-b border-gray-200 dark:border-dark-border z-10 shadow-sm">
        <div className="flex items-center gap-4">
          <Link to={backLink} data-testid="back-to-course-button" className="w-8 h-8 flex items-center justify-center rounded-lg hover:bg-gray-100 dark:hover:bg-dark-bg text-gray-500 transition-colors">
            <IconX className="w-5 h-5" />
          </Link>
          <div className="h-6 w-px bg-gray-200 dark:bg-dark-border"></div>

          {/* Navigation Controls */}
          {hasNavigation && (
            <div className="flex items-center gap-1">
              {/* Prev Button */}
              <button
                onClick={() => prevTaskUrl && navigate(prevTaskUrl)}
                disabled={!prevTaskUrl}
                data-testid="prev-task-button"
                className={`w-7 h-7 flex items-center justify-center rounded-md transition-colors ${
                  prevTaskUrl
                    ? 'hover:bg-gray-100 dark:hover:bg-dark-bg text-gray-600 dark:text-gray-400'
                    : 'text-gray-300 dark:text-gray-700 cursor-not-allowed'
                }`}
                title={prevTaskUrl ? tUI('task.prevTask') : undefined}
              >
                <IconChevronLeft className="w-4 h-4" />
              </button>

              {/* Task Dropdown */}
              <TaskNavigationDropdown
                modules={modules}
                flatTasks={flatTasks}
                currentTaskSlug={task.slug}
                courseId={courseId}
                currentIndex={currentIndex}
              />

              {/* Next Button */}
              <button
                onClick={() => nextTaskUrl && navigate(nextTaskUrl)}
                disabled={!nextTaskUrl}
                data-testid="next-task-button"
                className={`w-7 h-7 flex items-center justify-center rounded-md transition-colors ${
                  nextTaskUrl
                    ? 'hover:bg-gray-100 dark:hover:bg-dark-bg text-gray-600 dark:text-gray-400'
                    : 'text-gray-300 dark:text-gray-700 cursor-not-allowed'
                }`}
                title={nextTaskUrl ? tUI('task.nextTask') : undefined}
              >
                <IconChevronRight className="w-4 h-4" />
              </button>

              <div className="h-6 w-px bg-gray-200 dark:bg-dark-border ml-2"></div>
            </div>
          )}

          <div>
            <h2 data-testid="task-title" className="font-bold text-sm text-gray-900 dark:text-white flex items-center gap-2">
              {task.title}
            </h2>
            <div className="flex items-center gap-2 text-[10px] text-gray-500">
               <span data-testid="difficulty-badge" className={`px-1.5 rounded-sm uppercase font-bold tracking-wider ${
                 task.difficulty === 'easy' ? 'text-green-600 bg-green-50 dark:bg-green-900/20' :
                 task.difficulty === 'medium' ? 'text-yellow-600 bg-yellow-50 dark:bg-yellow-900/20' : 'text-red-600 bg-red-50 dark:bg-red-900/20'
               }`}>
                 {task.difficulty}
               </span>
               <span className="w-1 h-1 rounded-full bg-gray-300 dark:bg-gray-700"></span>
               <span className="font-medium text-gray-600 dark:text-gray-400">{langLabel}</span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Bug Report Button */}
          <button
            onClick={onBugReport}
            className="w-8 h-8 flex items-center justify-center rounded-md border border-gray-200 dark:border-dark-border text-gray-500 dark:text-gray-400 hover:text-red-500 dark:hover:text-red-400 hover:border-red-300 dark:hover:border-red-700 hover:bg-red-50 dark:hover:bg-red-900/10 transition-all"
            title={tUI('bugReport.title')}
          >
            <IconBug className="w-4 h-4" />
          </button>
          <div className="h-6 w-px bg-gray-200 dark:bg-dark-border"></div>

          {/* Run Button - hidden for prompt tasks */}
          {!isPromptTask && (
            <button
              onClick={onRun}
              disabled={isRunning || isOnCooldown}
              data-testid="run-button"
              className={`flex items-center gap-2 px-5 py-1.5 bg-green-600 hover:bg-green-500 text-white text-xs font-bold rounded-md transition-all shadow-lg shadow-green-900/20 ${
                isRunning || isOnCooldown ? 'opacity-75 cursor-wait' : ''
              }`}
              title={isOnCooldown ? `Wait ${cooldownSeconds}s` : undefined}
            >
              {isRunning ? (
                <span className="animate-spin">⟳</span>
              ) : isOnCooldown ? (
                <span className="text-[10px] font-mono">{cooldownSeconds}s</span>
              ) : (
                <IconPlay className="w-3 h-3" />
              )}
              {tUI('task.runCode')}
            </button>
          )}

          {/* Submit Button - purple for prompt tasks */}
          <button
            onClick={onSubmit}
            disabled={isSubmitting || isOnCooldown}
            data-testid="submit-button"
            className={`px-5 py-1.5 text-xs font-bold rounded-md transition-all ${
              isPromptTask
                ? `bg-purple-600 hover:bg-purple-500 text-white shadow-lg shadow-purple-900/20 ${isSubmitting || isOnCooldown ? 'opacity-75 cursor-wait' : ''}`
                : `bg-white dark:bg-dark-bg border border-gray-200 dark:border-dark-border text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:border-gray-600 ${isSubmitting || isOnCooldown ? 'opacity-75 cursor-wait' : ''}`
            }`}
            title={isOnCooldown ? `Wait ${cooldownSeconds}s` : undefined}
          >
            {isSubmitting ? (
              <span className="animate-spin inline-block mr-1">⟳</span>
            ) : isOnCooldown ? (
              <span className="text-[10px] font-mono mr-1">{cooldownSeconds}s</span>
            ) : null}
            {isPromptTask ? 'Submit Prompt' : tUI('task.submit')}
          </button>
        </div>
      </div>
  );
});
