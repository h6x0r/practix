import React, { useState, useRef, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { CourseModule } from '@/types';
import { FlatTask, MODULE_COLORS } from '../../model/useTaskNavigation';
import { IconChevronDown } from '@/components/Icons';
import { useLanguage } from '@/contexts/LanguageContext';

interface TaskNavigationDropdownProps {
  modules: CourseModule[];
  flatTasks: FlatTask[];
  currentTaskSlug: string;
  courseId: string;
  currentIndex: number;
}

export const TaskNavigationDropdown = ({
  modules,
  flatTasks,
  currentTaskSlug,
  courseId,
  currentIndex,
}: TaskNavigationDropdownProps) => {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const { t } = useLanguage();

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Close on Escape key
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener('keydown', handleKeyDown);
      return () => document.removeEventListener('keydown', handleKeyDown);
    }
  }, [isOpen]);

  // Group tasks by module for display
  const groupedByModule = modules.map((module, moduleIndex) => ({
    module,
    moduleIndex,
    color: MODULE_COLORS[moduleIndex % MODULE_COLORS.length],
    tasks: flatTasks.filter(t => t.moduleId === module.id),
  }));

  const currentTask = flatTasks[currentIndex];

  return (
    <div className="relative" ref={dropdownRef}>
      {/* Dropdown Trigger */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-1 px-2 py-1 rounded-md hover:bg-gray-100 dark:hover:bg-dark-bg transition-colors text-xs text-gray-600 dark:text-gray-400"
        title="Browse all tasks"
      >
        <span className="font-medium">
          {currentIndex + 1}/{flatTasks.length}
        </span>
        <IconChevronDown className={`w-3.5 h-3.5 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {/* Dropdown Menu */}
      {isOpen && (
        <div className="absolute top-full left-0 mt-2 w-80 max-h-[70vh] overflow-y-auto bg-white dark:bg-dark-surface rounded-xl shadow-xl border border-gray-200 dark:border-dark-border z-50">
          {/* Header */}
          <div className="sticky top-0 bg-white dark:bg-dark-surface border-b border-gray-100 dark:border-dark-border px-4 py-3">
            <h4 className="text-sm font-bold text-gray-900 dark:text-white">Course Tasks</h4>
            <p className="text-xs text-gray-500 mt-0.5">
              {flatTasks.length} tasks in {modules.length} modules
            </p>
          </div>

          {/* Modules and Tasks */}
          <div className="py-2">
            {groupedByModule.map(({ module, moduleIndex, color, tasks }) => {
              if (tasks.length === 0) return null;

              // Apply translations to module
              const translatedModule = t(module);

              return (
                <div key={module.id} className="mb-1">
                  {/* Module Header */}
                  <div className={`px-4 py-2 ${color.bg} border-l-4 ${color.border}`}>
                    <div className="flex items-center gap-2">
                      <span className={`w-2 h-2 rounded-full ${color.dot}`}></span>
                      <span className={`text-xs font-bold ${color.text}`}>
                        {translatedModule.title}
                      </span>
                    </div>
                  </div>

                  {/* Tasks in Module */}
                  <div className="pl-4">
                    {tasks.map((task, taskIndex) => {
                      const isCurrentTask = task.slug === currentTaskSlug;
                      const globalIndex = flatTasks.findIndex(t => t.slug === task.slug);

                      return (
                        <Link
                          key={task.slug}
                          to={`/course/${courseId}/task/${task.slug}`}
                          onClick={() => setIsOpen(false)}
                          className={`flex items-center gap-3 px-3 py-2 hover:bg-gray-50 dark:hover:bg-dark-bg transition-colors ${
                            isCurrentTask ? 'bg-brand-50 dark:bg-brand-900/20' : ''
                          }`}
                        >
                          {/* Task Number */}
                          <span className={`text-[10px] font-mono w-5 text-center ${
                            isCurrentTask ? 'text-brand-600 dark:text-brand-400 font-bold' : 'text-gray-400'
                          }`}>
                            {globalIndex + 1}
                          </span>

                          {/* Task Info */}
                          <div className="flex-1 min-w-0">
                            <p className={`text-xs truncate ${
                              isCurrentTask
                                ? 'text-brand-700 dark:text-brand-300 font-semibold'
                                : 'text-gray-700 dark:text-gray-300'
                            }`}>
                              {task.title}
                            </p>
                          </div>

                          {/* Difficulty Badge */}
                          <span className={`text-[9px] uppercase font-bold px-1.5 py-0.5 rounded ${
                            task.difficulty === 'easy'
                              ? 'text-green-600 bg-green-50 dark:bg-green-900/20'
                              : task.difficulty === 'medium'
                                ? 'text-yellow-600 bg-yellow-50 dark:bg-yellow-900/20'
                                : 'text-red-600 bg-red-50 dark:bg-red-900/20'
                          }`}>
                            {task.difficulty[0]}
                          </span>

                          {/* Current Indicator */}
                          {isCurrentTask && (
                            <span className="w-1.5 h-1.5 rounded-full bg-brand-500"></span>
                          )}
                        </Link>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};
