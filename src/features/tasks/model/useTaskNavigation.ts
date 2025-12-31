import { useState, useEffect, useMemo, useCallback } from 'react';
import { courseService } from '../../courses/api/courseService';
import { CourseModule, Topic, Task } from '@/types';
import { createLogger } from '@/lib/logger';

const log = createLogger('TaskNavigation');

export interface FlatTask {
  slug: string;
  title: string;
  difficulty: 'easy' | 'medium' | 'hard';
  moduleId: string;
  moduleTitle: string;
  topicId: string;
  topicTitle: string;
  moduleIndex: number;
}

export interface TaskNavigationResult {
  modules: CourseModule[];
  flatTasks: FlatTask[];
  currentIndex: number;
  prevTask: FlatTask | null;
  nextTask: FlatTask | null;
  isLoading: boolean;
  goToPrev: () => string | null;
  goToNext: () => string | null;
}

// Module colors for visual differentiation in dropdown
export const MODULE_COLORS = [
  { bg: 'bg-blue-50 dark:bg-blue-900/20', border: 'border-blue-200 dark:border-blue-800', text: 'text-blue-600 dark:text-blue-400', dot: 'bg-blue-500' },
  { bg: 'bg-purple-50 dark:bg-purple-900/20', border: 'border-purple-200 dark:border-purple-800', text: 'text-purple-600 dark:text-purple-400', dot: 'bg-purple-500' },
  { bg: 'bg-green-50 dark:bg-green-900/20', border: 'border-green-200 dark:border-green-800', text: 'text-green-600 dark:text-green-400', dot: 'bg-green-500' },
  { bg: 'bg-orange-50 dark:bg-orange-900/20', border: 'border-orange-200 dark:border-orange-800', text: 'text-orange-600 dark:text-orange-400', dot: 'bg-orange-500' },
  { bg: 'bg-pink-50 dark:bg-pink-900/20', border: 'border-pink-200 dark:border-pink-800', text: 'text-pink-600 dark:text-pink-400', dot: 'bg-pink-500' },
  { bg: 'bg-cyan-50 dark:bg-cyan-900/20', border: 'border-cyan-200 dark:border-cyan-800', text: 'text-cyan-600 dark:text-cyan-400', dot: 'bg-cyan-500' },
  { bg: 'bg-amber-50 dark:bg-amber-900/20', border: 'border-amber-200 dark:border-amber-800', text: 'text-amber-600 dark:text-amber-400', dot: 'bg-amber-500' },
  { bg: 'bg-indigo-50 dark:bg-indigo-900/20', border: 'border-indigo-200 dark:border-indigo-800', text: 'text-indigo-600 dark:text-indigo-400', dot: 'bg-indigo-500' },
];

export function useTaskNavigation(courseId: string | undefined, currentTaskSlug: string | undefined): TaskNavigationResult {
  const [modules, setModules] = useState<CourseModule[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  // Fetch course structure
  useEffect(() => {
    if (!courseId) {
      setIsLoading(false);
      return;
    }

    setIsLoading(true);
    courseService.getCourseStructure(courseId)
      .then(data => {
        setModules(data || []);
      })
      .catch(err => {
        log.error('Failed to load course structure', err);
        setModules([]);
      })
      .finally(() => setIsLoading(false));
  }, [courseId]);

  // Flatten all tasks from modules -> topics -> tasks
  const flatTasks = useMemo<FlatTask[]>(() => {
    const result: FlatTask[] = [];

    modules.forEach((module, moduleIndex) => {
      (module.topics || []).forEach((topic) => {
        (topic.tasks || []).forEach((task) => {
          if (task && task.slug) {
            result.push({
              slug: task.slug,
              title: task.title,
              difficulty: task.difficulty,
              moduleId: module.id,
              moduleTitle: module.title,
              topicId: topic.id,
              topicTitle: topic.title,
              moduleIndex,
            });
          }
        });
      });
    });

    return result;
  }, [modules]);

  // Find current task index
  const currentIndex = useMemo(() => {
    if (!currentTaskSlug) return -1;
    return flatTasks.findIndex(t => t.slug === currentTaskSlug);
  }, [flatTasks, currentTaskSlug]);

  // Determine prev/next tasks
  const prevTask = useMemo(() => {
    if (currentIndex <= 0) return null;
    return flatTasks[currentIndex - 1];
  }, [flatTasks, currentIndex]);

  const nextTask = useMemo(() => {
    if (currentIndex < 0 || currentIndex >= flatTasks.length - 1) return null;
    return flatTasks[currentIndex + 1];
  }, [flatTasks, currentIndex]);

  // Navigation helpers - return URL path for navigation
  const goToPrev = useCallback(() => {
    if (!prevTask || !courseId) return null;
    return `/course/${courseId}/task/${prevTask.slug}`;
  }, [prevTask, courseId]);

  const goToNext = useCallback(() => {
    if (!nextTask || !courseId) return null;
    return `/course/${courseId}/task/${nextTask.slug}`;
  }, [nextTask, courseId]);

  return {
    modules,
    flatTasks,
    currentIndex,
    prevTask,
    nextTask,
    isLoading,
    goToPrev,
    goToNext,
  };
}
