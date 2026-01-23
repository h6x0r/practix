import { describe, it, expect, beforeEach, vi } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { useTaskNavigation, MODULE_COLORS } from './useTaskNavigation';

vi.mock('../../courses/api/courseService', () => ({
  courseService: {
    getCourseStructure: vi.fn(),
  },
}));

vi.mock('@/lib/logger', () => ({
  createLogger: () => ({
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
    debug: vi.fn(),
  }),
}));

import { courseService } from '../../courses/api/courseService';

describe('useTaskNavigation', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  const mockModules = [
    {
      id: 'mod-1',
      title: 'Module 1',
      topics: [
        {
          id: 'topic-1',
          title: 'Topic 1',
          tasks: [
            { slug: 'task-1', title: 'Task 1', difficulty: 'easy' as const },
            { slug: 'task-2', title: 'Task 2', difficulty: 'medium' as const },
          ],
        },
      ],
    },
    {
      id: 'mod-2',
      title: 'Module 2',
      topics: [
        {
          id: 'topic-2',
          title: 'Topic 2',
          tasks: [
            { slug: 'task-3', title: 'Task 3', difficulty: 'hard' as const },
          ],
        },
      ],
    },
  ];

  describe('initial state', () => {
    it('should start with loading state', () => {
      vi.mocked(courseService.getCourseStructure).mockResolvedValue([]);

      const { result } = renderHook(() => useTaskNavigation('course-1', 'task-1'));

      expect(result.current.isLoading).toBe(true);
    });

    it('should set loading false when no courseId', async () => {
      const { result } = renderHook(() => useTaskNavigation(undefined, 'task-1'));

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.modules).toEqual([]);
    });
  });

  describe('fetching course structure', () => {
    it('should fetch and set modules', async () => {
      vi.mocked(courseService.getCourseStructure).mockResolvedValue(mockModules);

      const { result } = renderHook(() => useTaskNavigation('course-1', 'task-1'));

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(courseService.getCourseStructure).toHaveBeenCalledWith('course-1');
      expect(result.current.modules).toEqual(mockModules);
    });

    it('should handle API error', async () => {
      vi.mocked(courseService.getCourseStructure).mockRejectedValue(new Error('Network error'));

      const { result } = renderHook(() => useTaskNavigation('course-1', 'task-1'));

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.modules).toEqual([]);
    });

    it('should handle null response', async () => {
      vi.mocked(courseService.getCourseStructure).mockResolvedValue(null as any);

      const { result } = renderHook(() => useTaskNavigation('course-1', 'task-1'));

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.modules).toEqual([]);
    });
  });

  describe('flatTasks', () => {
    it('should flatten tasks from all modules and topics', async () => {
      vi.mocked(courseService.getCourseStructure).mockResolvedValue(mockModules);

      const { result } = renderHook(() => useTaskNavigation('course-1', 'task-1'));

      await waitFor(() => {
        expect(result.current.flatTasks).toHaveLength(3);
      });

      expect(result.current.flatTasks[0]).toEqual({
        slug: 'task-1',
        title: 'Task 1',
        difficulty: 'easy',
        moduleId: 'mod-1',
        moduleTitle: 'Module 1',
        topicId: 'topic-1',
        topicTitle: 'Topic 1',
        moduleIndex: 0,
      });

      expect(result.current.flatTasks[2].moduleIndex).toBe(1);
    });

    it('should handle modules with no topics', async () => {
      const modulesWithNoTopics = [{ id: 'mod-1', title: 'Empty', topics: [] }];
      vi.mocked(courseService.getCourseStructure).mockResolvedValue(modulesWithNoTopics);

      const { result } = renderHook(() => useTaskNavigation('course-1', 'task-1'));

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.flatTasks).toHaveLength(0);
    });

    it('should filter out tasks without slug', async () => {
      const modulesWithInvalidTasks = [
        {
          id: 'mod-1',
          title: 'Module 1',
          topics: [
            {
              id: 'topic-1',
              title: 'Topic 1',
              tasks: [
                { slug: 'valid-task', title: 'Valid', difficulty: 'easy' },
                { slug: '', title: 'No Slug', difficulty: 'easy' },
                { slug: null, title: 'Null Slug', difficulty: 'easy' },
              ],
            },
          ],
        },
      ];
      vi.mocked(courseService.getCourseStructure).mockResolvedValue(modulesWithInvalidTasks as any);

      const { result } = renderHook(() => useTaskNavigation('course-1', 'valid-task'));

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.flatTasks).toHaveLength(1);
      expect(result.current.flatTasks[0].slug).toBe('valid-task');
    });
  });

  describe('currentIndex', () => {
    it('should find current task index', async () => {
      vi.mocked(courseService.getCourseStructure).mockResolvedValue(mockModules);

      const { result } = renderHook(() => useTaskNavigation('course-1', 'task-2'));

      await waitFor(() => {
        expect(result.current.currentIndex).toBe(1);
      });
    });

    it('should return -1 for non-existent task', async () => {
      vi.mocked(courseService.getCourseStructure).mockResolvedValue(mockModules);

      const { result } = renderHook(() => useTaskNavigation('course-1', 'non-existent'));

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.currentIndex).toBe(-1);
    });

    it('should return -1 when no currentTaskSlug', async () => {
      vi.mocked(courseService.getCourseStructure).mockResolvedValue(mockModules);

      const { result } = renderHook(() => useTaskNavigation('course-1', undefined));

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.currentIndex).toBe(-1);
    });
  });

  describe('prevTask and nextTask', () => {
    it('should return null for prevTask when on first task', async () => {
      vi.mocked(courseService.getCourseStructure).mockResolvedValue(mockModules);

      const { result } = renderHook(() => useTaskNavigation('course-1', 'task-1'));

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.prevTask).toBeNull();
      expect(result.current.nextTask?.slug).toBe('task-2');
    });

    it('should return null for nextTask when on last task', async () => {
      vi.mocked(courseService.getCourseStructure).mockResolvedValue(mockModules);

      const { result } = renderHook(() => useTaskNavigation('course-1', 'task-3'));

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.prevTask?.slug).toBe('task-2');
      expect(result.current.nextTask).toBeNull();
    });

    it('should return both prevTask and nextTask for middle task', async () => {
      vi.mocked(courseService.getCourseStructure).mockResolvedValue(mockModules);

      const { result } = renderHook(() => useTaskNavigation('course-1', 'task-2'));

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.prevTask?.slug).toBe('task-1');
      expect(result.current.nextTask?.slug).toBe('task-3');
    });

    it('should return null for both when task not found', async () => {
      vi.mocked(courseService.getCourseStructure).mockResolvedValue(mockModules);

      const { result } = renderHook(() => useTaskNavigation('course-1', 'non-existent'));

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.prevTask).toBeNull();
      expect(result.current.nextTask).toBeNull();
    });
  });

  describe('goToPrev and goToNext', () => {
    it('should return correct URL for goToPrev', async () => {
      vi.mocked(courseService.getCourseStructure).mockResolvedValue(mockModules);

      const { result } = renderHook(() => useTaskNavigation('course-1', 'task-2'));

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.goToPrev()).toBe('/course/course-1/task/task-1');
    });

    it('should return correct URL for goToNext', async () => {
      vi.mocked(courseService.getCourseStructure).mockResolvedValue(mockModules);

      const { result } = renderHook(() => useTaskNavigation('course-1', 'task-2'));

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.goToNext()).toBe('/course/course-1/task/task-3');
    });

    it('should return null for goToPrev when no prevTask', async () => {
      vi.mocked(courseService.getCourseStructure).mockResolvedValue(mockModules);

      const { result } = renderHook(() => useTaskNavigation('course-1', 'task-1'));

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.goToPrev()).toBeNull();
    });

    it('should return null for goToNext when no nextTask', async () => {
      vi.mocked(courseService.getCourseStructure).mockResolvedValue(mockModules);

      const { result } = renderHook(() => useTaskNavigation('course-1', 'task-3'));

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.goToNext()).toBeNull();
    });

    it('should return null when no courseId', async () => {
      vi.mocked(courseService.getCourseStructure).mockResolvedValue(mockModules);

      const { result } = renderHook(() => useTaskNavigation(undefined, 'task-2'));

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.goToPrev()).toBeNull();
      expect(result.current.goToNext()).toBeNull();
    });
  });

  describe('MODULE_COLORS', () => {
    it('should have correct number of color schemes', () => {
      expect(MODULE_COLORS).toHaveLength(8);
    });

    it('should have all required properties in each color', () => {
      MODULE_COLORS.forEach((color) => {
        expect(color).toHaveProperty('bg');
        expect(color).toHaveProperty('border');
        expect(color).toHaveProperty('text');
        expect(color).toHaveProperty('dot');
      });
    });
  });
});
