import { describe, it, expect, beforeEach, vi } from 'vitest';
import { courseService } from './courseService';

// Mock the api module
vi.mock('@/lib/api', () => ({
  api: {
    get: vi.fn(),
  },
}));

import { api } from '@/lib/api';

describe('courseService', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('getAllCourses', () => {
    it('should fetch all courses', async () => {
      const mockCourses = [
        { id: 'go-basics', slug: 'go-basics', title: 'Go Basics' },
        { id: 'java-core', slug: 'java-core', title: 'Java Core' },
      ];

      vi.mocked(api.get).mockResolvedValueOnce(mockCourses);

      const result = await courseService.getAllCourses();

      expect(api.get).toHaveBeenCalledWith('/courses');
      expect(result).toEqual(mockCourses);
    });

    it('should return empty array when no courses', async () => {
      vi.mocked(api.get).mockResolvedValueOnce([]);

      const result = await courseService.getAllCourses();

      expect(result).toEqual([]);
    });
  });

  describe('getCourseById', () => {
    it('should fetch course by ID', async () => {
      const mockCourse = {
        id: 'go-basics',
        slug: 'go-basics',
        title: 'Go Basics',
        description: 'Learn Go programming',
      };

      vi.mocked(api.get).mockResolvedValueOnce(mockCourse);

      const result = await courseService.getCourseById('go-basics');

      expect(api.get).toHaveBeenCalledWith('/courses/go-basics');
      expect(result).toEqual(mockCourse);
    });

    it('should handle course not found', async () => {
      vi.mocked(api.get).mockRejectedValueOnce(new Error('Not found'));

      await expect(courseService.getCourseById('non-existent')).rejects.toThrow('Not found');
    });
  });

  describe('getCourseStructure', () => {
    it('should fetch course structure', async () => {
      const mockStructure = [
        {
          id: 'module-1',
          title: 'Getting Started',
          topics: [
            {
              id: 'topic-1',
              title: 'Hello World',
              tasks: [{ id: 'task-1', slug: 'hello-world', title: 'Hello World' }],
            },
          ],
        },
      ];

      vi.mocked(api.get).mockResolvedValueOnce(mockStructure);

      const result = await courseService.getCourseStructure('go-basics');

      expect(api.get).toHaveBeenCalledWith('/courses/go-basics/structure');
      expect(result).toEqual(mockStructure);
    });

    it('should return empty array for course with no modules', async () => {
      vi.mocked(api.get).mockResolvedValueOnce([]);

      const result = await courseService.getCourseStructure('empty-course');

      expect(result).toEqual([]);
    });
  });
});
