import { describe, it, expect, beforeEach, vi } from 'vitest';
import { userCoursesService } from './userCoursesService';

vi.mock('@/lib/api', () => ({
  api: {
    get: vi.fn(),
    post: vi.fn(),
    patch: vi.fn(),
  },
}));

import { api } from '@/lib/api';

describe('userCoursesService', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('getStartedCourses', () => {
    it('should fetch started courses', async () => {
      const mockCourses = [
        {
          id: 'course-1',
          slug: 'go-basics',
          title: 'Go Basics',
          progress: 50,
          startedAt: '2025-01-01T00:00:00Z',
          lastAccessedAt: '2025-01-15T00:00:00Z',
          completedAt: null,
        },
        {
          id: 'course-2',
          slug: 'python-basics',
          title: 'Python Basics',
          progress: 100,
          startedAt: '2024-12-01T00:00:00Z',
          lastAccessedAt: '2025-01-10T00:00:00Z',
          completedAt: '2025-01-10T00:00:00Z',
        },
      ];

      vi.mocked(api.get).mockResolvedValueOnce(mockCourses);

      const result = await userCoursesService.getStartedCourses();

      expect(api.get).toHaveBeenCalledWith('/users/me/courses');
      expect(result).toHaveLength(2);
      expect(result[0].progress).toBe(50);
    });

    it('should return empty array for new user', async () => {
      vi.mocked(api.get).mockResolvedValueOnce([]);

      const result = await userCoursesService.getStartedCourses();

      expect(result).toHaveLength(0);
    });

    it('should throw on API error', async () => {
      vi.mocked(api.get).mockRejectedValueOnce(new Error('Unauthorized'));

      await expect(userCoursesService.getStartedCourses()).rejects.toThrow('Unauthorized');
    });
  });

  describe('startCourse', () => {
    it('should start a new course', async () => {
      const mockCourse = {
        id: 'course-1',
        slug: 'go-basics',
        title: 'Go Basics',
        progress: 0,
        startedAt: '2025-01-16T00:00:00Z',
        lastAccessedAt: '2025-01-16T00:00:00Z',
        completedAt: null,
      };

      vi.mocked(api.post).mockResolvedValueOnce(mockCourse);

      const result = await userCoursesService.startCourse('go-basics');

      expect(api.post).toHaveBeenCalledWith('/users/me/courses/go-basics/start', {});
      expect(result.progress).toBe(0);
      expect(result.completedAt).toBeNull();
    });

    it('should throw on course not found', async () => {
      vi.mocked(api.post).mockRejectedValueOnce(new Error('Course not found'));

      await expect(userCoursesService.startCourse('invalid')).rejects.toThrow('Course not found');
    });
  });

  describe('updateProgress', () => {
    it('should update course progress', async () => {
      const mockResponse = {
        courseSlug: 'go-basics',
        progress: 75,
      };

      vi.mocked(api.patch).mockResolvedValueOnce(mockResponse);

      const result = await userCoursesService.updateProgress('go-basics', 75);

      expect(api.patch).toHaveBeenCalledWith('/users/me/courses/go-basics/progress', { progress: 75 });
      expect(result.progress).toBe(75);
    });

    it('should update to 100% for completion', async () => {
      vi.mocked(api.patch).mockResolvedValueOnce({ courseSlug: 'go-basics', progress: 100 });

      const result = await userCoursesService.updateProgress('go-basics', 100);

      expect(result.progress).toBe(100);
    });

    it('should throw on invalid progress', async () => {
      vi.mocked(api.patch).mockRejectedValueOnce(new Error('Invalid progress value'));

      await expect(userCoursesService.updateProgress('go-basics', 150)).rejects.toThrow('Invalid progress value');
    });
  });

  describe('hasStartedCourse', () => {
    it('should return true if course is started', async () => {
      vi.mocked(api.get).mockResolvedValueOnce([
        { slug: 'go-basics', progress: 50 },
        { slug: 'python-basics', progress: 25 },
      ]);

      const result = await userCoursesService.hasStartedCourse('go-basics');

      expect(result).toBe(true);
    });

    it('should return false if course is not started', async () => {
      vi.mocked(api.get).mockResolvedValueOnce([
        { slug: 'python-basics', progress: 25 },
      ]);

      const result = await userCoursesService.hasStartedCourse('go-basics');

      expect(result).toBe(false);
    });

    it('should return false for empty course list', async () => {
      vi.mocked(api.get).mockResolvedValueOnce([]);

      const result = await userCoursesService.hasStartedCourse('go-basics');

      expect(result).toBe(false);
    });
  });

  describe('updateLastAccessed', () => {
    it('should update last accessed time', async () => {
      vi.mocked(api.patch).mockResolvedValueOnce(undefined);

      await userCoursesService.updateLastAccessed('go-basics');

      expect(api.patch).toHaveBeenCalledWith('/users/me/courses/go-basics/access', {});
    });

    it('should throw on API error', async () => {
      vi.mocked(api.patch).mockRejectedValueOnce(new Error('Course not started'));

      await expect(userCoursesService.updateLastAccessed('invalid')).rejects.toThrow('Course not started');
    });
  });
});
