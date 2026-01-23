import { Course } from '@/types';
import { api } from '@/lib/api';

/**
 * User Course - course with user progress data
 */
export interface UserCourse extends Course {
  progress: number;
  startedAt: string;
  lastAccessedAt: string;
  completedAt: string | null;
}

/**
 * User Courses Service - Manages user's enrolled courses
 *
 * Endpoints:
 * - GET /users/me/courses - Get all started courses
 * - POST /users/me/courses/:courseSlug/start - Start a course
 * - PATCH /users/me/courses/:courseSlug/progress - Update progress
 */
export const userCoursesService = {

  /**
   * Get all courses the user has started
   * Ordered by lastAccessedAt (most recent first)
   */
  getStartedCourses: async (): Promise<UserCourse[]> => {
    return api.get<UserCourse[]>('/users/me/courses');
  },

  /**
   * Start a new course
   * Creates a UserCourse record with 0% progress
   */
  startCourse: async (courseSlug: string): Promise<UserCourse> => {
    return api.post<UserCourse>(`/users/me/courses/${courseSlug}/start`, {});
  },

  /**
   * Update course progress
   * Progress is a percentage (0-100)
   */
  updateProgress: async (courseSlug: string, progress: number): Promise<{ courseSlug: string; progress: number }> => {
    return api.patch<{ courseSlug: string; progress: number }>(`/users/me/courses/${courseSlug}/progress`, { progress });
  },

  /**
   * Check if user has started a specific course
   */
  hasStartedCourse: async (courseSlug: string): Promise<boolean> => {
    const courses = await userCoursesService.getStartedCourses();
    return courses.some(c => c.slug === courseSlug);
  },

  /**
   * Update last accessed time for a course
   * Moves the course to the top of My Tasks list
   */
  updateLastAccessed: async (courseSlug: string): Promise<void> => {
    await api.patch(`/users/me/courses/${courseSlug}/access`, {});
  }
};
