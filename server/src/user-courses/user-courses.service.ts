import { Injectable, NotFoundException, Logger } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';

@Injectable()
export class UserCoursesService {
  private readonly logger = new Logger(UserCoursesService.name);

  constructor(private prisma: PrismaService) {}

  /**
   * Get all courses started by a user
   * Returns courses with their progress and last accessed time
   */
  async getUserCourses(userId: string) {
    try {
      // Get user courses first
      const userCourses = await this.prisma.userCourse.findMany({
        where: { userId },
        orderBy: { lastAccessedAt: 'desc' },
      });

      if (userCourses.length === 0) {
        this.logger.log(`No courses found for user ${userId}`);
        return [];
      }

      // Optimized: Single query for all courses using WHERE IN instead of N+1
      const courseSlugs = userCourses.map((uc) => uc.courseSlug);
      const courses = await this.prisma.course.findMany({
        where: { slug: { in: courseSlugs } },
        select: {
          id: true,
          slug: true,
          title: true,
          description: true,
          category: true,
          icon: true,
          estimatedTime: true,
          translations: true,
        },
      });

      // Create a map for O(1) lookup
      const courseMap = new Map(courses.map((c) => [c.slug, c]));

      this.logger.log(`Retrieved ${userCourses.length} courses for user ${userId}`);

      // Map to flattened structure
      const coursesWithDetails = userCourses
        .filter((uc) => courseMap.has(uc.courseSlug)) // Filter out orphaned records
        .map((userCourse) => {
          const course = courseMap.get(userCourse.courseSlug)!;
          return {
            id: course.id,
            slug: course.slug,
            title: course.title,
            description: course.description,
            category: course.category,
            icon: course.icon,
            estimatedTime: course.estimatedTime,
            translations: course.translations,
            progress: userCourse.progress,
            startedAt: userCourse.startedAt,
            lastAccessedAt: userCourse.lastAccessedAt,
            completedAt: userCourse.completedAt,
          };
        });

      return coursesWithDetails;
    } catch (error) {
      this.logger.error(
        'Error in getUserCourses',
        error instanceof Error ? error.stack : String(error)
      );
      throw error;
    }
  }

  /**
   * Start a course for a user
   * Creates a new UserCourse record or updates existing one
   */
  async startCourse(userId: string, courseSlug: string) {
    try {
      // Verify course exists
      const course = await this.prisma.course.findUnique({
        where: { slug: courseSlug },
      });

      if (!course) {
        throw new NotFoundException(`Course not found: ${courseSlug}`);
      }

      // Create or update user course record
      const userCourse = await this.prisma.userCourse.upsert({
        where: {
          userId_courseSlug: {
            userId,
            courseSlug,
          },
        },
        update: {
          lastAccessedAt: new Date(),
        },
        create: {
          userId,
          courseSlug,
          progress: 0,
          startedAt: new Date(),
          lastAccessedAt: new Date(),
        },
      });

      this.logger.log(`User ${userId} started course ${courseSlug}`);

      return {
        id: course.id,
        slug: course.slug,
        title: course.title,
        description: course.description,
        category: course.category,
        icon: course.icon,
        estimatedTime: course.estimatedTime,
        translations: course.translations,
        progress: userCourse.progress,
        startedAt: userCourse.startedAt,
        lastAccessedAt: userCourse.lastAccessedAt,
        completedAt: userCourse.completedAt,
      };
    } catch (error) {
      this.logger.error(
        'Error in startCourse',
        error instanceof Error ? error.stack : String(error)
      );
      throw error;
    }
  }

  /**
   * Update course progress for a user
   * Progress is a percentage (0-100)
   */
  async updateProgress(userId: string, courseSlug: string, progress: number) {
    try {
      // Validate progress value
      if (progress < 0 || progress > 100) {
        throw new Error('Progress must be between 0 and 100');
      }

      // Check if user course exists
      const existingUserCourse = await this.prisma.userCourse.findUnique({
        where: {
          userId_courseSlug: {
            userId,
            courseSlug,
          },
        },
      });

      if (!existingUserCourse) {
        throw new NotFoundException(
          `User has not started course: ${courseSlug}`
        );
      }

      // Update progress and mark as completed if 100%
      const userCourse = await this.prisma.userCourse.update({
        where: {
          userId_courseSlug: {
            userId,
            courseSlug,
          },
        },
        data: {
          progress,
          lastAccessedAt: new Date(),
          completedAt: progress === 100 ? new Date() : null,
        },
      });

      this.logger.log(
        `Updated progress for user ${userId} in course ${courseSlug}: ${progress}%`
      );

      return {
        courseSlug: userCourse.courseSlug,
        progress: userCourse.progress,
        lastAccessedAt: userCourse.lastAccessedAt,
        completedAt: userCourse.completedAt,
      };
    } catch (error) {
      this.logger.error(
        'Error in updateProgress',
        error instanceof Error ? error.stack : String(error)
      );
      throw error;
    }
  }

  /**
   * Update last accessed time for a course
   * Called when user views course details or works on course tasks
   */
  async updateLastAccessed(userId: string, courseSlug: string) {
    try {
      // Check if user course exists, if not create it
      const userCourse = await this.prisma.userCourse.upsert({
        where: {
          userId_courseSlug: {
            userId,
            courseSlug,
          },
        },
        update: {
          lastAccessedAt: new Date(),
        },
        create: {
          userId,
          courseSlug,
          progress: 0,
          startedAt: new Date(),
          lastAccessedAt: new Date(),
        },
      });

      this.logger.log(
        `Updated last accessed for user ${userId} in course ${courseSlug}`
      );

      return {
        courseSlug: userCourse.courseSlug,
        lastAccessedAt: userCourse.lastAccessedAt,
      };
    } catch (error) {
      this.logger.error(
        'Error in updateLastAccessed',
        error instanceof Error ? error.stack : String(error)
      );
      throw error;
    }
  }
}
