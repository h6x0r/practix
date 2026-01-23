import { Injectable, NotFoundException, Logger } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';

// Grace period: 3 days after subscription expires, user still has access
const GRACE_PERIOD_DAYS = 3;

@Injectable()
export class UserCoursesService {
  private readonly logger = new Logger(UserCoursesService.name);

  constructor(private prisma: PrismaService) {}

  /**
   * Get the grace period cutoff date
   */
  private getGracePeriodCutoff(): Date {
    const cutoff = new Date();
    cutoff.setDate(cutoff.getDate() - GRACE_PERIOD_DAYS);
    return cutoff;
  }

  /**
   * Check if user has global premium access
   */
  private async hasGlobalAccess(userId: string): Promise<boolean> {
    const gracePeriodCutoff = this.getGracePeriodCutoff();
    const subscription = await this.prisma.subscription.findFirst({
      where: {
        userId,
        status: 'active',
        endDate: { gte: gracePeriodCutoff },
        plan: { type: 'global' },
      },
    });
    return !!subscription;
  }

  /**
   * Get course IDs that user has active subscriptions for
   */
  private async getAccessibleCourseIds(userId: string): Promise<Set<string>> {
    const gracePeriodCutoff = this.getGracePeriodCutoff();

    // Get all course-specific subscriptions
    const subscriptions = await this.prisma.subscription.findMany({
      where: {
        userId,
        status: 'active',
        endDate: { gte: gracePeriodCutoff },
        plan: { type: 'course' },
      },
      include: {
        plan: { select: { courseId: true } },
      },
    });

    const courseIds = new Set<string>();
    subscriptions.forEach((sub) => {
      if (sub.plan.courseId) {
        courseIds.add(sub.plan.courseId);
      }
    });

    return courseIds;
  }

  /**
   * Get all courses started by a user with active subscriptions
   * Only returns courses where user has global premium OR course-specific subscription
   * Progress is preserved in UserCourse even when subscription expires
   */
  async getUserCourses(userId: string) {
    try {
      // Check if user has global premium access
      const hasGlobal = await this.hasGlobalAccess(userId);

      // Get course IDs with active course-specific subscriptions
      const accessibleCourseIds = hasGlobal
        ? null // Global access means all courses
        : await this.getAccessibleCourseIds(userId);

      // Get user courses first
      const userCourses = await this.prisma.userCourse.findMany({
        where: { userId },
        orderBy: { lastAccessedAt: 'desc' },
      });

      if (userCourses.length === 0) {
        this.logger.log(`No courses found for user ${userId}`);
        return [];
      }

      // Get all courses with task counts
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
          modules: {
            select: {
              topics: {
                select: {
                  _count: { select: { tasks: true } },
                },
              },
            },
          },
        },
      });

      // Calculate total tasks per course
      const courseMap = new Map(
        courses.map((c) => {
          let totalTasks = 0;
          c.modules.forEach((m) =>
            m.topics.forEach((t) => (totalTasks += t._count.tasks))
          );
          return [c.slug, { ...c, totalTasks }];
        })
      );

      // Get user's passed submissions for these courses
      const passedSubmissions = await this.prisma.submission.findMany({
        where: {
          userId,
          status: 'passed',
          task: {
            topic: {
              module: {
                course: { slug: { in: courseSlugs } },
              },
            },
          },
        },
        select: {
          taskId: true,
          task: {
            select: {
              topic: {
                select: {
                  module: {
                    select: {
                      course: { select: { slug: true } },
                    },
                  },
                },
              },
            },
          },
        },
        distinct: ['taskId'],
      });

      // Group completed tasks by course
      const completedByCourse = new Map<string, Set<string>>();
      passedSubmissions.forEach((sub) => {
        const courseSlug = sub.task.topic.module.course.slug;
        if (!completedByCourse.has(courseSlug)) {
          completedByCourse.set(courseSlug, new Set());
        }
        completedByCourse.get(courseSlug)!.add(sub.taskId);
      });

      this.logger.log(`Retrieved ${userCourses.length} courses for user ${userId}`);

      // Map to flattened structure with calculated progress
      // Filter by: 1) course exists, 2) user has access (global or course-specific subscription)
      const coursesWithDetails = userCourses
        .filter((uc) => {
          if (!courseMap.has(uc.courseSlug)) return false;

          // Global access allows all courses
          if (hasGlobal) return true;

          // Check course-specific subscription
          const course = courseMap.get(uc.courseSlug)!;
          return accessibleCourseIds!.has(course.id);
        })
        .map((userCourse) => {
          const course = courseMap.get(userCourse.courseSlug)!;
          const completedTasks = completedByCourse.get(course.slug)?.size || 0;
          const calculatedProgress =
            course.totalTasks === 0
              ? 0
              : Math.round((completedTasks / course.totalTasks) * 100);

          return {
            id: course.id,
            slug: course.slug,
            title: course.title,
            description: course.description,
            category: course.category,
            icon: course.icon,
            estimatedTime: course.estimatedTime,
            translations: course.translations,
            progress: calculatedProgress, // Calculated from submissions, not stored value
            startedAt: userCourse.startedAt,
            lastAccessedAt: userCourse.lastAccessedAt,
            completedAt: calculatedProgress === 100 ? userCourse.completedAt : null,
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
