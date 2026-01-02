import { Injectable } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { TaskAccessDto, CourseAccessDto } from './dto/subscription.dto';

// Queue priority constants
const PRIORITY_HIGH = 1; // Premium/subscribed users
const PRIORITY_LOW = 10; // Free users

// Grace period: 3 days after subscription expires, user still has access
// This gives users time to renew and prevents abrupt access loss
const GRACE_PERIOD_DAYS = 3;

@Injectable()
export class AccessControlService {
  constructor(private prisma: PrismaService) {}

  /**
   * Get the grace period cutoff date (now - grace period)
   * Subscriptions expired within this window still grant access
   */
  private getGracePeriodCutoff(): Date {
    const cutoff = new Date();
    cutoff.setDate(cutoff.getDate() - GRACE_PERIOD_DAYS);
    return cutoff;
  }

  /**
   * Check if user has an active global subscription
   * Includes grace period - recently expired subscriptions still work
   */
  async hasGlobalAccess(userId: string): Promise<boolean> {
    const gracePeriodCutoff = this.getGracePeriodCutoff();

    const subscription = await this.prisma.subscription.findFirst({
      where: {
        userId,
        status: 'active',
        endDate: { gte: gracePeriodCutoff }, // Include grace period
        plan: { type: 'global' },
      },
    });
    return !!subscription;
  }

  /**
   * Check if user has access to a specific course (via course subscription or global)
   * Includes grace period - recently expired subscriptions still work
   */
  async hasCourseAccess(userId: string, courseId: string): Promise<boolean> {
    // Check global subscription first
    if (await this.hasGlobalAccess(userId)) {
      return true;
    }

    const gracePeriodCutoff = this.getGracePeriodCutoff();

    // Check course-specific subscription
    const subscription = await this.prisma.subscription.findFirst({
      where: {
        userId,
        status: 'active',
        endDate: { gte: gracePeriodCutoff }, // Include grace period
        plan: {
          type: 'course',
          courseId,
        },
      },
    });

    return !!subscription;
  }

  /**
   * Get queue priority for a user submitting to a specific course
   * Lower number = higher priority
   */
  async getQueuePriority(userId: string, courseId: string): Promise<number> {
    const hasAccess = await this.hasCourseAccess(userId, courseId);
    return hasAccess ? PRIORITY_HIGH : PRIORITY_LOW;
  }

  /**
   * Check if user can see the solution for a specific task
   * Free users can only see solutions for the first task in each topic
   */
  async canSeeSolution(userId: string, taskId: string): Promise<boolean> {
    // Get task with its topic and course info
    const task = await this.prisma.task.findUnique({
      where: { id: taskId },
      include: {
        topic: {
          include: {
            module: { include: { course: true } },
          },
        },
      },
    });

    if (!task?.topic) {
      return false;
    }

    const courseId = task.topic.module.courseId;

    // Premium/subscribed users can see all solutions
    if (await this.hasCourseAccess(userId, courseId)) {
      return true;
    }

    // Free users: only first task in topic
    const firstTaskInTopic = await this.prisma.task.findFirst({
      where: { topicId: task.topicId },
      orderBy: { order: 'asc' },
    });

    return task.id === firstTaskInTopic?.id;
  }

  /**
   * Check if user can use AI Tutor for a specific task
   * Only available for premium/subscribed users
   */
  async canUseAiTutor(userId: string, taskId: string): Promise<boolean> {
    const task = await this.prisma.task.findUnique({
      where: { id: taskId },
      include: {
        topic: {
          include: {
            module: { include: { course: true } },
          },
        },
      },
    });

    if (!task?.topic) {
      return false;
    }

    const courseId = task.topic.module.courseId;
    return this.hasCourseAccess(userId, courseId);
  }

  /**
   * Get comprehensive access info for a task
   */
  async getTaskAccess(userId: string, taskId: string): Promise<TaskAccessDto> {
    const task = await this.prisma.task.findUnique({
      where: { id: taskId },
      include: {
        topic: {
          include: {
            module: { include: { course: true } },
          },
        },
      },
    });

    if (!task?.topic) {
      return {
        canView: false,
        canRun: false,
        canSubmit: false,
        canSeeSolution: false,
        canUseAiTutor: false,
        queuePriority: PRIORITY_LOW,
      };
    }

    const courseId = task.topic.module.courseId;
    const hasAccess = await this.hasCourseAccess(userId, courseId);
    const canSeeSolution = await this.canSeeSolution(userId, taskId);

    return {
      canView: true, // Everyone can view tasks
      canRun: true, // Everyone can run code
      canSubmit: true, // Everyone can submit
      canSeeSolution,
      canUseAiTutor: hasAccess, // Only premium
      queuePriority: hasAccess ? PRIORITY_HIGH : PRIORITY_LOW,
    };
  }

  /**
   * Get access info for a course
   */
  async getCourseAccess(userId: string, courseId: string): Promise<CourseAccessDto> {
    const hasAccess = await this.hasCourseAccess(userId, courseId);

    return {
      hasAccess,
      queuePriority: hasAccess ? PRIORITY_HIGH : PRIORITY_LOW,
      canUseAiTutor: hasAccess,
    };
  }

  /**
   * Get user's active subscriptions
   * Includes subscriptions within grace period
   */
  async getUserSubscriptions(userId: string) {
    const gracePeriodCutoff = this.getGracePeriodCutoff();

    return this.prisma.subscription.findMany({
      where: {
        userId,
        status: 'active',
        endDate: { gte: gracePeriodCutoff }, // Include grace period
      },
      include: {
        plan: true,
      },
    });
  }
}
