import { Injectable, NotFoundException } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';

@Injectable()
export class TasksService {
  constructor(private prisma: PrismaService) {}

  /**
   * Fetch all tasks and enrich with completion status for the current user.
   */
  async findAll(userId?: string) {
    const tasks = await this.prisma.task.findMany({
      select: {
        id: true,
        slug: true,
        title: true,
        difficulty: true,
        tags: true,
        isPremium: true,
        estimatedTime: true,
        description: true, // Needed for search/preview sometimes
        // Exclude large fields if list is huge, but fine for MVP
      }
    });

    if (!userId) {
        return tasks.map(t => ({ ...t, status: 'pending' }));
    }

    // Fetch all passed submissions for the user in a single query
    const passedSubmissions = await this.prisma.submission.findMany({
        where: {
            userId: userId,
            status: 'passed'
        },
        select: {
            taskId: true
        }
    });

    // Create a Set for O(1) lookup performance
    const passedTaskIds = new Set(passedSubmissions.map(s => s.taskId));

    // Map tasks with completion status using the Set
    const enrichedTasks = tasks.map(task => ({
        ...task,
        status: passedTaskIds.has(task.id) ? 'completed' : 'pending'
    }));

    return enrichedTasks;
  }

  /**
   * Fetch single task with status
   */
  async findOne(slug: string, userId?: string) {
    const task = await this.prisma.task.findUnique({
      where: { slug },
    });

    if (!task) {
      throw new NotFoundException(`Task with slug ${slug} not found`);
    }

    let status = 'pending';

    if (userId) {
        const passedSubmission = await this.prisma.submission.findFirst({
            where: {
                userId: userId,
                taskId: task.id,
                status: 'passed'
            }
        });
        if (passedSubmission) status = 'completed';
    }

    return { ...task, status };
  }
}