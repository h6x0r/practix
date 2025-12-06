import { Injectable, NotFoundException } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';

@Injectable()
export class TasksService {
  constructor(private prisma: PrismaService) {}

  /**
   * Fetch all tasks and enrich with completion status for the current user.
   */
  async findAll(userId?: string) {
    const tasks = await (this.prisma as any).task.findMany({
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

    // Determine status for each task based on submissions
    const enrichedTasks = await Promise.all(tasks.map(async (task) => {
        const passedSubmission = await (this.prisma as any).submission.findFirst({
            where: {
                userId: userId,
                taskId: task.id,
                status: 'passed'
            }
        });

        return {
            ...task,
            status: passedSubmission ? 'completed' : 'pending'
        };
    }));

    return enrichedTasks;
  }

  /**
   * Fetch single task with status
   */
  async findOne(slug: string, userId?: string) {
    const task = await (this.prisma as any).task.findUnique({
      where: { slug },
    });

    if (!task) {
      throw new NotFoundException(`Task with slug ${slug} not found`);
    }

    let status = 'pending';

    if (userId) {
        const passedSubmission = await (this.prisma as any).submission.findFirst({
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