import { Injectable, NotFoundException } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { JudgeService } from '../judge/judge.service';
import { TasksService } from '../tasks/tasks.service';

@Injectable()
export class SubmissionsService {
  constructor(
    private prisma: PrismaService,
    private judgeService: JudgeService,
    private tasksService: TasksService
  ) {}

  async create(userId: string, taskIdentifier: string, code: string, language: string) {
    // 1. Resolve Task ID
    // The frontend might send a slug (e.g. 'two-sum') or a UUID.
    // We need the UUID for the Foreign Key relation.
    let task = await (this.prisma as any).task.findUnique({ where: { id: taskIdentifier } });
    
    if (!task) {
        // Try finding by slug
        task = await (this.prisma as any).task.findUnique({ where: { slug: taskIdentifier } });
    }

    if (!task) {
        throw new NotFoundException(`Task not found: ${taskIdentifier}`);
    }

    // 2. Execute Code (Remote Judge)
    // In a real scenario, we might want to validate the language against allowed languages for this task.
    const result = await this.judgeService.executeCode(language, code);

    // 3. Format Output Message
    let message = `> Status: ${result.description}\n> Runtime: ${result.time || 0}s\n\n`;
    if (result.stdout) message += `[STDOUT]:\n${result.stdout}\n`;
    if (result.stderr) message += `[STDERR]:\n${result.stderr}\n`;
    if (result.compile_output) message += `[COMPILER]:\n${result.compile_output}\n`;

    // 4. Save Submission to DB
    const submission = await (this.prisma as any).submission.create({
      data: {
        userId,
        taskId: task.id, // Use the resolved UUID
        code,
        status: result.status,
        score: result.status === 'passed' ? 100 : 0,
        runtime: `${Math.floor((parseFloat(result.time) || 0) * 1000)}ms`,
        message
      }
    });

    return submission;
  }
}