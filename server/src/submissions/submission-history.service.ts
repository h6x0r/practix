import { Injectable, NotFoundException } from "@nestjs/common";
import { PrismaService } from "../prisma/prisma.service";

/**
 * Handles submission history queries and run result persistence.
 * Pure read operations + run result upserts.
 */
@Injectable()
export class SubmissionHistoryService {
  constructor(private readonly prisma: PrismaService) {}

  async findOne(id: string): Promise<any> {
    const submission = await this.prisma.submission.findUnique({
      where: { id },
      include: { task: true },
    });

    if (!submission) {
      throw new NotFoundException(`Submission not found: ${id}`);
    }

    return submission;
  }

  async findByUserAndTask(userId: string, taskId: string): Promise<any[]> {
    return this.prisma.submission.findMany({
      where: { userId, taskId },
      orderBy: { createdAt: "desc" },
      take: 10,
      select: {
        id: true,
        status: true,
        score: true,
        runtime: true,
        memory: true,
        message: true,
        testsPassed: true,
        testsTotal: true,
        testCases: true,
        createdAt: true,
        code: true,
      },
    });
  }

  async getRunResult(
    userId: string,
    taskIdentifier: string,
  ): Promise<{
    status: string;
    testsPassed: number;
    testsTotal: number;
    testCases: any;
    runtime: string;
    message: string | null;
    code: string;
    updatedAt: Date;
  } | null> {
    const task = await this.prisma.task.findFirst({
      where: {
        OR: [{ id: taskIdentifier }, { slug: taskIdentifier }],
      },
      select: { id: true },
    });

    if (!task) return null;

    return this.prisma.runResult.findUnique({
      where: {
        userId_taskId: { userId, taskId: task.id },
      },
      select: {
        status: true,
        testsPassed: true,
        testsTotal: true,
        testCases: true,
        runtime: true,
        message: true,
        code: true,
        updatedAt: true,
      },
    });
  }

  async findRecentByUser(userId: string, limit = 10): Promise<any[]> {
    return this.prisma.submission.findMany({
      where: { userId },
      orderBy: { createdAt: "desc" },
      take: limit,
      include: { task: { select: { slug: true, title: true } } },
    });
  }
}
