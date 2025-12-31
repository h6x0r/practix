import { Injectable, Logger, BadRequestException } from '@nestjs/common';
import { Prisma } from '@prisma/client';
import { PrismaService } from '../prisma/prisma.service';
import { CreateBugReportDto, BugStatus } from './dto/bugreports.dto';

@Injectable()
export class BugReportsService {
  private readonly logger = new Logger(BugReportsService.name);

  constructor(private prisma: PrismaService) {}

  /**
   * Create a new bug report
   */
  async create(userId: string, dto: CreateBugReportDto) {
    this.logger.log(`User ${userId} submitting bug report: ${dto.category} - ${dto.title}`);

    const report = await this.prisma.bugReport.create({
      data: {
        userId,
        taskId: dto.taskId || null,
        category: dto.category,
        severity: dto.severity || 'medium',
        title: dto.title,
        description: dto.description,
        metadata: (dto.metadata as Prisma.InputJsonValue) || Prisma.JsonNull,
      },
      include: {
        user: { select: { name: true, email: true } },
        task: { select: { title: true, slug: true } },
      },
    });

    this.logger.log(`Bug report created: ${report.id}`);
    return report;
  }

  /**
   * Get user's own bug reports
   */
  async findByUser(userId: string) {
    return this.prisma.bugReport.findMany({
      where: { userId },
      orderBy: { createdAt: 'desc' },
      include: {
        task: { select: { title: true, slug: true } },
      },
    });
  }

  /**
   * Get all bug reports (for admin)
   */
  async findAll(filters?: { status?: string; severity?: string; category?: string }) {
    const where: Record<string, string> = {};
    if (filters?.status) where.status = filters.status;
    if (filters?.severity) where.severity = filters.severity;
    if (filters?.category) where.category = filters.category;

    return this.prisma.bugReport.findMany({
      where,
      orderBy: { createdAt: 'desc' },
      include: {
        user: { select: { name: true, email: true } },
        task: { select: { title: true, slug: true } },
      },
    });
  }

  /**
   * Get a single bug report by ID
   */
  async findOne(id: string) {
    return this.prisma.bugReport.findUnique({
      where: { id },
      include: {
        user: { select: { name: true, email: true } },
        task: { select: { title: true, slug: true } },
      },
    });
  }

  /**
   * Update bug report status (for admin)
   * @param id Bug report ID
   * @param status New status (must be a valid BugStatus enum value)
   */
  async updateStatus(id: string, status: BugStatus) {
    // Validate the status is a valid enum value
    const validStatuses = Object.values(BugStatus);
    if (!validStatuses.includes(status)) {
      throw new BadRequestException(
        `Invalid status: ${status}. Valid statuses are: ${validStatuses.join(', ')}`
      );
    }

    return this.prisma.bugReport.update({
      where: { id },
      data: { status },
    });
  }
}
