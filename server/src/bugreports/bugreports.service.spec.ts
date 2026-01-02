import { Test, TestingModule } from '@nestjs/testing';
import { BadRequestException } from '@nestjs/common';
import { BugReportsService } from './bugreports.service';
import { PrismaService } from '../prisma/prisma.service';
import { BugCategory, BugSeverity, BugStatus } from './dto/bugreports.dto';

describe('BugReportsService', () => {
  let service: BugReportsService;
  let prisma: PrismaService;

  const mockBugReport = {
    id: 'bug-123',
    userId: 'user-123',
    taskId: 'task-456',
    category: BugCategory.SOLUTION,
    severity: BugSeverity.MEDIUM,
    title: 'Test bug report',
    description: 'This is a test bug report',
    status: 'open',
    metadata: {},
    createdAt: new Date(),
    user: { name: 'Test User', email: 'test@example.com' },
    task: { title: 'Test Task', slug: 'test-task' },
  };

  const mockPrismaService = {
    bugReport: {
      create: jest.fn(),
      findMany: jest.fn(),
      findUnique: jest.fn(),
      update: jest.fn(),
    },
  };

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [
        BugReportsService,
        { provide: PrismaService, useValue: mockPrismaService },
      ],
    }).compile();

    service = module.get<BugReportsService>(BugReportsService);
    prisma = module.get<PrismaService>(PrismaService);

    jest.clearAllMocks();
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });

  // ============================================
  // create()
  // ============================================
  describe('create()', () => {
    it('should create a bug report with all fields', async () => {
      mockPrismaService.bugReport.create.mockResolvedValue(mockBugReport);

      const dto = {
        title: 'Test bug report',
        description: 'This is a test bug report',
        category: BugCategory.SOLUTION,
        severity: BugSeverity.MEDIUM,
        taskId: 'task-456',
      };

      const result = await service.create('user-123', dto);

      expect(result).toEqual(mockBugReport);
      expect(mockPrismaService.bugReport.create).toHaveBeenCalledWith({
        data: expect.objectContaining({
          userId: 'user-123',
          title: 'Test bug report',
          category: BugCategory.SOLUTION,
        }),
        include: expect.any(Object),
      });
    });

    it('should use default severity if not provided', async () => {
      mockPrismaService.bugReport.create.mockResolvedValue(mockBugReport);

      const dto = {
        title: 'Test',
        description: 'Description',
        category: BugCategory.EDITOR,
      };

      await service.create('user-123', dto);

      expect(mockPrismaService.bugReport.create).toHaveBeenCalledWith({
        data: expect.objectContaining({
          severity: 'medium',
        }),
        include: expect.any(Object),
      });
    });

    it('should handle null taskId', async () => {
      mockPrismaService.bugReport.create.mockResolvedValue({
        ...mockBugReport,
        taskId: null,
        task: null,
      });

      const dto = {
        title: 'General bug',
        description: 'Not related to any task',
        category: BugCategory.OTHER,
      };

      const result = await service.create('user-123', dto);

      expect(result.taskId).toBeNull();
    });

    it('should include metadata if provided', async () => {
      const metadata = { browser: 'Chrome', os: 'macOS' };
      mockPrismaService.bugReport.create.mockResolvedValue({
        ...mockBugReport,
        metadata,
      });

      const dto = {
        title: 'Test',
        description: 'Desc',
        category: BugCategory.EDITOR,
        metadata,
      };

      await service.create('user-123', dto);

      expect(mockPrismaService.bugReport.create).toHaveBeenCalledWith({
        data: expect.objectContaining({
          metadata,
        }),
        include: expect.any(Object),
      });
    });
  });

  // ============================================
  // findByUser()
  // ============================================
  describe('findByUser()', () => {
    it('should return user bug reports', async () => {
      mockPrismaService.bugReport.findMany.mockResolvedValue([mockBugReport]);

      const result = await service.findByUser('user-123');

      expect(result).toEqual([mockBugReport]);
      expect(mockPrismaService.bugReport.findMany).toHaveBeenCalledWith({
        where: { userId: 'user-123' },
        orderBy: { createdAt: 'desc' },
        include: expect.any(Object),
      });
    });

    it('should return empty array for user with no reports', async () => {
      mockPrismaService.bugReport.findMany.mockResolvedValue([]);

      const result = await service.findByUser('user-no-bugs');

      expect(result).toEqual([]);
    });
  });

  // ============================================
  // findAll()
  // ============================================
  describe('findAll()', () => {
    it('should return all bug reports without filters', async () => {
      mockPrismaService.bugReport.findMany.mockResolvedValue([mockBugReport]);

      const result = await service.findAll();

      expect(result).toEqual([mockBugReport]);
      expect(mockPrismaService.bugReport.findMany).toHaveBeenCalledWith({
        where: {},
        orderBy: { createdAt: 'desc' },
        include: expect.any(Object),
      });
    });

    it('should filter by status', async () => {
      mockPrismaService.bugReport.findMany.mockResolvedValue([mockBugReport]);

      await service.findAll({ status: 'open' });

      expect(mockPrismaService.bugReport.findMany).toHaveBeenCalledWith({
        where: { status: 'open' },
        orderBy: { createdAt: 'desc' },
        include: expect.any(Object),
      });
    });

    it('should filter by severity', async () => {
      mockPrismaService.bugReport.findMany.mockResolvedValue([]);

      await service.findAll({ severity: 'high' });

      expect(mockPrismaService.bugReport.findMany).toHaveBeenCalledWith({
        where: { severity: 'high' },
        orderBy: { createdAt: 'desc' },
        include: expect.any(Object),
      });
    });

    it('should filter by category', async () => {
      mockPrismaService.bugReport.findMany.mockResolvedValue([]);

      await service.findAll({ category: 'solution' });

      expect(mockPrismaService.bugReport.findMany).toHaveBeenCalledWith({
        where: { category: 'solution' },
        orderBy: { createdAt: 'desc' },
        include: expect.any(Object),
      });
    });

    it('should combine multiple filters', async () => {
      mockPrismaService.bugReport.findMany.mockResolvedValue([]);

      await service.findAll({ status: 'open', severity: 'high', category: 'solution' });

      expect(mockPrismaService.bugReport.findMany).toHaveBeenCalledWith({
        where: { status: 'open', severity: 'high', category: 'solution' },
        orderBy: { createdAt: 'desc' },
        include: expect.any(Object),
      });
    });
  });

  // ============================================
  // findOne()
  // ============================================
  describe('findOne()', () => {
    it('should return bug report by id', async () => {
      mockPrismaService.bugReport.findUnique.mockResolvedValue(mockBugReport);

      const result = await service.findOne('bug-123');

      expect(result).toEqual(mockBugReport);
      expect(mockPrismaService.bugReport.findUnique).toHaveBeenCalledWith({
        where: { id: 'bug-123' },
        include: expect.any(Object),
      });
    });

    it('should return null for non-existent report', async () => {
      mockPrismaService.bugReport.findUnique.mockResolvedValue(null);

      const result = await service.findOne('nonexistent');

      expect(result).toBeNull();
    });
  });

  // ============================================
  // updateStatus()
  // ============================================
  describe('updateStatus()', () => {
    it('should update bug report status', async () => {
      mockPrismaService.bugReport.update.mockResolvedValue({
        ...mockBugReport,
        status: BugStatus.RESOLVED,
      });

      const result = await service.updateStatus('bug-123', BugStatus.RESOLVED);

      expect(result.status).toBe(BugStatus.RESOLVED);
      expect(mockPrismaService.bugReport.update).toHaveBeenCalledWith({
        where: { id: 'bug-123' },
        data: { status: BugStatus.RESOLVED },
      });
    });

    it('should accept all valid status values', async () => {
      for (const status of Object.values(BugStatus)) {
        mockPrismaService.bugReport.update.mockResolvedValue({ ...mockBugReport, status });

        const result = await service.updateStatus('bug-123', status);

        expect(result.status).toBe(status);
      }
    });

    it('should throw BadRequestException for invalid status', async () => {
      await expect(
        service.updateStatus('bug-123', 'invalid-status' as BugStatus),
      ).rejects.toThrow(BadRequestException);
    });
  });
});
