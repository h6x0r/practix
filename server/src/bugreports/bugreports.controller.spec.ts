import { Test, TestingModule } from '@nestjs/testing';
import { BugReportsController } from './bugreports.controller';
import { BugReportsService } from './bugreports.service';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { AdminGuard } from '../auth/guards/admin.guard';
import { BugCategory, BugSeverity, BugStatus } from './dto/bugreports.dto';

describe('BugReportsController', () => {
  let controller: BugReportsController;
  let bugReportsService: BugReportsService;

  const mockBugReport = {
    id: 'bug-123',
    userId: 'user-123',
    taskId: 'task-456',
    category: BugCategory.SOLUTION,
    severity: BugSeverity.MEDIUM,
    title: 'Test bug report',
    description: 'This is a test bug report',
    status: BugStatus.OPEN,
    metadata: {},
    createdAt: new Date(),
    user: { name: 'Test User', email: 'test@example.com' },
    task: { title: 'Test Task', slug: 'test-task' },
  };

  const mockBugReportsService = {
    create: jest.fn(),
    findByUser: jest.fn(),
    findAll: jest.fn(),
    findOne: jest.fn(),
    updateStatus: jest.fn(),
  };

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      controllers: [BugReportsController],
      providers: [
        {
          provide: BugReportsService,
          useValue: mockBugReportsService,
        },
      ],
    })
      .overrideGuard(JwtAuthGuard)
      .useValue({ canActivate: () => true })
      .overrideGuard(AdminGuard)
      .useValue({ canActivate: () => true })
      .compile();

    controller = module.get<BugReportsController>(BugReportsController);
    bugReportsService = module.get<BugReportsService>(BugReportsService);

    jest.clearAllMocks();
  });

  it('should be defined', () => {
    expect(controller).toBeDefined();
  });

  describe('create', () => {
    const createDto = {
      title: 'Test bug report',
      description: 'This is a test bug report',
      category: BugCategory.SOLUTION,
      severity: BugSeverity.MEDIUM,
      taskId: 'task-456',
    };

    it('should create a bug report', async () => {
      mockBugReportsService.create.mockResolvedValue(mockBugReport);

      const result = await controller.create(
        { user: { userId: 'user-123' } },
        createDto
      );

      expect(result).toEqual(mockBugReport);
      expect(mockBugReportsService.create).toHaveBeenCalledWith('user-123', createDto);
    });

    it('should create bug report without taskId', async () => {
      const dtoWithoutTask = {
        title: 'General bug',
        description: 'Not related to any task',
        category: BugCategory.OTHER,
      };
      const reportWithoutTask = { ...mockBugReport, taskId: null, task: null };
      mockBugReportsService.create.mockResolvedValue(reportWithoutTask);

      const result = await controller.create(
        { user: { userId: 'user-123' } },
        dtoWithoutTask
      );

      expect(result.taskId).toBeNull();
    });

    it('should handle different categories', async () => {
      for (const category of Object.values(BugCategory)) {
        const dto = { ...createDto, category };
        mockBugReportsService.create.mockResolvedValue({ ...mockBugReport, category });

        const result = await controller.create({ user: { userId: 'user-123' } }, dto);

        expect(result.category).toBe(category);
      }
    });

    it('should handle different severities', async () => {
      for (const severity of Object.values(BugSeverity)) {
        const dto = { ...createDto, severity };
        mockBugReportsService.create.mockResolvedValue({ ...mockBugReport, severity });

        const result = await controller.create({ user: { userId: 'user-123' } }, dto);

        expect(result.severity).toBe(severity);
      }
    });

    it('should include metadata if provided', async () => {
      const dtoWithMetadata = {
        ...createDto,
        metadata: { browser: 'Chrome', os: 'macOS', version: '1.0.0' },
      };
      mockBugReportsService.create.mockResolvedValue({
        ...mockBugReport,
        metadata: dtoWithMetadata.metadata,
      });

      const result = await controller.create(
        { user: { userId: 'user-123' } },
        dtoWithMetadata
      );

      expect(result.metadata).toEqual(dtoWithMetadata.metadata);
    });
  });

  describe('findMy', () => {
    it('should return user bug reports', async () => {
      mockBugReportsService.findByUser.mockResolvedValue([mockBugReport]);

      const result = await controller.findMy({ user: { userId: 'user-123' } });

      expect(result).toEqual([mockBugReport]);
      expect(mockBugReportsService.findByUser).toHaveBeenCalledWith('user-123');
    });

    it('should return empty array for user with no reports', async () => {
      mockBugReportsService.findByUser.mockResolvedValue([]);

      const result = await controller.findMy({ user: { userId: 'user-no-bugs' } });

      expect(result).toEqual([]);
    });

    it('should return multiple bug reports', async () => {
      const multipleReports = [
        mockBugReport,
        { ...mockBugReport, id: 'bug-456', title: 'Another bug' },
        { ...mockBugReport, id: 'bug-789', title: 'Third bug' },
      ];
      mockBugReportsService.findByUser.mockResolvedValue(multipleReports);

      const result = await controller.findMy({ user: { userId: 'user-123' } });

      expect(result).toHaveLength(3);
    });
  });

  describe('findAll', () => {
    it('should return all bug reports without filters', async () => {
      mockBugReportsService.findAll.mockResolvedValue([mockBugReport]);

      const result = await controller.findAll();

      expect(result).toEqual([mockBugReport]);
      expect(mockBugReportsService.findAll).toHaveBeenCalledWith({
        status: undefined,
        severity: undefined,
        category: undefined,
      });
    });

    it('should filter by status', async () => {
      mockBugReportsService.findAll.mockResolvedValue([mockBugReport]);

      await controller.findAll('open');

      expect(mockBugReportsService.findAll).toHaveBeenCalledWith({
        status: 'open',
        severity: undefined,
        category: undefined,
      });
    });

    it('should filter by severity', async () => {
      mockBugReportsService.findAll.mockResolvedValue([]);

      await controller.findAll(undefined, 'high');

      expect(mockBugReportsService.findAll).toHaveBeenCalledWith({
        status: undefined,
        severity: 'high',
        category: undefined,
      });
    });

    it('should filter by category', async () => {
      mockBugReportsService.findAll.mockResolvedValue([]);

      await controller.findAll(undefined, undefined, 'solution');

      expect(mockBugReportsService.findAll).toHaveBeenCalledWith({
        status: undefined,
        severity: undefined,
        category: 'solution',
      });
    });

    it('should combine multiple filters', async () => {
      mockBugReportsService.findAll.mockResolvedValue([]);

      await controller.findAll('open', 'high', 'solution');

      expect(mockBugReportsService.findAll).toHaveBeenCalledWith({
        status: 'open',
        severity: 'high',
        category: 'solution',
      });
    });

    it('should return empty array when no matches', async () => {
      mockBugReportsService.findAll.mockResolvedValue([]);

      const result = await controller.findAll('resolved');

      expect(result).toEqual([]);
    });
  });

  describe('findOne', () => {
    it('should return bug report by id', async () => {
      mockBugReportsService.findOne.mockResolvedValue(mockBugReport);

      const result = await controller.findOne('bug-123');

      expect(result).toEqual(mockBugReport);
      expect(mockBugReportsService.findOne).toHaveBeenCalledWith('bug-123');
    });

    it('should return null for non-existent report', async () => {
      mockBugReportsService.findOne.mockResolvedValue(null);

      const result = await controller.findOne('nonexistent');

      expect(result).toBeNull();
    });

    it('should include user and task relations', async () => {
      mockBugReportsService.findOne.mockResolvedValue(mockBugReport);

      const result = await controller.findOne('bug-123');

      expect(result.user).toBeDefined();
      expect(result.task).toBeDefined();
    });
  });

  describe('updateStatus', () => {
    it('should update bug report status', async () => {
      const updatedReport = { ...mockBugReport, status: BugStatus.RESOLVED };
      mockBugReportsService.updateStatus.mockResolvedValue(updatedReport);

      const result = await controller.updateStatus('bug-123', {
        status: BugStatus.RESOLVED,
      });

      expect(result.status).toBe(BugStatus.RESOLVED);
      expect(mockBugReportsService.updateStatus).toHaveBeenCalledWith(
        'bug-123',
        BugStatus.RESOLVED
      );
    });

    it('should accept all valid status values', async () => {
      for (const status of Object.values(BugStatus)) {
        mockBugReportsService.updateStatus.mockResolvedValue({
          ...mockBugReport,
          status,
        });

        const result = await controller.updateStatus('bug-123', { status });

        expect(result.status).toBe(status);
      }
    });

    it('should update to in-progress status', async () => {
      mockBugReportsService.updateStatus.mockResolvedValue({
        ...mockBugReport,
        status: BugStatus.IN_PROGRESS,
      });

      const result = await controller.updateStatus('bug-123', {
        status: BugStatus.IN_PROGRESS,
      });

      expect(result.status).toBe(BugStatus.IN_PROGRESS);
    });

    it('should update to closed status', async () => {
      mockBugReportsService.updateStatus.mockResolvedValue({
        ...mockBugReport,
        status: BugStatus.CLOSED,
      });

      const result = await controller.updateStatus('bug-123', {
        status: BugStatus.CLOSED,
      });

      expect(result.status).toBe(BugStatus.CLOSED);
    });

    it('should update to wont-fix status', async () => {
      mockBugReportsService.updateStatus.mockResolvedValue({
        ...mockBugReport,
        status: BugStatus.WONT_FIX,
      });

      const result = await controller.updateStatus('bug-123', {
        status: BugStatus.WONT_FIX,
      });

      expect(result.status).toBe(BugStatus.WONT_FIX);
    });
  });

  describe('edge cases', () => {
    it('should handle service errors in create', async () => {
      mockBugReportsService.create.mockRejectedValue(new Error('Database error'));

      await expect(
        controller.create(
          { user: { userId: 'user-123' } },
          {
            title: 'Test',
            description: 'Desc',
            category: BugCategory.OTHER,
          }
        )
      ).rejects.toThrow('Database error');
    });

    it('should handle service errors in findAll', async () => {
      mockBugReportsService.findAll.mockRejectedValue(
        new Error('Database connection failed')
      );

      await expect(controller.findAll()).rejects.toThrow('Database connection failed');
    });

    it('should handle unicode in bug report title and description', async () => {
      const unicodeDto = {
        title: 'Ошибка в решении 解决方案中的错误',
        description: 'Описание ошибки 错误描述',
        category: BugCategory.SOLUTION,
      };
      mockBugReportsService.create.mockResolvedValue({
        ...mockBugReport,
        title: unicodeDto.title,
        description: unicodeDto.description,
      });

      const result = await controller.create(
        { user: { userId: 'user-123' } },
        unicodeDto
      );

      expect(result.title).toBe(unicodeDto.title);
      expect(result.description).toBe(unicodeDto.description);
    });

    it('should handle very long description', async () => {
      const longDescription = 'a'.repeat(5000);
      const dto = {
        title: 'Test',
        description: longDescription,
        category: BugCategory.OTHER,
      };
      mockBugReportsService.create.mockResolvedValue({
        ...mockBugReport,
        description: longDescription,
      });

      const result = await controller.create({ user: { userId: 'user-123' } }, dto);

      expect(result.description).toHaveLength(5000);
    });
  });
});
