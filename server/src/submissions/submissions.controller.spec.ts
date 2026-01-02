import { Test, TestingModule } from '@nestjs/testing';
import { SubmissionsController } from './submissions.controller';
import { SubmissionsService } from './submissions.service';
import { CodeExecutionService } from '../queue/code-execution.service';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { ThrottlerModule } from '@nestjs/throttler';
import { ForbiddenException, NotFoundException } from '@nestjs/common';
import { PlaygroundThrottlerGuard } from '../common/guards/playground-throttler.guard';

describe('SubmissionsController', () => {
  let controller: SubmissionsController;
  let submissionsService: SubmissionsService;
  let codeExecutionService: CodeExecutionService;

  const mockSubmissionsService = {
    create: jest.fn(),
    runCode: jest.fn(),
    runQuickTests: jest.fn(),
    getJudgeStatus: jest.fn(),
    findOne: jest.fn(),
    findRecentByUser: jest.fn(),
    findByUserAndTask: jest.fn(),
  };

  const mockCodeExecutionService = {
    getSupportedLanguages: jest.fn(),
  };

  const mockPlaygroundThrottler = {
    canActivate: jest.fn().mockResolvedValue(true),
    getRateLimitInfo: jest.fn().mockResolvedValue({ interval: 10, isPremium: false }),
  };

  // Mock request object with IP
  const createMockRequest = (userId?: string) => ({
    user: userId ? { userId } : undefined,
    ip: '127.0.0.1',
    headers: {},
    connection: { remoteAddress: '127.0.0.1' },
  });

  // Matches SubmissionResult interface
  const mockSubmission = {
    id: 'submission-123',
    status: 'passed',
    score: 100,
    runtime: '50ms',
    memory: '10MB',
    message: 'All tests passed',
    stdout: '',
    stderr: '',
    compileOutput: '',
    createdAt: new Date().toISOString(),
    testsPassed: 5,
    testsTotal: 5,
    testCases: [],
    // For findOne - includes full submission with userId
    userId: 'user-123',
    taskId: 'task-1',
    code: 'package main\n\nfunc main() {}',
    language: 'go',
  };

  // Matches ExecutionResult interface
  const mockRunResult = {
    status: 'passed' as const,
    statusId: 1,
    description: 'Executed successfully',
    stdout: 'Hello, World!',
    stderr: '',
    compileOutput: '',
    time: '0.05',
    memory: 1024,
    exitCode: 0,
    message: '',
  };

  // Matches runQuickTests return type
  const mockTestResult = {
    status: 'passed',
    testsPassed: 5,
    testsTotal: 5,
    testCases: [
      { name: 'Test 1', passed: true, message: '', expected: '5', actual: '5' },
      { name: 'Test 2', passed: true, message: '', expected: '10', actual: '10' },
    ],
    runtime: '100ms',
    message: 'All tests passed',
  };

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      imports: [
        ThrottlerModule.forRoot([{
          ttl: 60000,
          limit: 100,
        }]),
      ],
      controllers: [SubmissionsController],
      providers: [
        {
          provide: SubmissionsService,
          useValue: mockSubmissionsService,
        },
        {
          provide: CodeExecutionService,
          useValue: mockCodeExecutionService,
        },
        {
          provide: PlaygroundThrottlerGuard,
          useValue: mockPlaygroundThrottler,
        },
      ],
    })
      .overrideGuard(JwtAuthGuard)
      .useValue({ canActivate: () => true })
      .compile();

    controller = module.get<SubmissionsController>(SubmissionsController);
    submissionsService = module.get<SubmissionsService>(SubmissionsService);
    codeExecutionService = module.get<CodeExecutionService>(CodeExecutionService);

    jest.clearAllMocks();
  });

  it('should be defined', () => {
    expect(controller).toBeDefined();
  });

  describe('create', () => {
    const createDto = {
      taskId: 'task-1',
      code: 'package main\n\nfunc main() {}',
      language: 'go',
    };

    it('should create a submission successfully', async () => {
      mockSubmissionsService.create.mockResolvedValue(mockSubmission);
      const mockReq = { ...createMockRequest('user-123'), user: { userId: 'user-123' } };

      const result = await controller.create(mockReq, createDto);

      expect(result).toEqual(mockSubmission);
      expect(mockSubmissionsService.create).toHaveBeenCalledWith(
        'user-123',
        'task-1',
        createDto.code,
        'go',
        '127.0.0.1'
      );
    });

    it('should handle submission with failing tests', async () => {
      const failedSubmission = {
        ...mockSubmission,
        status: 'failed',
        testsPassed: 3,
        score: 60,
      };
      mockSubmissionsService.create.mockResolvedValue(failedSubmission);
      const mockReq = { ...createMockRequest('user-123'), user: { userId: 'user-123' } };

      const result = await controller.create(mockReq, createDto);

      expect(result.status).toBe('failed');
      expect(result.testsPassed).toBe(3);
    });

    it('should handle different languages', async () => {
      const javaDto = {
        taskId: 'task-1',
        code: 'public class Main { public static void main(String[] args) {} }',
        language: 'java',
      };
      mockSubmissionsService.create.mockResolvedValue(mockSubmission);
      const mockReq = { ...createMockRequest('user-123'), user: { userId: 'user-123' } };

      const result = await controller.create(mockReq, javaDto);

      expect(result).toEqual(mockSubmission);
      expect(mockSubmissionsService.create).toHaveBeenCalledWith(
        'user-123',
        'task-1',
        javaDto.code,
        'java',
        '127.0.0.1'
      );
    });

    it('should handle service errors', async () => {
      mockSubmissionsService.create.mockRejectedValue(
        new Error('Execution timeout')
      );
      const mockReq = { ...createMockRequest('user-123'), user: { userId: 'user-123' } };

      await expect(
        controller.create(mockReq, createDto)
      ).rejects.toThrow('Execution timeout');
    });
  });

  describe('runCode', () => {
    const runDto = {
      code: 'console.log("Hello");',
      language: 'javascript',
      stdin: '',
    };

    it('should run code successfully', async () => {
      mockSubmissionsService.runCode.mockResolvedValue(mockRunResult);
      const mockReq = createMockRequest();

      const result = await controller.runCode(mockReq, runDto);

      expect(result).toEqual(mockRunResult);
      expect(mockSubmissionsService.runCode).toHaveBeenCalledWith(
        runDto.code,
        'javascript',
        '',
        '127.0.0.1',
        undefined
      );
    });

    it('should handle code with stdin', async () => {
      const dtoWithStdin = {
        ...runDto,
        stdin: 'input data\n',
      };
      mockSubmissionsService.runCode.mockResolvedValue(mockRunResult);
      const mockReq = createMockRequest();

      await controller.runCode(mockReq, dtoWithStdin);

      expect(mockSubmissionsService.runCode).toHaveBeenCalledWith(
        runDto.code,
        'javascript',
        'input data\n',
        '127.0.0.1',
        undefined
      );
    });

    it('should handle code execution errors', async () => {
      const errorResult = {
        ...mockRunResult,
        stdout: '',
        stderr: 'SyntaxError: Unexpected token',
        status: 'error' as const,
      };
      mockSubmissionsService.runCode.mockResolvedValue(errorResult);
      const mockReq = createMockRequest();

      const result = await controller.runCode(mockReq, runDto);

      expect(result.stderr).toBe('SyntaxError: Unexpected token');
      expect(result.status).toBe('error');
    });

    it('should handle runtime errors', async () => {
      const runtimeError = {
        ...mockRunResult,
        stdout: '',
        stderr: 'panic: runtime error: index out of range',
        status: 'error' as const,
      };
      mockSubmissionsService.runCode.mockResolvedValue(runtimeError);
      const mockReq = createMockRequest();

      const result = await controller.runCode(mockReq, {
        code: 'go code with panic',
        language: 'go',
        stdin: '',
      });

      expect(result.status).toBe('error');
    });
  });

  describe('runTests', () => {
    const runTestsDto = {
      taskId: 'task-1',
      code: 'package main\n\nfunc Sum(a, b int) int { return a + b }',
      language: 'go',
    };

    it('should run quick tests successfully', async () => {
      mockSubmissionsService.runQuickTests.mockResolvedValue(mockTestResult);
      const mockReq = createMockRequest();

      const result = await controller.runTests(mockReq, runTestsDto);

      expect(result).toEqual(mockTestResult);
      expect(mockSubmissionsService.runQuickTests).toHaveBeenCalledWith(
        'task-1',
        runTestsDto.code,
        'go',
        '127.0.0.1',
        undefined
      );
    });

    it('should handle partial test failures', async () => {
      const partialResult = {
        ...mockTestResult,
        status: 'failed',
        testsPassed: 3,
        testCases: [
          { name: 'Test 1', passed: true, message: '', expected: '5', actual: '5' },
          { name: 'Test 2', passed: true, message: '', expected: '10', actual: '10' },
          { name: 'Test 3', passed: true, message: '', expected: '15', actual: '15' },
          { name: 'Test 4', passed: false, message: 'Expected 10, got 5', expected: '10', actual: '5' },
          { name: 'Test 5', passed: false, message: 'Expected 20, got 10', expected: '20', actual: '10' },
        ],
      };
      mockSubmissionsService.runQuickTests.mockResolvedValue(partialResult);
      const mockReq = createMockRequest();

      const result = await controller.runTests(mockReq, runTestsDto);

      expect(result.status).toBe('failed');
      expect(result.testsPassed).toBe(3);
    });

    it('should handle compilation errors', async () => {
      const compileError = {
        status: 'error',
        message: 'undefined: Sum',
        testsTotal: 0,
        testsPassed: 0,
        testCases: [],
        runtime: '0ms',
      };
      mockSubmissionsService.runQuickTests.mockResolvedValue(compileError);
      const mockReq = createMockRequest();

      const result = await controller.runTests(mockReq, runTestsDto);

      expect(result.message).toBe('undefined: Sum');
    });
  });

  describe('getJudgeStatus', () => {
    it('should return judge status', async () => {
      const status = {
        available: true,
        queueReady: true,
        queue: { waiting: 0, active: 0, completed: 100, failed: 0 },
        cache: { connected: true, keys: 50 },
        languages: ['go', 'java', 'python'],
      };
      mockSubmissionsService.getJudgeStatus.mockResolvedValue(status);

      const result = await controller.getJudgeStatus();

      expect(result).toEqual(status);
      expect(result.available).toBe(true);
    });

    it('should handle unhealthy judge', async () => {
      const unhealthyStatus = {
        available: false,
        queueReady: false,
        queue: { waiting: 100, active: 0, completed: 0, failed: 50 },
        cache: { connected: false },
        languages: [],
      };
      mockSubmissionsService.getJudgeStatus.mockResolvedValue(unhealthyStatus);

      const result = await controller.getJudgeStatus();

      expect(result.available).toBe(false);
      expect(result.queueReady).toBe(false);
    });
  });

  describe('getLanguages', () => {
    it('should return supported languages', () => {
      mockCodeExecutionService.getSupportedLanguages.mockReturnValue([
        { id: 'go', name: 'Go', version: '1.21' },
        { id: 'java', name: 'Java', version: '21' },
        { id: 'python', name: 'Python', version: '3.12' },
      ]);

      const result = controller.getLanguages();

      expect(result.languages).toHaveLength(3);
      expect(result.default).toBe('go');
    });

    it('should return empty list if no languages configured', () => {
      mockCodeExecutionService.getSupportedLanguages.mockReturnValue([]);

      const result = controller.getLanguages();

      expect(result.languages).toHaveLength(0);
      expect(result.default).toBe('go');
    });
  });

  describe('findOne', () => {
    it('should return submission by id for owner', async () => {
      mockSubmissionsService.findOne.mockResolvedValue(mockSubmission);

      const result = await controller.findOne(
        { user: { userId: 'user-123' } },
        'submission-123'
      );

      expect(result).toEqual(mockSubmission);
      expect(mockSubmissionsService.findOne).toHaveBeenCalledWith('submission-123');
    });

    it('should throw ForbiddenException for non-owner', async () => {
      mockSubmissionsService.findOne.mockResolvedValue(mockSubmission);

      await expect(
        controller.findOne({ user: { userId: 'other-user' } }, 'submission-123')
      ).rejects.toThrow(ForbiddenException);
    });

    it('should handle non-existent submission', async () => {
      mockSubmissionsService.findOne.mockRejectedValue(
        new NotFoundException('Submission not found')
      );

      await expect(
        controller.findOne({ user: { userId: 'user-123' } }, 'non-existent')
      ).rejects.toThrow(NotFoundException);
    });
  });

  describe('getUserRecent', () => {
    const recentSubmissions = [
      { ...mockSubmission, id: 'sub-1' },
      { ...mockSubmission, id: 'sub-2' },
      { ...mockSubmission, id: 'sub-3' },
    ];

    it('should return recent submissions with default limit', async () => {
      mockSubmissionsService.findRecentByUser.mockResolvedValue(recentSubmissions);

      const result = await controller.getUserRecent({ user: { userId: 'user-123' } });

      expect(result).toEqual(recentSubmissions);
      expect(mockSubmissionsService.findRecentByUser).toHaveBeenCalledWith('user-123', 10);
    });

    it('should respect custom limit', async () => {
      mockSubmissionsService.findRecentByUser.mockResolvedValue(recentSubmissions.slice(0, 2));

      const result = await controller.getUserRecent(
        { user: { userId: 'user-123' } },
        '2'
      );

      expect(mockSubmissionsService.findRecentByUser).toHaveBeenCalledWith('user-123', 2);
    });

    it('should handle empty submissions list', async () => {
      mockSubmissionsService.findRecentByUser.mockResolvedValue([]);

      const result = await controller.getUserRecent({ user: { userId: 'user-123' } });

      expect(result).toEqual([]);
    });

    it('should handle invalid limit gracefully', async () => {
      mockSubmissionsService.findRecentByUser.mockResolvedValue(recentSubmissions);

      await controller.getUserRecent(
        { user: { userId: 'user-123' } },
        'invalid'
      );

      // NaN is passed when parseInt fails
      expect(mockSubmissionsService.findRecentByUser).toHaveBeenCalled();
    });
  });

  describe('getByTask', () => {
    const taskSubmissions = [
      { ...mockSubmission, id: 'sub-1', passed: true },
      { ...mockSubmission, id: 'sub-2', passed: false },
    ];

    it('should return submissions for a task', async () => {
      mockSubmissionsService.findByUserAndTask.mockResolvedValue(taskSubmissions);

      const result = await controller.getByTask(
        { user: { userId: 'user-123' } },
        'task-1'
      );

      expect(result).toEqual(taskSubmissions);
      expect(mockSubmissionsService.findByUserAndTask).toHaveBeenCalledWith(
        'user-123',
        'task-1'
      );
    });

    it('should handle task with no submissions', async () => {
      mockSubmissionsService.findByUserAndTask.mockResolvedValue([]);

      const result = await controller.getByTask(
        { user: { userId: 'user-123' } },
        'new-task'
      );

      expect(result).toEqual([]);
    });

    it('should handle task slug formats', async () => {
      mockSubmissionsService.findByUserAndTask.mockResolvedValue(taskSubmissions);

      await controller.getByTask(
        { user: { userId: 'user-123' } },
        'hello-world-01'
      );

      expect(mockSubmissionsService.findByUserAndTask).toHaveBeenCalledWith(
        'user-123',
        'hello-world-01'
      );
    });
  });

  describe('edge cases', () => {
    it('should handle very long code submissions', async () => {
      const longCode = 'x'.repeat(100000);
      mockSubmissionsService.create.mockResolvedValue(mockSubmission);
      const mockReq = { ...createMockRequest('user-123'), user: { userId: 'user-123' } };

      await controller.create(mockReq, { taskId: 'task-1', code: longCode, language: 'go' });

      expect(mockSubmissionsService.create).toHaveBeenCalledWith(
        'user-123',
        'task-1',
        longCode,
        'go',
        '127.0.0.1'
      );
    });

    it('should handle unicode in code', async () => {
      const unicodeCode = 'fmt.Println("Привет мир! 你好世界!")';
      mockSubmissionsService.runCode.mockResolvedValue(mockRunResult);
      const mockReq = createMockRequest();

      await controller.runCode(mockReq, {
        code: unicodeCode,
        language: 'go',
        stdin: '',
      });

      expect(mockSubmissionsService.runCode).toHaveBeenCalledWith(
        unicodeCode,
        'go',
        '',
        '127.0.0.1',
        undefined
      );
    });

    it('should handle special characters in stdin', async () => {
      const specialStdin = 'line1\nline2\tspaced\r\nwindows';
      mockSubmissionsService.runCode.mockResolvedValue(mockRunResult);
      const mockReq = createMockRequest();

      await controller.runCode(mockReq, {
        code: 'scanner.Scan()',
        language: 'go',
        stdin: specialStdin,
      });

      expect(mockSubmissionsService.runCode).toHaveBeenCalledWith(
        'scanner.Scan()',
        'go',
        specialStdin,
        '127.0.0.1',
        undefined
      );
    });

    it('should handle concurrent submission requests', async () => {
      mockSubmissionsService.create.mockResolvedValue(mockSubmission);

      const mockReq = createMockRequest('user-123');
      const promises = Array.from({ length: 5 }, () =>
        controller.create(mockReq, { taskId: 'task-1', code: 'code', language: 'go' })
      );

      const results = await Promise.all(promises);

      expect(results).toHaveLength(5);
      expect(mockSubmissionsService.create).toHaveBeenCalledTimes(5);
    });
  });
});
