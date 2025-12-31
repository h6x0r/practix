import { Injectable, NotFoundException, BadRequestException, Logger } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { CodeExecutionService } from '../queue/code-execution.service';
import { CacheService } from '../cache/cache.service';
import { ExecutionResult, LANGUAGES } from '../piston/piston.service';
import { TasksService } from '../tasks/tasks.service';
import { AccessControlService } from '../subscriptions/access-control.service';
import { GamificationService } from '../gamification/gamification.service';

export interface SubmissionDto {
  code: string;
  taskId?: string;
  language?: string;
  stdin?: string;
}

export interface TestCaseResult {
  name: string;
  passed: boolean;
  input?: string;
  expectedOutput?: string;
  actualOutput?: string;
  error?: string;
}

export interface SubmissionResult {
  id: string;
  status: string;
  score: number;
  runtime: string;
  memory: string;
  message: string;
  stdout: string;
  stderr: string;
  compileOutput: string;
  createdAt: string;
  // Test case details
  testsPassed?: number;
  testsTotal?: number;
  testCases?: TestCaseResult[];
  // Gamification rewards
  xpEarned?: number;
  totalXp?: number;
  level?: number;
  leveledUp?: boolean;
  newBadges?: Array<{ slug: string; name: string; icon: string }>;
}

@Injectable()
export class SubmissionsService {
  private readonly logger = new Logger(SubmissionsService.name);

  constructor(
    private prisma: PrismaService,
    private codeExecutionService: CodeExecutionService,
    private cacheService: CacheService,
    private tasksService: TasksService,
    private accessControlService: AccessControlService,
    private gamificationService: GamificationService
  ) {}

  /**
   * Create a new code submission and execute it
   */
  async create(
    userId: string,
    taskIdentifier: string,
    code: string,
    language: string
  ): Promise<SubmissionResult> {
    // 1. Resolve Task with topic/module/course info for priority
    let task = await this.prisma.task.findUnique({
      where: { id: taskIdentifier },
      include: {
        topic: {
          include: {
            module: { select: { courseId: true } },
          },
        },
      },
    });

    if (!task) {
      task = await this.prisma.task.findUnique({
        where: { slug: taskIdentifier },
        include: {
          topic: {
            include: {
              module: { select: { courseId: true } },
            },
          },
        },
      });
    }

    if (!task) {
      throw new NotFoundException(`Task not found: ${taskIdentifier}`);
    }

    // 2. Get queue priority based on user's subscription
    const courseId = task.topic?.module?.courseId;
    const queuePriority = courseId
      ? await this.accessControlService.getQueuePriority(userId, courseId)
      : 10; // Default to low priority if no course

    // 3. Validate language
    const langConfig = LANGUAGES[language.toLowerCase()];
    if (!langConfig) {
      throw new BadRequestException(
        `Unsupported language: ${language}. Supported: ${Object.keys(LANGUAGES).join(', ')}`
      );
    }

    this.logger.log(
      `Submission: user=${userId}, task=${task.slug}, lang=${langConfig.name}, hasTests=${!!task.testCode}, priority=${queuePriority}`
    );

    // 4. Execute code via Piston + Queue
    // TODO: Pass queuePriority to async execution when queue mode is enabled
    let result;
    if (task.testCode) {
      // Execute user code with task's test code
      result = await this.codeExecutionService.executeSyncWithTests(
        code,
        task.testCode,
        language
      );
    } else {
      // No test code - just run the user's code
      result = await this.codeExecutionService.executeSync(code, language);
    }

    // 4. Parse test output (supports JSON and legacy formats)
    const { testCases, passed: testsPassed, total: testsTotal } = this.parseTestOutput(
      result.stdout,
      result.stderr
    );

    // 5. Determine final status:
    // - 'error' for compile errors
    // - 'failed' for test failures (some or all tests fail)
    // - 'passed' for all tests passing
    let finalStatus: string;
    if (result.status === 'compileError') {
      finalStatus = 'error';
    } else if (testsTotal > 0 && testsPassed < testsTotal) {
      finalStatus = 'failed';
    } else if (result.status === 'error' || result.status === 'timeout') {
      finalStatus = 'error';
    } else {
      finalStatus = 'passed';
    }

    // 6. Format output message
    const message = this.formatMessage(result);

    // 7. Calculate score based on tests passed
    const score = testsTotal > 0
      ? Math.round((testsPassed / testsTotal) * 100)
      : (finalStatus === 'passed' ? 100 : 0);

    // Format runtime - show '-' if not available, otherwise show time in ms
    const runtime = result.time === '-' ? '-' : `${Math.round(parseFloat(result.time) * 1000)}ms`;

    // Format memory - Piston returns unreliable memory values, so we don't show it
    // If memory is 0 or unrealistically high (>1GB), show '-'
    const memoryBytes = result.memory || 0;
    const memoryMB = memoryBytes / (1024 * 1024);
    const memory = (memoryBytes === 0 || memoryMB > 1024) ? '-' : `${memoryMB.toFixed(1)}MB`;

    // 8. Save to database with all fields
    const submission = await this.prisma.submission.create({
      data: {
        userId,
        taskId: task.id,
        code,
        status: finalStatus,
        score,
        runtime,
        memory,
        message,
        testsPassed: testsTotal > 0 ? testsPassed : null,
        testsTotal: testsTotal > 0 ? testsTotal : null,
        testCases: testCases.length > 0 ? JSON.parse(JSON.stringify(testCases)) : undefined,
      },
    });

    // 9. Award XP and check badges if task passed (first time)
    let gamificationResult = null;
    if (finalStatus === 'passed') {
      // Check if this is first successful submission for this task
      const previousPassed = await this.prisma.submission.count({
        where: {
          userId,
          taskId: task.id,
          status: 'passed',
          id: { not: submission.id },
        },
      });

      if (previousPassed === 0) {
        // First time solving this task - award XP
        gamificationResult = await this.gamificationService.awardTaskXp(userId, task.difficulty);
        this.logger.log(
          `XP awarded: user=${userId}, task=${task.slug}, xp=${gamificationResult.xpEarned}, level=${gamificationResult.level}, leveledUp=${gamificationResult.leveledUp}`
        );
      }
    }

    return {
      id: submission.id,
      status: finalStatus,
      score,
      runtime,
      memory,
      message,
      stdout: result.stdout,
      stderr: result.stderr,
      compileOutput: result.compileOutput,
      testsPassed: testsTotal > 0 ? testsPassed : undefined,
      testsTotal: testsTotal > 0 ? testsTotal : undefined,
      testCases: testCases.length > 0 ? testCases : undefined,
      createdAt: submission.createdAt.toISOString(),
      // Gamification rewards
      xpEarned: gamificationResult?.xpEarned,
      totalXp: gamificationResult?.totalXp,
      level: gamificationResult?.level,
      leveledUp: gamificationResult?.leveledUp,
      newBadges: gamificationResult?.newBadges,
    };
  }

  /**
   * Execute code without saving (for "Run" button)
   * Uses caching to avoid re-executing identical code
   */
  async runCode(
    code: string,
    language: string,
    stdin?: string
  ): Promise<ExecutionResult> {
    const langKey = language.toLowerCase();
    if (!LANGUAGES[langKey]) {
      throw new BadRequestException(
        `Unsupported language: ${language}. Supported: ${Object.keys(LANGUAGES).join(', ')}`
      );
    }

    // Check cache first
    const cached = await this.cacheService.getExecutionResult<ExecutionResult>(code, language, stdin);
    if (cached) {
      this.logger.log(`Cache hit: lang=${LANGUAGES[langKey].name}`);
      return cached;
    }

    this.logger.log(`Run code: lang=${LANGUAGES[langKey].name}, stdin=${stdin ? 'yes' : 'no'}`);

    // Execute code
    const result = await this.codeExecutionService.executeSync(code, language, stdin);

    // Cache the result (only successful executions are cached by the service)
    await this.cacheService.setExecutionResult(code, language, stdin, result);

    return result;
  }

  /**
   * Run quick tests (5 tests) without saving to database
   * Used for "Run Code" button - fast feedback
   */
  async runQuickTests(
    taskIdentifier: string,
    code: string,
    language: string
  ): Promise<{
    status: string;
    testsPassed: number;
    testsTotal: number;
    testCases: TestCaseResult[];
    runtime: string;
    message: string;
  }> {
    // 1. Resolve Task
    let task = await this.prisma.task.findUnique({
      where: { id: taskIdentifier },
    });

    if (!task) {
      task = await this.prisma.task.findUnique({
        where: { slug: taskIdentifier },
      });
    }

    if (!task) {
      throw new NotFoundException(`Task not found: ${taskIdentifier}`);
    }

    // 2. Validate language
    const langConfig = LANGUAGES[language.toLowerCase()];
    if (!langConfig) {
      throw new BadRequestException(
        `Unsupported language: ${language}. Supported: ${Object.keys(LANGUAGES).join(', ')}`
      );
    }

    this.logger.log(
      `Quick test: task=${task.slug}, lang=${langConfig.name}`
    );

    // 3. Execute code with tests (quick mode - only first 5 tests)
    let result;
    if (task.testCode) {
      result = await this.codeExecutionService.executeSyncWithTests(
        code,
        task.testCode,
        language,
        5 // Limit to 5 tests for quick mode
      );
    } else {
      // No test code - just run the user's code
      result = await this.codeExecutionService.executeSync(code, language);
    }

    // 4. Parse test output
    const { testCases, passed: testsPassed, total: testsTotal } = this.parseTestOutput(
      result.stdout,
      result.stderr
    );

    // 5. Determine status
    let status: string;
    if (result.status === 'compileError') {
      status = 'error';
    } else if (testsTotal > 0 && testsPassed < testsTotal) {
      status = 'failed';
    } else if (result.status === 'error' || result.status === 'timeout') {
      status = 'error';
    } else {
      status = 'passed';
    }

    // 6. Format runtime
    const runtime = result.time === '-' ? '-' : `${Math.round(parseFloat(result.time) * 1000)}ms`;

    // 7. Format message for errors
    const message = this.formatMessage(result);

    return {
      status,
      testsPassed,
      testsTotal,
      testCases,
      runtime,
      message,
    };
  }

  /**
   * Get submission by ID
   */
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

  /**
   * Get user's submissions for a task
   */
  async findByUserAndTask(userId: string, taskId: string): Promise<any[]> {
    return this.prisma.submission.findMany({
      where: { userId, taskId },
      orderBy: { createdAt: 'desc' },
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

  /**
   * Get user's recent submissions
   */
  async findRecentByUser(userId: string, limit = 10): Promise<any[]> {
    return this.prisma.submission.findMany({
      where: { userId },
      orderBy: { createdAt: 'desc' },
      take: limit,
      include: { task: { select: { slug: true, title: true } } },
    });
  }

  /**
   * Get execution engine health and queue status
   */
  async getJudgeStatus(): Promise<{
    available: boolean;
    queueReady: boolean;
    queue: { waiting: number; active: number; completed: number; failed: number };
    cache: { connected: boolean; keys?: number };
    languages: string[];
  }> {
    const health = await this.codeExecutionService.checkHealth();
    const queueStats = await this.codeExecutionService.getQueueStats();
    const cacheStats = await this.cacheService.getStats();
    const languages = Object.keys(LANGUAGES);

    return {
      available: health.available,
      queueReady: health.queueReady,
      queue: {
        waiting: queueStats.waiting,
        active: queueStats.active,
        completed: queueStats.completed,
        failed: queueStats.failed,
      },
      cache: cacheStats,
      languages,
    };
  }

  /**
   * Format execution result into readable message
   * Returns concise error information only - the UI handles detailed display
   */
  private formatMessage(result: ExecutionResult): string {
    // For passed submissions, no message needed (UI shows status badge)
    if (result.status === 'passed') {
      return '';
    }

    // For errors, include relevant error information
    if (result.status === 'compileError') {
      return result.compileOutput || 'Compilation failed';
    }

    if (result.status === 'timeout') {
      return 'Time limit exceeded';
    }

    if (result.status === 'error') {
      // Include stderr for runtime errors
      if (result.stderr) {
        return result.stderr;
      }
      return result.message || 'Runtime error';
    }

    // For failed tests, the UI uses testCases for detailed display
    // Just return a simple message
    if (result.status === 'failed') {
      return result.message || 'Tests failed';
    }

    return '';
  }

  /**
   * Get status emoji
   */
  private getStatusEmoji(status: string): string {
    switch (status) {
      case 'passed':
        return 'PASSED';
      case 'failed':
        return 'FAILED';
      case 'timeout':
        return 'TIMEOUT';
      case 'compileError':
        return 'COMPILE_ERROR';
      case 'error':
        return 'ERROR';
      default:
        return 'UNKNOWN';
    }
  }

  /**
   * Parse test output - supports JSON format from our test runners
   */
  private parseTestOutput(stdout: string, stderr: string): {
    testCases: TestCaseResult[];
    passed: number;
    total: number;
  } {
    const output = stdout.trim();

    // Try to parse JSON output from test runner
    try {
      // Find JSON in output (might have other text before/after)
      const jsonMatch = output.match(/\{[\s\S]*"tests"[\s\S]*\}/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
        if (parsed.tests && Array.isArray(parsed.tests)) {
          const testCases: TestCaseResult[] = parsed.tests.map((t: any) => ({
            name: t.name || 'test',
            passed: t.passed || false,
            expectedOutput: t.expected,
            actualOutput: t.output,
            error: t.error,
          }));
          return {
            testCases,
            passed: parsed.passed || 0,
            total: parsed.total || testCases.length,
          };
        }
      }
    } catch {
      // JSON parse failed, try legacy format
    }

    // Legacy format: === RUN / --- PASS/FAIL
    const testCases: TestCaseResult[] = [];
    const runPattern = /=== RUN\s+(\S+)/g;
    const passPattern = /--- PASS:\s*(\S+)/g;
    const failPattern = /--- FAIL:\s*(\S+)/g;

    const fullOutput = stdout + '\n' + stderr;
    const testNames: string[] = [];
    let match;

    while ((match = runPattern.exec(fullOutput)) !== null) {
      if (!testNames.includes(match[1])) {
        testNames.push(match[1]);
      }
    }

    const passedTests = new Set<string>();
    while ((match = passPattern.exec(fullOutput)) !== null) {
      passedTests.add(match[1]);
    }

    const failedTests = new Set<string>();
    while ((match = failPattern.exec(fullOutput)) !== null) {
      failedTests.add(match[1]);
    }

    for (const testName of testNames) {
      const passed = passedTests.has(testName);
      const failed = failedTests.has(testName);

      const testCase: TestCaseResult = {
        name: testName,
        passed: passed && !failed,
      };

      if (failed) {
        const errorMatch = fullOutput.match(new RegExp(
          `--- FAIL:\\s*${testName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}[\\s\\S]*?error:\\s*([^\\n]+)`,
          'i'
        ));
        if (errorMatch) {
          testCase.error = errorMatch[1].trim();
          const expectedActual = testCase.error.match(
            /(?:expected|want)[:\s]+(.+?)(?:,\s*|\s+)(?:got|actual|but was)[:\s]+(.+)/i
          );
          if (expectedActual) {
            testCase.expectedOutput = expectedActual[1].trim();
            testCase.actualOutput = expectedActual[2].trim();
          }
        }
      }

      testCases.push(testCase);
    }

    // Try to get count from RESULT line
    const resultMatch = fullOutput.match(/RESULT:\s*(\d+)\/(\d+)/);
    if (resultMatch) {
      return {
        testCases,
        passed: parseInt(resultMatch[1], 10),
        total: parseInt(resultMatch[2], 10),
      };
    }

    return {
      testCases,
      passed: testCases.filter(t => t.passed).length,
      total: testCases.length,
    };
  }
}
