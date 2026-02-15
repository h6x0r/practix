import {
  Injectable,
  BadRequestException,
  ForbiddenException,
  NotFoundException,
  Logger,
} from "@nestjs/common";
import { Prisma } from "@prisma/client";
import { PrismaService } from "../prisma/prisma.service";
import { CodeExecutionService } from "../queue/code-execution.service";
import { CacheService } from "../cache/cache.service";
import { ExecutionResult, LANGUAGES } from "../judge0/judge0.service";
import { AccessControlService } from "../subscriptions/access-control.service";
import { GamificationService } from "../gamification/gamification.service";
import { SecurityValidationService } from "../security/security-validation.service";
import { TestParserService } from "./test-parser.service";
import { ResultFormatterService } from "./result-formatter.service";
import { AiService } from "../ai/ai.service";
import { SubmissionPromptService } from "./submission-prompt.service";
import { SubmissionHistoryService } from "./submission-history.service";
import {
  SubmissionResult,
  PromptSubmissionResult,
  QuickTestResult,
  GamificationReward,
} from "./submission.types";

// Re-exports for backward compatibility
export type { SubmissionResult, PromptSubmissionResult, QuickTestResult };
export type { SubmissionDto, GamificationReward } from "./submission.types";
export { TestCaseResult } from "./test-parser.service";

/**
 * SubmissionsService — facade that orchestrates code submission flow.
 *
 * Delegates to:
 * - SubmissionPromptService: prompt engineering submissions
 * - SubmissionHistoryService: CRUD queries
 * - SecurityValidationService: code security scanning
 * - TestParserService: test output parsing
 * - ResultFormatterService: output formatting
 */
@Injectable()
export class SubmissionsService {
  private readonly logger = new Logger(SubmissionsService.name);

  constructor(
    private readonly prisma: PrismaService,
    private readonly codeExecutionService: CodeExecutionService,
    private readonly cacheService: CacheService,
    private readonly accessControlService: AccessControlService,
    private readonly gamificationService: GamificationService,
    private readonly securityValidation: SecurityValidationService,
    private readonly testParser: TestParserService,
    private readonly resultFormatter: ResultFormatterService,
    private readonly aiService: AiService,
    private readonly promptService: SubmissionPromptService,
    private readonly historyService: SubmissionHistoryService,
  ) {}

  // ============================================
  // Code Submission (Submit button)
  // ============================================

  async create(
    userId: string,
    taskIdentifier: string,
    code: string,
    language: string,
    ip?: string,
  ): Promise<SubmissionResult> {
    await this.securityValidation.validateCode(code, language, { ip, userId });

    const task = await this.findTaskByIdentifier(taskIdentifier);

    const taskAccess = await this.accessControlService.getTaskAccess(
      userId,
      task.id,
    );
    if (!taskAccess.canSubmit) {
      throw new ForbiddenException(
        "This task requires an active subscription. Upgrade to access premium content.",
      );
    }

    const runValidation = await this.cacheService.getRunValidation(
      userId,
      task.id,
    );
    if (!runValidation) {
      throw new ForbiddenException(
        "Please run your code first and pass at least 5 tests before submitting.",
      );
    }

    this.validateLanguage(language);

    const courseId = task.topic?.module?.courseId;
    const queuePriority = courseId
      ? await this.accessControlService.getQueuePriority(userId, courseId)
      : 10;

    this.logger.log(
      `Submission: user=${userId}, task=${task.slug}, lang=${language}, priority=${queuePriority}`,
    );

    const result = await this.executeCode(code, language, task.testCode);
    const testOutput = this.testParser.parseTestOutput(
      result.stdout,
      result.stderr,
    );
    const finalStatus = this.testParser.determineStatus(
      result.status,
      testOutput.passed,
      testOutput.total,
    );
    const score = this.testParser.calculateScore(
      testOutput.passed,
      testOutput.total,
      finalStatus,
    );
    const { runtime, memory } = this.resultFormatter.formatMetrics(result);
    const message = this.resultFormatter.formatMessage(result);

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
        testsPassed: testOutput.total > 0 ? testOutput.passed : null,
        testsTotal: testOutput.total > 0 ? testOutput.total : null,
        testCases:
          testOutput.testCases.length > 0
            ? JSON.parse(JSON.stringify(testOutput.testCases))
            : undefined,
      },
    });

    const gamificationResult = await this.awardXpIfFirstCompletion(
      userId,
      task.id,
      task.slug,
      task.difficulty,
      finalStatus,
    );

    if (finalStatus === "passed") {
      await this.cacheService.clearRunValidation(userId, task.id);
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
      testsPassed: testOutput.total > 0 ? testOutput.passed : undefined,
      testsTotal: testOutput.total > 0 ? testOutput.total : undefined,
      testCases:
        testOutput.testCases.length > 0 ? testOutput.testCases : undefined,
      createdAt: submission.createdAt.toISOString(),
      xpEarned: gamificationResult?.xpEarned,
      totalXp: gamificationResult?.totalXp,
      level: gamificationResult?.level,
      leveledUp: gamificationResult?.leveledUp,
      newBadges: gamificationResult?.newBadges,
    };
  }

  // ============================================
  // Run Code (no save)
  // ============================================

  async runCode(
    code: string,
    language: string,
    stdin?: string,
    ip?: string,
    userId?: string,
  ): Promise<ExecutionResult> {
    this.validateLanguage(language);
    await this.securityValidation.validateCode(code, language, { ip, userId });

    const cached = await this.cacheService.getExecutionResult<ExecutionResult>(
      code,
      language,
      stdin,
      userId,
    );
    if (cached) {
      this.logger.log(`Cache hit: lang=${language}, user=${userId}`);
      return cached;
    }

    this.logger.log(`Run code: lang=${language}, user=${userId}`);
    const result = await this.codeExecutionService.executeSync(
      code,
      language,
      stdin,
    );

    await this.cacheService.setExecutionResult(
      code,
      language,
      stdin,
      result,
      userId,
    );
    return result;
  }

  // ============================================
  // Quick Tests (Run button with tests)
  // ============================================

  async runQuickTests(
    taskIdentifier: string,
    code: string,
    language: string,
    ip?: string,
    userId?: string,
  ): Promise<QuickTestResult> {
    await this.securityValidation.validateCode(code, language, { ip, userId });

    const task = await this.findTaskByIdentifier(taskIdentifier, false);

    if (userId) {
      const taskAccess = await this.accessControlService.getTaskAccess(
        userId,
        task.id,
      );
      if (!taskAccess.canRun) {
        throw new ForbiddenException(
          "This task requires an active subscription. Upgrade to access premium content.",
        );
      }
    }

    this.validateLanguage(language);
    this.logger.log(`Quick test: task=${task.slug}, lang=${language}`);

    const result = await this.executeCode(code, language, task.testCode, 5);
    const testOutput = this.testParser.parseTestOutput(
      result.stdout,
      result.stderr,
    );
    const status = this.testParser.determineStatus(
      result.status,
      testOutput.passed,
      testOutput.total,
    );

    let runValidated = false;
    if (userId && testOutput.passed >= 5) {
      await this.cacheService.setRunValidated(
        userId,
        task.id,
        testOutput.passed,
      );
      runValidated = true;
    }

    const runResult: QuickTestResult = {
      status,
      testsPassed: testOutput.passed,
      testsTotal: testOutput.total,
      testCases: testOutput.testCases,
      runtime: this.resultFormatter.formatRuntime(result.time),
      message: this.resultFormatter.formatMessage(result),
      runValidated,
    };

    if (userId) {
      await this.saveRunResult(userId, task.id, task.slug, runResult, code);
    }

    return runResult;
  }

  // ============================================
  // Delegated methods
  // ============================================

  async submitPrompt(
    userId: string,
    taskIdentifier: string,
    prompt: string,
  ): Promise<PromptSubmissionResult> {
    return this.promptService.submitPrompt(userId, taskIdentifier, prompt);
  }

  async findOne(id: string) {
    return this.historyService.findOne(id);
  }

  async findByUserAndTask(userId: string, taskId: string) {
    return this.historyService.findByUserAndTask(userId, taskId);
  }

  async getRunResult(userId: string, taskIdentifier: string) {
    return this.historyService.getRunResult(userId, taskIdentifier);
  }

  async findRecentByUser(userId: string, limit = 10) {
    return this.historyService.findRecentByUser(userId, limit);
  }

  // ============================================
  // Judge Status
  // ============================================

  async getJudgeStatus() {
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

  // ============================================
  // Private helpers
  // ============================================

  private async findTaskByIdentifier(
    identifier: string,
    includeRelations = true,
  ): Promise<any> {
    const task = await this.prisma.task.findFirst({
      where: {
        OR: [{ id: identifier }, { slug: identifier }],
      },
      include: includeRelations
        ? { topic: { include: { module: { select: { courseId: true } } } } }
        : undefined,
    });

    if (!task) {
      throw new NotFoundException(`Task not found: ${identifier}`);
    }
    return task;
  }

  private validateLanguage(language: string): void {
    const langKey = language.toLowerCase();
    if (!LANGUAGES[langKey]) {
      throw new BadRequestException(
        `Unsupported language: ${language}. Supported: ${Object.keys(LANGUAGES).join(", ")}`,
      );
    }
  }

  private async executeCode(
    code: string,
    language: string,
    testCode?: string | null,
    maxTests?: number,
  ): Promise<ExecutionResult> {
    if (testCode) {
      return this.codeExecutionService.executeSyncWithTests(
        code,
        testCode,
        language,
        maxTests,
      );
    }
    return this.codeExecutionService.executeSync(code, language);
  }

  private async awardXpIfFirstCompletion(
    userId: string,
    taskId: string,
    taskSlug: string,
    difficulty: string,
    status: string,
  ): Promise<GamificationReward | null> {
    if (status !== "passed") return null;

    const xpEarned = this.resultFormatter.getXpForDifficulty(difficulty);

    try {
      await this.prisma.taskCompletion.create({
        data: { userId, taskId, xpAwarded: xpEarned },
      });

      const result = await this.gamificationService.awardTaskXp(
        userId,
        difficulty,
      );
      this.logger.log(
        `XP awarded: user=${userId}, task=${taskSlug}, xp=${result.xpEarned}, level=${result.level}`,
      );
      return result;
    } catch (error: unknown) {
      if (
        error instanceof Prisma.PrismaClientKnownRequestError &&
        error.code === "P2002"
      ) {
        return null;
      }
      throw error;
    }
  }

  private async saveRunResult(
    userId: string,
    taskId: string,
    taskSlug: string,
    runResult: QuickTestResult,
    code: string,
  ): Promise<void> {
    try {
      await this.prisma.runResult.upsert({
        where: { userId_taskId: { userId, taskId } },
        update: {
          status: runResult.status,
          testsPassed: runResult.testsPassed,
          testsTotal: runResult.testsTotal,
          runtime: runResult.runtime,
          message: runResult.message || "",
          testCases: runResult.testCases as any,
          code,
        },
        create: {
          userId,
          taskId,
          status: runResult.status,
          testsPassed: runResult.testsPassed,
          testsTotal: runResult.testsTotal,
          runtime: runResult.runtime,
          message: runResult.message || "",
          testCases: runResult.testCases as any,
          code,
        },
      });
      this.logger.log(`RunResult saved: user=${userId}, task=${taskSlug}`);
    } catch (error: any) {
      this.logger.warn(`Failed to save RunResult: ${error.message}`);
    }
  }
}
