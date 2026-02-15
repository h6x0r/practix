import {
  Injectable,
  NotFoundException,
  BadRequestException,
  ForbiddenException,
  Logger,
} from "@nestjs/common";
import { TaskType } from "@prisma/client";
import { PrismaService } from "../prisma/prisma.service";
import { AccessControlService } from "../subscriptions/access-control.service";
import { AiService } from "../ai/ai.service";
import { PromptSubmissionResult, GamificationReward } from "./submission.types";
import { GamificationService } from "../gamification/gamification.service";
import { ResultFormatterService } from "./result-formatter.service";

/**
 * Handles prompt engineering task submissions.
 * Uses AI-as-Judge for evaluation via AiService.
 */
@Injectable()
export class SubmissionPromptService {
  private readonly logger = new Logger(SubmissionPromptService.name);

  constructor(
    private readonly prisma: PrismaService,
    private readonly accessControlService: AccessControlService,
    private readonly aiService: AiService,
    private readonly gamificationService: GamificationService,
    private readonly resultFormatter: ResultFormatterService,
  ) {}

  async submitPrompt(
    userId: string,
    taskIdentifier: string,
    prompt: string,
  ): Promise<PromptSubmissionResult> {
    const task = await this.findPromptTask(taskIdentifier);

    const taskAccess = await this.accessControlService.getTaskAccess(
      userId,
      task.id,
    );
    if (!taskAccess.canSubmit) {
      throw new ForbiddenException(
        "This task requires an active subscription. Upgrade to access premium content.",
      );
    }

    this.logger.log(`Prompt submission: user=${userId}, task=${task.slug}`);

    const promptConfig = task.promptConfig as {
      testScenarios: Array<{
        input: string;
        expectedCriteria: string[];
        rubric?: string;
      }>;
      judgePrompt: string;
      passingScore: number;
    };

    const evaluation = await this.aiService.evaluatePrompt(
      userId,
      prompt,
      promptConfig,
    );

    const status = evaluation.passed ? "passed" : "failed";
    const score = Math.round(evaluation.score * 10);

    const submission = await this.prisma.submission.create({
      data: {
        userId,
        taskId: task.id,
        code: prompt,
        status,
        score,
        runtime: "0ms",
        memory: null,
        message: evaluation.summary,
        testsPassed: evaluation.scenarioResults.filter((r) => r.passed).length,
        testsTotal: evaluation.scenarioResults.length,
        testCases: evaluation.scenarioResults.map((r) => ({
          name: `Scenario ${r.scenarioIndex + 1}`,
          passed: r.passed,
          expected: r.input.substring(0, 100),
          actual: r.feedback,
        })),
      },
    });

    const gamificationResult = await this.awardXpIfFirstCompletion(
      userId,
      task.id,
      task.slug,
      task.difficulty,
      status,
    );

    return {
      id: submission.id,
      status,
      score,
      message: evaluation.summary,
      createdAt: submission.createdAt.toISOString(),
      scenarioResults: evaluation.scenarioResults,
      summary: evaluation.summary,
      xpEarned: gamificationResult?.xpEarned,
      totalXp: gamificationResult?.totalXp,
      level: gamificationResult?.level,
      leveledUp: gamificationResult?.leveledUp,
      newBadges: gamificationResult?.newBadges,
    };
  }

  private async findPromptTask(identifier: string) {
    const task = await this.prisma.task.findFirst({
      where: {
        OR: [{ id: identifier }, { slug: identifier }],
      },
      include: {
        topic: {
          include: {
            module: { select: { courseId: true } },
          },
        },
      },
    });

    if (!task) {
      throw new NotFoundException(`Task not found: ${identifier}`);
    }
    if (task.taskType !== TaskType.PROMPT) {
      throw new BadRequestException(
        `Task ${task.slug} is not a prompt engineering task`,
      );
    }
    if (!task.promptConfig) {
      throw new BadRequestException(
        `Task ${task.slug} has no prompt configuration`,
      );
    }

    return task;
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
        `XP awarded: user=${userId}, task=${taskSlug}, xp=${result.xpEarned}`,
      );
      return result;
    } catch (error: unknown) {
      if (
        error instanceof Error &&
        "code" in error &&
        (error as any).code === "P2002"
      ) {
        return null;
      }
      throw error;
    }
  }
}
