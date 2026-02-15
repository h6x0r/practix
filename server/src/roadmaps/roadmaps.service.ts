import { Injectable, ForbiddenException, Logger } from "@nestjs/common";
import { PrismaService } from "../prisma/prisma.service";
import { CacheService } from "../cache/cache.service";
import { UserRoadmap } from "@prisma/client";
import {
  GenerateRoadmapVariantsDto,
  SelectRoadmapVariantDto,
} from "./dto/roadmaps.dto";
import { VARIANTS_CACHE_TTL, PHASE_PALETTES } from "./roadmap.config";
import { RoadmapAIService } from "./roadmap-ai.service";
import { RoadmapVariantsService } from "./roadmap-variants.service";
import {
  RoadmapPhase,
  RoadmapVariantData,
  TaskWithCourseInfo,
} from "./roadmap.types";
import { COURSE_ICONS } from "./roadmap.config";

// Re-export types for backward compatibility
export type {
  RoadmapPhase,
  RoadmapStep,
  RoadmapVariantData,
} from "./roadmap.types";

@Injectable()
export class RoadmapsService {
  private readonly logger = new Logger(RoadmapsService.name);

  constructor(
    private prisma: PrismaService,
    private cacheService: CacheService,
    private aiService: RoadmapAIService,
    private variantsService: RoadmapVariantsService,
  ) {}

  // ============ Generation checks ============

  async canGenerateRoadmap(userId: string): Promise<{
    canGenerate: boolean;
    reason?: string;
    isPremium: boolean;
    generationCount: number;
  }> {
    const user = await this.prisma.user.findUnique({
      where: { id: userId },
      select: { roadmapGenerations: true },
    });

    if (!user) {
      return {
        canGenerate: false,
        reason: "user_not_found",
        isPremium: false,
        generationCount: 0,
      };
    }

    const activeSubscription = await this.prisma.subscription.findFirst({
      where: { userId, status: "active", endDate: { gt: new Date() } },
    });
    const isPremium = !!activeSubscription;

    if (isPremium) {
      return {
        canGenerate: true,
        isPremium: true,
        generationCount: user.roadmapGenerations,
      };
    }

    if (user.roadmapGenerations === 0) {
      return { canGenerate: true, isPremium: false, generationCount: 0 };
    }

    return {
      canGenerate: false,
      reason: "free_limit_reached",
      isPremium: false,
      generationCount: user.roadmapGenerations,
    };
  }

  // ============ CRUD operations ============

  async getUserRoadmap(userId: string) {
    const roadmap = await this.prisma.userRoadmap.findFirst({
      where: { userId },
      orderBy: { updatedAt: "desc" },
    });

    if (!roadmap) return null;

    const canGenStatus = await this.canGenerateRoadmap(userId);
    const hydratedRoadmap = await this.hydrateRoadmap(roadmap, userId);

    return {
      ...hydratedRoadmap,
      canRegenerate: canGenStatus.canGenerate,
      isPremium: canGenStatus.isPremium,
      generationCount: canGenStatus.generationCount,
    };
  }

  async deleteRoadmap(userId: string) {
    const roadmap = await this.prisma.userRoadmap.findFirst({
      where: { userId },
    });
    if (roadmap) {
      await this.prisma.userRoadmap.delete({ where: { id: roadmap.id } });
    }
    await this.clearUserVariants(userId);
    return { success: true };
  }

  async getTemplates() {
    return [
      {
        id: "backend-go",
        title: "Go Backend",
        description: "APIs, microservices, concurrency",
        icon: "🐹",
        gradient: "from-cyan-400 to-blue-500",
      },
      {
        id: "backend-java",
        title: "Java Backend",
        description: "Enterprise, Spring, production",
        icon: "☕",
        gradient: "from-orange-400 to-red-500",
      },
      {
        id: "python-data",
        title: "Python & Data",
        description: "ML basics, data analysis, algorithms",
        icon: "🐍",
        gradient: "from-green-400 to-emerald-500",
      },
      {
        id: "ai-ml",
        title: "AI & Machine Learning",
        description: "Deep learning, LLMs, inference",
        icon: "🤖",
        gradient: "from-purple-400 to-pink-500",
      },
      {
        id: "software-design",
        title: "Software Design",
        description: "SOLID, patterns, architecture",
        icon: "🏗️",
        gradient: "from-amber-400 to-orange-500",
      },
      {
        id: "algorithms",
        title: "Algorithms & DS",
        description: "Problem solving, interviews",
        icon: "🧮",
        gradient: "from-indigo-400 to-purple-500",
      },
      {
        id: "fullstack",
        title: "Fullstack",
        description: "Combined frontend + backend",
        icon: "🌐",
        gradient: "from-teal-400 to-cyan-500",
      },
    ];
  }

  // ============ Variant generation ============

  async generateRoadmapVariants(
    userId: string,
    dto: GenerateRoadmapVariantsDto,
  ) {
    const canGenStatus = await this.canGenerateRoadmap(userId);
    if (!canGenStatus.canGenerate) {
      throw new ForbiddenException(
        "Regeneration requires Premium subscription. Upgrade to create unlimited personalized roadmaps.",
      );
    }

    const allTasks = await this.getAllTasksWithCourseInfo();
    const completedTaskIds = await this.getUserCompletedTasks(userId);
    const availableTasks = allTasks.filter((t) => !completedTaskIds.has(t.id));

    let variants: RoadmapVariantData[];

    if (this.aiService.genAI && availableTasks.length > 0) {
      try {
        variants = await this.aiService.generateVariants(dto, availableTasks);
      } catch (error) {
        this.logger.warn("AI variant generation failed, using fallback", error);
        variants = this.variantsService.generateFallback(dto, availableTasks);
      }
    } else {
      variants = this.variantsService.generateFallback(dto, availableTasks);
    }

    await this.cacheService.set(
      this.getVariantsCacheKey(userId),
      variants,
      VARIANTS_CACHE_TTL,
    );
    this.logger.log(`Cached ${variants.length} variants for user ${userId}`);

    return {
      variants,
      canRegenerate: canGenStatus.canGenerate,
      isPremium: canGenStatus.isPremium,
      generationCount: canGenStatus.generationCount,
    };
  }

  async selectRoadmapVariant(userId: string, dto: SelectRoadmapVariantDto) {
    const canGenStatus = await this.canGenerateRoadmap(userId);
    if (!canGenStatus.canGenerate) {
      throw new ForbiddenException(
        "Regeneration requires Premium subscription.",
      );
    }

    const phases: RoadmapPhase[] = dto.phases || [];

    const roadmap = await this.prisma.userRoadmap.upsert({
      where: { id: (await this.findExistingRoadmapId(userId)) || "new" },
      create: {
        userId,
        role: dto.variantId,
        level: dto.difficulty,
        title: dto.name,
        phases: phases as unknown as any,
        totalProgress: 0,
      },
      update: {
        role: dto.variantId,
        level: dto.difficulty,
        title: dto.name,
        phases: phases as unknown as any,
        totalProgress: 0,
        updatedAt: new Date(),
      },
    });

    await this.prisma.user.update({
      where: { id: userId },
      data: {
        roadmapGenerations: { increment: 1 },
        lastRoadmapGeneration: new Date(),
      },
    });

    await this.clearUserVariants(userId);

    const hydratedRoadmap = await this.hydrateRoadmap(roadmap, userId);
    const newStatus = await this.canGenerateRoadmap(userId);

    return {
      ...hydratedRoadmap,
      canRegenerate: newStatus.canGenerate,
      isPremium: newStatus.isPremium,
      generationCount: newStatus.generationCount,
    };
  }

  async getUserVariants(userId: string) {
    const canGenStatus = await this.canGenerateRoadmap(userId);
    const cachedVariants = await this.cacheService.get<RoadmapVariantData[]>(
      this.getVariantsCacheKey(userId),
    );

    return {
      variants: cachedVariants || [],
      canRegenerate: canGenStatus.canGenerate,
      isPremium: canGenStatus.isPremium,
      generationCount: canGenStatus.generationCount,
    };
  }

  // ============ Private helpers ============

  private getVariantsCacheKey(userId: string): string {
    return `roadmap:variants:${userId}`;
  }

  private async clearUserVariants(userId: string): Promise<void> {
    await this.cacheService.delete(this.getVariantsCacheKey(userId));
    this.logger.log(`Cleared cached variants for user ${userId}`);
  }

  private async getUserCompletedTasks(userId: string): Promise<Set<string>> {
    const completedSubmissions = await this.prisma.submission.findMany({
      where: { userId, status: "passed" },
      select: { taskId: true },
    });
    return new Set(completedSubmissions.map((s) => s.taskId));
  }

  private async findExistingRoadmapId(userId: string): Promise<string | null> {
    const existing = await this.prisma.userRoadmap.findFirst({
      where: { userId },
      select: { id: true },
    });
    return existing?.id || null;
  }

  private async getAllTasksWithCourseInfo(): Promise<TaskWithCourseInfo[]> {
    const courses = await this.prisma.course.findMany({
      include: {
        modules: {
          include: {
            topics: {
              include: {
                tasks: {
                  select: {
                    id: true,
                    slug: true,
                    title: true,
                    difficulty: true,
                    estimatedTime: true,
                  },
                },
              },
            },
          },
          orderBy: { order: "asc" },
        },
      },
    });

    const tasks: TaskWithCourseInfo[] = [];
    for (const course of courses) {
      for (const module of course.modules) {
        for (const topic of module.topics) {
          for (const task of topic.tasks) {
            tasks.push({
              id: task.id,
              slug: task.slug,
              title: task.title,
              difficulty: task.difficulty,
              estimatedTime: task.estimatedTime,
              topicTitle: topic.title,
              moduleTitle: module.title,
              courseSlug: course.slug,
              courseTitle: course.title,
              courseIcon: COURSE_ICONS[course.slug] || "📚",
            });
          }
        }
      }
    }

    return tasks;
  }

  private getRoleTitle(role: string): string {
    const titles: Record<string, string> = {
      "backend-go": "Go Backend Developer",
      "backend-java": "Java Backend Developer",
      "python-data": "Python & Data Scientist",
      "ai-ml": "AI/ML Engineer",
      "software-design": "Software Architect",
      algorithms: "Algorithm Specialist",
      fullstack: "Fullstack Developer",
    };
    return titles[role] || "Developer";
  }

  private async hydrateRoadmap(roadmap: UserRoadmap, userId: string) {
    const completedSubmissions = await this.prisma.submission.findMany({
      where: { userId, status: "passed" },
      select: { taskId: true },
    });
    const completedTaskIds = new Set(completedSubmissions.map((s) => s.taskId));

    const allTasks = await this.prisma.task.findMany({
      select: { id: true, slug: true },
    });
    const taskSlugToId = new Map(allTasks.map((t) => [t.slug, t.id]));

    const phases = (roadmap.phases as unknown as RoadmapPhase[]).map(
      (phase, index) => {
        let completedCount = 0;

        const hydratedSteps = phase.steps.map((step) => {
          let isCompleted = false;
          if (step.resourceType === "task") {
            const taskId =
              taskSlugToId.get(step.relatedResourceId) ||
              step.relatedResourceId;
            isCompleted = completedTaskIds.has(taskId);
          }
          if (isCompleted) completedCount++;
          return { ...step, status: isCompleted ? "completed" : "available" };
        });

        return {
          ...phase,
          colorTheme: PHASE_PALETTES[index % PHASE_PALETTES.length],
          steps: hydratedSteps,
          progressPercentage:
            phase.steps.length > 0
              ? (completedCount / phase.steps.length) * 100
              : 0,
        };
      },
    );

    const totalSteps = phases.reduce((sum, p) => sum + p.steps.length, 0);
    const completedSteps = phases.reduce(
      (sum, p) => sum + p.steps.filter((s) => s.status === "completed").length,
      0,
    );

    return {
      id: roadmap.id,
      userId: roadmap.userId,
      role: roadmap.role,
      roleTitle: this.getRoleTitle(roadmap.role),
      level: roadmap.level,
      targetLevel: roadmap.level,
      title: roadmap.title,
      phases,
      totalProgress: totalSteps > 0 ? (completedSteps / totalSteps) * 100 : 0,
      createdAt: roadmap.createdAt,
      updatedAt: roadmap.updatedAt,
    };
  }
}
