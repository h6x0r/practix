import { Injectable, NotFoundException, Logger } from "@nestjs/common";
import { PrismaService } from "../prisma/prisma.service";
import { CacheService } from "../cache/cache.service";
import { Prisma } from "@prisma/client";

// Cache keys
const CACHE_KEYS = {
  COURSES_LIST: "courses:list",
  COURSE_STRUCTURE: (slug: string) => `courses:structure:${slug}`,
};

// 1 day TTL in seconds
const CACHE_TTL = 86400;

@Injectable()
export class CoursesService {
  private readonly logger = new Logger(CoursesService.name);

  constructor(
    private prisma: PrismaService,
    private cache: CacheService,
  ) {}

  async findAll(userId?: string) {
    try {
      // Try to get base courses from cache (without user progress)
      let baseCourses = await this.cache.get<any[]>(CACHE_KEYS.COURSES_LIST);

      if (!baseCourses) {
        this.logger.log("Cache MISS for courses list, fetching from DB");

        // Use _count to avoid N+1 query - only fetch counts, not all task data
        const courses = await this.prisma.course.findMany({
          orderBy: { order: "asc" },
          include: {
            modules: {
              include: {
                topics: {
                  include: {
                    _count: { select: { tasks: true } },
                    // Only fetch first 4 topics for sample preview
                    tasks: {
                      take: 1,
                      select: { id: true },
                    },
                  },
                },
              },
            },
          },
        });

        // Transform to cacheable format (without user-specific data)
        baseCourses = courses.map((c) => {
          let totalTasks = 0;
          c.modules.forEach((m) =>
            m.topics.forEach((t) => (totalTasks += t._count.tasks)),
          );

          // Get sample topics for preview (first 3-4 topics from different modules)
          const sampleTopics: {
            title: string;
            translations?: Prisma.JsonValue;
          }[] = [];
          for (const m of c.modules) {
            for (const t of m.topics) {
              if (sampleTopics.length < 4) {
                sampleTopics.push({
                  title: t.title,
                  translations: t.translations,
                });
              }
            }
            if (sampleTopics.length >= 4) break;
          }

          return {
            id: c.slug,
            slug: c.slug,
            uuid: c.id,
            title: c.title,
            description: c.description,
            category: c.category,
            icon: c.icon,
            estimatedTime: c.estimatedTime,
            totalModules: c.modules.length,
            totalTasks,
            translations: c.translations,
            sampleTopics,
          };
        });

        // Cache the base courses (without progress)
        await this.cache.set(CACHE_KEYS.COURSES_LIST, baseCourses, CACHE_TTL);
        this.logger.log(
          `Cached ${baseCourses.length} courses for ${CACHE_TTL}s`,
        );
      } else {
        this.logger.debug("Cache HIT for courses list");
      }

      // If no user, return courses with 0 progress
      if (!userId) {
        return baseCourses.map((c) => ({ ...c, progress: 0 }));
      }

      // Fetch user's progress (this is per-user, not cached)
      const passedSubmissions = await this.prisma.submission.findMany({
        where: {
          userId,
          status: "passed",
        },
        select: {
          taskId: true,
          task: {
            select: {
              topic: {
                select: {
                  module: {
                    select: {
                      courseId: true,
                    },
                  },
                },
              },
            },
          },
        },
        distinct: ["taskId"],
      });

      // Group submissions by courseId
      const completedTasksByCourse = new Map<string, number>();
      passedSubmissions.forEach((sub) => {
        const courseId = sub.task.topic?.module.courseId;
        if (courseId) {
          completedTasksByCourse.set(
            courseId,
            (completedTasksByCourse.get(courseId) || 0) + 1,
          );
        }
      });

      // Add progress to cached courses
      return baseCourses.map((c) => {
        const completed = completedTasksByCourse.get(c.uuid) || 0;
        const progress =
          c.totalTasks === 0 ? 0 : Math.round((completed / c.totalTasks) * 100);
        return { ...c, progress };
      });
    } catch (error) {
      this.logger.error(
        "Error in findAll",
        error instanceof Error ? error.stack : String(error),
      );
      throw error;
    }
  }

  async findOne(slug: string, userId?: string) {
    const course = await this.prisma.course.findUnique({
      where: { slug },
    });
    this.logger.debug(`findOne: slug=${slug}, found=${!!course}`);

    if (!course) {
      // Fallback for ID lookup if slug fails
      return this.prisma.course.findUnique({ where: { id: slug } });
    }

    // We reuse findAll logic for single entity hydration if needed,
    // or just return the metadata matching frontend type
    const list = await this.findAll(userId);
    return list.find((c) => c.id === slug);
  }

  async getStructure(slug: string, userId?: string) {
    // Try to get structure from cache
    const cacheKey = CACHE_KEYS.COURSE_STRUCTURE(slug);
    let baseStructure = await this.cache.get<any>(cacheKey);

    let courseId: string;

    if (!baseStructure) {
      this.logger.log(`Cache MISS for course structure: ${slug}`);

      const course = await this.prisma.course.findUnique({
        where: { slug },
        include: {
          modules: {
            orderBy: { order: "asc" },
            include: {
              topics: {
                orderBy: { order: "asc" },
                include: {
                  tasks: {
                    orderBy: { order: "asc" },
                    select: {
                      id: true,
                      slug: true,
                      title: true,
                      difficulty: true,
                      estimatedTime: true,
                      isPremium: true,
                      translations: true,
                    },
                  },
                },
              },
            },
          },
        },
      });

      if (!course) throw new NotFoundException("Course not found");

      courseId = course.id;

      // Transform to cacheable format
      baseStructure = {
        courseId: course.id,
        modules: course.modules.map((m) => ({
          id: m.id,
          title: m.title,
          description: m.description,
          section: m.section || "core",
          estimatedTime: m.estimatedTime,
          translations: m.translations,
          topics: m.topics.map((t) => ({
            id: t.id,
            title: t.title,
            description: t.description,
            difficulty: t.difficulty,
            estimatedTime: t.estimatedTime,
            translations: t.translations,
            tasks: t.tasks.map((task) => ({
              id: task.id,
              slug: task.slug,
              title: task.title,
              difficulty: task.difficulty,
              estimatedTime: task.estimatedTime,
              isPremium: task.isPremium,
              translations: task.translations,
            })),
          })),
        })),
      };

      // Cache the structure
      await this.cache.set(cacheKey, baseStructure, CACHE_TTL);
      this.logger.log(`Cached course structure for ${slug}`);
    } else {
      this.logger.debug(`Cache HIT for course structure: ${slug}`);
      courseId = baseStructure.courseId;
    }

    // If no user, return with all tasks as pending
    if (!userId) {
      return baseStructure.modules.map((m: any) => ({
        ...m,
        topics: m.topics.map((t: any) => ({
          ...t,
          tasks: t.tasks.map((task: any) => ({ ...task, status: "pending" })),
        })),
      }));
    }

    // Get completion status for this user (not cached)
    const subs = await this.prisma.submission.findMany({
      where: {
        userId,
        status: "passed",
        task: { topic: { module: { courseId } } },
      },
      select: { taskId: true },
    });
    const completedTaskIds = new Set(subs.map((s) => s.taskId));

    // Add completion status
    return baseStructure.modules.map((m: any) => ({
      ...m,
      topics: m.topics.map((t: any) => ({
        ...t,
        tasks: t.tasks.map((task: any) => ({
          ...task,
          status: completedTaskIds.has(task.id) ? "completed" : "pending",
        })),
      })),
    }));
  }

  /**
   * Invalidate all course caches (call after seed/update)
   */
  async invalidateCache(): Promise<{ deleted: number }> {
    this.logger.log("Invalidating all course caches");
    const deleted = await this.cache.deleteByPattern("courses:*");
    return { deleted };
  }
}
