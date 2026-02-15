import { Injectable } from '@nestjs/common';
import { PHASE_PALETTES, SALARY_RANGES } from './roadmap.config';
import { GenerateRoadmapVariantsDto } from './dto/roadmaps.dto';
import { RoadmapVariantData, RoadmapPhase, TaskWithCourseInfo } from './roadmap.types';
import { calculateTotalHours, buildSourcesFromTasks } from './roadmap-ai.service';

@Injectable()
export class RoadmapVariantsService {
  /**
   * Fallback variant generation (non-AI)
   */
  generateFallback(
    dto: GenerateRoadmapVariantsDto,
    tasks: TaskWithCourseInfo[],
  ): RoadmapVariantData[] {
    const sortedTasks = this.sortByDifficulty(tasks);
    const relevantTasks = this.filterByInterests(sortedTasks, dto.interests);

    const variants: RoadmapVariantData[] = [
      this.buildFromTasks(
        'quick-start', 'Quick Start',
        'Fast track to your goal with essential skills',
        relevantTasks.slice(0, 45), dto, 'easy', 'junior-plus',
      ),
      this.buildFromTasks(
        'balanced', 'Balanced Path',
        'Recommended path with comprehensive coverage',
        relevantTasks.slice(0, 100), dto, 'medium', 'middle',
      ),
      this.buildFromTasks(
        'deep-dive', 'Deep Dive',
        'Thorough learning path for complete mastery',
        relevantTasks.slice(0, 180), dto, 'hard', 'senior',
      ),
    ];

    variants[1].isRecommended = true;
    return variants;
  }

  /**
   * Build a single variant from a list of tasks
   */
  private buildFromTasks(
    id: string,
    name: string,
    description: string,
    tasks: TaskWithCourseInfo[],
    dto: GenerateRoadmapVariantsDto,
    difficulty: 'easy' | 'medium' | 'hard',
    salaryKey: string,
  ): RoadmapVariantData {
    const totalTasks = tasks.length;
    const estimatedHours = calculateTotalHours(tasks);
    const estimatedMonths = Math.ceil(estimatedHours / (dto.hoursPerWeek * 4));
    const salaryRange = SALARY_RANGES[salaryKey] || SALARY_RANGES['middle'];

    const phases = this.buildPhasesFromModules(tasks);
    const sources = buildSourcesFromTasks(tasks, totalTasks);
    const topics = this.extractTopics(tasks);

    const targetRoles: Record<string, string> = {
      easy: 'Junior Developer',
      medium: 'Middle Developer',
      hard: 'Senior Developer',
    };

    return {
      id,
      name,
      description,
      isRecommended: false,
      totalTasks,
      estimatedHours,
      estimatedMonths,
      salaryRange,
      targetRole: targetRoles[difficulty] || 'Developer',
      difficulty,
      topics,
      sources,
      previewTasks: tasks.slice(0, 5).map((t) => ({
        title: t.title,
        difficulty: t.difficulty,
        fromCourse: t.courseTitle,
      })),
      phases,
    };
  }

  private sortByDifficulty(tasks: TaskWithCourseInfo[]): TaskWithCourseInfo[] {
    const order = { easy: 0, medium: 1, hard: 2 };
    return [...tasks].sort(
      (a, b) =>
        (order[a.difficulty as keyof typeof order] || 1) -
        (order[b.difficulty as keyof typeof order] || 1),
    );
  }

  private filterByInterests(
    tasks: TaskWithCourseInfo[],
    interests: string[],
  ): TaskWithCourseInfo[] {
    if (interests.length === 0) return tasks;

    const patterns = interests.map((i) => new RegExp(i, 'i'));
    const filtered = tasks.filter((t) =>
      patterns.some(
        (p) => p.test(t.courseSlug) || p.test(t.courseTitle) || p.test(t.moduleTitle),
      ),
    );

    return filtered.length < 30 ? tasks : filtered;
  }

  private buildPhasesFromModules(tasks: TaskWithCourseInfo[]): RoadmapPhase[] {
    const moduleGroups = new Map<string, TaskWithCourseInfo[]>();
    for (const task of tasks) {
      if (!moduleGroups.has(task.moduleTitle)) moduleGroups.set(task.moduleTitle, []);
      moduleGroups.get(task.moduleTitle)!.push(task);
    }

    const phases: RoadmapPhase[] = [];
    let index = 0;
    for (const [moduleTitle, moduleTasks] of moduleGroups) {
      if (index >= 6) break;

      phases.push({
        id: `phase_${index + 1}`,
        title: moduleTitle,
        description: `Master ${moduleTitle.toLowerCase()} concepts`,
        colorTheme: PHASE_PALETTES[index % PHASE_PALETTES.length],
        order: index + 1,
        steps: moduleTasks.slice(0, 12).map((task) => ({
          id: `step_${task.id}`,
          title: task.title,
          type: 'practice' as const,
          durationEstimate: task.estimatedTime,
          deepLink: `/task/${task.slug}`,
          resourceType: 'task' as const,
          relatedResourceId: task.id,
          status: 'available' as const,
        })),
        progressPercentage: 0,
      });
      index++;
    }

    return phases;
  }

  private extractTopics(tasks: TaskWithCourseInfo[]): string[] {
    const modules = new Set<string>();
    for (const task of tasks) {
      modules.add(task.moduleTitle);
    }
    return Array.from(modules).slice(0, 8);
  }
}
