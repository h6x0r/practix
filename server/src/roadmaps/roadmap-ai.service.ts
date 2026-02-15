import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { GoogleGenAI } from '@google/genai';
import { DEFAULT_AI_MODEL, PHASE_PALETTES, SALARY_RANGES } from './roadmap.config';
import { GenerateRoadmapVariantsDto } from './dto/roadmaps.dto';
import {
  RoadmapVariantData,
  RoadmapPhase,
  TaskWithCourseInfo,
  AIVariantsResponse,
  AIPhaseResponse,
} from './roadmap.types';

@Injectable()
export class RoadmapAIService {
  private readonly logger = new Logger(RoadmapAIService.name);
  genAI: GoogleGenAI | null = null;

  constructor(private configService: ConfigService) {
    const apiKey =
      this.configService.get<string>('GEMINI_API_KEY') ||
      this.configService.get<string>('API_KEY');
    if (apiKey) {
      this.genAI = new GoogleGenAI({ apiKey });
    }
  }

  /**
   * Generate variants using AI
   */
  async generateVariants(
    dto: GenerateRoadmapVariantsDto,
    tasks: TaskWithCourseInfo[],
  ): Promise<RoadmapVariantData[]> {
    const prompt = this.buildPrompt(dto, tasks);

    const aiModel =
      this.configService.get<string>('AI_MODEL_NAME') || DEFAULT_AI_MODEL;
    const response = await this.genAI!.models.generateContent({
      model: aiModel,
      contents: prompt,
    });

    const text = response.text || '';
    const jsonMatch = text.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      throw new Error('No valid JSON in AI response');
    }

    const parsed = JSON.parse(jsonMatch[0]);
    const taskMap = new Map(tasks.map((t) => [t.slug, t]));

    return this.convertResponse(parsed, taskMap, dto);
  }

  /**
   * Build prompt for generating multiple variants
   */
  private buildPrompt(
    dto: GenerateRoadmapVariantsDto,
    tasks: TaskWithCourseInfo[],
  ): string {
    const goalDescriptions: Record<string, string> = {
      'first-job': 'Looking for first IT job, needs interview prep and portfolio',
      senior: 'Wants promotion to senior/lead, needs architecture and best practices',
      startup: 'Building a startup/product, needs full-stack and MVP skills',
      'master-skill': 'Deep expertise in specific technology',
    };

    const sampledTasks = tasks.slice(0, 300);
    const courseGroups = new Map<string, number>();
    for (const task of tasks) {
      courseGroups.set(task.courseTitle, (courseGroups.get(task.courseTitle) || 0) + 1);
    }

    return `
You are creating personalized learning roadmap VARIANTS for a developer.

USER PROFILE:
- Known Languages: ${dto.knownLanguages.join(', ') || 'None specified'}
- Years of Experience: ${dto.yearsOfExperience}
- Interested Areas: ${dto.interests.join(', ')}
- Main Goal: ${dto.goal} (${goalDescriptions[dto.goal] || dto.goal})
- Available Time: ${dto.hoursPerWeek} hours/week
- Target Timeline: ${dto.targetMonths} months

AVAILABLE COURSES (${courseGroups.size} courses, ${tasks.length} total tasks):
${Array.from(courseGroups.entries())
  .map(([name, count]) => `- ${name}: ${count} tasks`)
  .join('\n')}

SAMPLE TASKS (${sampledTasks.length} of ${tasks.length}):
${sampledTasks.map((t) => `- ${t.slug}: "${t.title}" [${t.difficulty}] (${t.estimatedTime}) from ${t.courseTitle}`).join('\n')}

Generate 3-5 DIFFERENT roadmap variants. Each variant should be a distinct learning path:

1. "Quick Start" - Minimal path to achieve goal fast (fewer tasks, essential only)
2. "Balanced" - Recommended path with good coverage (medium tasks)
3. "Deep Dive" - Comprehensive path for thorough learning (more tasks)
4. (Optional) "Specialized" - Focus on specific area from interests
5. (Optional) "Interview Ready" - If goal is job, focus on interview topics

Each variant should:
- Mix tasks from DIFFERENT courses based on interests
- Order tasks by logical progression (fundamentals → advanced)
- Consider user's experience level
- Estimate realistic completion time

Return ONLY valid JSON (no markdown):
{
  "variants": [
    {
      "id": "quick-start",
      "name": "Quick Start",
      "description": "Fast track to your goal - essential skills only",
      "targetRole": "Junior Go Developer",
      "difficulty": "easy",
      "taskSlugs": ["task-slug-1", "task-slug-2", ...],
      "topics": ["Go Basics", "REST APIs", "SQL Fundamentals"],
      "phases": [
        {
          "title": "Phase 1: Foundations",
          "description": "Core language fundamentals",
          "taskSlugs": ["slug1", "slug2"]
        }
      ]
    }
  ]
}

IMPORTANT:
- Use EXACT task slugs from the list above
- Each variant should have 3-6 phases
- Quick variant: 30-60 tasks, Balanced: 80-150, Deep: 150-250
- Ensure good variety - don't repeat the same tasks in different variants
`;
  }

  /**
   * Convert AI response to structured variants
   */
  private convertResponse(
    parsed: AIVariantsResponse,
    taskMap: Map<string, TaskWithCourseInfo>,
    dto: GenerateRoadmapVariantsDto,
  ): RoadmapVariantData[] {
    const variants: RoadmapVariantData[] = [];

    for (const variant of parsed.variants || []) {
      const variantTasks = this.collectVariantTasks(variant, taskMap);
      if (variantTasks.length === 0) continue;

      const totalTasks = variantTasks.length;
      const estimatedHours = calculateTotalHours(variantTasks);
      const estimatedMonths = Math.ceil(estimatedHours / (dto.hoursPerWeek * 4));

      const salaryKey =
        variant.difficulty === 'easy'
          ? 'junior-plus'
          : variant.difficulty === 'hard'
            ? 'senior'
            : 'middle';

      const phases = this.buildPhasesFromAI(variant.phases || [], taskMap);
      const sources = buildSourcesFromTasks(variantTasks, totalTasks);
      const previewTasks = variantTasks.slice(0, 5).map((t) => ({
        title: t.title,
        difficulty: t.difficulty,
        fromCourse: t.courseTitle,
      }));

      const variantId = variant.id || `variant_${variants.length + 1}`;
      variants.push({
        id: variantId,
        name: variant.name,
        description: variant.description,
        isRecommended:
          variantId === 'balanced' ||
          variant.name?.toLowerCase().includes('recommend'),
        totalTasks,
        estimatedHours,
        estimatedMonths,
        salaryRange: SALARY_RANGES[salaryKey] || SALARY_RANGES['middle'],
        targetRole: variant.targetRole || 'Developer',
        difficulty: variant.difficulty || 'medium',
        topics: variant.topics || [],
        sources,
        previewTasks,
        phases,
      });
    }

    return variants;
  }

  /**
   * Collect all valid tasks for a variant from phases + top-level slugs
   */
  private collectVariantTasks(
    variant: { phases?: AIPhaseResponse[]; taskSlugs?: string[] },
    taskMap: Map<string, TaskWithCourseInfo>,
  ): TaskWithCourseInfo[] {
    const tasks: TaskWithCourseInfo[] = [];
    const seen = new Set<string>();

    for (const phase of variant.phases || []) {
      for (const slug of phase.taskSlugs || []) {
        if (taskMap.has(slug) && !seen.has(slug)) {
          tasks.push(taskMap.get(slug)!);
          seen.add(slug);
        }
      }
    }

    for (const slug of variant.taskSlugs || []) {
      if (taskMap.has(slug) && !seen.has(slug)) {
        tasks.push(taskMap.get(slug)!);
        seen.add(slug);
      }
    }

    return tasks;
  }

  /**
   * Build phases from AI response
   */
  private buildPhasesFromAI(
    aiPhases: AIPhaseResponse[],
    taskMap: Map<string, TaskWithCourseInfo>,
  ): RoadmapPhase[] {
    return aiPhases
      .map((phase, index) => {
        const phaseTasks = (phase.taskSlugs || [])
          .map((slug) => taskMap.get(slug))
          .filter(Boolean) as TaskWithCourseInfo[];

        return {
          id: `phase_${index + 1}`,
          title: phase.title,
          description: phase.description,
          colorTheme: PHASE_PALETTES[index % PHASE_PALETTES.length],
          order: index + 1,
          steps: phaseTasks.map((task) => ({
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
        };
      })
      .filter((p) => p.steps.length > 0);
  }
}

// ============ Shared utility functions ============

/**
 * Calculate total hours from tasks
 */
export function calculateTotalHours(tasks: TaskWithCourseInfo[]): number {
  let totalMinutes = 0;
  for (const task of tasks) {
    totalMinutes += parseTimeToMinutes(task.estimatedTime);
  }
  return Math.round(totalMinutes / 60);
}

/**
 * Parse time string to minutes
 */
export function parseTimeToMinutes(time: string): number {
  if (!time) return 15;
  let total = 0;
  const hourMatch = time.match(/(\d+(?:\.\d+)?)\s*h/i);
  const minMatch = time.match(/(\d+)(?:-\d+)?\s*m/i);
  if (hourMatch) total += parseFloat(hourMatch[1]) * 60;
  if (minMatch) total += parseInt(minMatch[1]);
  return total || 15;
}

/**
 * Build sources array from tasks grouped by course
 */
export function buildSourcesFromTasks(
  tasks: TaskWithCourseInfo[],
  totalTasks: number,
) {
  const sourceMap = new Map<string, { count: number; title: string; icon: string }>();
  for (const task of tasks) {
    const existing = sourceMap.get(task.courseSlug) || {
      count: 0,
      title: task.courseTitle,
      icon: task.courseIcon,
    };
    existing.count++;
    sourceMap.set(task.courseSlug, existing);
  }

  return Array.from(sourceMap.entries())
    .map(([slug, data]) => ({
      courseSlug: slug,
      courseName: data.title,
      icon: data.icon,
      taskCount: data.count,
      percentage: Math.round((data.count / totalTasks) * 100),
    }))
    .sort((a, b) => b.taskCount - a.taskCount);
}
