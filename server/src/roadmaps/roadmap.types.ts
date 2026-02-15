/**
 * Shared types for the roadmaps module
 */

export interface RoadmapPhase {
  id: string;
  title: string;
  description: string;
  colorTheme: string;
  order: number;
  steps: RoadmapStep[];
  progressPercentage: number;
}

export interface RoadmapStep {
  id: string;
  title: string;
  type: 'practice' | 'video' | 'project' | 'reading';
  durationEstimate: string;
  deepLink: string;
  resourceType: 'task' | 'topic' | 'module';
  relatedResourceId: string;
  status: 'available' | 'completed' | 'locked';
}

export interface TaskInfo {
  id: string;
  slug: string;
  title: string;
  difficulty: string;
  estimatedTime: string;
  topicTitle: string;
  moduleTitle: string;
}

export interface TaskWithCourseInfo extends TaskInfo {
  courseSlug: string;
  courseTitle: string;
  courseIcon: string;
}

export interface RoadmapVariantData {
  id: string;
  name: string;
  description: string;
  isRecommended: boolean;
  totalTasks: number;
  estimatedHours: number;
  estimatedMonths: number;
  salaryRange: { min: number; max: number };
  targetRole: string;
  difficulty: 'easy' | 'medium' | 'hard';
  topics: string[];
  sources: {
    courseSlug: string;
    courseName: string;
    icon: string;
    taskCount: number;
    percentage: number;
  }[];
  previewTasks: {
    title: string;
    difficulty: string;
    fromCourse: string;
  }[];
  phases: RoadmapPhase[];
}

// AI response interfaces
export interface AIPhaseResponse {
  title: string;
  description: string;
  taskSlugs: string[];
}

export interface AIVariantResponse {
  id?: string;
  name: string;
  description: string;
  isRecommended?: boolean;
  difficulty: 'easy' | 'medium' | 'hard';
  targetRole: string;
  topics: string[];
  taskSlugs?: string[];
  phases?: AIPhaseResponse[];
}

export interface AIVariantsResponse {
  variants: AIVariantResponse[];
}
