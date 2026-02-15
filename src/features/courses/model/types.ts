import { Task, Translations } from "../../tasks/model/types";

export type CourseCategory = "language" | "cs" | "interview";

export interface SampleTopic {
  title: string;
  translations?: Translations;
}

export interface Course {
  id: string;
  slug: string;
  category: CourseCategory;
  title: string;
  description: string;
  icon: string; // Emoji or icon name (URLs are fine from backend)
  progress: number;
  totalModules: number;
  estimatedTime: string; // e.g. "56h"
  translations?: Translations; // Multi-language support
  sampleTopics?: SampleTopic[]; // Preview topics for course cards
}

export interface Topic {
  id: string;
  title: string;
  description: string;
  difficulty: "easy" | "medium" | "hard";
  estimatedTime: string; // e.g. "2h"
  tasks: Task[]; // Hierarchy: Course -> Topic -> Task
  translations?: Translations; // Multi-language support
}

export interface CourseModule {
  id: string;
  title: string;
  description: string;
  section?: "core" | "frameworks" | "interview" | "projects"; // New categorization
  estimatedTime: string; // Calculated from sum of task times during seed
  topics: Topic[];
  translations?: Translations; // Multi-language support
}

export interface Track {
  id: string;
  courseId: string; // Link to parent course
  title: string;
  description: string;
  progress: number; // 0-100
  totalTasks: number;
  completedTasks: number;
}
