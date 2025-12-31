/**
 * TypeScript interfaces for KODLA learning platform seed data
 * Defines hierarchical structure: Course → Module → Topic → Task
 */

/**
 * Translation interface for task content in different languages
 */
export interface TaskTranslation {
	title: string;
	description: string;
	hint1: string;
	hint2: string;
	hint3?: string; // Optional third hint for complex tasks
	whyItMatters: string;
	solutionCode?: string;
}

/**
 * Visualization type for ML tasks
 * Used to render charts in frontend via Recharts
 */
export type VisualizationType =
	| 'none'
	| 'line'
	| 'bar'
	| 'scatter'
	| 'heatmap'
	| 'confusion_matrix'
	| 'multi'; // For tasks with multiple chart types

/**
 * Task - the smallest unit of learning content
 * Contains code challenge, solution, hints, and multilingual translations
 */
export interface Task {
	slug: string;
	title: string;
	difficulty: 'easy' | 'medium' | 'hard';
	tags: string[];
	estimatedTime: string;
	isPremium: boolean;
	youtubeUrl?: string;
	description: string;
	initialCode: string;
	solutionCode: string;
	testCode?: string; // Optional test code for validation
	hint1: string;
	hint2: string;
	hint3?: string; // Optional third hint for complex tasks
	whyItMatters: string;
	order: number;
	// ML visualization support
	visualizationType?: VisualizationType; // Chart type for ML tasks
	expectedVisualization?: object; // Expected chart data for validation
	translations: {
		ru: TaskTranslation;
		uz: TaskTranslation;
	};
}

/**
 * Topic - a collection of related tasks within a module
 * Example: "Error Handling Fundamentals" with 7 tasks
 */
export interface Topic {
	slug?: string; // Optional unique identifier for the topic
	title: string;
	description: string;
	difficulty?: 'easy' | 'medium' | 'hard'; // Optional, can be inferred from tasks
	estimatedTime?: string; // Optional, can be calculated from tasks
	isPremium?: boolean; // Optional premium flag for topic-level access control
	order: number;
	tasks?: Task[]; // Optional when using separate task files
	translations?: {
		ru: TopicTranslation;
		uz: TopicTranslation;
	};
}

/**
 * Module - a major section within a course
 * Example: "Error Handling", "HTTP Middleware"
 */
export interface Module {
	slug?: string;
	title: string;
	description: string;
	section?: string; // Optional section grouping (e.g., 'core', 'advanced')
	order: number;
	difficulty?: 'easy' | 'medium' | 'hard';
	estimatedTime?: string;
	isPremium?: boolean;
	topics: Topic[];
	translations?: {
		ru: ModuleTranslation;
		uz: ModuleTranslation;
	};
}

/**
 * Course - top-level learning path
 * Example: "Go Language Mastery", "Java Enterprise"
 */
export interface Course {
	slug: string;
	title: string;
	description: string;
	category: 'language' | 'cs' | 'framework';
	icon: string;
	estimatedTime: string;
	order: number;
	modules: Module[];
}

/**
 * Translation interfaces for Course, Module, and Topic
 */
export interface CourseTranslation {
	title: string;
	description: string;
}

export interface ModuleTranslation {
	title: string;
	description: string;
}

export interface TopicTranslation {
	title: string;
	description: string;
}

/**
 * Metadata types for each level (without children)
 * Used in individual metadata files (course.ts, module.ts, topic.ts)
 */
export interface CourseMeta {
	slug: string;
	title: string;
	description: string;
	category: 'language' | 'cs' | 'framework';
	icon: string;
	estimatedTime: string;
	order: number;
	translations?: {
		ru: CourseTranslation;
		uz: CourseTranslation;
	};
}

export interface ModuleMeta {
	slug?: string;
	title: string;
	description: string;
	section?: string; // Optional section grouping
	order: number;
	isPremium?: boolean; // Optional premium flag
	translations?: {
		ru: ModuleTranslation;
		uz: ModuleTranslation;
	};
}

export interface TopicMeta {
	slug?: string; // Optional unique identifier
	title: string;
	description: string;
	difficulty?: 'easy' | 'medium' | 'hard'; // Optional, can be inferred from tasks
	estimatedTime?: string; // Optional, can be calculated from tasks
	isPremium?: boolean; // Optional premium flag for topic-level access control
	order: number;
	translations?: {
		ru: TopicTranslation;
		uz: TopicTranslation;
	};
}
