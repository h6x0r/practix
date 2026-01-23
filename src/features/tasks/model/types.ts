// Translation object structure for multi-language support
export interface Translations {
  ru?: Record<string, string>;
  uz?: Record<string, string>;
}

// Task types for different submission flows
export type TaskType = 'CODE' | 'PROMPT';

// Visualization types for ML tasks
export type VisualizationType =
  | 'none'
  | 'line'
  | 'bar'
  | 'scatter'
  | 'heatmap'
  | 'confusion_matrix'
  | 'multi';

// Prompt test scenario for prompt engineering tasks
export interface PromptTestScenario {
  input: string;
  expectedCriteria: string[];
  rubric?: string;
}

// Configuration for prompt engineering tasks
export interface PromptConfig {
  testScenarios: PromptTestScenario[];
  judgePrompt: string;
  passingScore: number;
}

// Result of a single prompt test scenario
export interface PromptScenarioResult {
  scenarioIndex: number;
  input: string;
  output: string;
  score: number;
  feedback: string;
  passed: boolean;
}

// Chart data structure returned by ML tasks
export interface ChartData {
  type: VisualizationType;
  data: Array<Record<string, number | string>>;
  config?: {
    title?: string;
    xLabel?: string;
    yLabel?: string;
    colors?: string[];
  };
}

export interface Task {
  id: string;
  slug: string;
  title: string;
  description: string;
  difficulty: 'easy' | 'medium' | 'hard';
  tags: string[];
  isPremium: boolean;
  initialCode: string;
  solutionCode?: string; // Canonical solution code
  solutionExplanation?: string; // Line-by-line explanation of the solution
  whyItMatters?: string; // Real-world usage and importance of this pattern
  testCode?: string;     // Test code for validation
  hint1?: string;        // First hint (gentle nudge, 8-10 words)
  hint2?: string;        // Second hint (more detailed, 8-10 words)
  estimatedTime: string; // e.g. "30m"
  status?: 'pending' | 'completed'; // For UI display
  youtubeUrl?: string; // Optional link to video solution
  translations?: Translations; // Multi-language support
  // ML visualization support
  visualizationType?: VisualizationType; // Chart type for ML tasks
  // Task type: CODE (default) or PROMPT (for prompt engineering)
  taskType?: TaskType;
  // Prompt Engineering configuration (only for taskType: PROMPT)
  promptConfig?: PromptConfig;
}

// Prompt submission result (returned from backend)
export interface PromptSubmission {
  id: string;
  status: 'pending' | 'running' | 'passed' | 'failed' | 'error';
  score: number;
  message: string;
  createdAt: string;
  scenarioResults: PromptScenarioResult[];
  summary: string;
  // Gamification
  xpEarned?: number;
  totalXp?: number;
  level?: number;
  leveledUp?: boolean;
  newBadges?: Array<{ slug: string; name: string; icon: string }>;
}

export interface TestCaseResult {
  name: string;
  passed: boolean;
  input?: string;
  expectedOutput?: string;
  actualOutput?: string;
  error?: string;
}

export interface Submission {
  id: string;
  status: 'pending' | 'running' | 'passed' | 'failed' | 'error';
  score: number;
  runtime: string;
  memory?: string;
  createdAt: string;
  code: string;
  message?: string; // Console output / Test results
  // Test case details
  testsPassed?: number;
  testsTotal?: number;
  testCases?: TestCaseResult[];
}

export interface AiChatMessage {
  role: 'user' | 'model';
  text: string;
}
