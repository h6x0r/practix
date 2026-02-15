import { TestCaseResult } from "./test-parser.service";
import { PromptTestResult } from "../ai/ai.service";

// Re-export for backward compatibility
export { TestCaseResult };

export interface SubmissionDto {
  code: string;
  taskId?: string;
  language?: string;
  stdin?: string;
}

export interface SubmissionResult {
  id: string;
  status: string;
  score: number;
  runtime: string;
  memory: string;
  message: string;
  stdout: string;
  stderr: string;
  compileOutput: string;
  createdAt: string;
  testsPassed?: number;
  testsTotal?: number;
  testCases?: TestCaseResult[];
  xpEarned?: number;
  totalXp?: number;
  level?: number;
  leveledUp?: boolean;
  newBadges?: Array<{ slug: string; name: string; icon: string }>;
}

export interface PromptSubmissionResult {
  id: string;
  status: string;
  score: number;
  message: string;
  createdAt: string;
  scenarioResults: PromptTestResult[];
  summary: string;
  xpEarned?: number;
  totalXp?: number;
  level?: number;
  leveledUp?: boolean;
  newBadges?: Array<{ slug: string; name: string; icon: string }>;
}

export interface GamificationReward {
  xpEarned: number;
  totalXp: number;
  level: number;
  leveledUp: boolean;
  newBadges: Array<{ slug: string; name: string; icon: string }>;
}

export interface QuickTestResult {
  status: string;
  testsPassed: number;
  testsTotal: number;
  testCases: TestCaseResult[];
  runtime: string;
  message: string;
  runValidated?: boolean;
}
