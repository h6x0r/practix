/**
 * Judge0 types and configuration constants
 */

export interface Judge0Language {
  id: number;
  name: string;
}

export interface Judge0SubmissionRequest {
  source_code: string;
  language_id: number;
  stdin?: string;
  expected_output?: string;
  cpu_time_limit?: number;
  wall_time_limit?: number;
  memory_limit?: number;
  compiler_options?: string;
}

export interface Judge0SubmissionResponse {
  token?: string;
  stdout: string | null;
  stderr: string | null;
  compile_output: string | null;
  message: string | null;
  exit_code: number | null;
  exit_signal: number | null;
  status: {
    id: number;
    description: string;
  };
  time: string | null;
  wall_time: string | null;
  memory: number | null;
}

export interface ExecutionResult {
  status: 'passed' | 'failed' | 'error' | 'timeout' | 'compileError';
  statusId: number;
  description: string;
  stdout: string;
  stderr: string;
  compileOutput: string;
  time: string;
  memory: number;
  exitCode: number | null;
  message?: string;
}

export interface LanguageConfig {
  judge0Id: number;
  name: string;
  extension: string;
  monacoId: string;
  timeLimit: number;
  memoryLimit: number;
}

export const LANGUAGES: Record<string, LanguageConfig> = {
  go: { judge0Id: 60, name: 'Go', extension: '.go', monacoId: 'go', timeLimit: 15, memoryLimit: 512000 },
  java: { judge0Id: 62, name: 'Java', extension: '.java', monacoId: 'java', timeLimit: 15, memoryLimit: 512000 },
  javascript: { judge0Id: 63, name: 'JavaScript', extension: '.js', monacoId: 'javascript', timeLimit: 10, memoryLimit: 256000 },
  typescript: { judge0Id: 74, name: 'TypeScript', extension: '.ts', monacoId: 'typescript', timeLimit: 15, memoryLimit: 256000 },
  python: { judge0Id: 71, name: 'Python', extension: '.py', monacoId: 'python', timeLimit: 15, memoryLimit: 256000 },
  rust: { judge0Id: 73, name: 'Rust', extension: '.rs', monacoId: 'rust', timeLimit: 15, memoryLimit: 256000 },
  cpp: { judge0Id: 54, name: 'C++', extension: '.cpp', monacoId: 'cpp', timeLimit: 10, memoryLimit: 256000 },
  c: { judge0Id: 50, name: 'C', extension: '.c', monacoId: 'c', timeLimit: 10, memoryLimit: 256000 },
};

export const STATUS = {
  IN_QUEUE: 1,
  PROCESSING: 2,
  ACCEPTED: 3,
  WRONG_ANSWER: 4,
  TIME_LIMIT_EXCEEDED: 5,
  COMPILATION_ERROR: 6,
  RUNTIME_ERROR_SIGSEGV: 7,
  RUNTIME_ERROR_SIGXFSZ: 8,
  RUNTIME_ERROR_SIGFPE: 9,
  RUNTIME_ERROR_SIGABRT: 10,
  RUNTIME_ERROR_NZEC: 11,
  RUNTIME_ERROR_OTHER: 12,
  INTERNAL_ERROR: 13,
  EXEC_FORMAT_ERROR: 14,
};
