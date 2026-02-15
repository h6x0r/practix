import {
  Injectable,
  Logger,
  OnModuleInit,
  OnModuleDestroy,
} from "@nestjs/common";
import { ConfigService } from "@nestjs/config";
import axios, { AxiosInstance } from "axios";
import * as http from "http";
import * as https from "https";
import {
  Judge0Language,
  Judge0SubmissionRequest,
  Judge0SubmissionResponse,
  ExecutionResult,
  LanguageConfig,
  LANGUAGES,
  STATUS,
} from "./judge0.types";
import {
  buildPythonTestCode,
  buildGoTestCode,
  buildJavaTestCode,
} from "./judge0-test-builders";

// Re-export types for backward compatibility
export type {
  Judge0Language,
  Judge0SubmissionRequest,
  Judge0SubmissionResponse,
  ExecutionResult,
  LanguageConfig,
} from "./judge0.types";
export { LANGUAGES } from "./judge0.types";

@Injectable()
export class Judge0Service implements OnModuleInit, OnModuleDestroy {
  private readonly logger = new Logger(Judge0Service.name);
  private readonly client: AxiosInstance;
  private readonly judge0Url: string;
  private readonly httpAgent: http.Agent;
  private readonly httpsAgent: https.Agent;
  private availableLanguages: Judge0Language[] = [];
  private isAvailable = false;

  constructor(private config: ConfigService) {
    this.judge0Url = this.config.get("JUDGE0_URL") || "http://judge0:2358";

    this.httpAgent = new http.Agent({
      keepAlive: true,
      maxSockets: 20,
      maxFreeSockets: 10,
      timeout: 60000,
    });

    this.httpsAgent = new https.Agent({
      keepAlive: true,
      maxSockets: 20,
      maxFreeSockets: 10,
      timeout: 60000,
    });

    this.client = axios.create({
      baseURL: this.judge0Url,
      timeout: 120000,
      httpAgent: this.httpAgent,
      httpsAgent: this.httpsAgent,
      headers: { "Content-Type": "application/json" },
    });

    this.logger.log(`Judge0 configured: ${this.judge0Url}`);
  }

  async onModuleDestroy() {
    this.httpAgent.destroy();
    this.httpsAgent.destroy();
    this.logger.log("HTTP agents destroyed");
  }

  async onModuleInit() {
    await this.loadLanguages();
  }

  async loadLanguages(): Promise<void> {
    try {
      const response = await this.client.get<Judge0Language[]>("/languages");
      this.availableLanguages = response.data;
      this.isAvailable = true;
      this.logger.log(
        `Judge0 ready: ${this.availableLanguages.length} languages available`,
      );
    } catch (error) {
      this.isAvailable = false;
      this.logger.warn("Judge0 not available - code execution disabled");
    }
  }

  async checkHealth(): Promise<boolean> {
    if (this.isAvailable && this.availableLanguages.length > 0) return true;
    await this.loadLanguages();
    return this.isAvailable && this.availableLanguages.length > 0;
  }

  isLanguageAvailable(language: string): boolean {
    const langConfig = this.getLanguageConfig(language);
    if (!langConfig) return false;
    return this.availableLanguages.some((l) => l.id === langConfig.judge0Id);
  }

  getSupportedLanguages(): LanguageConfig[] {
    return Object.values(LANGUAGES);
  }

  getLanguageConfig(language: string): LanguageConfig | null {
    const key = language.toLowerCase().replace(/\s+/g, "");

    if (LANGUAGES[key]) return LANGUAGES[key];

    if (key.includes("java") && !key.includes("script")) return LANGUAGES.java;
    if (key.includes("go") || key === "golang") return LANGUAGES.go;
    if (key.includes("python") || key === "py") return LANGUAGES.python;
    if (key.includes("javascript") || key === "js" || key === "node")
      return LANGUAGES.javascript;
    if (key.includes("typescript") || key === "ts") return LANGUAGES.typescript;
    if (key.includes("rust") || key === "rs") return LANGUAGES.rust;
    if (key === "c++" || key === "cpp") return LANGUAGES.cpp;
    if (key === "c") return LANGUAGES.c;

    return null;
  }

  async executeWithTests(
    solutionCode: string,
    testCode: string,
    language: string,
    maxTests?: number,
  ): Promise<ExecutionResult> {
    const langConfig = this.getLanguageConfig(language);
    if (!langConfig)
      return this.errorResult(`Unsupported language: ${language}`);

    const available = await this.checkHealth();
    if (!available) return this.serviceUnavailableResult();

    let combinedCode: string;

    if (language === "python" || language === "py") {
      combinedCode = buildPythonTestCode(solutionCode, testCode, maxTests);
    } else if (language === "go" || language === "golang") {
      combinedCode = buildGoTestCode(solutionCode, testCode, maxTests);
    } else if (language === "java") {
      combinedCode = buildJavaTestCode(solutionCode, testCode);
    } else {
      combinedCode = solutionCode;
    }

    return this.execute(combinedCode, language);
  }

  async execute(
    code: string,
    language: string,
    stdin?: string,
  ): Promise<ExecutionResult> {
    const langConfig = this.getLanguageConfig(language);
    if (!langConfig)
      return this.errorResult(`Unsupported language: ${language}`);

    const available = await this.checkHealth();
    if (!available) return this.serviceUnavailableResult();

    try {
      const startTime = Date.now();

      const request: Judge0SubmissionRequest = {
        source_code: code,
        language_id: langConfig.judge0Id,
        stdin: stdin || "",
        cpu_time_limit: Math.min(langConfig.timeLimit, 15),
        wall_time_limit: Math.min(langConfig.timeLimit * 1.5, 20),
        memory_limit: Math.min(langConfig.memoryLimit, 512000),
      };

      this.logger.debug(
        `Executing ${langConfig.name} via Judge0 (timeout=${langConfig.timeLimit}s)`,
      );

      const response = await this.client.post<Judge0SubmissionResponse>(
        "/submissions?base64_encoded=false&wait=true",
        request,
      );

      const elapsed = ((Date.now() - startTime) / 1000).toFixed(3);
      return this.parseResponse(response.data, elapsed);
    } catch (error: unknown) {
      return this.handleExecutionError(error);
    }
  }

  // ============ Private helpers ============

  private handleExecutionError(error: unknown): ExecutionResult {
    const errorMessage =
      error instanceof Error ? error.message : "Unknown error";
    const errorCode = (error as any)?.code;

    this.logger.error(`Judge0 execution failed: ${errorMessage}`);

    if (errorCode === "ECONNABORTED" || errorMessage?.includes("timeout")) {
      return {
        status: "timeout",
        statusId: 5,
        description: "Time Limit Exceeded",
        stdout: "",
        stderr: "Your code took too long to execute",
        compileOutput: "",
        time: "-",
        memory: 0,
        exitCode: null,
        message: "Execution timed out. Try optimizing your code.",
      };
    }

    return this.serviceUnavailableResult();
  }

  private parseResponse(
    response: Judge0SubmissionResponse,
    fallbackTime: string,
  ): ExecutionResult {
    const status = response.status;
    const time = response.time || fallbackTime;
    const memory = response.memory || 0;

    if (!status || typeof status.id !== "number") {
      this.logger.error(
        `Invalid Judge0 response: missing status field. Response: ${JSON.stringify(response).substring(0, 500)}`,
      );
      return this.errorResult("Invalid response from code execution service");
    }

    if (status.id === STATUS.COMPILATION_ERROR) {
      return {
        status: "compileError",
        statusId: 6,
        description: "Compilation Error",
        stdout: "",
        stderr: "",
        compileOutput: this.truncateOutput(response.compile_output || ""),
        time: "-",
        memory: 0,
        exitCode: null,
        message: "Compilation failed",
      };
    }

    if (status.id === STATUS.TIME_LIMIT_EXCEEDED) {
      return {
        status: "timeout",
        statusId: 5,
        description: "Time Limit Exceeded",
        stdout: this.truncateOutput(response.stdout || ""),
        stderr: "",
        compileOutput: "",
        time,
        memory,
        exitCode: null,
        message: "Your code took too long to execute",
      };
    }

    if (
      status.id >= STATUS.RUNTIME_ERROR_SIGSEGV &&
      status.id <= STATUS.RUNTIME_ERROR_OTHER
    ) {
      return {
        status: "error",
        statusId: status.id,
        description: status.description,
        stdout: this.truncateOutput(response.stdout || ""),
        stderr: this.truncateOutput(response.stderr || ""),
        compileOutput: this.truncateOutput(response.compile_output || ""),
        time,
        memory,
        exitCode: response.exit_code,
        message: response.message || status.description,
      };
    }

    if (status.id === STATUS.INTERNAL_ERROR) {
      return this.serviceUnavailableResult();
    }

    return {
      status: "passed",
      statusId: 3,
      description: "Accepted",
      stdout: this.truncateOutput(response.stdout || ""),
      stderr: this.truncateOutput(response.stderr || ""),
      compileOutput: this.truncateOutput(response.compile_output || ""),
      time,
      memory,
      exitCode: response.exit_code,
    };
  }

  private serviceUnavailableResult(): ExecutionResult {
    return {
      status: "error",
      statusId: 13,
      description: "Service Temporarily Unavailable",
      stdout: "",
      stderr: "",
      compileOutput: "",
      time: "-",
      memory: 0,
      exitCode: null,
      message: "Code execution service is temporarily unavailable.",
    };
  }

  private errorResult(message: string): ExecutionResult {
    return {
      status: "error",
      statusId: 13,
      description: "Error",
      stdout: "",
      stderr: message,
      compileOutput: "",
      time: "-",
      memory: 0,
      exitCode: null,
      message,
    };
  }

  private truncateOutput(output: string, maxLength = 10000): string {
    if (!output) return "";
    if (output.length <= maxLength) return output;
    return output.substring(0, maxLength) + "\n\n... [Output truncated]";
  }
}
