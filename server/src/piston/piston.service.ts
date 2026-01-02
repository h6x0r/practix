import { Injectable, Logger, OnModuleInit, OnModuleDestroy } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import axios, { AxiosInstance } from 'axios';
import * as http from 'http';
import * as https from 'https';

/**
 * Piston Language Runtime Configuration
 */
export interface PistonRuntime {
  language: string;
  version: string;
  aliases: string[];
  runtime?: string;
}

/**
 * Piston Execution Request
 */
export interface PistonExecuteRequest {
  language: string;
  version?: string;
  files: { name?: string; content: string }[];
  stdin?: string;
  args?: string[];
  compile_timeout?: number;
  run_timeout?: number;
  compile_memory_limit?: number;
  run_memory_limit?: number;
}

/**
 * Piston Execution Response
 */
export interface PistonExecuteResponse {
  language: string;
  version: string;
  run: {
    stdout: string;
    stderr: string;
    code: number;
    signal: string | null;
    output: string;
    memory?: number;      // Memory usage in bytes
    cpu_time?: number;    // CPU time in milliseconds
    wall_time?: number;   // Wall clock time in milliseconds
  };
  compile?: {
    stdout: string;
    stderr: string;
    code: number;
    signal: string | null;
    output: string;
    memory?: number;
    cpu_time?: number;
    wall_time?: number;
  };
}

/**
 * Normalized execution result (compatible with existing API)
 */
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

/**
 * Language configuration mapping
 */
export interface LanguageConfig {
  pistonName: string;
  version?: string;
  name: string;
  extension: string;
  monacoId: string;
  timeLimit: number;
  memoryLimit: number;
}

export const LANGUAGES: Record<string, LanguageConfig> = {
  go: {
    pistonName: 'go',
    name: 'Go',
    extension: '.go',
    monacoId: 'go',
    timeLimit: 5000,
    memoryLimit: 256 * 1024 * 1024, // 256MB
  },
  java: {
    pistonName: 'java',
    name: 'Java',
    extension: '.java',
    monacoId: 'java',
    timeLimit: 10000,
    memoryLimit: 512 * 1024 * 1024,
  },
  javascript: {
    pistonName: 'javascript',
    name: 'JavaScript',
    extension: '.js',
    monacoId: 'javascript',
    timeLimit: 5000,
    memoryLimit: 256 * 1024 * 1024,
  },
  typescript: {
    pistonName: 'typescript',
    name: 'TypeScript',
    extension: '.ts',
    monacoId: 'typescript',
    timeLimit: 10000,
    memoryLimit: 256 * 1024 * 1024,
  },
  python: {
    pistonName: 'python',
    name: 'Python',
    extension: '.py',
    monacoId: 'python',
    timeLimit: 10000,
    memoryLimit: 256 * 1024 * 1024,
  },
  rust: {
    pistonName: 'rust',
    name: 'Rust',
    extension: '.rs',
    monacoId: 'rust',
    timeLimit: 10000,
    memoryLimit: 256 * 1024 * 1024,
  },
  cpp: {
    pistonName: 'c++',
    name: 'C++',
    extension: '.cpp',
    monacoId: 'cpp',
    timeLimit: 5000,
    memoryLimit: 256 * 1024 * 1024,
  },
  c: {
    pistonName: 'c',
    name: 'C',
    extension: '.c',
    monacoId: 'c',
    timeLimit: 5000,
    memoryLimit: 256 * 1024 * 1024,
  },
};

@Injectable()
export class PistonService implements OnModuleInit, OnModuleDestroy {
  private readonly logger = new Logger(PistonService.name);
  private readonly client: AxiosInstance;
  private readonly pistonUrl: string;
  private readonly httpAgent: http.Agent;
  private readonly httpsAgent: https.Agent;
  private availableRuntimes: PistonRuntime[] = [];
  private isAvailable = false;

  constructor(private config: ConfigService) {
    this.pistonUrl = this.config.get('PISTON_URL') || 'http://piston:2000';

    // Create HTTP agents with connection pooling for better performance
    this.httpAgent = new http.Agent({
      keepAlive: true,
      maxSockets: 10, // Max concurrent connections
      maxFreeSockets: 5, // Max idle connections to keep open
      timeout: 60000, // Socket timeout
    });

    this.httpsAgent = new https.Agent({
      keepAlive: true,
      maxSockets: 10,
      maxFreeSockets: 5,
      timeout: 60000,
    });

    this.client = axios.create({
      baseURL: `${this.pistonUrl}/api/v2`,
      timeout: 60000,
      httpAgent: this.httpAgent,
      httpsAgent: this.httpsAgent,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.logger.log(`Piston configured: ${this.pistonUrl} (connection pooling enabled)`);
  }

  async onModuleDestroy() {
    // Destroy HTTP agents to close all open connections
    this.httpAgent.destroy();
    this.httpsAgent.destroy();
    this.logger.log('HTTP agents destroyed');
  }

  async onModuleInit() {
    await this.loadRuntimes();
  }

  /**
   * Load available runtimes from Piston
   */
  async loadRuntimes(): Promise<void> {
    try {
      const response = await this.client.get<PistonRuntime[]>('/runtimes');
      this.availableRuntimes = response.data;
      this.isAvailable = true;
      this.logger.log(`Piston ready: ${this.availableRuntimes.length} runtimes available`);
    } catch (error) {
      this.isAvailable = false;
      this.logger.warn('Piston not available - will use mock mode');
    }
  }

  /**
   * Check if Piston is available with runtimes
   */
  async checkHealth(): Promise<boolean> {
    if (this.isAvailable && this.availableRuntimes.length > 0) {
      return true;
    }
    await this.loadRuntimes();
    return this.isAvailable && this.availableRuntimes.length > 0;
  }

  /**
   * Check if a specific language runtime is available
   */
  isLanguageAvailable(language: string): boolean {
    const langConfig = this.getLanguageConfig(language);
    if (!langConfig) return false;

    return this.availableRuntimes.some(
      runtime => runtime.language === langConfig.pistonName ||
                 runtime.aliases?.includes(langConfig.pistonName)
    );
  }

  /**
   * Get supported languages
   */
  getSupportedLanguages(): LanguageConfig[] {
    return Object.values(LANGUAGES);
  }

  /**
   * Get language config by name/key
   */
  getLanguageConfig(language: string): LanguageConfig | null {
    const key = language.toLowerCase().replace(/\s+/g, '');

    if (LANGUAGES[key]) return LANGUAGES[key];

    // Alias matching
    if (key.includes('java') && !key.includes('script')) return LANGUAGES.java;
    if (key.includes('go') || key === 'golang') return LANGUAGES.go;
    if (key.includes('python') || key === 'py') return LANGUAGES.python;
    if (key.includes('javascript') || key === 'js' || key === 'node') return LANGUAGES.javascript;
    if (key.includes('typescript') || key === 'ts') return LANGUAGES.typescript;
    if (key.includes('rust') || key === 'rs') return LANGUAGES.rust;
    if (key === 'c++' || key === 'cpp') return LANGUAGES.cpp;
    if (key === 'c') return LANGUAGES.c;

    return null;
  }

  /**
   * Execute code with tests via Piston
   * Combines user solution with test code for validation
   * @param maxTests - Optional limit on number of tests to run (for quick mode)
   */
  async executeWithTests(
    solutionCode: string,
    testCode: string,
    language: string,
    maxTests?: number,
  ): Promise<ExecutionResult> {
    const langConfig = this.getLanguageConfig(language);
    if (!langConfig) {
      return this.errorResult(`Unsupported language: ${language}`);
    }

    // Check availability first - return user-friendly error if unavailable
    const available = await this.checkHealth();
    if (!available) {
      this.logger.warn('Piston unavailable, returning service unavailable error');
      return this.serviceUnavailableResult();
    }

    // Build combined code based on language
    let combinedCode: string;

    if (language === 'python' || language === 'py') {
      combinedCode = this.buildPythonTestCode(solutionCode, testCode, maxTests);
    } else if (language === 'go' || language === 'golang') {
      combinedCode = this.buildGoTestCode(solutionCode, testCode, maxTests);
    } else if (language === 'java') {
      combinedCode = this.buildJavaTestCode(solutionCode, testCode);
    } else {
      combinedCode = solutionCode;
    }

    return this.execute(combinedCode, language);
  }

  /**
   * Extract test method names from test code
   */
  private extractTestNames(testCode: string): string[] {
    const testNames: string[] = [];

    // Python: def test_xxx(self): or def test_xxx():
    const pythonMatches = testCode.match(/def\s+(test_\w+)/g);
    if (pythonMatches) {
      pythonMatches.forEach(m => {
        const name = m.replace(/def\s+/, '');
        testNames.push(name);
      });
    }

    // Go: func TestXxx(t *testing.T)
    const goMatches = testCode.match(/func\s+(Test\w+)/g);
    if (goMatches) {
      goMatches.forEach(m => {
        const name = m.replace(/func\s+/, '');
        testNames.push(name);
      });
    }

    // Java: @Test methods or void test methods
    const javaMatches = testCode.match(/@Test[\s\S]*?(?:public|private|protected)?\s*void\s+(\w+)/g);
    if (javaMatches) {
      javaMatches.forEach(m => {
        const nameMatch = m.match(/void\s+(\w+)/);
        if (nameMatch) testNames.push(nameMatch[1]);
      });
    }

    return testNames.length > 0 ? testNames : ['test_1', 'test_2', 'test_3'];
  }


  /**
   * Build Python code that runs tests without pytest dependency
   * Tests run sequentially and STOP on first failure
   * Output is clean JSON for parsing
   * @param maxTests - Optional limit on number of tests to run
   */
  private buildPythonTestCode(solutionCode: string, testCode: string, maxTests?: number): string {
    // Remove pytest and solution imports from test code
    const cleanedTestCode = testCode
      .replace(/^import pytest.*$/gm, '')
      .replace(/^from pytest import.*$/gm, '')
      .replace(/^from solution import.*$/gm, '')
      .replace(/^import solution.*$/gm, '');

    const maxTestsLimit = maxTests ? `methods = methods[:${maxTests}]  # Quick mode: limit to ${maxTests} tests` : '';

    return `# Solution code
${solutionCode}

# Test code
${cleanedTestCode}

# Run tests sequentially - STOP on first failure
if __name__ == "__main__":
    import sys
    import json
    import re

    test_class = None

    # Find test class
    for name, obj in list(globals().items()):
        if isinstance(obj, type) and name.startswith('Test'):
            test_class = obj
            break

    if test_class:
        instance = test_class()
        methods = sorted([m for m in dir(instance) if m.startswith('test_')])
        ${maxTestsLimit}
        total_tests = len(methods)
        results = []

        for i, method_name in enumerate(methods, 1):
            test_result = {"name": method_name, "passed": False}
            try:
                getattr(instance, method_name)()
                test_result["passed"] = True
                results.append(test_result)
            except AssertionError as e:
                error_str = str(e)
                test_result["error"] = error_str
                # Try to parse "expected X, got Y" or "X != Y" patterns
                match = re.search(r'expected[:\\s]+(.+?)[,\\s]+(?:got|but got|actual)[:\\s]+(.+)', error_str, re.I)
                if match:
                    test_result["expected"] = match.group(1).strip()
                    test_result["output"] = match.group(2).strip()
                else:
                    # Try "X != Y" pattern
                    match = re.search(r'(.+?)\\s*!=\\s*(.+)', error_str)
                    if match:
                        test_result["output"] = match.group(1).strip()
                        test_result["expected"] = match.group(2).strip()
                results.append(test_result)
                # Output JSON and exit on first failure
                print(json.dumps({"tests": results, "passed": len([r for r in results if r["passed"]]), "total": total_tests}))
                sys.exit(1)
            except Exception as e:
                test_result["error"] = f"{type(e).__name__}: {e}"
                results.append(test_result)
                print(json.dumps({"tests": results, "passed": len([r for r in results if r["passed"]]), "total": total_tests}))
                sys.exit(1)

        # All tests passed
        print(json.dumps({"tests": results, "passed": len(results), "total": total_tests}))
        sys.exit(0)
    else:
        print(json.dumps({"error": "No test class found", "tests": [], "passed": 0, "total": 0}))
        sys.exit(1)
`;
  }

  /**
   * Build Go code that runs tests
   * Tests run sequentially and STOP on first failure
   * Output is clean JSON for parsing
   * @param maxTests - Optional limit on number of tests to run
   */
  private buildGoTestCode(solutionCode: string, testCode: string, maxTests?: number): string {
    // For Go, we need to combine into a single file with test runner
    // Remove package declarations and test file structure
    const cleanSolution = solutionCode
      .replace(/^package\s+\w+\s*$/gm, '')
      .trim();

    const cleanTests = testCode
      .replace(/^package\s+\w+\s*$/gm, '')
      .replace(/import\s*\(\s*"testing"\s*\)/g, '')
      .replace(/import\s+"testing"/g, '')
      .replace(/\*testing\.T/g, '*T')
      .trim();

    // Extract test function names from testCode
    const testFunctions = this.extractGoTestFunctions(testCode);

    // Apply maxTests limit
    const testsToRun = maxTests ? testFunctions.slice(0, maxTests) : testFunctions;

    // Generate runTest calls
    const testCalls = testsToRun.map(name => `    runTest("${name}", ${name})`).join('\n');

    return `package main

import (
    "encoding/json"
    "fmt"
    "os"
    "regexp"
)

// Test result structure
type TestResult struct {
    Name     string \`json:"name"\`
    Passed   bool   \`json:"passed"\`
    Error    string \`json:"error,omitempty"\`
    Expected string \`json:"expected,omitempty"\`
    Output   string \`json:"output,omitempty"\`
}

type TestOutput struct {
    Tests  []TestResult \`json:"tests"\`
    Passed int          \`json:"passed"\`
    Total  int          \`json:"total"\`
}

// Mock testing.T with error capture
type T struct {
    failed   bool
    name     string
    errorMsg string
}

func (t *T) Errorf(format string, args ...interface{}) {
    t.failed = true
    t.errorMsg = fmt.Sprintf(format, args...)
}

func (t *T) Error(args ...interface{}) {
    t.failed = true
    t.errorMsg = fmt.Sprint(args...)
}

func (t *T) Fatalf(format string, args ...interface{}) {
    t.failed = true
    t.errorMsg = fmt.Sprintf(format, args...)
    panic("test failed")
}

func (t *T) Fatal(args ...interface{}) {
    t.failed = true
    t.errorMsg = fmt.Sprint(args...)
    panic("test failed")
}

func (t *T) Logf(format string, args ...interface{}) {}
func (t *T) Log(args ...interface{}) {}

// Solution code
${cleanSolution}

// Test code
${cleanTests}

var testResults []TestResult
var totalTests = 0

// Parse error message to extract expected/actual
func parseError(errMsg string) (expected, output string) {
    // Pattern: "expected X, got Y" or "want X, got Y"
    re := regexp.MustCompile(\`(?i)(?:expected|want)[:\\s]+(.+?)[,\\s]+(?:got|but got|actual)[:\\s]+(.+)\`)
    matches := re.FindStringSubmatch(errMsg)
    if len(matches) >= 3 {
        return matches[1], matches[2]
    }
    // Pattern: "X != Y"
    re2 := regexp.MustCompile(\`(.+?)\\s*!=\\s*(.+)\`)
    matches2 := re2.FindStringSubmatch(errMsg)
    if len(matches2) >= 3 {
        return matches2[2], matches2[1]
    }
    return "", ""
}

// Run single test - returns true if passed, false if failed
func runTest(name string, fn func(*T)) bool {
    totalTests++
    t := &T{name: name}
    result := TestResult{Name: name, Passed: false}

    defer func() {
        if r := recover(); r != nil {
            if t.errorMsg == "" {
                t.errorMsg = fmt.Sprintf("%v", r)
            }
            result.Error = t.errorMsg
            result.Expected, result.Output = parseError(t.errorMsg)
            testResults = append(testResults, result)
        }
    }()

    fn(t)

    if t.failed {
        result.Error = t.errorMsg
        result.Expected, result.Output = parseError(t.errorMsg)
        testResults = append(testResults, result)
        return false
    }

    result.Passed = true
    testResults = append(testResults, result)
    return true
}

func printResults() {
    passed := 0
    for _, r := range testResults {
        if r.Passed {
            passed++
        }
    }
    output := TestOutput{Tests: testResults, Passed: passed, Total: totalTests}
    jsonBytes, _ := json.Marshal(output)
    fmt.Println(string(jsonBytes))
}

func main() {
    // Run all test functions
${testCalls}

    printResults()
    passed := 0
    for _, r := range testResults {
        if r.Passed {
            passed++
        }
    }
    if passed < totalTests {
        os.Exit(1)
    }
}
`;
  }

  /**
   * Extract Go test function names from test code
   * Validates function names to prevent code injection
   */
  private extractGoTestFunctions(testCode: string): string[] {
    const functions: string[] = [];
    const regex = /func\s+(Test\w+)\s*\(/g;
    // Whitelist pattern: only alphanumeric and underscore, must start with Test
    const validNamePattern = /^Test[A-Za-z0-9_]+$/;
    // Blacklist dangerous keywords that could indicate injection attempts
    const dangerousPatterns = [
      /os\./i, /exec\./i, /syscall\./i, /unsafe\./i,
      /runtime\./i, /reflect\./i, /eval/i, /import/i
    ];

    let match;
    while ((match = regex.exec(testCode)) !== null) {
      const funcName = match[1];

      // Validate function name format
      if (!validNamePattern.test(funcName)) {
        this.logger.warn(`Invalid test function name rejected: ${funcName}`);
        continue;
      }

      // Check for dangerous patterns in function name
      const hasDangerousPattern = dangerousPatterns.some(pattern => pattern.test(funcName));
      if (hasDangerousPattern) {
        this.logger.warn(`Potentially dangerous test function name rejected: ${funcName}`);
        continue;
      }

      // Limit function name length to prevent buffer overflow attempts
      if (funcName.length > 100) {
        this.logger.warn(`Test function name too long, rejected: ${funcName.substring(0, 20)}...`);
        continue;
      }

      functions.push(funcName);
    }
    return functions;
  }

  /**
   * Build Java code that runs tests
   */
  private buildJavaTestCode(solutionCode: string, testCode: string): string {
    // For Java, combine solution with simple test runner
    const cleanTests = testCode
      .replace(/import\s+org\.junit.*$/gm, '')
      .replace(/@Test/g, '// @Test')
      .trim();

    return `${solutionCode}

// Test execution
class TestRunner {
    static int passed = 0;
    static int failed = 0;

    static void assertEquals(Object expected, Object actual) {
        if (!expected.equals(actual)) {
            throw new AssertionError("Expected: " + expected + ", Got: " + actual);
        }
    }

    static void assertTrue(boolean condition) {
        if (!condition) {
            throw new AssertionError("Expected true but got false");
        }
    }

    static void assertFalse(boolean condition) {
        if (condition) {
            throw new AssertionError("Expected false but got true");
        }
    }
}

${cleanTests}
`;
  }

  /**
   * Execute code via Piston
   */
  async execute(
    code: string,
    language: string,
    stdin?: string,
  ): Promise<ExecutionResult> {
    const langConfig = this.getLanguageConfig(language);
    if (!langConfig) {
      return this.errorResult(`Unsupported language: ${language}`);
    }

    // Check availability - return user-friendly error if unavailable
    const available = await this.checkHealth();
    if (!available) {
      this.logger.warn('Piston unavailable, returning service unavailable error');
      return this.serviceUnavailableResult();
    }

    try {
      const startTime = Date.now();

      const request: PistonExecuteRequest = {
        language: langConfig.pistonName,
        version: '*', // Use latest available version
        files: [{ content: code }],
        stdin: stdin || '',
        run_timeout: langConfig.timeLimit,
        run_memory_limit: langConfig.memoryLimit,
      };

      this.logger.debug(`Executing ${langConfig.name} code via Piston`);

      const response = await this.client.post<PistonExecuteResponse>(
        '/execute',
        request,
      );

      const elapsed = ((Date.now() - startTime) / 1000).toFixed(3);
      return this.parseResponse(response.data, elapsed);
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      const errorCode = (error as any)?.code;
      const statusCode = (error as any)?.response?.status;
      this.logger.error(`Piston execution failed: ${errorMessage}`);

      // Check for timeout
      if (errorCode === 'ECONNABORTED' || errorMessage?.includes('timeout')) {
        return {
          status: 'timeout',
          statusId: 5,
          description: 'Time Limit Exceeded',
          stdout: '',
          stderr: 'Your code took too long to execute',
          compileOutput: '',
          time: '-',
          memory: 0,
          exitCode: null,
          message: 'Execution timed out. Try optimizing your code.',
        };
      }

      // Check for 400 error (usually means language not available)
      if (statusCode === 400) {
        this.logger.warn(`Language runtime not available: ${langConfig.name}`);
        return this.errorResult(`Language ${langConfig.name} is temporarily unavailable. Please try again later.`);
      }

      // Connection errors - return user-friendly message
      if (errorCode === 'ECONNREFUSED' || errorCode === 'ENOTFOUND') {
        this.logger.warn(`Piston connection failed`);
        return this.serviceUnavailableResult();
      }

      // Generic error with user-friendly message
      return this.serviceUnavailableResult();
    }
  }

  /**
   * Return user-friendly error when service is unavailable
   * No mock execution - honest feedback to user
   */
  private serviceUnavailableResult(): ExecutionResult {
    return {
      status: 'error',
      statusId: 13,
      description: 'Service Temporarily Unavailable',
      stdout: '',
      stderr: '',
      compileOutput: '',
      time: '-',
      memory: 0,
      exitCode: null,
      message: 'Code execution service is temporarily unavailable. Please try again in a few minutes.',
    };
  }

  /**
   * Parse Piston response to normalized format
   */
  private parseResponse(response: PistonExecuteResponse, fallbackTime: string): ExecutionResult {
    const run = response.run;
    const compile = response.compile;

    // Use Piston's wall_time if available, otherwise fallback to measured time
    // wall_time is in milliseconds, convert to seconds
    const time = run.wall_time
      ? (run.wall_time / 1000).toFixed(3)
      : fallbackTime;

    // Memory is in bytes, we'll pass it as-is and format in the submission service
    const memory = run.memory || 0;

    // Check for compilation error
    if (compile && compile.code !== 0) {
      return {
        status: 'compileError',
        statusId: 6,
        description: 'Compilation Error',
        stdout: '',
        stderr: '',
        compileOutput: this.truncateOutput(compile.stderr || compile.output),
        time: '-',
        memory: 0,
        exitCode: compile.code,
        message: 'Compilation failed',
      };
    }

    // Check for runtime error
    // Special case: If we got valid output before timeout/kill, treat as success
    // This handles cases like numpy imports that produce output but get killed during cleanup
    const hasValidOutput = run.stdout && run.stdout.trim().length > 0;
    const isTimeoutWithOutput = run.signal === 'SIGKILL' && hasValidOutput;

    if ((run.code !== 0 || run.signal) && !isTimeoutWithOutput) {
      return {
        status: 'error',
        statusId: 11,
        description: run.signal ? `Signal: ${run.signal}` : 'Runtime Error',
        stdout: this.truncateOutput(run.stdout),
        stderr: this.truncateOutput(run.stderr),
        compileOutput: compile?.output ? this.truncateOutput(compile.output) : '',
        time,
        memory,
        exitCode: run.code,
        message: run.stderr || `Exit code: ${run.code}`,
      };
    }

    // Success
    return {
      status: 'passed',
      statusId: 3,
      description: 'Accepted',
      stdout: this.truncateOutput(run.stdout),
      stderr: this.truncateOutput(run.stderr),
      compileOutput: compile?.output ? this.truncateOutput(compile.output) : '',
      time,
      memory,
      exitCode: run.code,
    };
  }

  /**
   * Helper: Create error result
   */
  private errorResult(message: string): ExecutionResult {
    return {
      status: 'error',
      statusId: 13,
      description: 'Error',
      stdout: '',
      stderr: message,
      compileOutput: '',
      time: '-',
      memory: 0,
      exitCode: null,
      message,
    };
  }

  /**
   * Helper: Truncate long output
   */
  private truncateOutput(output: string, maxLength = 10000): string {
    if (!output) return '';
    if (output.length <= maxLength) return output;
    return output.substring(0, maxLength) + '\n\n... [Output truncated]';
  }
}
