import { Injectable } from '@nestjs/common';

/**
 * Result of a single test case
 */
export interface TestCaseResult {
  name: string;
  passed: boolean;
  input?: string;
  expectedOutput?: string;
  actualOutput?: string;
  error?: string;
}

/**
 * Parsed test output
 */
export interface ParsedTestOutput {
  testCases: TestCaseResult[];
  passed: number;
  total: number;
}

/**
 * Raw JSON test result from test runner
 */
interface RawTestResult {
  name?: string;
  passed?: boolean;
  expected?: string;
  output?: string;
  error?: string;
}

/**
 * TestParserService
 *
 * Parses test output from various test runners.
 * Extracted from SubmissionsService to follow Single Responsibility Principle.
 *
 * Supported formats:
 * - JSON format: { "tests": [...], "passed": N, "total": N }
 * - Go-style format: === RUN / --- PASS/FAIL
 * - RESULT line format: RESULT: N/M
 */
@Injectable()
export class TestParserService {
  /**
   * Parse test output from stdout/stderr
   *
   * @param stdout - Standard output from test runner
   * @param stderr - Standard error from test runner
   * @returns Parsed test results
   */
  parseTestOutput(stdout: string, stderr: string): ParsedTestOutput {
    const output = stdout.trim();

    // Try JSON format first (preferred)
    const jsonResult = this.tryParseJsonFormat(output);
    if (jsonResult) {
      return jsonResult;
    }

    // Fall back to legacy Go-style format
    return this.parseLegacyFormat(stdout, stderr);
  }

  /**
   * Try to parse JSON format output
   * Format: { "tests": [...], "passed": N, "total": N }
   */
  private tryParseJsonFormat(output: string): ParsedTestOutput | null {
    try {
      // Find JSON in output (might have other text before/after)
      const jsonMatch = output.match(/\{[\s\S]*"tests"[\s\S]*\}/);
      if (!jsonMatch) {
        return null;
      }

      const parsed = JSON.parse(jsonMatch[0]);
      if (!parsed.tests || !Array.isArray(parsed.tests)) {
        return null;
      }

      const testCases: TestCaseResult[] = parsed.tests.map((t: RawTestResult) => ({
        name: t.name || 'test',
        passed: t.passed || false,
        expectedOutput: t.expected,
        actualOutput: t.output,
        error: t.error,
      }));

      return {
        testCases,
        passed: parsed.passed || 0,
        total: parsed.total || testCases.length,
      };
    } catch {
      return null;
    }
  }

  /**
   * Parse legacy Go-style format
   * Format: === RUN TestName / --- PASS: TestName / --- FAIL: TestName
   */
  private parseLegacyFormat(stdout: string, stderr: string): ParsedTestOutput {
    const testCases: TestCaseResult[] = [];
    const fullOutput = stdout + '\n' + stderr;

    // Extract test names from === RUN lines
    const testNames = this.extractTestNames(fullOutput);

    // Find passed and failed tests
    const passedTests = this.extractTestsByPattern(fullOutput, /--- PASS:\s*(\S+)/g);
    const failedTests = this.extractTestsByPattern(fullOutput, /--- FAIL:\s*(\S+)/g);

    // Build test case results
    for (const testName of testNames) {
      const passed = passedTests.has(testName);
      const failed = failedTests.has(testName);

      const testCase: TestCaseResult = {
        name: testName,
        passed: passed && !failed,
      };

      // Extract error details for failed tests
      if (failed) {
        const errorInfo = this.extractErrorInfo(fullOutput, testName);
        if (errorInfo) {
          testCase.error = errorInfo.error;
          testCase.expectedOutput = errorInfo.expected;
          testCase.actualOutput = errorInfo.actual;
        }
      }

      testCases.push(testCase);
    }

    // Try to get count from RESULT line
    const resultMatch = fullOutput.match(/RESULT:\s*(\d+)\/(\d+)/);
    if (resultMatch) {
      return {
        testCases,
        passed: parseInt(resultMatch[1], 10),
        total: parseInt(resultMatch[2], 10),
      };
    }

    return {
      testCases,
      passed: testCases.filter(t => t.passed).length,
      total: testCases.length,
    };
  }

  /**
   * Extract test names from === RUN lines
   */
  private extractTestNames(output: string): string[] {
    const runPattern = /=== RUN\s+(\S+)/g;
    const testNames: string[] = [];
    let match;

    while ((match = runPattern.exec(output)) !== null) {
      if (!testNames.includes(match[1])) {
        testNames.push(match[1]);
      }
    }

    return testNames;
  }

  /**
   * Extract test names matching a pattern (PASS/FAIL)
   */
  private extractTestsByPattern(output: string, pattern: RegExp): Set<string> {
    const tests = new Set<string>();
    let match;

    while ((match = pattern.exec(output)) !== null) {
      tests.add(match[1]);
    }

    return tests;
  }

  /**
   * Extract error info for a failed test
   */
  private extractErrorInfo(
    output: string,
    testName: string,
  ): { error: string; expected?: string; actual?: string } | null {
    // Escape special regex characters in test name
    const escapedName = testName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

    const errorMatch = output.match(
      new RegExp(
        `--- FAIL:\\s*${escapedName}[\\s\\S]*?error:\\s*([^\\n]+)`,
        'i',
      ),
    );

    if (!errorMatch) {
      return null;
    }

    const error = errorMatch[1].trim();

    // Try to extract expected/actual from error message
    const expectedActual = error.match(
      /(?:expected|want)[:\s]+(.+?)(?:,\s*|\s+)(?:got|actual|but was)[:\s]+(.+)/i,
    );

    return {
      error,
      expected: expectedActual?.[1]?.trim(),
      actual: expectedActual?.[2]?.trim(),
    };
  }

  /**
   * Determine submission status based on execution result and test results
   */
  determineStatus(
    executionStatus: string,
    testsPassed: number,
    testsTotal: number,
  ): 'passed' | 'failed' | 'error' {
    if (executionStatus === 'compileError') {
      return 'error';
    }

    if (testsTotal > 0 && testsPassed < testsTotal) {
      return 'failed';
    }

    if (executionStatus === 'error' || executionStatus === 'timeout') {
      return 'error';
    }

    return 'passed';
  }

  /**
   * Calculate score based on tests passed
   */
  calculateScore(testsPassed: number, testsTotal: number, status: string): number {
    if (testsTotal > 0) {
      return Math.round((testsPassed / testsTotal) * 100);
    }
    return status === 'passed' ? 100 : 0;
  }
}
