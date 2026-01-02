import { TestParserService } from './test-parser.service';

describe('TestParserService', () => {
  let service: TestParserService;

  beforeEach(() => {
    service = new TestParserService();
  });

  describe('parseTestOutput()', () => {
    describe('JSON format parsing', () => {
      it('should parse valid JSON test output', () => {
        const stdout = '{"tests":[{"name":"Test1","passed":true},{"name":"Test2","passed":false,"error":"expected 2, got 1"}],"passed":1,"total":2}';

        const result = service.parseTestOutput(stdout, '');

        expect(result.passed).toBe(1);
        expect(result.total).toBe(2);
        expect(result.testCases).toHaveLength(2);
        expect(result.testCases[0].name).toBe('Test1');
        expect(result.testCases[0].passed).toBe(true);
        expect(result.testCases[1].name).toBe('Test2');
        expect(result.testCases[1].passed).toBe(false);
        expect(result.testCases[1].error).toBe('expected 2, got 1');
      });

      it('should extract JSON from mixed output', () => {
        const stdout = 'Some log output\n{"tests":[{"name":"Test1","passed":true}],"passed":1,"total":1}\nMore output';

        const result = service.parseTestOutput(stdout, '');

        expect(result.passed).toBe(1);
        expect(result.total).toBe(1);
        expect(result.testCases).toHaveLength(1);
      });

      it('should handle JSON with expected/actual fields', () => {
        const stdout = '{"tests":[{"name":"Test1","passed":false,"expected":"hello","output":"world"}],"passed":0,"total":1}';

        const result = service.parseTestOutput(stdout, '');

        expect(result.testCases[0].expectedOutput).toBe('hello');
        expect(result.testCases[0].actualOutput).toBe('world');
      });

      it('should default test name to "test" if not provided', () => {
        const stdout = '{"tests":[{"passed":true}],"passed":1,"total":1}';

        const result = service.parseTestOutput(stdout, '');

        expect(result.testCases[0].name).toBe('test');
      });
    });

    describe('Legacy Go-style format parsing', () => {
      it('should parse Go test output with PASS', () => {
        const stdout = '=== RUN   TestHelloWorld\n--- PASS: TestHelloWorld (0.00s)\nPASS\nRESULT: 1/1';

        const result = service.parseTestOutput(stdout, '');

        expect(result.passed).toBe(1);
        expect(result.total).toBe(1);
        expect(result.testCases[0].name).toBe('TestHelloWorld');
        expect(result.testCases[0].passed).toBe(true);
      });

      it('should parse Go test output with FAIL', () => {
        const stdout = '=== RUN   TestAdd\n--- FAIL: TestAdd (0.00s)\n    main_test.go:10: error: expected 5, got 3\nFAIL\nRESULT: 0/1';

        const result = service.parseTestOutput(stdout, '');

        expect(result.passed).toBe(0);
        expect(result.total).toBe(1);
        expect(result.testCases[0].name).toBe('TestAdd');
        expect(result.testCases[0].passed).toBe(false);
      });

      it('should parse multiple test results', () => {
        const stdout = `=== RUN   TestAdd
--- PASS: TestAdd (0.00s)
=== RUN   TestSubtract
--- FAIL: TestSubtract (0.00s)
=== RUN   TestMultiply
--- PASS: TestMultiply (0.00s)
RESULT: 2/3`;

        const result = service.parseTestOutput(stdout, '');

        expect(result.passed).toBe(2);
        expect(result.total).toBe(3);
        expect(result.testCases).toHaveLength(3);
      });

      it('should extract count from RESULT line', () => {
        const stdout = '=== RUN   Test1\n--- PASS: Test1\nRESULT: 5/10';

        const result = service.parseTestOutput(stdout, '');

        expect(result.passed).toBe(5);
        expect(result.total).toBe(10);
      });

      it('should count from test cases if no RESULT line', () => {
        const stdout = '=== RUN   Test1\n--- PASS: Test1\n=== RUN   Test2\n--- FAIL: Test2';

        const result = service.parseTestOutput(stdout, '');

        expect(result.passed).toBe(1);
        expect(result.total).toBe(2);
      });
    });

    describe('edge cases', () => {
      it('should handle empty output', () => {
        const result = service.parseTestOutput('', '');

        expect(result.testCases).toHaveLength(0);
        expect(result.passed).toBe(0);
        expect(result.total).toBe(0);
      });

      it('should handle output with only whitespace', () => {
        const result = service.parseTestOutput('   \n   ', '');

        expect(result.testCases).toHaveLength(0);
      });

      it('should handle invalid JSON gracefully', () => {
        const stdout = '{invalid json}';

        const result = service.parseTestOutput(stdout, '');

        // Should fall back to legacy parsing
        expect(result.testCases).toHaveLength(0);
      });

      it('should deduplicate test names', () => {
        const stdout = '=== RUN   TestDuplicate\n=== RUN   TestDuplicate\n--- PASS: TestDuplicate\nRESULT: 1/1';

        const result = service.parseTestOutput(stdout, '');

        expect(result.testCases).toHaveLength(1);
      });
    });
  });

  describe('determineStatus()', () => {
    it('should return "error" for compile errors', () => {
      expect(service.determineStatus('compileError', 0, 0)).toBe('error');
    });

    it('should return "error" for execution errors', () => {
      expect(service.determineStatus('error', 0, 0)).toBe('error');
    });

    it('should return "error" for timeout', () => {
      expect(service.determineStatus('timeout', 0, 0)).toBe('error');
    });

    it('should return "failed" when some tests fail', () => {
      expect(service.determineStatus('passed', 2, 5)).toBe('failed');
    });

    it('should return "passed" when all tests pass', () => {
      expect(service.determineStatus('passed', 5, 5)).toBe('passed');
    });

    it('should return "passed" when no tests (execution passed)', () => {
      expect(service.determineStatus('passed', 0, 0)).toBe('passed');
    });
  });

  describe('calculateScore()', () => {
    it('should calculate percentage score', () => {
      expect(service.calculateScore(3, 10, 'failed')).toBe(30);
    });

    it('should return 100 for all tests passed', () => {
      expect(service.calculateScore(10, 10, 'passed')).toBe(100);
    });

    it('should return 0 for no tests passed', () => {
      expect(service.calculateScore(0, 10, 'failed')).toBe(0);
    });

    it('should return 100 for passed status with no tests', () => {
      expect(service.calculateScore(0, 0, 'passed')).toBe(100);
    });

    it('should return 0 for failed status with no tests', () => {
      expect(service.calculateScore(0, 0, 'failed')).toBe(0);
    });

    it('should round to nearest integer', () => {
      expect(service.calculateScore(1, 3, 'failed')).toBe(33);
    });
  });
});
