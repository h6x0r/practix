/**
 * Language-specific test code builders for Judge0 execution
 */

/**
 * Build Python test code combining solution and tests
 */
export function buildPythonTestCode(
  solutionCode: string,
  testCode: string,
  maxTests?: number,
): string {
  const cleanedTestCode = testCode
    .replace(/^import pytest.*$/gm, '')
    .replace(/^from pytest import.*$/gm, '')
    .replace(/^from solution import.*$/gm, '')
    .replace(/^import solution.*$/gm, '')
    .replace(
      /if\s+__name__\s*==\s*['"]__main__['"]\s*:\s*\n?\s*unittest\.main\(\)/gm,
      '',
    );

  const maxTestsLimit = maxTests
    ? `methods = methods[:${maxTests}]  # Quick mode`
    : '';

  return `# Solution code
${solutionCode}

# Test code
${cleanedTestCode}

import json
import sys
import re

if __name__ == "__main__":
    test_class = None
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

        for method_name in methods:
            test_result = {"name": method_name, "passed": False}
            # Call setUp() before each test if it exists
            if hasattr(instance, 'setUp'):
                instance.setUp()
            method = getattr(instance, method_name)
            try:
                method()
                test_result["passed"] = True
                results.append(test_result)
            except AssertionError as e:
                test_result["error"] = str(e)
                results.append(test_result)
                print(json.dumps({"tests": results, "passed": len([r for r in results if r["passed"]]), "total": total_tests}))
                sys.exit(1)
            except Exception as e:
                test_result["error"] = f"{type(e).__name__}: {e}"
                results.append(test_result)
                print(json.dumps({"tests": results, "passed": len([r for r in results if r["passed"]]), "total": total_tests}))
                sys.exit(1)

        print(json.dumps({"tests": results, "passed": len(results), "total": total_tests}))
        sys.exit(0)
    else:
        print(json.dumps({"error": "No test class found", "tests": [], "passed": 0, "total": 0}))
        sys.exit(1)
`;
}

/**
 * Build Go test code combining solution and tests
 */
export function buildGoTestCode(
  solutionCode: string,
  testCode: string,
  maxTests?: number,
): string {
  const cleanSolution = cleanGoCode(solutionCode);
  const cleanTests = cleanGoCode(testCode);

  const solutionImports = extractGoImports(solutionCode);
  const testImports = extractGoImports(testCode);
  const candidateImports = [...new Set([...solutionImports, ...testImports])];

  const combinedCode = cleanSolution + '\n' + cleanTests;
  const additionalImports = candidateImports.filter((imp) => {
    const pkgName = imp.split('/').pop() || imp;
    const usagePattern = new RegExp(`\\b${pkgName}\\.`, 'g');
    return usagePattern.test(combinedCode);
  });

  const testFunctions = extractGoTestFunctions(testCode);
  const testsToRun = maxTests ? testFunctions.slice(0, maxTests) : testFunctions;
  const testCalls = testsToRun
    .map((name) => `    runTest("${name}", ${name}, "")`)
    .join('\n');

  const baseImports = ['encoding/json', 'fmt', 'os', 'regexp'];
  const allImports = [...new Set([...baseImports, ...additionalImports])];
  const importBlock = allImports.map((i) => `    "${i}"`).join('\n');

  return `package main

import (
${importBlock}
)

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
func (t *T) Helper() {}
func (t *T) Parallel() {}
func (t *T) Cleanup(f func()) { defer f() }
func (t *T) Run(name string, f func(*T)) bool {
    subT := &T{name: t.name + "/" + name}
    defer func() {
        if r := recover(); r != nil {
            subT.failed = true
        }
        if subT.failed {
            t.failed = true
            t.errorMsg = subT.errorMsg
        }
    }()
    f(subT)
    return !subT.failed
}

type B struct { N int }
func (b *B) ResetTimer() {}
func (b *B) StopTimer() {}
func (b *B) StartTimer() {}

// Solution code
${cleanSolution}

// Test code
${cleanTests}

var testResults []TestResult
var totalTests = 0

func parseError(errMsg string) (expected, output string) {
    re := regexp.MustCompile(\`(?i)(?:expected|want)[:\\s]+(.+?)[,\\s]+(?:got|but got|actual)[:\\s]+(.+)\`)
    matches := re.FindStringSubmatch(errMsg)
    if len(matches) >= 3 {
        return matches[1], matches[2]
    }
    return "", ""
}

func runTest(name string, fn func(*T), description string) bool {
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
 * Build Java test code combining solution and tests
 */
export function buildJavaTestCode(
  solutionCode: string,
  testCode: string,
): string {
  const testClasses = extractJavaTestClasses(testCode);
  const additionalImports = extractJavaImports(testCode);

  const cleanSolution = solutionCode
    .replace(/import\s+java\.util\.\*;/gm, '')
    .replace(/import\s+java\.util\.\w+;/gm, '')
    .replace(/public\s+class\s+/g, 'class ')
    .trim();

  const cleanTests = testCode
    .replace(/import\s+org\.junit.*$/gm, '')
    .replace(/import\s+static\s+org\.junit.*$/gm, '')
    .replace(/import\s+java\.\w+(\.\w+)*(\.\*)?;/gm, '')
    .replace(/class\s+(Test\d+)\s*\{/g, 'class $1 implements Testable {')
    .replace(/@Test\s*/g, '')
    .replace(/(?<!Assert\.)assertEquals\(/g, 'Assert.assertEquals(')
    .replace(/(?<!Assert\.)assertTrue\(/g, 'Assert.assertTrue(')
    .replace(/(?<!Assert\.)assertFalse\(/g, 'Assert.assertFalse(')
    .replace(/(?<!Assert\.)assertNull\(/g, 'Assert.assertNull(')
    .replace(/(?<!Assert\.)assertNotNull\(/g, 'Assert.assertNotNull(')
    .replace(/(?<!Assert\.)fail\(/g, 'Assert.fail(')
    .trim();

  const testCalls = testClasses
    .map((name) => `        runTest("${name}", new ${name}());`)
    .join('\n');

  const baseImports = ['java.util.ArrayList', 'java.util.List'];
  const allImports = [...new Set([...baseImports, ...additionalImports])];
  const importBlock = allImports.map((imp) => `import ${imp};`).join('\n');

  return `${importBlock}

public class Main {
    static List<String> results = new ArrayList<>();
    static int totalTests = 0;
    static int passed = 0;

    static void runTest(String name, Testable t) {
        totalTests++;
        Assert.reset();
        try {
            t.test();
            passed++;
            results.add("{\\"name\\":\\"" + esc(name) + "\\",\\"passed\\":true}");
        } catch (AssertionError e) {
            String err = e.getMessage() != null ? e.getMessage() : "Assertion failed";
            results.add("{\\"name\\":\\"" + esc(name) + "\\",\\"passed\\":false,\\"error\\":\\"" + esc(err) + "\\"}");
            printResults();
            System.exit(1);
        } catch (Exception e) {
            String err = e.getClass().getSimpleName() + ": " + e.getMessage();
            results.add("{\\"name\\":\\"" + esc(name) + "\\",\\"passed\\":false,\\"error\\":\\"" + esc(err) + "\\"}");
            printResults();
            System.exit(1);
        }
    }

    static String esc(String s) {
        if (s == null) return "";
        return s.replace("\\\\", "\\\\\\\\").replace("\\"", "\\\\\\"").replace("\\n", "\\\\n");
    }

    static void printResults() {
        StringBuilder sb = new StringBuilder();
        sb.append("{\\"tests\\":[");
        for (int i = 0; i < results.size(); i++) {
            if (i > 0) sb.append(",");
            sb.append(results.get(i));
        }
        sb.append("],\\"passed\\":").append(passed).append(",\\"total\\":").append(totalTests).append("}");
        System.out.println(sb.toString());
    }

    public static void main(String[] args) {
${testCalls}
        printResults();
        System.exit(0);
    }
}

// Solution code
${cleanSolution}

interface Testable { void test() throws Exception; }

class Assert {
    static void assertEquals(Object expected, Object actual) {
        if (!java.util.Objects.equals(expected, actual))
            throw new AssertionError("Expected: " + expected + ", Got: " + actual);
    }
    static void assertEquals(String msg, Object expected, Object actual) {
        if (!java.util.Objects.equals(expected, actual))
            throw new AssertionError(msg + " - Expected: " + expected + ", Got: " + actual);
    }
    static void assertEquals(int expected, int actual) {
        if (expected != actual) throw new AssertionError("Expected: " + expected + ", Got: " + actual);
    }
    static void assertEquals(String msg, int expected, int actual) {
        if (expected != actual) throw new AssertionError(msg + " - Expected: " + expected + ", Got: " + actual);
    }
    static void assertEquals(long expected, long actual) {
        if (expected != actual) throw new AssertionError("Expected: " + expected + ", Got: " + actual);
    }
    static void assertEquals(double expected, double actual, double delta) {
        if (Math.abs(expected - actual) > delta) throw new AssertionError("Expected: " + expected + ", Got: " + actual);
    }
    static void assertTrue(boolean condition) {
        if (!condition) throw new AssertionError("Expected true but got false");
    }
    static void assertTrue(String msg, boolean condition) {
        if (!condition) throw new AssertionError(msg);
    }
    static void assertFalse(boolean condition) {
        if (condition) throw new AssertionError("Expected false but got true");
    }
    static void assertFalse(String msg, boolean condition) {
        if (condition) throw new AssertionError(msg);
    }
    static void assertNull(Object obj) {
        if (obj != null) throw new AssertionError("Expected null but got: " + obj);
    }
    static void assertNull(String msg, Object obj) {
        if (obj != null) throw new AssertionError(msg + " - Expected null but got: " + obj);
    }
    static void assertNotNull(Object obj) {
        if (obj == null) throw new AssertionError("Expected not null but got null");
    }
    static void assertNotNull(String msg, Object obj) {
        if (obj == null) throw new AssertionError(msg + " - Expected not null but got null");
    }
    static void fail(String msg) {
        throw new AssertionError(msg);
    }
    static void reset() {}
}

// Test classes
${cleanTests}
`;
}

// ============ Helper functions ============

function cleanGoCode(code: string): string {
  return code
    .replace(/^package\s+\w+\s*$/gm, '')
    .replace(/import\s*\([\s\S]*?\)/g, '')
    .replace(/import\s+"[^"]+"/g, '')
    .replace(/\*testing\.T/g, '*T')
    .trim();
}

export function extractGoImports(code: string): string[] {
  const imports: string[] = [];
  const multiImportMatch = code.match(/import\s*\(([\s\S]*?)\)/);
  if (multiImportMatch) {
    const packageMatches = multiImportMatch[1].match(/"([^"]+)"/g);
    if (packageMatches) {
      packageMatches.forEach((pkg) => {
        const cleanPkg = pkg.replace(/"/g, '');
        if (cleanPkg !== 'testing') imports.push(cleanPkg);
      });
    }
  }
  const singleImportMatches = code.match(/import\s+"([^"]+)"/g);
  if (singleImportMatches) {
    singleImportMatches.forEach((match) => {
      const pkgMatch = match.match(/"([^"]+)"/);
      if (pkgMatch && pkgMatch[1] !== 'testing') imports.push(pkgMatch[1]);
    });
  }
  return imports;
}

export function extractGoTestFunctions(testCode: string): string[] {
  const functions: string[] = [];
  const regex = /func\s+(Test\w+)\s*\(/g;
  let match;
  while ((match = regex.exec(testCode)) !== null) {
    if (/^Test[A-Za-z0-9_]+$/.test(match[1])) {
      functions.push(match[1]);
    }
  }
  return functions;
}

export function extractJavaTestClasses(testCode: string): string[] {
  const classes: string[] = [];
  const classRegex = /class\s+(Test\d+)\s*\{/g;
  let match;
  while ((match = classRegex.exec(testCode)) !== null) {
    classes.push(match[1]);
  }
  return classes;
}

export function extractJavaImports(testCode: string): string[] {
  const imports: string[] = [];
  const importRegex = /import\s+(java\.\w+(?:\.\w+)*(?:\.\*)?);/g;
  let match;
  while ((match = importRegex.exec(testCode)) !== null) {
    if (!match[1].startsWith('java.util.') && !match[1].startsWith('java.util')) {
      imports.push(match[1]);
    }
  }
  return imports;
}
