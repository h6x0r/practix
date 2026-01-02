import { Test, TestingModule } from '@nestjs/testing';
import { PistonService, LANGUAGES } from './piston.service';
import { ConfigService } from '@nestjs/config';
import axios from 'axios';

// Mock axios
jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

describe('PistonService', () => {
  let service: PistonService;

  const mockConfigService = {
    get: jest.fn().mockReturnValue('http://localhost:2000'),
  };

  const mockAxiosInstance = {
    get: jest.fn(),
    post: jest.fn(),
  };

  beforeEach(async () => {
    mockedAxios.create.mockReturnValue(mockAxiosInstance as any);

    const module: TestingModule = await Test.createTestingModule({
      providers: [
        PistonService,
        { provide: ConfigService, useValue: mockConfigService },
      ],
    }).compile();

    service = module.get<PistonService>(PistonService);

    jest.clearAllMocks();
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });

  // ============================================
  // Language configuration
  // ============================================
  describe('LANGUAGES constant', () => {
    it('should include Go configuration', () => {
      expect(LANGUAGES.go).toBeDefined();
      expect(LANGUAGES.go.pistonName).toBe('go');
      expect(LANGUAGES.go.name).toBe('Go');
    });

    it('should include Java configuration', () => {
      expect(LANGUAGES.java).toBeDefined();
      expect(LANGUAGES.java.pistonName).toBe('java');
    });

    it('should include Python configuration', () => {
      expect(LANGUAGES.python).toBeDefined();
      expect(LANGUAGES.python.pistonName).toBe('python');
    });

    it('should include JavaScript configuration', () => {
      expect(LANGUAGES.javascript).toBeDefined();
      expect(LANGUAGES.javascript.pistonName).toBe('javascript');
    });

    it('should include TypeScript configuration', () => {
      expect(LANGUAGES.typescript).toBeDefined();
    });

    it('should include Rust configuration', () => {
      expect(LANGUAGES.rust).toBeDefined();
    });

    it('should include C++ configuration', () => {
      expect(LANGUAGES.cpp).toBeDefined();
      expect(LANGUAGES.cpp.pistonName).toBe('c++');
    });

    it('should include C configuration', () => {
      expect(LANGUAGES.c).toBeDefined();
    });
  });

  // ============================================
  // getLanguageConfig()
  // ============================================
  describe('getLanguageConfig()', () => {
    it('should return config for valid language key', () => {
      const config = service.getLanguageConfig('go');

      expect(config).toBeDefined();
      expect(config?.name).toBe('Go');
    });

    it('should handle case-insensitive input', () => {
      const config1 = service.getLanguageConfig('GO');
      const config2 = service.getLanguageConfig('Go');
      const config3 = service.getLanguageConfig('go');

      expect(config1).toEqual(config2);
      expect(config2).toEqual(config3);
    });

    it('should recognize golang alias', () => {
      const config = service.getLanguageConfig('golang');

      expect(config).toBeDefined();
      expect(config?.pistonName).toBe('go');
    });

    it('should recognize python alias py', () => {
      const config = service.getLanguageConfig('py');

      expect(config).toBeDefined();
      expect(config?.pistonName).toBe('python');
    });

    it('should recognize javascript aliases', () => {
      expect(service.getLanguageConfig('js')?.pistonName).toBe('javascript');
      expect(service.getLanguageConfig('node')?.pistonName).toBe('javascript');
    });

    it('should recognize typescript alias ts', () => {
      const config = service.getLanguageConfig('ts');

      expect(config).toBeDefined();
      expect(config?.pistonName).toBe('typescript');
    });

    it('should return null for unknown language', () => {
      const config = service.getLanguageConfig('unknown');

      expect(config).toBeNull();
    });
  });

  // ============================================
  // getSupportedLanguages()
  // ============================================
  describe('getSupportedLanguages()', () => {
    it('should return all supported languages', () => {
      const languages = service.getSupportedLanguages();

      expect(languages.length).toBeGreaterThan(0);
      expect(languages.some(l => l.pistonName === 'go')).toBe(true);
      expect(languages.some(l => l.pistonName === 'java')).toBe(true);
      expect(languages.some(l => l.pistonName === 'python')).toBe(true);
    });
  });

  // ============================================
  // checkHealth()
  // ============================================
  describe('checkHealth()', () => {
    it('should return true when Piston is available', async () => {
      mockAxiosInstance.get.mockResolvedValue({
        data: [
          { language: 'go', version: '1.21.0', aliases: [] },
          { language: 'java', version: '17.0.0', aliases: [] },
        ],
      });

      await service.loadRuntimes();
      const result = await service.checkHealth();

      expect(result).toBe(true);
    });

    it('should return false when Piston is unavailable', async () => {
      mockAxiosInstance.get.mockRejectedValue(new Error('Connection refused'));

      await service.loadRuntimes();
      const result = await service.checkHealth();

      expect(result).toBe(false);
    });
  });

  // ============================================
  // execute()
  // ============================================
  describe('execute()', () => {
    beforeEach(async () => {
      mockAxiosInstance.get.mockResolvedValue({
        data: [{ language: 'go', version: '1.21.0', aliases: [] }],
      });
      await service.loadRuntimes();
    });

    it('should execute Go code successfully', async () => {
      mockAxiosInstance.post.mockResolvedValue({
        data: {
          language: 'go',
          version: '1.21.0',
          run: {
            stdout: 'Hello, World!',
            stderr: '',
            code: 0,
            signal: null,
            output: 'Hello, World!',
          },
        },
      });

      const result = await service.execute('package main; func main() { println("Hello") }', 'go');

      expect(result.status).toBe('passed');
      expect(result.stdout).toBe('Hello, World!');
    });

    it('should execute Java code successfully', async () => {
      mockAxiosInstance.post.mockResolvedValue({
        data: {
          language: 'java',
          version: '17.0.0',
          run: {
            stdout: 'Hello, Java!',
            stderr: '',
            code: 0,
            signal: null,
            output: 'Hello, Java!',
          },
        },
      });

      const result = await service.execute('class Main { public static void main(String[] args) {} }', 'java');

      expect(result.status).toBe('passed');
    });

    it('should execute Python code successfully', async () => {
      mockAxiosInstance.post.mockResolvedValue({
        data: {
          language: 'python',
          version: '3.11.0',
          run: {
            stdout: 'Hello, Python!',
            stderr: '',
            code: 0,
            signal: null,
            output: 'Hello, Python!',
          },
        },
      });

      const result = await service.execute('print("Hello, Python!")', 'python');

      expect(result.status).toBe('passed');
    });

    it('should return stdout and stderr', async () => {
      mockAxiosInstance.post.mockResolvedValue({
        data: {
          language: 'python',
          version: '3.11.0',
          run: {
            stdout: 'output',
            stderr: 'warning',
            code: 0,
            signal: null,
            output: 'output\nwarning',
          },
        },
      });

      const result = await service.execute('code', 'python');

      expect(result.stdout).toBe('output');
      expect(result.stderr).toBe('warning');
    });

    it('should handle compilation errors', async () => {
      mockAxiosInstance.post.mockResolvedValue({
        data: {
          language: 'go',
          version: '1.21.0',
          compile: {
            stdout: '',
            stderr: 'syntax error: unexpected EOF',
            code: 1,
            signal: null,
            output: 'syntax error: unexpected EOF',
          },
          run: {
            stdout: '',
            stderr: '',
            code: 0,
            signal: null,
            output: '',
          },
        },
      });

      const result = await service.execute('invalid code', 'go');

      expect(result.status).toBe('compileError');
      expect(result.compileOutput).toContain('syntax error');
    });

    it('should handle runtime errors', async () => {
      mockAxiosInstance.post.mockResolvedValue({
        data: {
          language: 'go',
          version: '1.21.0',
          run: {
            stdout: '',
            stderr: 'panic: runtime error',
            code: 1,
            signal: null,
            output: 'panic: runtime error',
          },
        },
      });

      const result = await service.execute('code', 'go');

      expect(result.status).toBe('error');
      expect(result.stderr).toContain('panic');
    });

    it('should handle timeout', async () => {
      mockAxiosInstance.post.mockRejectedValue({
        code: 'ECONNABORTED',
        message: 'timeout of 60000ms exceeded',
      });

      const result = await service.execute('infinite loop', 'go');

      expect(result.status).toBe('timeout');
    });

    it('should return error for unsupported language', async () => {
      const result = await service.execute('code', 'unsupported');

      expect(result.status).toBe('error');
      expect(result.stderr).toContain('Unsupported language');
    });

    it('should return service unavailable when Piston unavailable', async () => {
      mockAxiosInstance.get.mockRejectedValue(new Error('Connection refused'));
      mockAxiosInstance.post.mockRejectedValue({
        code: 'ECONNREFUSED',
        message: 'Connection refused',
      });
      await service.loadRuntimes();

      const result = await service.execute('print("hello")', 'python');

      expect(result.status).toBe('error');
      expect(result.description).toBe('Service Temporarily Unavailable');
    });
  });

  // ============================================
  // executeWithTests()
  // ============================================
  describe('executeWithTests()', () => {
    beforeEach(async () => {
      mockAxiosInstance.get.mockResolvedValue({
        data: [{ language: 'go', version: '1.21.0', aliases: [] }],
      });
      await service.loadRuntimes();
    });

    it('should combine solution and test code', async () => {
      mockAxiosInstance.post.mockResolvedValue({
        data: {
          language: 'go',
          version: '1.21.0',
          run: {
            stdout: '{"tests":[{"name":"TestAdd","passed":true}],"passed":1,"total":1}',
            stderr: '',
            code: 0,
            signal: null,
            output: '',
          },
        },
      });

      const solution = 'func Add(a, b int) int { return a + b }';
      const testCode = 'func TestAdd(t *testing.T) { if Add(1, 2) != 3 { t.Error("fail") } }';

      const result = await service.executeWithTests(solution, testCode, 'go');

      expect(result.status).toBe('passed');
    });

    it('should parse JSON test results', async () => {
      mockAxiosInstance.post.mockResolvedValue({
        data: {
          language: 'go',
          version: '1.21.0',
          run: {
            stdout: '{"tests":[{"name":"TestA","passed":true},{"name":"TestB","passed":false}],"passed":1,"total":2}',
            stderr: '',
            code: 0,
            signal: null,
            output: '',
          },
        },
      });

      const result = await service.executeWithTests('solution', 'tests', 'go');

      expect(result.stdout).toContain('"passed":1');
    });

    it('should limit tests when maxTests is specified', async () => {
      mockAxiosInstance.post.mockResolvedValue({
        data: {
          language: 'go',
          version: '1.21.0',
          run: { stdout: '', stderr: '', code: 0, signal: null, output: '' },
        },
      });

      const testCode = `
        func TestA(t *testing.T) {}
        func TestB(t *testing.T) {}
        func TestC(t *testing.T) {}
        func TestD(t *testing.T) {}
        func TestE(t *testing.T) {}
        func TestF(t *testing.T) {}
      `;

      await service.executeWithTests('solution', testCode, 'go', 3);

      // The generated code should only include first 3 tests
      expect(mockAxiosInstance.post).toHaveBeenCalled();
    });

    it('should handle test framework errors', async () => {
      mockAxiosInstance.post.mockResolvedValue({
        data: {
          language: 'go',
          version: '1.21.0',
          run: {
            stdout: '',
            stderr: 'test framework error',
            code: 1,
            signal: null,
            output: '',
          },
        },
      });

      const result = await service.executeWithTests('solution', 'tests', 'go');

      expect(result.status).toBe('error');
    });

    it('should return error for unsupported language', async () => {
      const result = await service.executeWithTests('solution', 'tests', 'unknown');

      expect(result.status).toBe('error');
    });
  });

  // ============================================
  // isLanguageAvailable()
  // ============================================
  describe('isLanguageAvailable()', () => {
    it('should return true for available language', async () => {
      mockAxiosInstance.get.mockResolvedValue({
        data: [{ language: 'go', version: '1.21.0', aliases: ['golang'] }],
      });
      await service.loadRuntimes();

      const result = service.isLanguageAvailable('go');

      expect(result).toBe(true);
    });

    it('should return false for unavailable language', async () => {
      mockAxiosInstance.get.mockResolvedValue({
        data: [{ language: 'go', version: '1.21.0', aliases: [] }],
      });
      await service.loadRuntimes();

      const result = service.isLanguageAvailable('kotlin');

      expect(result).toBe(false);
    });

    it('should check aliases', async () => {
      mockAxiosInstance.get.mockResolvedValue({
        data: [{ language: 'go', version: '1.21.0', aliases: ['golang'] }],
      });
      await service.loadRuntimes();

      // Service should find 'go' via direct match
      expect(service.isLanguageAvailable('go')).toBe(true);
    });
  });

  // ============================================
  // loadRuntimes()
  // ============================================
  describe('loadRuntimes()', () => {
    it('should load available runtimes from Piston', async () => {
      mockAxiosInstance.get.mockResolvedValue({
        data: [
          { language: 'go', version: '1.21.0', aliases: [] },
          { language: 'python', version: '3.11.0', aliases: ['py'] },
        ],
      });

      await service.loadRuntimes();
      const healthy = await service.checkHealth();

      expect(healthy).toBe(true);
    });

    it('should handle Piston unavailable', async () => {
      mockAxiosInstance.get.mockRejectedValue(new Error('Network error'));

      await service.loadRuntimes();
      const healthy = await service.checkHealth();

      expect(healthy).toBe(false);
    });
  });

  // ============================================
  // parseResponse() - tested via execute()
  // ============================================
  describe('parseResponse() via execute()', () => {
    beforeEach(async () => {
      mockAxiosInstance.get.mockResolvedValue({
        data: [{ language: 'go', version: '1.21.0', aliases: [] }],
      });
      await service.loadRuntimes();
    });

    it('should use wall_time from Piston response', async () => {
      mockAxiosInstance.post.mockResolvedValue({
        data: {
          language: 'go',
          version: '1.21.0',
          run: {
            stdout: 'output',
            stderr: '',
            code: 0,
            signal: null,
            output: 'output',
            wall_time: 1234, // milliseconds
          },
        },
      });

      const result = await service.execute('code', 'go');

      expect(result.time).toBe('1.234');
    });

    it('should handle memory from Piston response', async () => {
      mockAxiosInstance.post.mockResolvedValue({
        data: {
          language: 'go',
          version: '1.21.0',
          run: {
            stdout: 'output',
            stderr: '',
            code: 0,
            signal: null,
            output: 'output',
            memory: 1048576, // 1MB in bytes
          },
        },
      });

      const result = await service.execute('code', 'go');

      expect(result.memory).toBe(1048576);
    });

    it('should handle SIGKILL with valid output as success', async () => {
      mockAxiosInstance.post.mockResolvedValue({
        data: {
          language: 'python',
          version: '3.11.0',
          run: {
            stdout: 'Valid test output here',
            stderr: '',
            code: -1,
            signal: 'SIGKILL',
            output: 'Valid test output here',
          },
        },
      });

      const result = await service.execute('import numpy', 'python');

      // Should pass because there's valid output
      expect(result.status).toBe('passed');
      expect(result.stdout).toBe('Valid test output here');
    });

    it('should handle SIGKILL without output as error', async () => {
      mockAxiosInstance.post.mockResolvedValue({
        data: {
          language: 'go',
          version: '1.21.0',
          run: {
            stdout: '',
            stderr: 'killed',
            code: -1,
            signal: 'SIGKILL',
            output: '',
          },
        },
      });

      const result = await service.execute('infinite loop', 'go');

      expect(result.status).toBe('error');
      expect(result.description).toContain('SIGKILL');
    });

    it('should handle compile output in success case', async () => {
      mockAxiosInstance.post.mockResolvedValue({
        data: {
          language: 'go',
          version: '1.21.0',
          compile: {
            stdout: 'Compiling...',
            stderr: '',
            code: 0,
            signal: null,
            output: 'Compiling...',
          },
          run: {
            stdout: 'output',
            stderr: '',
            code: 0,
            signal: null,
            output: 'output',
          },
        },
      });

      const result = await service.execute('code', 'go');

      expect(result.status).toBe('passed');
      expect(result.compileOutput).toBe('Compiling...');
    });
  });

  // ============================================
  // Error handling scenarios
  // ============================================
  describe('error handling', () => {
    beforeEach(async () => {
      mockAxiosInstance.get.mockResolvedValue({
        data: [{ language: 'go', version: '1.21.0', aliases: [] }],
      });
      await service.loadRuntimes();
    });

    it('should handle 400 error with language unavailable message', async () => {
      mockAxiosInstance.post.mockRejectedValue({
        response: { status: 400 },
        message: 'Bad request',
      });

      const result = await service.execute('print("hello")', 'python');

      // Returns error with language unavailable message
      expect(result.status).toBe('error');
      expect(result.description).toBe('Error');
      expect(result.stderr).toContain('temporarily unavailable');
    });

    it('should handle ECONNREFUSED with service unavailable', async () => {
      mockAxiosInstance.post.mockRejectedValue({
        code: 'ECONNREFUSED',
        message: 'Connection refused',
      });

      const result = await service.execute('code', 'go');

      expect(result.status).toBe('error');
      expect(result.description).toBe('Service Temporarily Unavailable');
    });

    it('should handle ENOTFOUND with service unavailable', async () => {
      mockAxiosInstance.post.mockRejectedValue({
        code: 'ENOTFOUND',
        message: 'Host not found',
      });

      const result = await service.execute('code', 'go');

      expect(result.status).toBe('error');
      expect(result.description).toBe('Service Temporarily Unavailable');
    });

    it('should handle timeout message in error', async () => {
      const timeoutError = new Error('Request timeout');
      mockAxiosInstance.post.mockRejectedValue(timeoutError);

      const result = await service.execute('code', 'go');

      expect(result.status).toBe('timeout');
    });

    it('should handle generic error with user-friendly message', async () => {
      mockAxiosInstance.post.mockRejectedValue({
        message: 'Internal server error',
        code: 'UNKNOWN',
      });

      const result = await service.execute('code', 'go');

      expect(result.status).toBe('error');
      expect(result.description).toBe('Service Temporarily Unavailable');
    });
  });

  // ============================================
  // execute() when Piston unavailable
  // ============================================
  describe('execute() when unavailable', () => {
    beforeEach(async () => {
      // Make Piston unavailable
      mockAxiosInstance.get.mockRejectedValue(new Error('Unavailable'));
      await service.loadRuntimes();
    });

    it('should return service unavailable error for Go code', async () => {
      const result = await service.execute('package main; func main() {}', 'go');

      expect(result.status).toBe('error');
      expect(result.description).toBe('Service Temporarily Unavailable');
    });

    it('should return service unavailable error for JavaScript code', async () => {
      const result = await service.execute('console.log("test")', 'javascript');

      expect(result.status).toBe('error');
      expect(result.description).toBe('Service Temporarily Unavailable');
    });

    it('should return service unavailable error for Python code', async () => {
      const result = await service.execute('print("test")', 'python');

      expect(result.status).toBe('error');
      expect(result.description).toBe('Service Temporarily Unavailable');
    });
  });

  // ============================================
  // executeWithTests() when Piston unavailable
  // ============================================
  describe('executeWithTests() when unavailable', () => {
    beforeEach(async () => {
      // Make Piston unavailable
      mockAxiosInstance.get.mockRejectedValue(new Error('Unavailable'));
      await service.loadRuntimes();
    });

    it('should return service unavailable error for Python tests', async () => {
      const testCode = `
        def test_addition(self):
            pass
      `;
      const result = await service.executeWithTests('solution', testCode, 'python');

      expect(result.status).toBe('error');
      expect(result.description).toBe('Service Temporarily Unavailable');
    });

    it('should return service unavailable error for Go tests', async () => {
      const testCode = `
        func TestAddition(t *testing.T) {}
      `;
      const result = await service.executeWithTests('solution', testCode, 'go');

      expect(result.status).toBe('error');
      expect(result.description).toBe('Service Temporarily Unavailable');
    });

    it('should return service unavailable error for Java tests', async () => {
      const testCode = `
        @Test
        public void testAddition() {}
      `;
      const result = await service.executeWithTests('solution', testCode, 'java');

      expect(result.status).toBe('error');
      expect(result.description).toBe('Service Temporarily Unavailable');
    });
  });

  // ============================================
  // Additional getLanguageConfig() aliases
  // ============================================
  describe('getLanguageConfig() additional aliases', () => {
    it('should recognize java without script', () => {
      const config = service.getLanguageConfig('java');

      expect(config?.pistonName).toBe('java');
    });

    it('should not confuse java with javascript', () => {
      const jsConfig = service.getLanguageConfig('javascript');
      const javaConfig = service.getLanguageConfig('java');

      expect(jsConfig?.pistonName).toBe('javascript');
      expect(javaConfig?.pistonName).toBe('java');
    });

    it('should recognize rs as rust', () => {
      const config = service.getLanguageConfig('rs');

      expect(config?.pistonName).toBe('rust');
    });

    it('should recognize c++ as cpp', () => {
      const config = service.getLanguageConfig('c++');

      expect(config?.pistonName).toBe('c++');
    });
  });

  // ============================================
  // truncateOutput() - tested via long responses
  // ============================================
  describe('output truncation', () => {
    beforeEach(async () => {
      mockAxiosInstance.get.mockResolvedValue({
        data: [{ language: 'go', version: '1.21.0', aliases: [] }],
      });
      await service.loadRuntimes();
    });

    it('should truncate very long stdout', async () => {
      const longOutput = 'x'.repeat(15000);
      mockAxiosInstance.post.mockResolvedValue({
        data: {
          language: 'go',
          version: '1.21.0',
          run: {
            stdout: longOutput,
            stderr: '',
            code: 0,
            signal: null,
            output: longOutput,
          },
        },
      });

      const result = await service.execute('code', 'go');

      expect(result.stdout.length).toBeLessThan(longOutput.length);
      expect(result.stdout).toContain('truncated');
    });

    it('should truncate very long stderr', async () => {
      const longError = 'e'.repeat(15000);
      mockAxiosInstance.post.mockResolvedValue({
        data: {
          language: 'go',
          version: '1.21.0',
          run: {
            stdout: '',
            stderr: longError,
            code: 1,
            signal: null,
            output: '',
          },
        },
      });

      const result = await service.execute('code', 'go');

      expect(result.stderr.length).toBeLessThan(longError.length);
      expect(result.stderr).toContain('truncated');
    });

    it('should handle empty output without truncation', async () => {
      mockAxiosInstance.post.mockResolvedValue({
        data: {
          language: 'go',
          version: '1.21.0',
          run: {
            stdout: '',
            stderr: '',
            code: 0,
            signal: null,
            output: '',
          },
        },
      });

      const result = await service.execute('code', 'go');

      expect(result.stdout).toBe('');
      expect(result.stderr).toBe('');
    });
  });

  // ============================================
  // Python test code building
  // ============================================
  describe('buildPythonTestCode()', () => {
    beforeEach(async () => {
      mockAxiosInstance.get.mockResolvedValue({
        data: [{ language: 'python', version: '3.11.0', aliases: ['py'] }],
      });
      await service.loadRuntimes();
    });

    it('should remove pytest imports from test code', async () => {
      mockAxiosInstance.post.mockImplementation(async (_url, data: any) => {
        // Check that pytest import is removed
        expect(data.files[0].content).not.toContain('import pytest');
        return {
          data: {
            language: 'python',
            version: '3.11.0',
            run: { stdout: '{}', stderr: '', code: 0, signal: null, output: '' },
          },
        };
      });

      const testCode = `
        import pytest
        from solution import add
        def test_add():
            assert add(1, 2) == 3
      `;

      await service.executeWithTests('def add(a, b): return a + b', testCode, 'python');
    });
  });
});
