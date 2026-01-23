import { Test, TestingModule } from '@nestjs/testing';
import { ForbiddenException } from '@nestjs/common';
import { SecurityValidationService } from './security-validation.service';
import { CodeScannerService, CodeScanResult, ThreatLevel } from './code-scanner.service';
import { ActivityLoggerService } from './activity-logger.service';
import { IpBanService } from './ip-ban.service';

describe('SecurityValidationService', () => {
  let service: SecurityValidationService;
  let codeScannerService: jest.Mocked<CodeScannerService>;
  let activityLogger: jest.Mocked<ActivityLoggerService>;
  let ipBanService: jest.Mocked<IpBanService>;

  const mockCodeScannerService = {
    scan: jest.fn(),
  };

  const mockActivityLogger = {
    logMaliciousCode: jest.fn(),
  };

  const mockIpBanService = {
    handleMaliciousCode: jest.fn(),
  };

  beforeEach(async () => {
    jest.clearAllMocks();

    const module: TestingModule = await Test.createTestingModule({
      providers: [
        SecurityValidationService,
        {
          provide: CodeScannerService,
          useValue: mockCodeScannerService,
        },
        {
          provide: ActivityLoggerService,
          useValue: mockActivityLogger,
        },
        {
          provide: IpBanService,
          useValue: mockIpBanService,
        },
      ],
    }).compile();

    service = module.get<SecurityValidationService>(SecurityValidationService);
    codeScannerService = module.get(CodeScannerService);
    activityLogger = module.get(ActivityLoggerService);
    ipBanService = module.get(IpBanService);
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });

  // ============================================
  // validateCode() - Validate code for security threats
  // ============================================
  describe('validateCode()', () => {
    it('should pass for safe code', async () => {
      mockCodeScannerService.scan.mockReturnValue({
        isSafe: true,
        message: 'Code is safe',
        threatLevel: ThreatLevel.NONE,
        threats: [],
      });

      await expect(
        service.validateCode('print("hello")', 'python'),
      ).resolves.not.toThrow();
    });

    it('should throw ForbiddenException for malicious code', async () => {
      mockCodeScannerService.scan.mockReturnValue({
        isSafe: false,
        message: 'Malicious code detected',
        threatLevel: ThreatLevel.HIGH,
        threats: [{ pattern: 'os.system', description: 'Test threat', threatLevel: ThreatLevel.HIGH }],
      });

      await expect(
        service.validateCode('import os; os.system("rm -rf /")', 'python'),
      ).rejects.toThrow(ForbiddenException);
    });

    it('should log security event when IP is provided', async () => {
      mockCodeScannerService.scan.mockReturnValue({
        isSafe: false,
        message: 'Malicious code',
        threatLevel: ThreatLevel.CRITICAL,
        threats: [{ pattern: 'subprocess', description: 'Test threat', threatLevel: ThreatLevel.CRITICAL }],
      });

      await expect(
        service.validateCode('import subprocess', 'python', {
          ip: '192.168.1.1',
          userId: 'user-123',
        }),
      ).rejects.toThrow();

      expect(mockActivityLogger.logMaliciousCode).toHaveBeenCalledWith(
        '192.168.1.1',
        'user-123',
        'import subprocess',
        'python',
        ThreatLevel.CRITICAL,
        expect.any(Array),
      );
    });

    it('should add strikes to IP when malicious code detected', async () => {
      mockCodeScannerService.scan.mockReturnValue({
        isSafe: false,
        message: 'Malicious code',
        threatLevel: ThreatLevel.HIGH,
        threats: [{ pattern: 'eval', description: 'Test threat', threatLevel: ThreatLevel.HIGH }],
      });

      await expect(
        service.validateCode('eval(code)', 'javascript', { ip: '10.0.0.1' }),
      ).rejects.toThrow();

      expect(mockIpBanService.handleMaliciousCode).toHaveBeenCalledWith(
        '10.0.0.1',
        ThreatLevel.HIGH,
      );
    });

    it('should not log or ban when no IP provided', async () => {
      mockCodeScannerService.scan.mockReturnValue({
        isSafe: false,
        message: 'Malicious code',
        threatLevel: ThreatLevel.MEDIUM,
        threats: [{ pattern: 'exec', description: 'Test threat', threatLevel: ThreatLevel.MEDIUM }],
      });

      await expect(
        service.validateCode('exec("code")', 'python'),
      ).rejects.toThrow();

      expect(mockActivityLogger.logMaliciousCode).not.toHaveBeenCalled();
      expect(mockIpBanService.handleMaliciousCode).not.toHaveBeenCalled();
    });

    it('should include threat info in exception', async () => {
      mockCodeScannerService.scan.mockReturnValue({
        isSafe: false,
        message: 'Network access not allowed',
        threatLevel: ThreatLevel.HIGH,
        threats: [
          { pattern: 'socket', description: 'Test threat', threatLevel: ThreatLevel.HIGH },
          { pattern: 'urllib', description: 'Test threat', threatLevel: ThreatLevel.HIGH },
        ],
      });

      try {
        await service.validateCode('import socket', 'python');
        fail('Should have thrown');
      } catch (error) {
        expect(error).toBeInstanceOf(ForbiddenException);
        const response = (error as ForbiddenException).getResponse() as any;
        expect(response.message).toBe('Network access not allowed');
        expect(response.threatLevel).toBe(ThreatLevel.HIGH);
        expect(response.threats).toContain('socket');
        expect(response.threats).toContain('urllib');
      }
    });

    it('should handle code with only userId context', async () => {
      mockCodeScannerService.scan.mockReturnValue({
        isSafe: false,
        message: 'Malicious code',
        threatLevel: ThreatLevel.LOW,
        threats: [{ pattern: 'suspicious', description: 'Test threat', threatLevel: ThreatLevel.LOW }],
      });

      await expect(
        service.validateCode('suspicious()', 'python', { userId: 'user-123' }),
      ).rejects.toThrow();

      // Should not log or ban without IP
      expect(mockActivityLogger.logMaliciousCode).not.toHaveBeenCalled();
      expect(mockIpBanService.handleMaliciousCode).not.toHaveBeenCalled();
    });
  });

  // ============================================
  // isCodeSafe() - Quick safety check
  // ============================================
  describe('isCodeSafe()', () => {
    it('should return true for safe code', () => {
      mockCodeScannerService.scan.mockReturnValue({
        isSafe: true,
        message: 'Code is safe',
        threatLevel: ThreatLevel.NONE,
        threats: [],
      });

      const result = service.isCodeSafe('console.log("hello")', 'javascript');

      expect(result).toBe(true);
    });

    it('should return false for unsafe code', () => {
      mockCodeScannerService.scan.mockReturnValue({
        isSafe: false,
        message: 'Malicious code',
        threatLevel: ThreatLevel.HIGH,
        threats: [{ pattern: 'fs.unlink', description: 'Test threat', threatLevel: ThreatLevel.HIGH }],
      });

      const result = service.isCodeSafe('fs.unlink("/etc/passwd")', 'javascript');

      expect(result).toBe(false);
    });

    it('should not throw exceptions', () => {
      mockCodeScannerService.scan.mockReturnValue({
        isSafe: false,
        message: 'Dangerous code',
        threatLevel: ThreatLevel.CRITICAL,
        threats: [{ pattern: 'rm -rf', description: 'Test threat', threatLevel: ThreatLevel.CRITICAL }],
      });

      // Should not throw, just return false
      expect(() => service.isCodeSafe('os.system("rm -rf /")', 'python')).not.toThrow();
    });
  });

  // ============================================
  // scanCode() - Get detailed scan result
  // ============================================
  describe('scanCode()', () => {
    it('should return scan result for safe code', () => {
      const expectedResult: CodeScanResult = {
        isSafe: true,
        message: 'Code is safe',
        threatLevel: ThreatLevel.NONE,
        threats: [],
      };
      mockCodeScannerService.scan.mockReturnValue(expectedResult);

      const result = service.scanCode('x = 1 + 1', 'python');

      expect(result).toEqual(expectedResult);
    });

    it('should return scan result for unsafe code', () => {
      const expectedResult: CodeScanResult = {
        isSafe: false,
        message: 'System access detected',
        threatLevel: ThreatLevel.CRITICAL,
        threats: [
          { pattern: 'os.system', description: 'Test threat', threatLevel: ThreatLevel.CRITICAL },
          { pattern: 'subprocess', description: 'Test threat', threatLevel: ThreatLevel.CRITICAL },
        ],
      };
      mockCodeScannerService.scan.mockReturnValue(expectedResult);

      const result = service.scanCode(
        'import os, subprocess; os.system("cat /etc/passwd")',
        'python',
      );

      expect(result).toEqual(expectedResult);
      expect(result.threats).toHaveLength(2);
    });

    it('should not throw for any code', () => {
      mockCodeScannerService.scan.mockReturnValue({
        isSafe: false,
        message: 'Critical threat',
        threatLevel: ThreatLevel.CRITICAL,
        threats: [{ pattern: 'hack', description: 'Test threat', threatLevel: ThreatLevel.CRITICAL }],
      });

      expect(() => service.scanCode('hack()', 'python')).not.toThrow();
    });

    it('should pass language to scanner', () => {
      mockCodeScannerService.scan.mockReturnValue({
        isSafe: true,
        message: 'Safe',
        threatLevel: ThreatLevel.NONE,
        threats: [],
      });

      service.scanCode('code', 'golang');

      expect(mockCodeScannerService.scan).toHaveBeenCalledWith('code', 'golang');
    });
  });

  // ============================================
  // Different languages
  // ============================================
  describe('Language support', () => {
    beforeEach(() => {
      mockCodeScannerService.scan.mockReturnValue({
        isSafe: true,
        message: 'Safe',
        threatLevel: ThreatLevel.NONE,
        threats: [],
      });
    });

    it('should validate Python code', async () => {
      await service.validateCode('def hello(): pass', 'python');
      expect(mockCodeScannerService.scan).toHaveBeenCalledWith('def hello(): pass', 'python');
    });

    it('should validate JavaScript code', async () => {
      await service.validateCode('const x = 1;', 'javascript');
      expect(mockCodeScannerService.scan).toHaveBeenCalledWith('const x = 1;', 'javascript');
    });

    it('should validate Go code', async () => {
      await service.validateCode('package main', 'go');
      expect(mockCodeScannerService.scan).toHaveBeenCalledWith('package main', 'go');
    });

    it('should validate Java code', async () => {
      await service.validateCode('class Main {}', 'java');
      expect(mockCodeScannerService.scan).toHaveBeenCalledWith('class Main {}', 'java');
    });
  });
});
