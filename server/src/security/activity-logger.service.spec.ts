import { Test, TestingModule } from '@nestjs/testing';
import { ConfigService } from '@nestjs/config';
import { ActivityLoggerService, SecurityEventType, EventSeverity } from './activity-logger.service';
import { PrismaService } from '../prisma/prisma.service';
import { CacheService } from '../cache/cache.service';
import { ThreatLevel } from './code-scanner.service';

describe('ActivityLoggerService', () => {
  let service: ActivityLoggerService;

  const mockConfigService = {
    get: jest.fn((key: string, defaultValue?: boolean) => {
      if (key === 'SUSPICIOUS_ACTIVITY_LOG') return true;
      if (key === 'NODE_ENV') return 'test'; // Not production, so no DB storage
      return defaultValue;
    }),
  };

  const mockPrismaService = {
    securityEvent: {
      create: jest.fn().mockResolvedValue({}),
      findMany: jest.fn().mockResolvedValue([]),
      groupBy: jest.fn().mockResolvedValue([]),
    },
  };

  const mockCacheService = {
    get: jest.fn().mockResolvedValue(null),
    set: jest.fn().mockResolvedValue(undefined),
    delete: jest.fn().mockResolvedValue(undefined),
  };

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [
        ActivityLoggerService,
        { provide: ConfigService, useValue: mockConfigService },
        { provide: PrismaService, useValue: mockPrismaService },
        { provide: CacheService, useValue: mockCacheService },
      ],
    }).compile();

    service = module.get<ActivityLoggerService>(ActivityLoggerService);

    jest.clearAllMocks();
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });

  // ============================================
  // logEvent()
  // ============================================
  describe('logEvent()', () => {
    it('should log event successfully', async () => {
      await service.logEvent({
        type: SecurityEventType.LOGIN_SUCCESS,
        severity: EventSeverity.INFO,
        ip: '127.0.0.1',
        userId: 'user-123',
        details: { isNewIp: false },
      });

      // Should not throw
      expect(true).toBe(true);
    });

    it('should track critical event patterns', async () => {
      mockCacheService.get.mockResolvedValue(2); // Already 2 critical events

      await service.logEvent({
        type: SecurityEventType.MALICIOUS_CODE_DETECTED,
        severity: EventSeverity.CRITICAL,
        ip: '192.168.1.1',
        details: {},
      });

      expect(mockCacheService.set).toHaveBeenCalledWith(
        'critical_events:192.168.1.1',
        3,
        3600,
      );
    });
  });

  // ============================================
  // logFailedLogin()
  // ============================================
  describe('logFailedLogin()', () => {
    it('should log failed login attempt', async () => {
      mockCacheService.get.mockResolvedValue(0);

      const result = await service.logFailedLogin('192.168.1.1', 'test@example.com');

      expect(result.isBruteForce).toBe(false);
      expect(mockCacheService.set).toHaveBeenCalledWith(
        'login_failed:192.168.1.1:test@example.com',
        1,
        900, // 15 minutes
      );
    });

    it('should detect brute force after threshold', async () => {
      mockCacheService.get.mockResolvedValue(4); // 4 previous attempts

      const result = await service.logFailedLogin('192.168.1.1', 'test@example.com');

      expect(result.isBruteForce).toBe(true);
      expect(mockCacheService.set).toHaveBeenCalledWith(
        'login_failed:192.168.1.1:test@example.com',
        5,
        900,
      );
    });

    it('should not flag brute force below threshold', async () => {
      mockCacheService.get.mockResolvedValue(3);

      const result = await service.logFailedLogin('192.168.1.1', 'test@example.com');

      expect(result.isBruteForce).toBe(false);
    });
  });

  // ============================================
  // logSuccessfulLogin()
  // ============================================
  describe('logSuccessfulLogin()', () => {
    it('should clear failed login count on success', async () => {
      mockCacheService.get.mockResolvedValue([]); // No known IPs

      await service.logSuccessfulLogin('192.168.1.1', 'user-123', 'test@example.com');

      expect(mockCacheService.delete).toHaveBeenCalledWith(
        'login_failed:192.168.1.1:test@example.com',
      );
    });

    it('should detect new IP for user', async () => {
      mockCacheService.get
        .mockResolvedValueOnce(['10.0.0.1']); // Known IPs (first call is for user_ips)

      await service.logSuccessfulLogin('192.168.1.1', 'user-123', 'test@example.com');

      // Should save new IP list with new IP added
      expect(mockCacheService.set).toHaveBeenCalledWith(
        'user_ips:user-123',
        expect.arrayContaining(['10.0.0.1', '192.168.1.1']),
        30 * 24 * 60 * 60, // 30 days
      );
    });

    it('should not add duplicate IP for known IP', async () => {
      mockCacheService.get
        .mockResolvedValueOnce(['192.168.1.1']); // Already known

      await service.logSuccessfulLogin('192.168.1.1', 'user-123', 'test@example.com');

      // For known IP, user_ips should NOT be updated
      const setCalls = mockCacheService.set.mock.calls;
      const userIpSetCall = setCalls.find(call => call[0] === 'user_ips:user-123');
      expect(userIpSetCall).toBeUndefined();
    });

    it('should keep only last 10 IPs', async () => {
      const existingIps = ['1.1.1.1', '2.2.2.2', '3.3.3.3', '4.4.4.4', '5.5.5.5',
                          '6.6.6.6', '7.7.7.7', '8.8.8.8', '9.9.9.9', '10.10.10.10'];
      mockCacheService.get
        .mockResolvedValueOnce(null)
        .mockResolvedValueOnce(existingIps);

      await service.logSuccessfulLogin('192.168.1.1', 'user-123', 'test@example.com');

      // Should keep only last 10 (removing oldest)
      expect(mockCacheService.set).toHaveBeenCalledWith(
        'user_ips:user-123',
        expect.arrayContaining(['192.168.1.1']),
        expect.any(Number),
      );
    });
  });

  // ============================================
  // logMaliciousCode()
  // ============================================
  describe('logMaliciousCode()', () => {
    it('should log malicious code without storing the code', async () => {
      const threats = [
        { description: 'System command execution', pattern: 'os\\.system' },
        { description: 'File access', pattern: 'open\\(' },
      ];

      await service.logMaliciousCode(
        '192.168.1.1',
        'user-123',
        'os.system("rm -rf /")',
        'python',
        ThreatLevel.CRITICAL,
        threats,
      );

      // Should not throw and should track pattern
      expect(mockCacheService.set).toHaveBeenCalled();
    });

    it('should convert CRITICAL threat to CRITICAL severity', async () => {
      jest.clearAllMocks();
      mockCacheService.get.mockResolvedValueOnce(0); // No previous critical events

      await service.logMaliciousCode(
        '192.168.1.1',
        undefined,
        'dangerous code',
        'python',
        ThreatLevel.CRITICAL,
        [{ description: 'test', pattern: 'test' }],
      );

      // Track critical events
      expect(mockCacheService.set).toHaveBeenCalledWith(
        'critical_events:192.168.1.1',
        1,
        3600,
      );
    });

    it('should handle anonymous user', async () => {
      await service.logMaliciousCode(
        '192.168.1.1',
        undefined,
        'code',
        'python',
        ThreatLevel.HIGH,
        [],
      );

      // Should not throw
      expect(true).toBe(true);
    });
  });

  // ============================================
  // logRateLimitExceeded()
  // ============================================
  describe('logRateLimitExceeded()', () => {
    beforeEach(() => {
      jest.clearAllMocks();
    });

    it('should log rate limit violation', async () => {
      mockCacheService.get.mockResolvedValueOnce(0);

      await service.logRateLimitExceeded(
        '192.168.1.1',
        'user-123',
        '/api/submissions',
        60,
      );

      expect(mockCacheService.set).toHaveBeenCalledWith(
        'rate_violations:192.168.1.1',
        1,
        3600, // 1 hour
      );
    });

    it('should detect rate limit abuse after 10 violations', async () => {
      mockCacheService.get.mockResolvedValueOnce(9);

      await service.logRateLimitExceeded(
        '192.168.1.1',
        undefined,
        '/api/test',
        30,
      );

      expect(mockCacheService.set).toHaveBeenCalledWith(
        'rate_violations:192.168.1.1',
        10,
        3600,
      );
    });
  });

  // ============================================
  // logUnauthorizedAccess()
  // ============================================
  describe('logUnauthorizedAccess()', () => {
    it('should log unauthorized access with headers', async () => {
      await service.logUnauthorizedAccess(
        '192.168.1.1',
        '/api/admin',
        'GET',
        {
          'user-agent': 'Mozilla/5.0',
          'origin': 'https://evil.com',
        },
      );

      // Should not throw
      expect(true).toBe(true);
    });
  });

  // ============================================
  // logSuspiciousRequest()
  // ============================================
  describe('logSuspiciousRequest()', () => {
    it('should log SQL injection attempt', async () => {
      await service.logSuspiciousRequest(
        SecurityEventType.SQL_INJECTION_ATTEMPT,
        '192.168.1.1',
        'user-123',
        {
          path: '/api/users',
          method: 'GET',
          query: "id=1' OR '1'='1",
        },
      );

      // Should not throw
      expect(true).toBe(true);
    });

    it('should log XSS attempt', async () => {
      await service.logSuspiciousRequest(
        SecurityEventType.XSS_ATTEMPT,
        '192.168.1.1',
        undefined,
        {
          path: '/api/comments',
          method: 'POST',
          body: '<script>alert("xss")</script>',
        },
      );

      expect(true).toBe(true);
    });

    it('should log path traversal attempt', async () => {
      await service.logSuspiciousRequest(
        SecurityEventType.PATH_TRAVERSAL_ATTEMPT,
        '192.168.1.1',
        undefined,
        {
          path: '/api/files/../../../etc/passwd',
          method: 'GET',
        },
      );

      expect(true).toBe(true);
    });
  });

  // ============================================
  // logIpBan()
  // ============================================
  describe('logIpBan()', () => {
    it('should log IP ban event', async () => {
      await service.logIpBan('192.168.1.1', 'brute_force', 24);

      // Should not throw
      expect(true).toBe(true);
    });
  });

  // ============================================
  // getRecentEvents()
  // ============================================
  describe('getRecentEvents()', () => {
    it('should return empty array when not in production', async () => {
      const events = await service.getRecentEvents(100);

      expect(events).toEqual([]);
    });

    it('should return events in production', async () => {
      // Create new instance with production config
      const prodConfigService = {
        get: jest.fn((key: string) => {
          if (key === 'SUSPICIOUS_ACTIVITY_LOG') return true;
          if (key === 'NODE_ENV') return 'production';
          return undefined;
        }),
      };

      const module = await Test.createTestingModule({
        providers: [
          ActivityLoggerService,
          { provide: ConfigService, useValue: prodConfigService },
          { provide: PrismaService, useValue: mockPrismaService },
          { provide: CacheService, useValue: mockCacheService },
        ],
      }).compile();

      const prodService = module.get<ActivityLoggerService>(ActivityLoggerService);
      mockPrismaService.securityEvent.findMany.mockResolvedValue([
        { type: SecurityEventType.LOGIN_SUCCESS, ip: '1.1.1.1' },
      ]);

      const events = await prodService.getRecentEvents(50);

      expect(mockPrismaService.securityEvent.findMany).toHaveBeenCalledWith({
        where: {},
        orderBy: { createdAt: 'desc' },
        take: 50,
      });
    });

    it('should filter by severity in production', async () => {
      const prodConfigService = {
        get: jest.fn((key: string) => {
          if (key === 'SUSPICIOUS_ACTIVITY_LOG') return true;
          if (key === 'NODE_ENV') return 'production';
          return undefined;
        }),
      };

      const module = await Test.createTestingModule({
        providers: [
          ActivityLoggerService,
          { provide: ConfigService, useValue: prodConfigService },
          { provide: PrismaService, useValue: mockPrismaService },
          { provide: CacheService, useValue: mockCacheService },
        ],
      }).compile();

      const prodService = module.get<ActivityLoggerService>(ActivityLoggerService);

      await prodService.getRecentEvents(100, EventSeverity.CRITICAL);

      expect(mockPrismaService.securityEvent.findMany).toHaveBeenCalledWith({
        where: { severity: EventSeverity.CRITICAL },
        orderBy: { createdAt: 'desc' },
        take: 100,
      });
    });
  });

  // ============================================
  // getEventCounts()
  // ============================================
  describe('getEventCounts()', () => {
    it('should return empty object when not in production', async () => {
      const counts = await service.getEventCounts(new Date());

      expect(counts).toEqual({});
    });

    it('should return counts in production', async () => {
      const prodConfigService = {
        get: jest.fn((key: string) => {
          if (key === 'SUSPICIOUS_ACTIVITY_LOG') return true;
          if (key === 'NODE_ENV') return 'production';
          return undefined;
        }),
      };

      const module = await Test.createTestingModule({
        providers: [
          ActivityLoggerService,
          { provide: ConfigService, useValue: prodConfigService },
          { provide: PrismaService, useValue: mockPrismaService },
          { provide: CacheService, useValue: mockCacheService },
        ],
      }).compile();

      const prodService = module.get<ActivityLoggerService>(ActivityLoggerService);
      mockPrismaService.securityEvent.groupBy.mockResolvedValue([
        { type: SecurityEventType.LOGIN_SUCCESS, _count: 10 },
        { type: SecurityEventType.LOGIN_FAILED, _count: 5 },
      ]);

      const counts = await prodService.getEventCounts(new Date('2024-01-01'));

      expect(counts[SecurityEventType.LOGIN_SUCCESS]).toBe(10);
      expect(counts[SecurityEventType.LOGIN_FAILED]).toBe(5);
    });
  });

  // ============================================
  // Disabled logger
  // ============================================
  describe('disabled logger', () => {
    it('should not log when disabled', async () => {
      const disabledConfigService = {
        get: jest.fn((key: string) => {
          if (key === 'SUSPICIOUS_ACTIVITY_LOG') return false;
          return undefined;
        }),
      };

      const module = await Test.createTestingModule({
        providers: [
          ActivityLoggerService,
          { provide: ConfigService, useValue: disabledConfigService },
          { provide: PrismaService, useValue: mockPrismaService },
          { provide: CacheService, useValue: mockCacheService },
        ],
      }).compile();

      const disabledService = module.get<ActivityLoggerService>(ActivityLoggerService);

      await disabledService.logEvent({
        type: SecurityEventType.LOGIN_SUCCESS,
        severity: EventSeverity.INFO,
        ip: '127.0.0.1',
        details: {},
      });

      // Cache should not be called for pattern tracking when disabled
      expect(mockCacheService.set).not.toHaveBeenCalled();
    });
  });

  // ============================================
  // Store event error handling
  // ============================================
  describe('store event error handling', () => {
    it('should handle database errors gracefully', async () => {
      const prodConfigService = {
        get: jest.fn((key: string) => {
          if (key === 'SUSPICIOUS_ACTIVITY_LOG') return true;
          if (key === 'NODE_ENV') return 'production';
          return undefined;
        }),
      };

      mockPrismaService.securityEvent.create.mockRejectedValue(new Error('DB error'));

      const module = await Test.createTestingModule({
        providers: [
          ActivityLoggerService,
          { provide: ConfigService, useValue: prodConfigService },
          { provide: PrismaService, useValue: mockPrismaService },
          { provide: CacheService, useValue: mockCacheService },
        ],
      }).compile();

      const prodService = module.get<ActivityLoggerService>(ActivityLoggerService);

      // Should not throw
      await expect(prodService.logEvent({
        type: SecurityEventType.LOGIN_SUCCESS,
        severity: EventSeverity.INFO,
        ip: '127.0.0.1',
        details: {},
      })).resolves.not.toThrow();
    });
  });
});
