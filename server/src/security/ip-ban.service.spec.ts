import { Test, TestingModule } from '@nestjs/testing';
import { HttpException, HttpStatus } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { IpBanService, BanReason, BanRecord } from './ip-ban.service';
import { CacheService } from '../cache/cache.service';
import { ActivityLoggerService, SecurityEventType } from './activity-logger.service';
import { ThreatLevel } from './code-scanner.service';

describe('IpBanService', () => {
  let service: IpBanService;
  let cacheService: jest.Mocked<CacheService>;
  let activityLogger: jest.Mocked<ActivityLoggerService>;

  const mockConfigValues = {
    ENABLE_IP_BAN: true,
    IP_BAN_THRESHOLD: 5,
    IP_BAN_DURATION_HOURS: 24,
  };

  const mockCacheService = {
    get: jest.fn(),
    set: jest.fn(),
    delete: jest.fn(),
  };

  const mockActivityLogger = {
    logIpBan: jest.fn(),
    logEvent: jest.fn(),
  };

  beforeEach(async () => {
    jest.clearAllMocks();

    const module: TestingModule = await Test.createTestingModule({
      providers: [
        IpBanService,
        {
          provide: ConfigService,
          useValue: {
            get: jest.fn((key: string, defaultValue?: any) =>
              mockConfigValues[key] ?? defaultValue,
            ),
          },
        },
        {
          provide: CacheService,
          useValue: mockCacheService,
        },
        {
          provide: ActivityLoggerService,
          useValue: mockActivityLogger,
        },
      ],
    }).compile();

    service = module.get<IpBanService>(IpBanService);
    cacheService = module.get(CacheService);
    activityLogger = module.get(ActivityLoggerService);
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });

  // ============================================
  // isBanned() - Check if IP is banned
  // ============================================
  describe('isBanned()', () => {
    it('should return false when ban is disabled', async () => {
      const disabledModule = await Test.createTestingModule({
        providers: [
          IpBanService,
          {
            provide: ConfigService,
            useValue: {
              get: jest.fn((key: string, defaultValue?: any) =>
                key === 'ENABLE_IP_BAN' ? false : mockConfigValues[key] ?? defaultValue,
              ),
            },
          },
          { provide: CacheService, useValue: mockCacheService },
          { provide: ActivityLoggerService, useValue: mockActivityLogger },
        ],
      }).compile();

      const disabledService = disabledModule.get<IpBanService>(IpBanService);
      const result = await disabledService.isBanned('192.168.1.1');

      expect(result).toBe(false);
    });

    it('should return false for non-banned IP', async () => {
      mockCacheService.get.mockResolvedValue(null);

      const result = await service.isBanned('192.168.1.1');

      expect(result).toBe(false);
    });

    it('should return true for banned IP', async () => {
      const banRecord: BanRecord = {
        ip: '192.168.1.1',
        reason: BanReason.BRUTE_FORCE,
        bannedAt: Date.now(),
        expiresAt: Date.now() + 1000 * 60 * 60, // 1 hour from now
        strikes: 5,
      };
      mockCacheService.get.mockResolvedValue(banRecord);

      const result = await service.isBanned('192.168.1.1');

      expect(result).toBe(true);
    });

    it('should return false and unban if ban has expired', async () => {
      const expiredBanRecord: BanRecord = {
        ip: '192.168.1.1',
        reason: BanReason.BRUTE_FORCE,
        bannedAt: Date.now() - 2000 * 60 * 60 * 1000, // 2000 hours ago
        expiresAt: Date.now() - 1000, // expired 1 second ago
        strikes: 5,
      };
      mockCacheService.get.mockResolvedValue(expiredBanRecord);

      const result = await service.isBanned('192.168.1.1');

      expect(result).toBe(false);
      expect(mockCacheService.delete).toHaveBeenCalled();
    });
  });

  // ============================================
  // getBanRecord() - Get ban record
  // ============================================
  describe('getBanRecord()', () => {
    it('should return ban record from cache', async () => {
      const banRecord: BanRecord = {
        ip: '192.168.1.1',
        reason: BanReason.MALICIOUS_CODE,
        bannedAt: Date.now(),
        expiresAt: Date.now() + 1000 * 60 * 60,
        strikes: 3,
        details: 'Test ban',
      };
      mockCacheService.get.mockResolvedValue(banRecord);

      const result = await service.getBanRecord('192.168.1.1');

      expect(result).toEqual(banRecord);
      expect(mockCacheService.get).toHaveBeenCalledWith('ip_ban:192.168.1.1');
    });

    it('should return null for non-banned IP', async () => {
      mockCacheService.get.mockResolvedValue(null);

      const result = await service.getBanRecord('192.168.1.1');

      expect(result).toBeNull();
    });
  });

  // ============================================
  // addStrike() - Add strike to IP
  // ============================================
  describe('addStrike()', () => {
    it('should return false when ban is disabled', async () => {
      const disabledModule = await Test.createTestingModule({
        providers: [
          IpBanService,
          {
            provide: ConfigService,
            useValue: {
              get: jest.fn((key: string, defaultValue?: any) =>
                key === 'ENABLE_IP_BAN' ? false : mockConfigValues[key] ?? defaultValue,
              ),
            },
          },
          { provide: CacheService, useValue: mockCacheService },
          { provide: ActivityLoggerService, useValue: mockActivityLogger },
        ],
      }).compile();

      const disabledService = disabledModule.get<IpBanService>(IpBanService);
      const result = await disabledService.addStrike('192.168.1.1', BanReason.BRUTE_FORCE);

      expect(result).toBe(false);
    });

    it('should add first strike', async () => {
      mockCacheService.get.mockResolvedValueOnce(null); // No existing strikes

      const result = await service.addStrike('192.168.1.1', BanReason.BRUTE_FORCE);

      expect(result).toBe(false); // Not banned yet
      expect(mockCacheService.set).toHaveBeenCalledWith(
        'ip_strikes:192.168.1.1',
        1,
        24 * 60 * 60,
      );
    });

    it('should increment existing strikes', async () => {
      mockCacheService.get.mockResolvedValueOnce(3); // 3 existing strikes

      const result = await service.addStrike('192.168.1.1', BanReason.RATE_LIMIT_ABUSE);

      expect(result).toBe(false); // Not banned yet (4 < 5)
      expect(mockCacheService.set).toHaveBeenCalledWith(
        'ip_strikes:192.168.1.1',
        4,
        24 * 60 * 60,
      );
    });

    it('should ban IP when threshold reached', async () => {
      mockCacheService.get
        .mockResolvedValueOnce(4) // 4 existing strikes
        .mockResolvedValueOnce(5); // Current strikes for ban record

      const result = await service.addStrike('192.168.1.1', BanReason.SUSPICIOUS_ACTIVITY);

      expect(result).toBe(true); // Banned
      expect(mockCacheService.set).toHaveBeenCalledWith(
        expect.stringContaining('ip_ban:'),
        expect.objectContaining({
          ip: '192.168.1.1',
          reason: BanReason.SUSPICIOUS_ACTIVITY,
        }),
        24 * 60 * 60,
      );
    });

    it('should include details in strike', async () => {
      mockCacheService.get.mockResolvedValueOnce(4).mockResolvedValueOnce(5);

      await service.addStrike('192.168.1.1', BanReason.MALICIOUS_CODE, 'Detected crypto miner');

      expect(mockCacheService.set).toHaveBeenCalledWith(
        expect.stringContaining('ip_ban:'),
        expect.objectContaining({
          details: 'Detected crypto miner',
        }),
        expect.any(Number),
      );
    });
  });

  // ============================================
  // ban() - Ban an IP
  // ============================================
  describe('ban()', () => {
    it('should not ban when disabled', async () => {
      const disabledModule = await Test.createTestingModule({
        providers: [
          IpBanService,
          {
            provide: ConfigService,
            useValue: {
              get: jest.fn((key: string, defaultValue?: any) =>
                key === 'ENABLE_IP_BAN' ? false : mockConfigValues[key] ?? defaultValue,
              ),
            },
          },
          { provide: CacheService, useValue: mockCacheService },
          { provide: ActivityLoggerService, useValue: mockActivityLogger },
        ],
      }).compile();

      const disabledService = disabledModule.get<IpBanService>(IpBanService);
      await disabledService.ban('192.168.1.1', BanReason.MANUAL);

      expect(mockCacheService.set).not.toHaveBeenCalled();
    });

    it('should create ban record', async () => {
      mockCacheService.get.mockResolvedValue(3); // Current strikes

      await service.ban('192.168.1.1', BanReason.MANUAL, 'Admin ban');

      expect(mockCacheService.set).toHaveBeenCalledWith(
        'ip_ban:192.168.1.1',
        expect.objectContaining({
          ip: '192.168.1.1',
          reason: BanReason.MANUAL,
          strikes: 3,
          details: 'Admin ban',
        }),
        24 * 60 * 60,
      );
    });

    it('should log ban event', async () => {
      mockCacheService.get.mockResolvedValue(0);

      await service.ban('192.168.1.1', BanReason.BRUTE_FORCE);

      expect(mockActivityLogger.logIpBan).toHaveBeenCalledWith(
        '192.168.1.1',
        BanReason.BRUTE_FORCE,
        24,
      );
    });
  });

  // ============================================
  // unban() - Unban an IP
  // ============================================
  describe('unban()', () => {
    it('should delete ban and strike records', async () => {
      await service.unban('192.168.1.1', 'manual');

      expect(mockCacheService.delete).toHaveBeenCalledWith('ip_ban:192.168.1.1');
      expect(mockCacheService.delete).toHaveBeenCalledWith('ip_strikes:192.168.1.1');
    });

    it('should log unban event', async () => {
      await service.unban('192.168.1.1', 'expired');

      expect(mockActivityLogger.logEvent).toHaveBeenCalledWith({
        type: SecurityEventType.IP_UNBANNED,
        severity: 'info',
        ip: '192.168.1.1',
        details: { reason: 'expired' },
      });
    });
  });

  // ============================================
  // getStrikes() - Get strike count
  // ============================================
  describe('getStrikes()', () => {
    it('should return strike count', async () => {
      mockCacheService.get.mockResolvedValue(3);

      const result = await service.getStrikes('192.168.1.1');

      expect(result).toBe(3);
    });

    it('should return 0 for no strikes', async () => {
      mockCacheService.get.mockResolvedValue(null);

      const result = await service.getStrikes('192.168.1.1');

      expect(result).toBe(0);
    });
  });

  // ============================================
  // handleBruteForce() - Handle brute force
  // ============================================
  describe('handleBruteForce()', () => {
    it('should add strike for brute force', async () => {
      mockCacheService.get.mockResolvedValue(null);

      await service.handleBruteForce('192.168.1.1');

      expect(mockCacheService.set).toHaveBeenCalledWith(
        'ip_strikes:192.168.1.1',
        1,
        expect.any(Number),
      );
    });
  });

  // ============================================
  // handleRateLimitAbuse() - Handle rate limit abuse
  // ============================================
  describe('handleRateLimitAbuse()', () => {
    it('should add strike with endpoint info', async () => {
      mockCacheService.get.mockResolvedValue(null);

      await service.handleRateLimitAbuse('192.168.1.1', '/api/submissions');

      expect(mockCacheService.set).toHaveBeenCalled();
    });
  });

  // ============================================
  // handleMaliciousCode() - Handle malicious code
  // ============================================
  describe('handleMaliciousCode()', () => {
    it('should immediately ban for CRITICAL threat', async () => {
      mockCacheService.get.mockResolvedValue(0);

      await service.handleMaliciousCode('192.168.1.1', ThreatLevel.CRITICAL);

      expect(mockCacheService.set).toHaveBeenCalledWith(
        'ip_ban:192.168.1.1',
        expect.objectContaining({
          reason: BanReason.MALICIOUS_CODE,
        }),
        expect.any(Number),
      );
    });

    it('should add 2 strikes for HIGH threat', async () => {
      mockCacheService.get.mockResolvedValue(0);

      await service.handleMaliciousCode('192.168.1.1', ThreatLevel.HIGH);

      // Should call set twice (2 strikes)
      expect(mockCacheService.set).toHaveBeenCalledTimes(2);
    });

    it('should add 1 strike for MEDIUM threat', async () => {
      mockCacheService.get.mockResolvedValue(0);

      await service.handleMaliciousCode('192.168.1.1', ThreatLevel.MEDIUM);

      expect(mockCacheService.set).toHaveBeenCalledTimes(1);
    });

    it('should not add strike for LOW threat', async () => {
      await service.handleMaliciousCode('192.168.1.1', ThreatLevel.LOW);

      // LOW threat doesn't trigger any action
      expect(mockCacheService.set).not.toHaveBeenCalled();
    });
  });

  // ============================================
  // checkAndThrow() - Middleware check
  // ============================================
  describe('checkAndThrow()', () => {
    it('should not throw when ban is disabled', async () => {
      const disabledModule = await Test.createTestingModule({
        providers: [
          IpBanService,
          {
            provide: ConfigService,
            useValue: {
              get: jest.fn((key: string, defaultValue?: any) =>
                key === 'ENABLE_IP_BAN' ? false : mockConfigValues[key] ?? defaultValue,
              ),
            },
          },
          { provide: CacheService, useValue: mockCacheService },
          { provide: ActivityLoggerService, useValue: mockActivityLogger },
        ],
      }).compile();

      const disabledService = disabledModule.get<IpBanService>(IpBanService);

      await expect(disabledService.checkAndThrow('192.168.1.1')).resolves.not.toThrow();
    });

    it('should not throw for non-banned IP', async () => {
      mockCacheService.get.mockResolvedValue(null);

      await expect(service.checkAndThrow('192.168.1.1')).resolves.not.toThrow();
    });

    it('should throw HttpException for banned IP', async () => {
      const banRecord: BanRecord = {
        ip: '192.168.1.1',
        reason: BanReason.BRUTE_FORCE,
        bannedAt: Date.now(),
        expiresAt: Date.now() + 1000 * 60 * 60, // 1 hour from now
        strikes: 5,
      };
      mockCacheService.get.mockResolvedValue(banRecord);

      await expect(service.checkAndThrow('192.168.1.1')).rejects.toThrow(HttpException);
    });

    it('should include ban info in exception', async () => {
      const banRecord: BanRecord = {
        ip: '192.168.1.1',
        reason: BanReason.MALICIOUS_CODE,
        bannedAt: Date.now(),
        expiresAt: Date.now() + 2 * 60 * 60 * 1000, // 2 hours from now
        strikes: 5,
      };
      mockCacheService.get.mockResolvedValue(banRecord);

      try {
        await service.checkAndThrow('192.168.1.1');
        fail('Should have thrown');
      } catch (error) {
        expect(error).toBeInstanceOf(HttpException);
        expect((error as HttpException).getStatus()).toBe(HttpStatus.FORBIDDEN);
        const response = (error as HttpException).getResponse() as any;
        expect(response.error).toBe('IP_BANNED');
        expect(response.reason).toBe(BanReason.MALICIOUS_CODE);
      }
    });
  });

  // ============================================
  // getBanInfo() - Get ban info for display
  // ============================================
  describe('getBanInfo()', () => {
    it('should return isBanned: false for non-banned IP', async () => {
      mockCacheService.get.mockResolvedValue(null);

      const result = await service.getBanInfo('192.168.1.1');

      expect(result).toEqual({ isBanned: false });
    });

    it('should return ban details for banned IP', async () => {
      const banRecord: BanRecord = {
        ip: '192.168.1.1',
        reason: BanReason.BRUTE_FORCE,
        bannedAt: Date.now(),
        expiresAt: Date.now() + 3 * 60 * 60 * 1000, // 3 hours from now
        strikes: 5,
      };
      mockCacheService.get.mockResolvedValue(banRecord);

      const result = await service.getBanInfo('192.168.1.1');

      expect(result.isBanned).toBe(true);
      expect(result.remainingHours).toBe(3);
      expect(result.reason).toBe(BanReason.BRUTE_FORCE);
    });

    it('should unban and return false for expired ban', async () => {
      const expiredBanRecord: BanRecord = {
        ip: '192.168.1.1',
        reason: BanReason.BRUTE_FORCE,
        bannedAt: Date.now() - 2000 * 60 * 60 * 1000,
        expiresAt: Date.now() - 1000, // expired
        strikes: 5,
      };
      mockCacheService.get.mockResolvedValue(expiredBanRecord);

      const result = await service.getBanInfo('192.168.1.1');

      expect(result.isBanned).toBe(false);
      expect(mockCacheService.delete).toHaveBeenCalled();
    });
  });
});
