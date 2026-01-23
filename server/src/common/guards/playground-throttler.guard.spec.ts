import { Test, TestingModule } from '@nestjs/testing';
import { ExecutionContext, HttpException, HttpStatus } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import {
  PlaygroundThrottlerGuard,
  RateLimitResponse,
} from './playground-throttler.guard';
import { CacheService } from '../../cache/cache.service';
import { AccessControlService } from '../../subscriptions/access-control.service';

describe('PlaygroundThrottlerGuard', () => {
  let guard: PlaygroundThrottlerGuard;
  let cacheService: jest.Mocked<CacheService>;
  let accessControlService: jest.Mocked<AccessControlService>;

  const mockConfigValues = {
    RATE_LIMIT_RUN_FREE: 10,
    RATE_LIMIT_RUN_AUTH: 10,
    RATE_LIMIT_RUN_PREMIUM: 5,
  };

  const mockCacheService = {
    get: jest.fn(),
    set: jest.fn(),
    del: jest.fn(),
  };

  const mockAccessControlService = {
    hasGlobalAccess: jest.fn(),
    hasCourseAccess: jest.fn(),
  };

  const createMockContext = (
    user?: { userId: string },
    ip?: string,
    headers: Record<string, string | string[]> = {},
  ): ExecutionContext => {
    const request = {
      user,
      ip: ip || '127.0.0.1',
      socket: { remoteAddress: ip || '127.0.0.1' },
      headers,
    };
    const response = {
      setHeader: jest.fn(),
    };

    return {
      switchToHttp: () => ({
        getRequest: () => request,
        getResponse: () => response,
      }),
    } as unknown as ExecutionContext;
  };

  beforeEach(async () => {
    jest.clearAllMocks();

    const module: TestingModule = await Test.createTestingModule({
      providers: [
        PlaygroundThrottlerGuard,
        {
          provide: ConfigService,
          useValue: {
            get: jest.fn((key: string, defaultValue?: number) =>
              mockConfigValues[key] ?? defaultValue,
            ),
          },
        },
        {
          provide: CacheService,
          useValue: mockCacheService,
        },
        {
          provide: AccessControlService,
          useValue: mockAccessControlService,
        },
      ],
    }).compile();

    guard = module.get<PlaygroundThrottlerGuard>(PlaygroundThrottlerGuard);
    cacheService = module.get(CacheService);
    accessControlService = module.get(AccessControlService);
  });

  it('should be defined', () => {
    expect(guard).toBeDefined();
  });

  // ============================================
  // checkRateLimit() - Rate limit logic
  // ============================================
  describe('checkRateLimit()', () => {
    describe('Unauthenticated users', () => {
      it('should allow first request from unauthenticated user', async () => {
        mockCacheService.get.mockResolvedValue(null);
        const context = createMockContext(undefined, '192.168.1.1');

        const result = await guard.checkRateLimit(context);

        expect(result).toBeNull();
        expect(mockCacheService.set).toHaveBeenCalledWith(
          'playground_rate:ip:192.168.1.1',
          expect.any(Number),
          10, // Free rate limit
        );
      });

      it('should throttle unauthenticated user within rate limit window', async () => {
        const now = Date.now();
        mockCacheService.get.mockResolvedValue(now - 3000); // 3 seconds ago
        const context = createMockContext(undefined, '192.168.1.1');

        const result = await guard.checkRateLimit(context);

        expect(result).not.toBeNull();
        expect(result!.statusCode).toBe(429);
        expect(result!.isPremium).toBe(false);
        expect(result!.rateLimitSeconds).toBe(10);
        expect(result!.retryAfter).toBe(7); // 10 - 3 = 7 seconds remaining
      });

      it('should allow unauthenticated user after rate limit window expires', async () => {
        const now = Date.now();
        mockCacheService.get.mockResolvedValue(now - 15000); // 15 seconds ago
        const context = createMockContext(undefined, '192.168.1.1');

        const result = await guard.checkRateLimit(context);

        expect(result).toBeNull();
      });
    });

    describe('Authenticated (non-premium) users', () => {
      it('should allow first request from authenticated user', async () => {
        mockCacheService.get.mockResolvedValue(null);
        mockAccessControlService.hasGlobalAccess.mockResolvedValue(false);
        const context = createMockContext({ userId: 'user-123' });

        const result = await guard.checkRateLimit(context);

        expect(result).toBeNull();
        expect(mockCacheService.set).toHaveBeenCalledWith(
          'playground_rate:user:user-123',
          expect.any(Number),
          10, // Auth rate limit
        );
      });

      it('should throttle authenticated user within rate limit window', async () => {
        const now = Date.now();
        mockCacheService.get.mockResolvedValue(now - 5000); // 5 seconds ago
        mockAccessControlService.hasGlobalAccess.mockResolvedValue(false);
        const context = createMockContext({ userId: 'user-123' });

        const result = await guard.checkRateLimit(context);

        expect(result).not.toBeNull();
        expect(result!.retryAfter).toBe(5); // 10 - 5 = 5 seconds remaining
        expect(result!.isPremium).toBe(false);
      });

      it('should use user ID for cache key, not IP', async () => {
        mockCacheService.get.mockResolvedValue(null);
        mockAccessControlService.hasGlobalAccess.mockResolvedValue(false);
        const context = createMockContext({ userId: 'user-123' }, '192.168.1.1');

        await guard.checkRateLimit(context);

        expect(mockCacheService.set).toHaveBeenCalledWith(
          'playground_rate:user:user-123',
          expect.any(Number),
          expect.any(Number),
        );
        expect(mockCacheService.set).not.toHaveBeenCalledWith(
          expect.stringContaining('ip:'),
          expect.any(Number),
          expect.any(Number),
        );
      });
    });

    describe('Premium users', () => {
      it('should allow first request from premium user', async () => {
        mockCacheService.get.mockResolvedValue(null);
        mockAccessControlService.hasGlobalAccess.mockResolvedValue(true);
        const context = createMockContext({ userId: 'premium-user-123' });

        const result = await guard.checkRateLimit(context);

        expect(result).toBeNull();
        expect(mockCacheService.set).toHaveBeenCalledWith(
          'playground_rate:user:premium-user-123',
          expect.any(Number),
          5, // Premium rate limit
        );
      });

      it('should use shorter rate limit for premium users', async () => {
        const now = Date.now();
        mockCacheService.get.mockResolvedValue(now - 3000); // 3 seconds ago
        mockAccessControlService.hasGlobalAccess.mockResolvedValue(true);
        const context = createMockContext({ userId: 'premium-user-123' });

        const result = await guard.checkRateLimit(context);

        expect(result).not.toBeNull();
        expect(result!.retryAfter).toBe(2); // 5 - 3 = 2 seconds remaining
        expect(result!.isPremium).toBe(true);
        expect(result!.rateLimitSeconds).toBe(5);
      });

      it('should allow premium user after 5 seconds', async () => {
        const now = Date.now();
        mockCacheService.get.mockResolvedValue(now - 6000); // 6 seconds ago
        mockAccessControlService.hasGlobalAccess.mockResolvedValue(true);
        const context = createMockContext({ userId: 'premium-user-123' });

        const result = await guard.checkRateLimit(context);

        expect(result).toBeNull();
      });
    });
  });

  // ============================================
  // canActivate() - Guard activation
  // ============================================
  describe('canActivate()', () => {
    it('should return true when request is allowed', async () => {
      mockCacheService.get.mockResolvedValue(null);
      const context = createMockContext(undefined, '192.168.1.1');

      const result = await guard.canActivate(context);

      expect(result).toBe(true);
    });

    it('should throw HttpException when rate limited', async () => {
      const now = Date.now();
      mockCacheService.get.mockResolvedValue(now - 3000); // 3 seconds ago
      const context = createMockContext(undefined, '192.168.1.1');

      await expect(guard.canActivate(context)).rejects.toThrow(HttpException);
    });

    it('should throw with 429 status code', async () => {
      const now = Date.now();
      mockCacheService.get.mockResolvedValue(now - 3000);
      const context = createMockContext(undefined, '192.168.1.1');

      try {
        await guard.canActivate(context);
        fail('Should have thrown');
      } catch (error) {
        expect(error).toBeInstanceOf(HttpException);
        expect((error as HttpException).getStatus()).toBe(HttpStatus.TOO_MANY_REQUESTS);
      }
    });

    it('should set Retry-After header when rate limited', async () => {
      const now = Date.now();
      mockCacheService.get.mockResolvedValue(now - 3000);
      const context = createMockContext(undefined, '192.168.1.1');
      const response = context.switchToHttp().getResponse();

      try {
        await guard.canActivate(context);
      } catch {
        expect(response.setHeader).toHaveBeenCalledWith('Retry-After', '7');
      }
    });

    it('should set X-RateLimit-* headers when rate limited', async () => {
      const now = Date.now();
      mockCacheService.get.mockResolvedValue(now - 3000);
      const context = createMockContext(undefined, '192.168.1.1');
      const response = context.switchToHttp().getResponse();

      try {
        await guard.canActivate(context);
      } catch {
        expect(response.setHeader).toHaveBeenCalledWith('X-RateLimit-Limit', '10');
        expect(response.setHeader).toHaveBeenCalledWith('X-RateLimit-Remaining', '0');
        expect(response.setHeader).toHaveBeenCalledWith(
          'X-RateLimit-Reset',
          expect.any(String),
        );
      }
    });
  });

  // ============================================
  // getClientIp() - IP extraction
  // ============================================
  describe('getClientIp() via checkRateLimit', () => {
    it('should use x-forwarded-for header when present', async () => {
      mockCacheService.get.mockResolvedValue(null);
      const context = createMockContext(undefined, '127.0.0.1', {
        'x-forwarded-for': '203.0.113.1, 198.51.100.1',
      });

      await guard.checkRateLimit(context);

      expect(mockCacheService.set).toHaveBeenCalledWith(
        'playground_rate:ip:203.0.113.1',
        expect.any(Number),
        expect.any(Number),
      );
    });

    it('should use first IP from x-forwarded-for chain', async () => {
      mockCacheService.get.mockResolvedValue(null);
      const context = createMockContext(undefined, '127.0.0.1', {
        'x-forwarded-for': '  10.0.0.1  , 192.168.1.1, 172.16.0.1',
      });

      await guard.checkRateLimit(context);

      expect(mockCacheService.set).toHaveBeenCalledWith(
        'playground_rate:ip:10.0.0.1',
        expect.any(Number),
        expect.any(Number),
      );
    });

    it('should use x-real-ip header when x-forwarded-for is missing', async () => {
      mockCacheService.get.mockResolvedValue(null);
      const context = createMockContext(undefined, '127.0.0.1', {
        'x-real-ip': '192.0.2.1',
      });

      await guard.checkRateLimit(context);

      expect(mockCacheService.set).toHaveBeenCalledWith(
        'playground_rate:ip:192.0.2.1',
        expect.any(Number),
        expect.any(Number),
      );
    });

    it('should fallback to request.ip when no headers present', async () => {
      mockCacheService.get.mockResolvedValue(null);
      const context = createMockContext(undefined, '192.168.100.1', {});

      await guard.checkRateLimit(context);

      expect(mockCacheService.set).toHaveBeenCalledWith(
        'playground_rate:ip:192.168.100.1',
        expect.any(Number),
        expect.any(Number),
      );
    });
  });

  // ============================================
  // getRateLimitInfo() - UI display info
  // ============================================
  describe('getRateLimitInfo()', () => {
    it('should return free tier info for unauthenticated users', async () => {
      const result = await guard.getRateLimitInfo();

      expect(result).toEqual({
        rateLimitSeconds: 10,
        isPremium: false,
      });
    });

    it('should return auth tier info for non-premium authenticated users', async () => {
      mockAccessControlService.hasGlobalAccess.mockResolvedValue(false);

      const result = await guard.getRateLimitInfo('user-123');

      expect(result).toEqual({
        rateLimitSeconds: 10,
        isPremium: false,
      });
    });

    it('should return premium tier info for premium users', async () => {
      mockAccessControlService.hasGlobalAccess.mockResolvedValue(true);

      const result = await guard.getRateLimitInfo('premium-user-123');

      expect(result).toEqual({
        rateLimitSeconds: 5,
        isPremium: true,
      });
    });

    it('should call hasGlobalAccess when userId is provided', async () => {
      mockAccessControlService.hasGlobalAccess.mockResolvedValue(false);

      await guard.getRateLimitInfo('user-123');

      expect(mockAccessControlService.hasGlobalAccess).toHaveBeenCalledWith('user-123');
    });

    it('should not call hasGlobalAccess when no userId', async () => {
      await guard.getRateLimitInfo();

      expect(mockAccessControlService.hasGlobalAccess).not.toHaveBeenCalled();
    });
  });

  // ============================================
  // Edge cases
  // ============================================
  describe('Edge cases', () => {
    it('should handle exactly at rate limit boundary', async () => {
      const now = Date.now();
      mockCacheService.get.mockResolvedValue(now - 10000); // Exactly 10 seconds ago
      const context = createMockContext(undefined, '192.168.1.1');

      const result = await guard.checkRateLimit(context);

      // Should allow since 10 - 10 = 0, which is not > 0
      expect(result).toBeNull();
    });

    it('should handle cache errors gracefully', async () => {
      mockCacheService.get.mockRejectedValue(new Error('Cache error'));
      const context = createMockContext(undefined, '192.168.1.1');

      // Should throw the cache error (not swallow it)
      await expect(guard.checkRateLimit(context)).rejects.toThrow('Cache error');
    });

    it('should round up remaining seconds', async () => {
      const now = Date.now();
      mockCacheService.get.mockResolvedValue(now - 3500); // 3.5 seconds ago
      const context = createMockContext(undefined, '192.168.1.1');

      const result = await guard.checkRateLimit(context);

      // 10 - 3.5 = 6.5, should ceil to 7
      expect(result!.retryAfter).toBe(7);
    });

    it('should handle x-forwarded-for as array', async () => {
      mockCacheService.get.mockResolvedValue(null);
      const context = createMockContext(undefined, '127.0.0.1', {
        'x-forwarded-for': ['10.0.0.1', '192.168.1.1'],
      });

      await guard.checkRateLimit(context);

      expect(mockCacheService.set).toHaveBeenCalledWith(
        'playground_rate:ip:10.0.0.1',
        expect.any(Number),
        expect.any(Number),
      );
    });

    it('should handle x-real-ip as array', async () => {
      mockCacheService.get.mockResolvedValue(null);
      const context = createMockContext(undefined, '127.0.0.1', {
        'x-real-ip': ['172.16.0.1', '192.168.1.1'],
      });

      await guard.checkRateLimit(context);

      expect(mockCacheService.set).toHaveBeenCalledWith(
        'playground_rate:ip:172.16.0.1',
        expect.any(Number),
        expect.any(Number),
      );
    });
  });
});
