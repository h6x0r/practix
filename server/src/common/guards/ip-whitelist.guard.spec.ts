import { Test, TestingModule } from '@nestjs/testing';
import { ExecutionContext, ForbiddenException } from '@nestjs/common';
import { Reflector } from '@nestjs/core';
import { ConfigService } from '@nestjs/config';
import { IpWhitelistGuard, IP_WHITELIST_KEY } from './ip-whitelist.guard';

describe('IpWhitelistGuard', () => {
  let guard: IpWhitelistGuard;
  let reflector: Reflector;
  let configService: ConfigService;

  const mockConfigService = {
    get: jest.fn(),
  };

  const mockReflector = {
    get: jest.fn(),
  };

  const createMockContext = (ip: string, headers: Record<string, string> = {}): ExecutionContext => {
    const request = {
      ip,
      socket: { remoteAddress: ip },
      headers,
    };
    return {
      switchToHttp: () => ({
        getRequest: () => request,
      }),
      getHandler: () => ({}),
    } as unknown as ExecutionContext;
  };

  beforeEach(async () => {
    jest.clearAllMocks();

    // Default config values
    mockConfigService.get.mockImplementation((key: string, defaultValue?: any) => {
      const config: Record<string, any> = {
        NODE_ENV: 'production',
        WEBHOOK_IP_WHITELIST_ENABLED: true,
        PAYME_ALLOWED_IPS: '185.8.212.0/24,195.158.31.0/24',
        CLICK_ALLOWED_IPS: '185.8.212.0/24,195.158.28.0/24',
      };
      return config[key] ?? defaultValue;
    });

    const module: TestingModule = await Test.createTestingModule({
      providers: [
        IpWhitelistGuard,
        { provide: Reflector, useValue: mockReflector },
        { provide: ConfigService, useValue: mockConfigService },
      ],
    }).compile();

    guard = module.get<IpWhitelistGuard>(IpWhitelistGuard);
    reflector = module.get<Reflector>(Reflector);
    configService = module.get<ConfigService>(ConfigService);
  });

  describe('canActivate', () => {
    it('should allow request when IP whitelist is disabled', async () => {
      // Re-create guard with disabled whitelist
      mockConfigService.get.mockImplementation((key: string, defaultValue?: any) => {
        if (key === 'WEBHOOK_IP_WHITELIST_ENABLED') return false;
        return defaultValue;
      });

      const module: TestingModule = await Test.createTestingModule({
        providers: [
          IpWhitelistGuard,
          { provide: Reflector, useValue: mockReflector },
          { provide: ConfigService, useValue: mockConfigService },
        ],
      }).compile();

      const disabledGuard = module.get<IpWhitelistGuard>(IpWhitelistGuard);
      mockReflector.get.mockReturnValue('payme');

      const context = createMockContext('1.2.3.4');
      const result = await disabledGuard.canActivate(context);

      expect(result).toBe(true);
    });

    it('should allow request when no provider is specified', async () => {
      mockReflector.get.mockReturnValue(undefined);

      const context = createMockContext('1.2.3.4');
      const result = await guard.canActivate(context);

      expect(result).toBe(true);
    });

    it('should allow Payme IP in CIDR range', async () => {
      mockReflector.get.mockReturnValue('payme');

      const context = createMockContext('185.8.212.100');
      const result = await guard.canActivate(context);

      expect(result).toBe(true);
    });

    it('should allow Click IP in CIDR range', async () => {
      mockReflector.get.mockReturnValue('click');

      const context = createMockContext('185.8.212.50');
      const result = await guard.canActivate(context);

      expect(result).toBe(true);
    });

    it('should block unauthorized IP for Payme', async () => {
      mockReflector.get.mockReturnValue('payme');

      const context = createMockContext('1.2.3.4');

      await expect(guard.canActivate(context)).rejects.toThrow(ForbiddenException);
    });

    it('should block unauthorized IP for Click', async () => {
      mockReflector.get.mockReturnValue('click');

      const context = createMockContext('8.8.8.8');

      await expect(guard.canActivate(context)).rejects.toThrow(ForbiddenException);
    });

    it('should include IP address in error message', async () => {
      mockReflector.get.mockReturnValue('payme');

      const context = createMockContext('1.2.3.4');

      try {
        await guard.canActivate(context);
        fail('Expected ForbiddenException');
      } catch (error) {
        expect(error.message).toContain('1.2.3.4');
        expect(error.message).toContain('payme');
      }
    });
  });

  describe('IP extraction', () => {
    it('should extract IP from x-forwarded-for header', async () => {
      mockReflector.get.mockReturnValue('payme');

      const context = createMockContext('1.2.3.4', {
        'x-forwarded-for': '185.8.212.100, 10.0.0.1',
      });
      const result = await guard.canActivate(context);

      expect(result).toBe(true);
    });

    it('should extract IP from x-real-ip header', async () => {
      mockReflector.get.mockReturnValue('payme');

      const context = createMockContext('1.2.3.4', {
        'x-real-ip': '185.8.212.100',
      });
      const result = await guard.canActivate(context);

      expect(result).toBe(true);
    });

    it('should prefer x-forwarded-for over x-real-ip', async () => {
      mockReflector.get.mockReturnValue('payme');

      // x-forwarded-for has allowed IP, x-real-ip has blocked IP
      const context = createMockContext('1.2.3.4', {
        'x-forwarded-for': '185.8.212.100',
        'x-real-ip': '8.8.8.8',
      });
      const result = await guard.canActivate(context);

      expect(result).toBe(true);
    });
  });

  describe('development mode', () => {
    beforeEach(async () => {
      mockConfigService.get.mockImplementation((key: string, defaultValue?: any) => {
        const config: Record<string, any> = {
          NODE_ENV: 'development',
          WEBHOOK_IP_WHITELIST_ENABLED: true,
          PAYME_ALLOWED_IPS: '',
          CLICK_ALLOWED_IPS: '',
        };
        return config[key] ?? defaultValue;
      });

      const module: TestingModule = await Test.createTestingModule({
        providers: [
          IpWhitelistGuard,
          { provide: Reflector, useValue: mockReflector },
          { provide: ConfigService, useValue: mockConfigService },
        ],
      }).compile();

      guard = module.get<IpWhitelistGuard>(IpWhitelistGuard);
    });

    it('should allow localhost in development', async () => {
      mockReflector.get.mockReturnValue('payme');

      const context = createMockContext('127.0.0.1');
      const result = await guard.canActivate(context);

      expect(result).toBe(true);
    });

    it('should allow ::1 (IPv6 localhost) in development', async () => {
      mockReflector.get.mockReturnValue('payme');

      const context = createMockContext('::1');
      const result = await guard.canActivate(context);

      expect(result).toBe(true);
    });

    it('should allow ::ffff:127.0.0.1 (IPv4-mapped IPv6) in development', async () => {
      mockReflector.get.mockReturnValue('payme');

      const context = createMockContext('::ffff:127.0.0.1');
      const result = await guard.canActivate(context);

      expect(result).toBe(true);
    });

    it('should allow private IPs (10.x.x.x) in development', async () => {
      mockReflector.get.mockReturnValue('payme');

      const context = createMockContext('10.0.0.5');
      const result = await guard.canActivate(context);

      expect(result).toBe(true);
    });

    it('should allow private IPs (192.168.x.x) in development', async () => {
      mockReflector.get.mockReturnValue('payme');

      const context = createMockContext('192.168.1.100');
      const result = await guard.canActivate(context);

      expect(result).toBe(true);
    });

    it('should allow private IPs (172.16-31.x.x) in development', async () => {
      mockReflector.get.mockReturnValue('payme');

      const context = createMockContext('172.17.0.1');
      const result = await guard.canActivate(context);

      expect(result).toBe(true);
    });
  });

  describe('CIDR validation', () => {
    it('should correctly validate /24 CIDR range', async () => {
      mockReflector.get.mockReturnValue('payme');

      // 185.8.212.0/24 should match 185.8.212.0 - 185.8.212.255
      const validIps = ['185.8.212.0', '185.8.212.1', '185.8.212.255'];
      for (const ip of validIps) {
        const context = createMockContext(ip);
        const result = await guard.canActivate(context);
        expect(result).toBe(true);
      }

      // Should NOT match 185.8.213.x
      const context = createMockContext('185.8.213.1');
      await expect(guard.canActivate(context)).rejects.toThrow(ForbiddenException);
    });

    it('should handle exact IP match (no CIDR)', async () => {
      mockConfigService.get.mockImplementation((key: string, defaultValue?: any) => {
        const config: Record<string, any> = {
          NODE_ENV: 'production',
          WEBHOOK_IP_WHITELIST_ENABLED: true,
          PAYME_ALLOWED_IPS: '1.2.3.4,5.6.7.8',
          CLICK_ALLOWED_IPS: '',
        };
        return config[key] ?? defaultValue;
      });

      const module: TestingModule = await Test.createTestingModule({
        providers: [
          IpWhitelistGuard,
          { provide: Reflector, useValue: mockReflector },
          { provide: ConfigService, useValue: mockConfigService },
        ],
      }).compile();

      const exactMatchGuard = module.get<IpWhitelistGuard>(IpWhitelistGuard);
      mockReflector.get.mockReturnValue('payme');

      // Exact match should work
      let context = createMockContext('1.2.3.4');
      let result = await exactMatchGuard.canActivate(context);
      expect(result).toBe(true);

      context = createMockContext('5.6.7.8');
      result = await exactMatchGuard.canActivate(context);
      expect(result).toBe(true);

      // Non-matching should fail
      context = createMockContext('1.2.3.5');
      await expect(exactMatchGuard.canActivate(context)).rejects.toThrow(ForbiddenException);
    });
  });

  describe('custom IP configuration', () => {
    it('should use custom IPs from environment', async () => {
      mockConfigService.get.mockImplementation((key: string, defaultValue?: any) => {
        const config: Record<string, any> = {
          NODE_ENV: 'production',
          WEBHOOK_IP_WHITELIST_ENABLED: true,
          PAYME_ALLOWED_IPS: '10.10.10.0/24',
          CLICK_ALLOWED_IPS: '20.20.20.0/24',
        };
        return config[key] ?? defaultValue;
      });

      const module: TestingModule = await Test.createTestingModule({
        providers: [
          IpWhitelistGuard,
          { provide: Reflector, useValue: mockReflector },
          { provide: ConfigService, useValue: mockConfigService },
        ],
      }).compile();

      const customGuard = module.get<IpWhitelistGuard>(IpWhitelistGuard);

      // Custom Payme IP
      mockReflector.get.mockReturnValue('payme');
      let context = createMockContext('10.10.10.50');
      let result = await customGuard.canActivate(context);
      expect(result).toBe(true);

      // Custom Click IP
      mockReflector.get.mockReturnValue('click');
      context = createMockContext('20.20.20.100');
      result = await customGuard.canActivate(context);
      expect(result).toBe(true);

      // Default IPs should NOT work anymore
      mockReflector.get.mockReturnValue('payme');
      context = createMockContext('185.8.212.100');
      await expect(customGuard.canActivate(context)).rejects.toThrow(ForbiddenException);
    });
  });
});
