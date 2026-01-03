import { Test, TestingModule } from '@nestjs/testing';
import { ExecutionContext, ForbiddenException } from '@nestjs/common';
import { IpBanGuard } from './ip-ban.guard';
import { IpBanService } from '../ip-ban.service';

describe('IpBanGuard', () => {
  let guard: IpBanGuard;
  let ipBanService: IpBanService;

  const mockIpBanService = {
    checkAndThrow: jest.fn().mockResolvedValue(undefined),
  };

  const createMockExecutionContext = (
    ip = '127.0.0.1',
    headers: Record<string, string | string[]> = {},
  ): ExecutionContext => {
    const request = {
      ip,
      headers,
      socket: { remoteAddress: ip },
    };

    return {
      switchToHttp: () => ({
        getRequest: () => request,
      }),
    } as unknown as ExecutionContext;
  };

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [
        IpBanGuard,
        { provide: IpBanService, useValue: mockIpBanService },
      ],
    }).compile();

    guard = module.get<IpBanGuard>(IpBanGuard);
    ipBanService = module.get<IpBanService>(IpBanService);

    jest.clearAllMocks();
  });

  it('should be defined', () => {
    expect(guard).toBeDefined();
  });

  // ============================================
  // canActivate()
  // ============================================
  describe('canActivate()', () => {
    it('should allow non-banned IPs', async () => {
      const context = createMockExecutionContext('192.168.1.1');

      const result = await guard.canActivate(context);

      expect(result).toBe(true);
      expect(mockIpBanService.checkAndThrow).toHaveBeenCalledWith('192.168.1.1');
    });

    it('should throw for banned IPs', async () => {
      mockIpBanService.checkAndThrow.mockRejectedValueOnce(
        new ForbiddenException('IP banned'),
      );
      const context = createMockExecutionContext('10.0.0.1');

      await expect(guard.canActivate(context)).rejects.toThrow(ForbiddenException);
    });
  });

  // ============================================
  // IP Detection
  // ============================================
  describe('IP detection', () => {
    it('should extract IP from x-forwarded-for header', async () => {
      const context = createMockExecutionContext('127.0.0.1', {
        'x-forwarded-for': '192.168.1.100, 10.0.0.1',
      });

      await guard.canActivate(context);

      expect(mockIpBanService.checkAndThrow).toHaveBeenCalledWith('192.168.1.100');
    });

    it('should extract IP from x-forwarded-for array', async () => {
      const context = createMockExecutionContext('127.0.0.1', {
        'x-forwarded-for': ['10.10.10.10', '20.20.20.20'],
      });

      await guard.canActivate(context);

      expect(mockIpBanService.checkAndThrow).toHaveBeenCalledWith('10.10.10.10');
    });

    it('should extract IP from x-real-ip header', async () => {
      const context = createMockExecutionContext('127.0.0.1', {
        'x-real-ip': '172.16.0.1',
      });

      await guard.canActivate(context);

      expect(mockIpBanService.checkAndThrow).toHaveBeenCalledWith('172.16.0.1');
    });

    it('should extract IP from x-real-ip array', async () => {
      const context = createMockExecutionContext('127.0.0.1', {
        'x-real-ip': ['172.16.0.1', '172.16.0.2'],
      });

      await guard.canActivate(context);

      expect(mockIpBanService.checkAndThrow).toHaveBeenCalledWith('172.16.0.1');
    });

    it('should use request.ip when no proxy headers', async () => {
      const context = createMockExecutionContext('192.168.0.100');

      await guard.canActivate(context);

      expect(mockIpBanService.checkAndThrow).toHaveBeenCalledWith('192.168.0.100');
    });

    it('should fallback to socket.remoteAddress when ip is undefined', async () => {
      const request = {
        ip: undefined,
        headers: {},
        socket: { remoteAddress: '10.20.30.40' },
      };
      const context = {
        switchToHttp: () => ({
          getRequest: () => request,
        }),
      } as unknown as ExecutionContext;

      await guard.canActivate(context);

      expect(mockIpBanService.checkAndThrow).toHaveBeenCalledWith('10.20.30.40');
    });

    it('should return "unknown" when no IP available', async () => {
      const request = {
        ip: undefined,
        headers: {},
        socket: { remoteAddress: undefined },
      };
      const context = {
        switchToHttp: () => ({
          getRequest: () => request,
        }),
      } as unknown as ExecutionContext;

      await guard.canActivate(context);

      expect(mockIpBanService.checkAndThrow).toHaveBeenCalledWith('unknown');
    });

    it('should trim whitespace from forwarded IP', async () => {
      const context = createMockExecutionContext('127.0.0.1', {
        'x-forwarded-for': '  192.168.1.1  , 10.0.0.1',
      });

      await guard.canActivate(context);

      expect(mockIpBanService.checkAndThrow).toHaveBeenCalledWith('192.168.1.1');
    });
  });
});
