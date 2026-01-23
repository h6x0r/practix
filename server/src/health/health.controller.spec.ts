import { Test, TestingModule } from '@nestjs/testing';
import { HealthController } from './health.controller';
import {
  HealthCheckService,
  DiskHealthIndicator,
  MemoryHealthIndicator,
  HealthCheckResult,
} from '@nestjs/terminus';
import { PrismaHealthIndicator } from './prisma.health';
import { RedisHealthIndicator } from './redis.health';
import { MetricsService } from './metrics.service';
import { Response } from 'express';

describe('HealthController', () => {
  let controller: HealthController;
  let healthCheckService: jest.Mocked<HealthCheckService>;
  let prismaHealth: jest.Mocked<PrismaHealthIndicator>;
  let redisHealth: jest.Mocked<RedisHealthIndicator>;
  let diskHealth: jest.Mocked<DiskHealthIndicator>;
  let memoryHealth: jest.Mocked<MemoryHealthIndicator>;
  let metricsService: jest.Mocked<MetricsService>;

  const mockHealthCheckService = {
    check: jest.fn(),
  };

  const mockPrismaHealth = {
    isHealthy: jest.fn(),
  };

  const mockRedisHealth = {
    isHealthy: jest.fn(),
  };

  const mockDiskHealth = {
    checkStorage: jest.fn(),
  };

  const mockMemoryHealth = {
    checkHeap: jest.fn(),
    checkRSS: jest.fn(),
  };

  const mockMetricsService = {
    getMetrics: jest.fn(),
  };

  const mockHealthyResult: HealthCheckResult = {
    status: 'ok',
    info: {
      database: { status: 'up' },
      redis: { status: 'up' },
    },
    error: {},
    details: {
      database: { status: 'up' },
      redis: { status: 'up' },
    },
  };

  beforeEach(async () => {
    jest.clearAllMocks();

    const module: TestingModule = await Test.createTestingModule({
      controllers: [HealthController],
      providers: [
        { provide: HealthCheckService, useValue: mockHealthCheckService },
        { provide: PrismaHealthIndicator, useValue: mockPrismaHealth },
        { provide: RedisHealthIndicator, useValue: mockRedisHealth },
        { provide: DiskHealthIndicator, useValue: mockDiskHealth },
        { provide: MemoryHealthIndicator, useValue: mockMemoryHealth },
        { provide: MetricsService, useValue: mockMetricsService },
      ],
    }).compile();

    controller = module.get<HealthController>(HealthController);
    healthCheckService = module.get(HealthCheckService);
    prismaHealth = module.get(PrismaHealthIndicator);
    redisHealth = module.get(RedisHealthIndicator);
    diskHealth = module.get(DiskHealthIndicator);
    memoryHealth = module.get(MemoryHealthIndicator);
    metricsService = module.get(MetricsService);
  });

  it('should be defined', () => {
    expect(controller).toBeDefined();
  });

  // ============================================
  // live() - Liveness probe
  // ============================================
  describe('live()', () => {
    it('should return status ok with timestamp', () => {
      const result = controller.live();

      expect(result.status).toBe('ok');
      expect(result.timestamp).toBeDefined();
      expect(new Date(result.timestamp)).toBeInstanceOf(Date);
    });

    it('should return valid ISO timestamp', () => {
      const result = controller.live();

      // Validate ISO 8601 format
      const isoRegex = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/;
      expect(result.timestamp).toMatch(isoRegex);
    });
  });

  // ============================================
  // ready() - Readiness probe
  // ============================================
  describe('ready()', () => {
    it('should check database and redis health', async () => {
      mockHealthCheckService.check.mockResolvedValue(mockHealthyResult);

      const result = await controller.ready();

      expect(result).toEqual(mockHealthyResult);
      expect(mockHealthCheckService.check).toHaveBeenCalledWith([
        expect.any(Function),
        expect.any(Function),
      ]);
    });

    it('should call prisma health indicator', async () => {
      mockPrismaHealth.isHealthy.mockResolvedValue({ database: { status: 'up' } });
      mockRedisHealth.isHealthy.mockResolvedValue({ redis: { status: 'up' } });
      mockHealthCheckService.check.mockImplementation(async (checks) => {
        for (const check of checks) {
          await check();
        }
        return mockHealthyResult;
      });

      await controller.ready();

      expect(mockPrismaHealth.isHealthy).toHaveBeenCalledWith('database');
    });

    it('should call redis health indicator', async () => {
      mockPrismaHealth.isHealthy.mockResolvedValue({ database: { status: 'up' } });
      mockRedisHealth.isHealthy.mockResolvedValue({ redis: { status: 'up' } });
      mockHealthCheckService.check.mockImplementation(async (checks) => {
        for (const check of checks) {
          await check();
        }
        return mockHealthyResult;
      });

      await controller.ready();

      expect(mockRedisHealth.isHealthy).toHaveBeenCalledWith('redis');
    });

    it('should handle unhealthy state', async () => {
      const unhealthyResult: HealthCheckResult = {
        status: 'error',
        info: {},
        error: {
          database: { status: 'down', message: 'Connection refused' },
        },
        details: {
          database: { status: 'down', message: 'Connection refused' },
        },
      };
      mockHealthCheckService.check.mockResolvedValue(unhealthyResult);

      const result = await controller.ready();

      expect(result.status).toBe('error');
    });
  });

  // ============================================
  // check() - Full health check
  // ============================================
  describe('check()', () => {
    it('should check all health indicators', async () => {
      const fullHealthResult: HealthCheckResult = {
        status: 'ok',
        info: {
          database: { status: 'up' },
          redis: { status: 'up' },
          storage: { status: 'up' },
          memory_heap: { status: 'up' },
          memory_rss: { status: 'up' },
        },
        error: {},
        details: {
          database: { status: 'up' },
          redis: { status: 'up' },
          storage: { status: 'up' },
          memory_heap: { status: 'up' },
          memory_rss: { status: 'up' },
        },
      };
      mockHealthCheckService.check.mockResolvedValue(fullHealthResult);

      const result = await controller.check();

      expect(result).toEqual(fullHealthResult);
      expect(mockHealthCheckService.check).toHaveBeenCalledWith([
        expect.any(Function),
        expect.any(Function),
        expect.any(Function),
        expect.any(Function),
        expect.any(Function),
      ]);
    });

    it('should check disk storage', async () => {
      mockDiskHealth.checkStorage.mockResolvedValue({ storage: { status: 'up' } });
      mockHealthCheckService.check.mockImplementation(async (checks) => {
        for (const check of checks) {
          await check();
        }
        return mockHealthyResult;
      });

      await controller.check();

      expect(mockDiskHealth.checkStorage).toHaveBeenCalledWith('storage', {
        path: '/',
        thresholdPercent: 0.9,
      });
    });

    it('should check memory heap', async () => {
      mockMemoryHealth.checkHeap.mockResolvedValue({ memory_heap: { status: 'up' } });
      mockHealthCheckService.check.mockImplementation(async (checks) => {
        for (const check of checks) {
          await check();
        }
        return mockHealthyResult;
      });

      await controller.check();

      expect(mockMemoryHealth.checkHeap).toHaveBeenCalledWith(
        'memory_heap',
        300 * 1024 * 1024
      );
    });

    it('should check memory RSS', async () => {
      mockMemoryHealth.checkRSS.mockResolvedValue({ memory_rss: { status: 'up' } });
      mockHealthCheckService.check.mockImplementation(async (checks) => {
        for (const check of checks) {
          await check();
        }
        return mockHealthyResult;
      });

      await controller.check();

      expect(mockMemoryHealth.checkRSS).toHaveBeenCalledWith(
        'memory_rss',
        500 * 1024 * 1024
      );
    });

    it('should handle partial failure', async () => {
      const partialResult: HealthCheckResult = {
        status: 'error',
        info: {
          database: { status: 'up' },
          redis: { status: 'up' },
        },
        error: {
          storage: { status: 'down', message: 'Disk full' },
        },
        details: {
          database: { status: 'up' },
          redis: { status: 'up' },
          storage: { status: 'down', message: 'Disk full' },
        },
      };
      mockHealthCheckService.check.mockResolvedValue(partialResult);

      const result = await controller.check();

      expect(result.status).toBe('error');
      expect(result.error).toHaveProperty('storage');
    });
  });

  // ============================================
  // metrics() - Prometheus metrics
  // ============================================
  describe('metrics()', () => {
    it('should return prometheus metrics', async () => {
      const metricsOutput = `
# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="GET",path="/api/health"} 100
      `.trim();
      mockMetricsService.getMetrics.mockResolvedValue(metricsOutput);

      const mockRes = {
        set: jest.fn(),
        send: jest.fn(),
      } as unknown as Response;

      await controller.metrics(mockRes);

      expect(mockMetricsService.getMetrics).toHaveBeenCalled();
      expect(mockRes.set).toHaveBeenCalledWith('Content-Type', 'text/plain');
      expect(mockRes.send).toHaveBeenCalledWith(metricsOutput);
    });

    it('should handle empty metrics', async () => {
      mockMetricsService.getMetrics.mockResolvedValue('');

      const mockRes = {
        set: jest.fn(),
        send: jest.fn(),
      } as unknown as Response;

      await controller.metrics(mockRes);

      expect(mockRes.send).toHaveBeenCalledWith('');
    });

    it('should set correct content type', async () => {
      mockMetricsService.getMetrics.mockResolvedValue('metric 1');

      const mockRes = {
        set: jest.fn(),
        send: jest.fn(),
      } as unknown as Response;

      await controller.metrics(mockRes);

      expect(mockRes.set).toHaveBeenCalledWith('Content-Type', 'text/plain');
    });
  });
});
