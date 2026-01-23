import { Test, TestingModule } from '@nestjs/testing';
import { HealthCheckError } from '@nestjs/terminus';
import { PrismaHealthIndicator } from './prisma.health';
import { RedisHealthIndicator } from './redis.health';
import { PrismaService } from '../prisma/prisma.service';
import { CacheService } from '../cache/cache.service';

describe('Health Indicators', () => {
  // ============================================
  // PrismaHealthIndicator
  // ============================================
  describe('PrismaHealthIndicator', () => {
    let indicator: PrismaHealthIndicator;
    let prisma: jest.Mocked<PrismaService>;

    beforeEach(async () => {
      const mockPrisma = {
        $queryRaw: jest.fn(),
      };

      const module: TestingModule = await Test.createTestingModule({
        providers: [
          PrismaHealthIndicator,
          {
            provide: PrismaService,
            useValue: mockPrisma,
          },
        ],
      }).compile();

      indicator = module.get<PrismaHealthIndicator>(PrismaHealthIndicator);
      prisma = module.get(PrismaService);
    });

    it('should be defined', () => {
      expect(indicator).toBeDefined();
    });

    describe('isHealthy()', () => {
      it('should return healthy status when database is reachable', async () => {
        prisma.$queryRaw.mockResolvedValue([{ '1': 1 }]);

        const result = await indicator.isHealthy('database');

        expect(result).toEqual({
          database: {
            status: 'up',
            message: 'Database is reachable',
          },
        });
      });

      it('should use provided key in response', async () => {
        prisma.$queryRaw.mockResolvedValue([{ '1': 1 }]);

        const result = await indicator.isHealthy('postgres');

        expect(result.postgres).toBeDefined();
        expect(result.postgres.status).toBe('up');
      });

      it('should throw HealthCheckError when database is unreachable', async () => {
        prisma.$queryRaw.mockRejectedValue(new Error('Connection refused'));

        await expect(indicator.isHealthy('database')).rejects.toThrow(HealthCheckError);
      });

      it('should include error message in health check error', async () => {
        prisma.$queryRaw.mockRejectedValue(new Error('ECONNREFUSED'));

        try {
          await indicator.isHealthy('database');
          fail('Should have thrown');
        } catch (error) {
          expect(error).toBeInstanceOf(HealthCheckError);
          expect((error as HealthCheckError).causes).toEqual({
            database: {
              status: 'down',
              message: 'ECONNREFUSED',
            },
          });
        }
      });

      it('should handle non-Error exceptions', async () => {
        prisma.$queryRaw.mockRejectedValue('String error');

        try {
          await indicator.isHealthy('database');
          fail('Should have thrown');
        } catch (error) {
          expect(error).toBeInstanceOf(HealthCheckError);
          expect((error as HealthCheckError).causes.database.message).toBe(
            'Database check failed',
          );
        }
      });
    });
  });

  // ============================================
  // RedisHealthIndicator
  // ============================================
  describe('RedisHealthIndicator', () => {
    let indicator: RedisHealthIndicator;
    let cacheService: jest.Mocked<CacheService>;

    beforeEach(async () => {
      const mockCache = {
        set: jest.fn(),
        get: jest.fn(),
        delete: jest.fn(),
      };

      const module: TestingModule = await Test.createTestingModule({
        providers: [
          RedisHealthIndicator,
          {
            provide: CacheService,
            useValue: mockCache,
          },
        ],
      }).compile();

      indicator = module.get<RedisHealthIndicator>(RedisHealthIndicator);
      cacheService = module.get(CacheService);
    });

    it('should be defined', () => {
      expect(indicator).toBeDefined();
    });

    describe('isHealthy()', () => {
      it('should return healthy status when Redis is reachable', async () => {
        const testValue = Date.now().toString();
        cacheService.set.mockResolvedValue(undefined);
        cacheService.get.mockImplementation(async () => testValue);
        cacheService.delete.mockResolvedValue(undefined);

        // Mock Date.now to return consistent value
        const originalNow = Date.now;
        Date.now = jest.fn(() => parseInt(testValue));

        const result = await indicator.isHealthy('redis');

        Date.now = originalNow;

        expect(result).toEqual({
          redis: {
            status: 'up',
            message: 'Redis is reachable',
          },
        });
      });

      it('should use provided key in response', async () => {
        const testValue = '12345';
        cacheService.set.mockResolvedValue(undefined);
        cacheService.get.mockResolvedValue(testValue);
        cacheService.delete.mockResolvedValue(undefined);

        const originalNow = Date.now;
        Date.now = jest.fn(() => parseInt(testValue));

        const result = await indicator.isHealthy('cache');

        Date.now = originalNow;

        expect(result.cache).toBeDefined();
        expect(result.cache.status).toBe('up');
      });

      it('should clean up health check key after successful check', async () => {
        const testValue = '12345';
        cacheService.set.mockResolvedValue(undefined);
        cacheService.get.mockResolvedValue(testValue);
        cacheService.delete.mockResolvedValue(undefined);

        const originalNow = Date.now;
        Date.now = jest.fn(() => parseInt(testValue));

        await indicator.isHealthy('redis');

        Date.now = originalNow;

        expect(cacheService.delete).toHaveBeenCalledWith('__health_check__');
      });

      it('should throw HealthCheckError when Redis set fails', async () => {
        cacheService.set.mockRejectedValue(new Error('NOAUTH'));

        await expect(indicator.isHealthy('redis')).rejects.toThrow(HealthCheckError);
      });

      it('should throw HealthCheckError when value mismatch occurs', async () => {
        cacheService.set.mockResolvedValue(undefined);
        cacheService.get.mockResolvedValue('different-value'); // Mismatch

        await expect(indicator.isHealthy('redis')).rejects.toThrow(HealthCheckError);

        try {
          cacheService.get.mockResolvedValue('different-value');
          await indicator.isHealthy('redis');
        } catch (error) {
          expect((error as HealthCheckError).causes.redis.message).toBe(
            'Redis read/write mismatch',
          );
        }
      });

      it('should include error message in health check error', async () => {
        cacheService.set.mockRejectedValue(new Error('Connection timeout'));

        try {
          await indicator.isHealthy('redis');
          fail('Should have thrown');
        } catch (error) {
          expect(error).toBeInstanceOf(HealthCheckError);
          expect((error as HealthCheckError).causes).toEqual({
            redis: {
              status: 'down',
              message: 'Connection timeout',
            },
          });
        }
      });

      it('should handle non-Error exceptions', async () => {
        cacheService.set.mockRejectedValue('String error');

        try {
          await indicator.isHealthy('redis');
          fail('Should have thrown');
        } catch (error) {
          expect(error).toBeInstanceOf(HealthCheckError);
          expect((error as HealthCheckError).causes.redis.message).toBe(
            'Redis check failed',
          );
        }
      });
    });
  });
});
