import { Test, TestingModule } from '@nestjs/testing';
import { CacheService } from './cache.service';
import { ConfigService } from '@nestjs/config';

describe('CacheService', () => {
  let service: CacheService;

  const mockConfigService = {
    get: jest.fn().mockReturnValue('localhost'),
  };

  // Mock Redis instance
  const mockRedis = {
    on: jest.fn(),
    ping: jest.fn().mockResolvedValue('PONG'),
    get: jest.fn(),
    setex: jest.fn(),
    del: jest.fn(),
    keys: jest.fn(),
    scan: jest.fn(),
    dbsize: jest.fn(),
    quit: jest.fn().mockResolvedValue(undefined),
  };

  // Mock ioredis constructor
  jest.mock('ioredis', () => {
    return jest.fn().mockImplementation(() => mockRedis);
  });

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [
        CacheService,
        { provide: ConfigService, useValue: mockConfigService },
      ],
    }).compile();

    service = module.get<CacheService>(CacheService);

    // Manually set connection status for testing
    (service as any).isConnected = true;
    (service as any).redis = mockRedis;

    jest.clearAllMocks();
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });

  // ============================================
  // generateHash()
  // ============================================
  describe('generateHash()', () => {
    it('should generate consistent hash for same input', () => {
      const hash1 = service.generateHash('test data');
      const hash2 = service.generateHash('test data');

      expect(hash1).toBe(hash2);
    });

    it('should generate different hash for different input', () => {
      const hash1 = service.generateHash('test data 1');
      const hash2 = service.generateHash('test data 2');

      expect(hash1).not.toBe(hash2);
    });

    it('should return 16 character hash', () => {
      const hash = service.generateHash('some data');

      expect(hash).toHaveLength(16);
    });
  });

  // ============================================
  // getExecutionCacheKey()
  // ============================================
  describe('getExecutionCacheKey()', () => {
    it('should generate key with language prefix', () => {
      const key = service.getExecutionCacheKey('code', 'python');

      expect(key).toMatch(/^exec:python:/);
    });

    it('should include stdin in hash if provided', () => {
      const keyWithStdin = service.getExecutionCacheKey('code', 'go', 'input');
      const keyWithoutStdin = service.getExecutionCacheKey('code', 'go');

      expect(keyWithStdin).not.toBe(keyWithoutStdin);
    });

    it('should generate same key for same inputs', () => {
      const key1 = service.getExecutionCacheKey('func main()', 'go', 'stdin');
      const key2 = service.getExecutionCacheKey('func main()', 'go', 'stdin');

      expect(key1).toBe(key2);
    });
  });

  // ============================================
  // getExecutionResult()
  // ============================================
  describe('getExecutionResult()', () => {
    it('should return null when not connected', async () => {
      (service as any).isConnected = false;

      const result = await service.getExecutionResult('code', 'python');

      expect(result).toBeNull();
    });

    it('should return cached result if exists', async () => {
      const cachedData = { status: 'passed', stdout: 'output' };
      mockRedis.get.mockResolvedValue(JSON.stringify(cachedData));

      const result = await service.getExecutionResult('code', 'python');

      expect(result).toEqual(cachedData);
    });

    it('should return null if cache miss', async () => {
      mockRedis.get.mockResolvedValue(null);

      const result = await service.getExecutionResult('code', 'python');

      expect(result).toBeNull();
    });

    it('should handle Redis errors gracefully', async () => {
      mockRedis.get.mockRejectedValue(new Error('Redis error'));

      const result = await service.getExecutionResult('code', 'python');

      expect(result).toBeNull();
    });
  });

  // ============================================
  // setExecutionResult()
  // ============================================
  describe('setExecutionResult()', () => {
    it('should not cache when not connected', async () => {
      (service as any).isConnected = false;

      await service.setExecutionResult('code', 'python', undefined, { status: 'passed' });

      expect(mockRedis.setex).not.toHaveBeenCalled();
    });

    it('should not cache failed results', async () => {
      await service.setExecutionResult('code', 'python', undefined, { status: 'failed' });

      expect(mockRedis.setex).not.toHaveBeenCalled();
    });

    it('should cache successful results', async () => {
      await service.setExecutionResult('code', 'python', undefined, { status: 'passed' });

      expect(mockRedis.setex).toHaveBeenCalled();
    });

    it('should use correct TTL', async () => {
      await service.setExecutionResult('code', 'python', undefined, { status: 'passed' });

      expect(mockRedis.setex).toHaveBeenCalledWith(
        expect.any(String),
        1800, // EXECUTION_CACHE_TTL
        expect.any(String),
      );
    });
  });

  // ============================================
  // get()
  // ============================================
  describe('get()', () => {
    it('should return null when not connected', async () => {
      (service as any).isConnected = false;

      const result = await service.get('some-key');

      expect(result).toBeNull();
    });

    it('should return parsed value if exists', async () => {
      const data = { user: 'test', level: 5 };
      mockRedis.get.mockResolvedValue(JSON.stringify(data));

      const result = await service.get('user:123');

      expect(result).toEqual(data);
    });

    it('should return null for missing keys', async () => {
      mockRedis.get.mockResolvedValue(null);

      const result = await service.get('nonexistent');

      expect(result).toBeNull();
    });
  });

  // ============================================
  // set()
  // ============================================
  describe('set()', () => {
    it('should not set when not connected', async () => {
      (service as any).isConnected = false;

      await service.set('key', { data: 'value' });

      expect(mockRedis.setex).not.toHaveBeenCalled();
    });

    it('should set value with default TTL', async () => {
      await service.set('key', { data: 'value' });

      expect(mockRedis.setex).toHaveBeenCalledWith(
        'key',
        3600, // DEFAULT_TTL
        JSON.stringify({ data: 'value' }),
      );
    });

    it('should respect custom TTL', async () => {
      await service.set('key', 'value', 600);

      expect(mockRedis.setex).toHaveBeenCalledWith('key', 600, '"value"');
    });
  });

  // ============================================
  // delete()
  // ============================================
  describe('delete()', () => {
    it('should not delete when not connected', async () => {
      (service as any).isConnected = false;

      await service.delete('key');

      expect(mockRedis.del).not.toHaveBeenCalled();
    });

    it('should delete key from Redis', async () => {
      await service.delete('key-to-delete');

      expect(mockRedis.del).toHaveBeenCalledWith('key-to-delete');
    });
  });

  // ============================================
  // deleteByPattern()
  // ============================================
  describe('deleteByPattern()', () => {
    it('should return 0 when not connected', async () => {
      (service as any).isConnected = false;

      const result = await service.deleteByPattern('courses:*');

      expect(result).toBe(0);
    });

    it('should delete matching keys using SCAN', async () => {
      // First scan returns keys and cursor 0 (end of iteration)
      mockRedis.scan.mockResolvedValue(['0', ['courses:1', 'courses:2', 'courses:3']]);
      mockRedis.del.mockResolvedValue(3);

      const result = await service.deleteByPattern('courses:*');

      expect(result).toBe(3);
      expect(mockRedis.scan).toHaveBeenCalled();
      expect(mockRedis.del).toHaveBeenCalledWith('courses:1', 'courses:2', 'courses:3');
    });

    it('should return 0 when no keys match', async () => {
      mockRedis.scan.mockResolvedValue(['0', []]);

      const result = await service.deleteByPattern('nonexistent:*');

      expect(result).toBe(0);
      expect(mockRedis.del).not.toHaveBeenCalled();
    });
  });

  // ============================================
  // getStats()
  // ============================================
  describe('getStats()', () => {
    it('should return disconnected status when not connected', async () => {
      (service as any).isConnected = false;

      const result = await service.getStats();

      expect(result).toEqual({ connected: false });
    });

    it('should return connected status with key count', async () => {
      mockRedis.dbsize.mockResolvedValue(150);

      const result = await service.getStats();

      expect(result).toEqual({ connected: true, keys: 150 });
    });

    it('should handle dbsize error', async () => {
      mockRedis.dbsize.mockRejectedValue(new Error('Error'));

      const result = await service.getStats();

      expect(result).toEqual({ connected: false });
    });
  });

  // ============================================
  // Error handling edge cases
  // ============================================
  describe('error handling', () => {
    it('should handle setExecutionResult Redis errors gracefully', async () => {
      mockRedis.setex.mockRejectedValue(new Error('Redis write error'));

      // Should not throw, just log the error
      await expect(
        service.setExecutionResult('code', 'python', undefined, { status: 'passed' })
      ).resolves.not.toThrow();
    });

    it('should handle get Redis errors gracefully', async () => {
      mockRedis.get.mockRejectedValue(new Error('Redis read error'));

      const result = await service.get('some-key');

      expect(result).toBeNull();
    });

    it('should handle set Redis errors gracefully', async () => {
      mockRedis.setex.mockRejectedValue(new Error('Redis write error'));

      // Should not throw, just log the error
      await expect(
        service.set('key', { data: 'value' })
      ).resolves.not.toThrow();
    });

    it('should handle delete Redis errors gracefully', async () => {
      mockRedis.del.mockRejectedValue(new Error('Redis delete error'));

      // Should not throw, just log the error
      await expect(
        service.delete('key-to-delete')
      ).resolves.not.toThrow();
    });

    it('should handle deleteByPattern Redis errors gracefully', async () => {
      mockRedis.scan.mockRejectedValue(new Error('Redis scan error'));

      const result = await service.deleteByPattern('pattern:*');

      expect(result).toBe(0);
    });

    it('should handle deleteByPattern del errors gracefully', async () => {
      mockRedis.scan.mockResolvedValue(['0', ['key1', 'key2']]);
      mockRedis.del.mockRejectedValue(new Error('Redis del error'));

      const result = await service.deleteByPattern('pattern:*');

      expect(result).toBe(0);
    });
  });

  // ============================================
  // onModuleDestroy
  // ============================================
  describe('onModuleDestroy()', () => {
    it('should quit Redis connection on destroy', async () => {
      await service.onModuleDestroy();

      expect(mockRedis.quit).toHaveBeenCalled();
    });

    it('should handle missing Redis instance', async () => {
      (service as any).redis = undefined;

      // Should not throw
      await expect(service.onModuleDestroy()).resolves.not.toThrow();
    });
  });
});
