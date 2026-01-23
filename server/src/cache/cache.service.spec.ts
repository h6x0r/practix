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
    set: jest.fn(),
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

  // ============================================
  // getOrSet() - Cache stampede protection
  // ============================================
  describe('getOrSet()', () => {
    beforeEach(() => {
      // Clear mocks and ensure connected state for each getOrSet test
      jest.clearAllMocks();
      (service as any).isConnected = true;
      (service as any).redis = mockRedis;
    });

    it('should return cached value if exists', async () => {
      const cachedData = { name: 'cached' };
      mockRedis.get.mockResolvedValue(JSON.stringify(cachedData));
      const factory = jest.fn();

      const result = await service.getOrSet('test-key', 300, factory);

      expect(result).toEqual(cachedData);
      expect(factory).not.toHaveBeenCalled();
    });

    it('should call factory and cache when cache miss', async () => {
      mockRedis.get.mockResolvedValue(null);
      mockRedis.set.mockResolvedValue('OK');
      const factoryResult = { name: 'generated' };
      const factory = jest.fn().mockResolvedValue(factoryResult);

      const result = await service.getOrSet('test-key', 300, factory);

      expect(result).toEqual(factoryResult);
      expect(factory).toHaveBeenCalled();
      expect(mockRedis.setex).toHaveBeenCalledWith('test-key', 300, JSON.stringify(factoryResult));
    });

    it('should acquire lock before generating value', async () => {
      mockRedis.get.mockResolvedValue(null);
      mockRedis.set.mockResolvedValue('OK');
      const factory = jest.fn().mockResolvedValue({ data: 'test' });

      await service.getOrSet('test-key', 300, factory);

      expect(mockRedis.set).toHaveBeenCalledWith('lock:test-key', '1', 'EX', 10, 'NX');
    });

    it('should call factory when not connected', async () => {
      (service as any).isConnected = false;
      const factoryResult = { name: 'direct' };
      const factory = jest.fn().mockResolvedValue(factoryResult);

      const result = await service.getOrSet('test-key', 300, factory);

      expect(result).toEqual(factoryResult);
      expect(factory).toHaveBeenCalled();
      expect(mockRedis.get).not.toHaveBeenCalled();
    });

    it('should wait and retry when lock is held by another process', async () => {
      // First get returns null, lock not acquired, then cache populated by other process
      mockRedis.get
        .mockResolvedValueOnce(null) // First check
        .mockResolvedValueOnce(JSON.stringify({ name: 'populated-by-other' })); // After wait
      mockRedis.set.mockResolvedValue(null); // Lock not acquired
      const factory = jest.fn();

      const result = await service.getOrSet('test-key', 300, factory);

      expect(result).toEqual({ name: 'populated-by-other' });
      expect(factory).not.toHaveBeenCalled();
    });

    it('should fallback to factory after lock wait timeout', async () => {
      mockRedis.get.mockResolvedValue(null); // Always cache miss
      mockRedis.set.mockResolvedValue(null); // Lock not acquired
      const factoryResult = { name: 'fallback' };
      const factory = jest.fn().mockResolvedValue(factoryResult);

      const result = await service.getOrSet('test-key', 300, factory);

      expect(result).toEqual(factoryResult);
      expect(factory).toHaveBeenCalled();
    }, 15000); // Longer timeout for wait loops

    it('should handle factory errors and release lock', async () => {
      mockRedis.get.mockResolvedValue(null);
      mockRedis.set.mockResolvedValue('OK');
      const factory = jest.fn().mockRejectedValue(new Error('Factory error'));

      await expect(service.getOrSet('test-key', 300, factory)).rejects.toThrow('Factory error');
      expect(mockRedis.del).toHaveBeenCalledWith('lock:test-key');
    });

    it('should handle Redis errors and call factory', async () => {
      mockRedis.get.mockRejectedValue(new Error('Redis error'));
      const factoryResult = { name: 'fallback-on-error' };
      const factory = jest.fn().mockResolvedValue(factoryResult);

      const result = await service.getOrSet('test-key', 300, factory);

      expect(result).toEqual(factoryResult);
      expect(factory).toHaveBeenCalled();
    });
  });

  // ============================================
  // getExecutionCacheKey() with userId
  // ============================================
  describe('getExecutionCacheKey() with userId', () => {
    it('should include userId in cache key', () => {
      const keyWithUser = service.getExecutionCacheKey('code', 'python', '', 'user-123');
      const keyWithoutUser = service.getExecutionCacheKey('code', 'python', '');

      expect(keyWithUser).not.toBe(keyWithoutUser);
    });

    it('should generate different keys for different users', () => {
      const keyUser1 = service.getExecutionCacheKey('code', 'python', '', 'user-1');
      const keyUser2 = service.getExecutionCacheKey('code', 'python', '', 'user-2');

      expect(keyUser1).not.toBe(keyUser2);
    });

    it('should use "anon" for missing userId', () => {
      const keyUndefined = service.getExecutionCacheKey('code', 'python', '', undefined);
      const keyAnon = service.getExecutionCacheKey('code', 'python', '');

      expect(keyUndefined).toBe(keyAnon);
    });
  });

  // ============================================
  // Run Validation Methods
  // ============================================
  describe('getRunValidationKey()', () => {
    it('should generate correct key format', () => {
      const key = service.getRunValidationKey('user-123', 'task-456');

      expect(key).toBe('run_valid:user-123:task-456');
    });
  });

  describe('setRunValidated()', () => {
    it('should cache run validation with correct TTL', async () => {
      await service.setRunValidated('user-123', 'task-456', 5);

      expect(mockRedis.setex).toHaveBeenCalledWith(
        'run_valid:user-123:task-456',
        3600, // RUN_VALIDATION_TTL
        expect.stringContaining('"testsPassed":5'),
      );
    });

    it('should include validatedAt timestamp', async () => {
      await service.setRunValidated('user-123', 'task-456', 8);

      const setexCall = mockRedis.setex.mock.calls[0];
      const data = JSON.parse(setexCall[2]);
      expect(data.validatedAt).toBeDefined();
      expect(data.testsPassed).toBe(8);
    });

    it('should not cache when not connected', async () => {
      (service as any).isConnected = false;

      await service.setRunValidated('user-123', 'task-456', 5);

      expect(mockRedis.setex).not.toHaveBeenCalled();
    });

    it('should handle Redis errors gracefully', async () => {
      mockRedis.setex.mockRejectedValue(new Error('Redis error'));

      await expect(
        service.setRunValidated('user-123', 'task-456', 5)
      ).resolves.not.toThrow();
    });
  });

  describe('getRunValidation()', () => {
    it('should return cached validation data', async () => {
      const validationData = { testsPassed: 7, validatedAt: '2024-01-01T00:00:00.000Z' };
      mockRedis.get.mockResolvedValue(JSON.stringify(validationData));

      const result = await service.getRunValidation('user-123', 'task-456');

      expect(result).toEqual(validationData);
      expect(mockRedis.get).toHaveBeenCalledWith('run_valid:user-123:task-456');
    });

    it('should return null when not validated', async () => {
      mockRedis.get.mockResolvedValue(null);

      const result = await service.getRunValidation('user-123', 'task-456');

      expect(result).toBeNull();
    });

    it('should return fail-open validation when not connected', async () => {
      (service as any).isConnected = false;

      const result = await service.getRunValidation('user-123', 'task-456');

      expect(result).toEqual({
        testsPassed: 5,
        validatedAt: expect.any(String),
      });
    });

    it('should handle Redis errors gracefully', async () => {
      mockRedis.get.mockRejectedValue(new Error('Redis error'));

      const result = await service.getRunValidation('user-123', 'task-456');

      expect(result).toBeNull();
    });
  });

  describe('clearRunValidation()', () => {
    it('should delete run validation key', async () => {
      await service.clearRunValidation('user-123', 'task-456');

      expect(mockRedis.del).toHaveBeenCalledWith('run_valid:user-123:task-456');
    });

    it('should not delete when not connected', async () => {
      (service as any).isConnected = false;

      await service.clearRunValidation('user-123', 'task-456');

      expect(mockRedis.del).not.toHaveBeenCalled();
    });

    it('should handle Redis errors gracefully', async () => {
      mockRedis.del.mockRejectedValue(new Error('Redis error'));

      await expect(
        service.clearRunValidation('user-123', 'task-456')
      ).resolves.not.toThrow();
    });
  });

  // ============================================
  // getExecutionResult/setExecutionResult with userId
  // ============================================
  describe('getExecutionResult() with userId', () => {
    it('should use userId in cache key', async () => {
      mockRedis.get.mockResolvedValue(null);

      await service.getExecutionResult('code', 'python', 'stdin', 'user-123');

      const expectedKey = service.getExecutionCacheKey('code', 'python', 'stdin', 'user-123');
      expect(mockRedis.get).toHaveBeenCalledWith(expectedKey);
    });
  });

  describe('setExecutionResult() with userId', () => {
    it('should use userId in cache key', async () => {
      await service.setExecutionResult('code', 'python', 'stdin', { status: 'passed' }, 'user-123');

      const expectedKey = service.getExecutionCacheKey('code', 'python', 'stdin', 'user-123');
      expect(mockRedis.setex).toHaveBeenCalledWith(
        expectedKey,
        1800,
        expect.any(String),
      );
    });
  });

  // ============================================
  // deleteByPattern() multi-iteration
  // ============================================
  describe('deleteByPattern() multi-iteration', () => {
    it('should iterate through multiple cursors', async () => {
      // First iteration returns cursor 123, second returns 0 (end)
      mockRedis.scan
        .mockResolvedValueOnce(['123', ['key1', 'key2']])
        .mockResolvedValueOnce(['0', ['key3']]);
      mockRedis.del.mockResolvedValue(1);

      const result = await service.deleteByPattern('test:*');

      expect(result).toBe(3);
      expect(mockRedis.scan).toHaveBeenCalledTimes(2);
      expect(mockRedis.del).toHaveBeenCalledTimes(2);
    });
  });
});
