import { Test, TestingModule } from '@nestjs/testing';
import { ConfigService } from '@nestjs/config';
import { JwtStrategy } from './jwt.strategy';

describe('JwtStrategy', () => {
  describe('constructor', () => {
    it('should initialize successfully with valid JWT_SECRET', async () => {
      const mockConfigService = {
        get: jest.fn().mockReturnValue('this-is-a-very-long-jwt-secret-for-testing-purposes'),
      };

      const module: TestingModule = await Test.createTestingModule({
        providers: [
          JwtStrategy,
          { provide: ConfigService, useValue: mockConfigService },
        ],
      }).compile();

      const strategy = module.get<JwtStrategy>(JwtStrategy);
      expect(strategy).toBeDefined();
    });

    it('should throw error if JWT_SECRET is not configured', () => {
      const mockConfigService = {
        get: jest.fn().mockReturnValue(undefined),
      };

      expect(() => {
        new JwtStrategy(mockConfigService as any);
      }).toThrow('FATAL: JWT_SECRET environment variable is not configured');
    });

    it('should throw error if JWT_SECRET is empty string', () => {
      const mockConfigService = {
        get: jest.fn().mockReturnValue(''),
      };

      expect(() => {
        new JwtStrategy(mockConfigService as any);
      }).toThrow('FATAL: JWT_SECRET environment variable is not configured');
    });

    it('should throw error if JWT_SECRET is too short', () => {
      const mockConfigService = {
        get: jest.fn().mockReturnValue('short-secret'),
      };

      expect(() => {
        new JwtStrategy(mockConfigService as any);
      }).toThrow('FATAL: JWT_SECRET is too short');
    });

    it('should throw error if JWT_SECRET is exactly 31 characters', () => {
      const mockConfigService = {
        get: jest.fn().mockReturnValue('a'.repeat(31)),
      };

      expect(() => {
        new JwtStrategy(mockConfigService as any);
      }).toThrow('FATAL: JWT_SECRET is too short');
    });

    it('should accept JWT_SECRET with exactly 32 characters', async () => {
      const mockConfigService = {
        get: jest.fn().mockReturnValue('a'.repeat(32)),
      };

      const module: TestingModule = await Test.createTestingModule({
        providers: [
          JwtStrategy,
          { provide: ConfigService, useValue: mockConfigService },
        ],
      }).compile();

      const strategy = module.get<JwtStrategy>(JwtStrategy);
      expect(strategy).toBeDefined();
    });
  });

  describe('validate', () => {
    let strategy: JwtStrategy;

    beforeEach(async () => {
      const mockConfigService = {
        get: jest.fn().mockReturnValue('this-is-a-very-long-jwt-secret-for-testing-purposes'),
      };

      const module: TestingModule = await Test.createTestingModule({
        providers: [
          JwtStrategy,
          { provide: ConfigService, useValue: mockConfigService },
        ],
      }).compile();

      strategy = module.get<JwtStrategy>(JwtStrategy);
    });

    it('should return user object from payload', async () => {
      const payload = {
        sub: 'user-123',
        email: 'test@example.com',
        iat: Date.now(),
        exp: Date.now() + 3600000,
      };

      const result = await strategy.validate(payload);

      expect(result).toEqual({
        userId: 'user-123',
        email: 'test@example.com',
      });
    });

    it('should handle payload with different user id', async () => {
      const payload = {
        sub: 'another-user-456',
        email: 'another@example.com',
        iat: Date.now(),
        exp: Date.now() + 3600000,
      };

      const result = await strategy.validate(payload);

      expect(result).toEqual({
        userId: 'another-user-456',
        email: 'another@example.com',
      });
    });

    it('should handle unicode email', async () => {
      const payload = {
        sub: 'user-789',
        email: 'тест@example.com',
        iat: Date.now(),
        exp: Date.now() + 3600000,
      };

      const result = await strategy.validate(payload);

      expect(result.email).toBe('тест@example.com');
    });
  });
});
