import { Test, TestingModule } from "@nestjs/testing";
import { ConfigService } from "@nestjs/config";
import { UnauthorizedException } from "@nestjs/common";
import { JwtStrategy } from "./jwt.strategy";
import { PrismaService } from "../../prisma/prisma.service";

describe("JwtStrategy", () => {
  const mockPrismaService = {
    user: {
      findUnique: jest.fn(),
    },
  };

  const mockConfigService = {
    get: jest
      .fn()
      .mockReturnValue("this-is-a-very-long-jwt-secret-for-testing-purposes"),
  };

  describe("constructor", () => {
    it("should initialize successfully with valid JWT_SECRET", async () => {
      const module: TestingModule = await Test.createTestingModule({
        providers: [
          JwtStrategy,
          { provide: ConfigService, useValue: mockConfigService },
          { provide: PrismaService, useValue: mockPrismaService },
        ],
      }).compile();

      const strategy = module.get<JwtStrategy>(JwtStrategy);
      expect(strategy).toBeDefined();
    });

    it("should throw error if JWT_SECRET is not configured", () => {
      const badConfigService = {
        get: jest.fn().mockReturnValue(undefined),
      };

      expect(() => {
        new JwtStrategy(badConfigService as any, mockPrismaService as any);
      }).toThrow("FATAL: JWT_SECRET environment variable is not configured");
    });

    it("should throw error if JWT_SECRET is empty string", () => {
      const badConfigService = {
        get: jest.fn().mockReturnValue(""),
      };

      expect(() => {
        new JwtStrategy(badConfigService as any, mockPrismaService as any);
      }).toThrow("FATAL: JWT_SECRET environment variable is not configured");
    });

    it("should throw error if JWT_SECRET is too short", () => {
      const badConfigService = {
        get: jest.fn().mockReturnValue("short-secret"),
      };

      expect(() => {
        new JwtStrategy(badConfigService as any, mockPrismaService as any);
      }).toThrow("FATAL: JWT_SECRET is too short");
    });

    it("should throw error if JWT_SECRET is exactly 31 characters", () => {
      const badConfigService = {
        get: jest.fn().mockReturnValue("a".repeat(31)),
      };

      expect(() => {
        new JwtStrategy(badConfigService as any, mockPrismaService as any);
      }).toThrow("FATAL: JWT_SECRET is too short");
    });

    it("should accept JWT_SECRET with exactly 32 characters", async () => {
      const goodConfigService = {
        get: jest.fn().mockReturnValue("a".repeat(32)),
      };

      const module: TestingModule = await Test.createTestingModule({
        providers: [
          JwtStrategy,
          { provide: ConfigService, useValue: goodConfigService },
          { provide: PrismaService, useValue: mockPrismaService },
        ],
      }).compile();

      const strategy = module.get<JwtStrategy>(JwtStrategy);
      expect(strategy).toBeDefined();
    });
  });

  describe("validate", () => {
    let strategy: JwtStrategy;

    beforeEach(async () => {
      jest.clearAllMocks();
      mockPrismaService.user.findUnique.mockResolvedValue({ isBanned: false });

      const module: TestingModule = await Test.createTestingModule({
        providers: [
          JwtStrategy,
          { provide: ConfigService, useValue: mockConfigService },
          { provide: PrismaService, useValue: mockPrismaService },
        ],
      }).compile();

      strategy = module.get<JwtStrategy>(JwtStrategy);
    });

    it("should return user object from payload", async () => {
      const payload = {
        sub: "user-123",
        email: "test@example.com",
        iat: Date.now(),
        exp: Date.now() + 3600000,
      };

      const result = await strategy.validate(payload);

      expect(result).toEqual({
        userId: "user-123",
        email: "test@example.com",
      });
    });

    it("should handle payload with different user id", async () => {
      const payload = {
        sub: "another-user-456",
        email: "another@example.com",
        iat: Date.now(),
        exp: Date.now() + 3600000,
      };

      const result = await strategy.validate(payload);

      expect(result).toEqual({
        userId: "another-user-456",
        email: "another@example.com",
      });
    });

    it("should handle unicode email", async () => {
      const payload = {
        sub: "user-789",
        email: "тест@example.com",
        iat: Date.now(),
        exp: Date.now() + 3600000,
      };

      const result = await strategy.validate(payload);

      expect(result.email).toBe("тест@example.com");
    });

    it("should throw UnauthorizedException if user is banned", async () => {
      mockPrismaService.user.findUnique.mockResolvedValue({ isBanned: true });

      const payload = {
        sub: "banned-user",
        email: "banned@example.com",
        iat: Date.now(),
        exp: Date.now() + 3600000,
      };

      await expect(strategy.validate(payload)).rejects.toThrow(
        UnauthorizedException,
      );
      await expect(strategy.validate(payload)).rejects.toThrow(
        "Your account has been suspended",
      );
    });

    it("should allow non-banned user", async () => {
      mockPrismaService.user.findUnique.mockResolvedValue({ isBanned: false });

      const payload = {
        sub: "active-user",
        email: "active@example.com",
        iat: Date.now(),
        exp: Date.now() + 3600000,
      };

      const result = await strategy.validate(payload);

      expect(result).toEqual({
        userId: "active-user",
        email: "active@example.com",
      });
    });

    it("should allow user if not found in database (user field is null)", async () => {
      mockPrismaService.user.findUnique.mockResolvedValue(null);

      const payload = {
        sub: "unknown-user",
        email: "unknown@example.com",
        iat: Date.now(),
        exp: Date.now() + 3600000,
      };

      const result = await strategy.validate(payload);

      expect(result).toEqual({
        userId: "unknown-user",
        email: "unknown@example.com",
      });
    });
  });
});
