import { Test, TestingModule } from '@nestjs/testing';
import { SessionsService } from './sessions.service';
import { PrismaService } from '../prisma/prisma.service';
import { DeviceType } from '@prisma/client';
import * as crypto from 'crypto';

// Helper to hash tokens like the service does
const hashToken = (token: string): string => {
  return crypto.createHash('sha256').update(token).digest('hex');
};

describe('SessionsService', () => {
  let service: SessionsService;
  let prisma: PrismaService;

  const testToken = 'jwt-token-abc';
  const hashedToken = hashToken(testToken);

  const mockSession = {
    id: 'session-123',
    userId: 'user-123',
    token: hashedToken, // Session stores hashed token
    deviceType: DeviceType.DESKTOP,
    deviceInfo: 'Chrome on macOS',
    ipAddress: '192.168.1.1',
    expiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000), // 7 days from now
    isActive: true,
    createdAt: new Date(),
    lastActiveAt: new Date(),
  };

  const mockPrismaService = {
    session: {
      create: jest.fn(),
      findUnique: jest.fn(),
      findMany: jest.fn(),
      update: jest.fn(),
      updateMany: jest.fn(),
      deleteMany: jest.fn(),
      count: jest.fn(),
    },
  };

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [
        SessionsService,
        { provide: PrismaService, useValue: mockPrismaService },
      ],
    }).compile();

    service = module.get<SessionsService>(SessionsService);
    prisma = module.get<PrismaService>(PrismaService);

    jest.clearAllMocks();
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });

  // ============================================
  // createSession()
  // ============================================
  describe('createSession()', () => {
    it('should create a new session with hashed token', async () => {
      mockPrismaService.session.create.mockResolvedValue(mockSession);

      const result = await service.createSession('user-123', testToken);

      expect(result).toEqual(mockSession);
      expect(mockPrismaService.session.create).toHaveBeenCalledWith({
        data: expect.objectContaining({
          userId: 'user-123',
          token: hashedToken, // Token should be hashed
          isActive: true,
        }),
      });
    });

    it('should include device type and info if provided', async () => {
      mockPrismaService.session.create.mockResolvedValue(mockSession);

      await service.createSession('user-123', 'token', DeviceType.DESKTOP, 'Chrome on macOS');

      expect(mockPrismaService.session.create).toHaveBeenCalledWith({
        data: expect.objectContaining({
          deviceType: DeviceType.DESKTOP,
          deviceInfo: 'Chrome on macOS',
        }),
      });
    });

    it('should include IP address if provided', async () => {
      mockPrismaService.session.create.mockResolvedValue(mockSession);

      await service.createSession('user-123', 'token', DeviceType.UNKNOWN, undefined, '192.168.1.1');

      expect(mockPrismaService.session.create).toHaveBeenCalledWith({
        data: expect.objectContaining({
          ipAddress: '192.168.1.1',
        }),
      });
    });

    it('should set expiresAt to 7 days from now', async () => {
      mockPrismaService.session.create.mockResolvedValue(mockSession);

      await service.createSession('user-123', 'token');

      const callData = mockPrismaService.session.create.mock.calls[0][0].data;
      const expectedExpiry = new Date();
      expectedExpiry.setDate(expectedExpiry.getDate() + 7);

      // Allow 1 second tolerance
      expect(callData.expiresAt.getTime()).toBeCloseTo(expectedExpiry.getTime(), -3);
    });
  });

  // ============================================
  // validateSession()
  // ============================================
  describe('validateSession()', () => {
    it('should return session if valid and active', async () => {
      mockPrismaService.session.findUnique.mockResolvedValue(mockSession);

      const result = await service.validateSession(testToken);

      expect(result).toEqual(mockSession);
      // Should query using hashed token
      expect(mockPrismaService.session.findUnique).toHaveBeenCalledWith({
        where: { token: hashedToken },
      });
    });

    it('should return null if session not found', async () => {
      mockPrismaService.session.findUnique.mockResolvedValue(null);

      const result = await service.validateSession('invalid-token');

      expect(result).toBeNull();
    });

    it('should return null if session is inactive', async () => {
      mockPrismaService.session.findUnique.mockResolvedValue({
        ...mockSession,
        isActive: false,
      });

      const result = await service.validateSession(testToken);

      expect(result).toBeNull();
    });

    it('should return null if session is expired', async () => {
      mockPrismaService.session.findUnique.mockResolvedValue({
        ...mockSession,
        expiresAt: new Date(Date.now() - 1000), // Expired
      });

      const result = await service.validateSession(testToken);

      expect(result).toBeNull();
    });
  });

  // ============================================
  // invalidateUserSessions()
  // ============================================
  describe('invalidateUserSessions()', () => {
    it('should invalidate all user sessions', async () => {
      mockPrismaService.session.updateMany.mockResolvedValue({ count: 3 });

      const result = await service.invalidateUserSessions('user-123');

      expect(result).toBe(3);
      expect(mockPrismaService.session.updateMany).toHaveBeenCalledWith({
        where: {
          userId: 'user-123',
          isActive: true,
        },
        data: {
          isActive: false,
        },
      });
    });

    it('should return 0 if no active sessions', async () => {
      mockPrismaService.session.updateMany.mockResolvedValue({ count: 0 });

      const result = await service.invalidateUserSessions('user-no-sessions');

      expect(result).toBe(0);
    });
  });

  // ============================================
  // invalidateUserSessionsByDevice()
  // ============================================
  describe('invalidateUserSessionsByDevice()', () => {
    it('should invalidate sessions for specific device type', async () => {
      mockPrismaService.session.updateMany.mockResolvedValue({ count: 1 });

      const result = await service.invalidateUserSessionsByDevice('user-123', DeviceType.MOBILE);

      expect(result).toBe(1);
      expect(mockPrismaService.session.updateMany).toHaveBeenCalledWith({
        where: {
          userId: 'user-123',
          deviceType: DeviceType.MOBILE,
          isActive: true,
        },
        data: {
          isActive: false,
        },
      });
    });

    it('should not affect other device types', async () => {
      mockPrismaService.session.updateMany.mockResolvedValue({ count: 0 });

      const result = await service.invalidateUserSessionsByDevice('user-123', DeviceType.DESKTOP);

      expect(result).toBe(0);
      expect(mockPrismaService.session.updateMany).toHaveBeenCalledWith({
        where: {
          userId: 'user-123',
          deviceType: DeviceType.DESKTOP,
          isActive: true,
        },
        data: {
          isActive: false,
        },
      });
    });
  });

  // ============================================
  // getActiveSessionCountByDevice()
  // ============================================
  describe('getActiveSessionCountByDevice()', () => {
    it('should return count of active sessions for specific device type', async () => {
      mockPrismaService.session.count.mockResolvedValue(1);

      const result = await service.getActiveSessionCountByDevice('user-123', DeviceType.MOBILE);

      expect(result).toBe(1);
      expect(mockPrismaService.session.count).toHaveBeenCalledWith({
        where: {
          userId: 'user-123',
          deviceType: DeviceType.MOBILE,
          isActive: true,
          expiresAt: { gte: expect.any(Date) },
        },
      });
    });

    it('should return 0 for device type with no active sessions', async () => {
      mockPrismaService.session.count.mockResolvedValue(0);

      const result = await service.getActiveSessionCountByDevice('user-123', DeviceType.DESKTOP);

      expect(result).toBe(0);
    });
  });

  // ============================================
  // invalidateSession()
  // ============================================
  describe('invalidateSession()', () => {
    it('should invalidate specific session', async () => {
      mockPrismaService.session.update.mockResolvedValue({
        ...mockSession,
        isActive: false,
      });

      const result = await service.invalidateSession('session-123');

      expect(result.isActive).toBe(false);
      expect(mockPrismaService.session.update).toHaveBeenCalledWith({
        where: { id: 'session-123' },
        data: { isActive: false },
      });
    });
  });

  // ============================================
  // getUserSessions()
  // ============================================
  describe('getUserSessions()', () => {
    it('should return active sessions for user', async () => {
      mockPrismaService.session.findMany.mockResolvedValue([mockSession]);

      const result = await service.getUserSessions('user-123');

      expect(result).toEqual([mockSession]);
    });

    it('should filter by active status and non-expired', async () => {
      mockPrismaService.session.findMany.mockResolvedValue([]);

      await service.getUserSessions('user-123');

      expect(mockPrismaService.session.findMany).toHaveBeenCalledWith({
        where: {
          userId: 'user-123',
          isActive: true,
          expiresAt: {
            gte: expect.any(Date),
          },
        },
        orderBy: {
          lastActiveAt: 'desc',
        },
      });
    });

    it('should return empty array for user with no sessions', async () => {
      mockPrismaService.session.findMany.mockResolvedValue([]);

      const result = await service.getUserSessions('user-no-sessions');

      expect(result).toEqual([]);
    });
  });

  // ============================================
  // updateLastActive()
  // ============================================
  describe('updateLastActive()', () => {
    it('should update lastActiveAt using hashed token', async () => {
      const updatedSession = {
        ...mockSession,
        lastActiveAt: new Date(),
      };
      mockPrismaService.session.update.mockResolvedValue(updatedSession);

      const result = await service.updateLastActive(testToken);

      expect(result).toEqual(updatedSession);
      expect(mockPrismaService.session.update).toHaveBeenCalledWith({
        where: { token: hashedToken }, // Should use hashed token
        data: {
          lastActiveAt: expect.any(Date),
        },
      });
    });

    it('should return null if session not found', async () => {
      mockPrismaService.session.update.mockRejectedValue(new Error('Not found'));

      const result = await service.updateLastActive('invalid-token');

      expect(result).toBeNull();
    });
  });

  // ============================================
  // cleanupOldSessions()
  // ============================================
  describe('cleanupOldSessions()', () => {
    it('should delete old invalidated and expired sessions', async () => {
      mockPrismaService.session.deleteMany.mockResolvedValue({ count: 15 });

      const result = await service.cleanupOldSessions(30);

      expect(result).toBe(15);
      expect(mockPrismaService.session.deleteMany).toHaveBeenCalledWith({
        where: {
          OR: [
            {
              isActive: false,
              lastActiveAt: { lt: expect.any(Date) },
            },
            {
              expiresAt: { lt: expect.any(Date) },
            },
          ],
        },
      });
    });

    it('should use default 30 days if not specified', async () => {
      mockPrismaService.session.deleteMany.mockResolvedValue({ count: 0 });

      await service.cleanupOldSessions();

      const callArgs = mockPrismaService.session.deleteMany.mock.calls[0][0];
      const cutoffDate = callArgs.where.OR[0].lastActiveAt.lt;
      const expectedCutoff = new Date();
      expectedCutoff.setDate(expectedCutoff.getDate() - 30);

      // Allow 1 second tolerance
      expect(cutoffDate.getTime()).toBeCloseTo(expectedCutoff.getTime(), -3);
    });

    it('should return 0 if no sessions to cleanup', async () => {
      mockPrismaService.session.deleteMany.mockResolvedValue({ count: 0 });

      const result = await service.cleanupOldSessions();

      expect(result).toBe(0);
    });
  });

  // ============================================
  // getActiveSessionCount()
  // ============================================
  describe('getActiveSessionCount()', () => {
    it('should return count of active sessions for user', async () => {
      mockPrismaService.session.count.mockResolvedValue(3);

      const result = await service.getActiveSessionCount('user-123');

      expect(result).toBe(3);
      expect(mockPrismaService.session.count).toHaveBeenCalledWith({
        where: {
          userId: 'user-123',
          isActive: true,
          expiresAt: { gte: expect.any(Date) },
        },
      });
    });

    it('should return 0 for user with no active sessions', async () => {
      mockPrismaService.session.count.mockResolvedValue(0);

      const result = await service.getActiveSessionCount('user-no-sessions');

      expect(result).toBe(0);
    });
  });
});
