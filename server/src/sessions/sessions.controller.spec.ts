import { Test, TestingModule } from '@nestjs/testing';
import { HttpException, HttpStatus } from '@nestjs/common';
import { SessionsController } from './sessions.controller';
import { SessionsService } from './sessions.service';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';

describe('SessionsController', () => {
  let controller: SessionsController;
  let sessionsService: SessionsService;

  const mockSession = {
    id: 'session-123',
    userId: 'user-123',
    token: 'jwt-token-abc',
    deviceInfo: 'Chrome on macOS',
    ipAddress: '192.168.1.1',
    expiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000),
    isActive: true,
    createdAt: new Date(),
    lastActiveAt: new Date(),
  };

  const mockSessionsService = {
    getUserSessions: jest.fn(),
    invalidateSession: jest.fn(),
  };

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      controllers: [SessionsController],
      providers: [
        {
          provide: SessionsService,
          useValue: mockSessionsService,
        },
      ],
    })
      .overrideGuard(JwtAuthGuard)
      .useValue({ canActivate: () => true })
      .compile();

    controller = module.get<SessionsController>(SessionsController);
    sessionsService = module.get<SessionsService>(SessionsService);

    jest.clearAllMocks();
  });

  it('should be defined', () => {
    expect(controller).toBeDefined();
  });

  describe('getUserSessions', () => {
    it('should return user sessions without sensitive data', async () => {
      mockSessionsService.getUserSessions.mockResolvedValue([mockSession]);

      const result = await controller.getUserSessions({ user: { userId: 'user-123' } });

      expect(result).toHaveLength(1);
      expect(result[0]).toEqual({
        id: mockSession.id,
        deviceInfo: mockSession.deviceInfo,
        ipAddress: mockSession.ipAddress,
        createdAt: mockSession.createdAt,
        lastActiveAt: mockSession.lastActiveAt,
        expiresAt: mockSession.expiresAt,
        isActive: mockSession.isActive,
      });
      // Should NOT include token
      expect(result[0]).not.toHaveProperty('token');
      expect(result[0]).not.toHaveProperty('userId');
    });

    it('should return empty array for user with no sessions', async () => {
      mockSessionsService.getUserSessions.mockResolvedValue([]);

      const result = await controller.getUserSessions({ user: { userId: 'user-no-sessions' } });

      expect(result).toEqual([]);
    });

    it('should return multiple sessions', async () => {
      const multipleSessions = [
        mockSession,
        { ...mockSession, id: 'session-456', deviceInfo: 'Firefox on Windows' },
        { ...mockSession, id: 'session-789', deviceInfo: 'Safari on iOS' },
      ];
      mockSessionsService.getUserSessions.mockResolvedValue(multipleSessions);

      const result = await controller.getUserSessions({ user: { userId: 'user-123' } });

      expect(result).toHaveLength(3);
      expect(mockSessionsService.getUserSessions).toHaveBeenCalledWith('user-123');
    });
  });

  describe('invalidateSession', () => {
    it('should invalidate session that belongs to user', async () => {
      mockSessionsService.getUserSessions.mockResolvedValue([mockSession]);
      mockSessionsService.invalidateSession.mockResolvedValue({
        ...mockSession,
        isActive: false,
      });

      const result = await controller.invalidateSession(
        { user: { userId: 'user-123' } },
        'session-123'
      );

      expect(result).toEqual({
        message: 'Session invalidated successfully',
        sessionId: 'session-123',
      });
      expect(mockSessionsService.invalidateSession).toHaveBeenCalledWith('session-123');
    });

    it('should throw HttpException if session not found', async () => {
      mockSessionsService.getUserSessions.mockResolvedValue([mockSession]);

      await expect(
        controller.invalidateSession(
          { user: { userId: 'user-123' } },
          'nonexistent-session'
        )
      ).rejects.toThrow(HttpException);

      await expect(
        controller.invalidateSession(
          { user: { userId: 'user-123' } },
          'nonexistent-session'
        )
      ).rejects.toThrow('Session not found or does not belong to you');
    });

    it('should throw HttpException if session belongs to another user', async () => {
      // User has no sessions
      mockSessionsService.getUserSessions.mockResolvedValue([]);

      await expect(
        controller.invalidateSession(
          { user: { userId: 'other-user' } },
          'session-123'
        )
      ).rejects.toThrow(HttpException);
    });

    it('should use correct HTTP status for not found', async () => {
      mockSessionsService.getUserSessions.mockResolvedValue([]);

      try {
        await controller.invalidateSession(
          { user: { userId: 'user-123' } },
          'nonexistent'
        );
      } catch (error) {
        expect(error).toBeInstanceOf(HttpException);
        expect(error.getStatus()).toBe(HttpStatus.NOT_FOUND);
      }
    });

    it('should only invalidate the specified session', async () => {
      const multipleSessions = [
        mockSession,
        { ...mockSession, id: 'session-456' },
      ];
      mockSessionsService.getUserSessions.mockResolvedValue(multipleSessions);
      mockSessionsService.invalidateSession.mockResolvedValue({
        ...mockSession,
        isActive: false,
      });

      await controller.invalidateSession(
        { user: { userId: 'user-123' } },
        'session-123'
      );

      expect(mockSessionsService.invalidateSession).toHaveBeenCalledTimes(1);
      expect(mockSessionsService.invalidateSession).toHaveBeenCalledWith('session-123');
    });
  });
});
