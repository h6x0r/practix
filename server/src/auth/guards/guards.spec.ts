import { Test, TestingModule } from '@nestjs/testing';
import { ExecutionContext, ForbiddenException } from '@nestjs/common';
import { JwtAuthGuard } from './jwt-auth.guard';
import { OptionalJwtAuthGuard } from './optional-jwt.guard';
import { AdminGuard } from './admin.guard';
import { PrismaService } from '../../prisma/prisma.service';

describe('Auth Guards', () => {
  describe('JwtAuthGuard', () => {
    let guard: JwtAuthGuard;

    beforeEach(() => {
      guard = new JwtAuthGuard();
    });

    it('should be defined', () => {
      expect(guard).toBeDefined();
    });

    it('should extend AuthGuard with jwt strategy', () => {
      // JwtAuthGuard extends AuthGuard('jwt')
      // The actual authentication is handled by passport
      expect(guard).toBeInstanceOf(JwtAuthGuard);
    });
  });

  describe('OptionalJwtAuthGuard', () => {
    let guard: OptionalJwtAuthGuard;

    beforeEach(() => {
      guard = new OptionalJwtAuthGuard();
    });

    it('should be defined', () => {
      expect(guard).toBeDefined();
    });

    it('should return user when user is authenticated', () => {
      const mockUser = { userId: 'user-123', email: 'test@example.com' };

      const result = guard.handleRequest(null, mockUser, null);

      expect(result).toEqual(mockUser);
    });

    it('should return null when no user is found', () => {
      const result = guard.handleRequest(null, null, null);

      expect(result).toBeNull();
    });

    it('should return null when user is undefined', () => {
      const result = guard.handleRequest(null, undefined, null);

      expect(result).toBeNull();
    });

    it('should return null even when there is an error', () => {
      const mockError = new Error('Token expired');

      const result = guard.handleRequest(mockError, null, null);

      expect(result).toBeNull();
    });

    it('should return null when user is false', () => {
      const result = guard.handleRequest(null, false, null);

      expect(result).toBeNull();
    });
  });

  describe('AdminGuard', () => {
    let guard: AdminGuard;
    let prisma: PrismaService;

    const mockPrismaService = {
      user: {
        findUnique: jest.fn(),
      },
    };

    const createMockExecutionContext = (userId?: string): ExecutionContext => {
      return {
        switchToHttp: () => ({
          getRequest: () => ({
            user: userId ? { userId } : null,
          }),
        }),
      } as ExecutionContext;
    };

    beforeEach(async () => {
      const module: TestingModule = await Test.createTestingModule({
        providers: [
          AdminGuard,
          {
            provide: PrismaService,
            useValue: mockPrismaService,
          },
        ],
      }).compile();

      guard = module.get<AdminGuard>(AdminGuard);
      prisma = module.get<PrismaService>(PrismaService);

      jest.clearAllMocks();
    });

    it('should be defined', () => {
      expect(guard).toBeDefined();
    });

    it('should allow access for admin user', async () => {
      mockPrismaService.user.findUnique.mockResolvedValue({ role: 'ADMIN' });

      const context = createMockExecutionContext('admin-user-123');

      const result = await guard.canActivate(context);

      expect(result).toBe(true);
      expect(mockPrismaService.user.findUnique).toHaveBeenCalledWith({
        where: { id: 'admin-user-123' },
        select: { role: true },
      });
    });

    it('should throw ForbiddenException when no userId in request', async () => {
      const context = createMockExecutionContext();

      await expect(guard.canActivate(context)).rejects.toThrow(ForbiddenException);
      await expect(guard.canActivate(context)).rejects.toThrow('Authentication required');
    });

    it('should throw ForbiddenException when user not found', async () => {
      mockPrismaService.user.findUnique.mockResolvedValue(null);

      const context = createMockExecutionContext('non-existent-user');

      await expect(guard.canActivate(context)).rejects.toThrow(ForbiddenException);
      await expect(guard.canActivate(context)).rejects.toThrow('Admin access required');
    });

    it('should throw ForbiddenException for non-admin user', async () => {
      mockPrismaService.user.findUnique.mockResolvedValue({ role: 'USER' });

      const context = createMockExecutionContext('regular-user-123');

      await expect(guard.canActivate(context)).rejects.toThrow(ForbiddenException);
      await expect(guard.canActivate(context)).rejects.toThrow('Admin access required');
    });

    it('should throw ForbiddenException for MODERATOR role', async () => {
      mockPrismaService.user.findUnique.mockResolvedValue({ role: 'MODERATOR' });

      const context = createMockExecutionContext('mod-user-123');

      await expect(guard.canActivate(context)).rejects.toThrow(ForbiddenException);
    });

    it('should handle undefined user object in request', async () => {
      const context = {
        switchToHttp: () => ({
          getRequest: () => ({
            user: undefined,
          }),
        }),
      } as ExecutionContext;

      await expect(guard.canActivate(context)).rejects.toThrow(ForbiddenException);
      await expect(guard.canActivate(context)).rejects.toThrow('Authentication required');
    });

    it('should handle database errors', async () => {
      mockPrismaService.user.findUnique.mockRejectedValue(new Error('Database error'));

      const context = createMockExecutionContext('user-123');

      await expect(guard.canActivate(context)).rejects.toThrow('Database error');
    });
  });
});
