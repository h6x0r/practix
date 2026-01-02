import { Test, TestingModule } from '@nestjs/testing';
import { AuthController } from './auth.controller';
import { AuthService } from './auth.service';
import { JwtAuthGuard } from './guards/jwt-auth.guard';
import { ConflictException, UnauthorizedException } from '@nestjs/common';
import { ThrottlerModule } from '@nestjs/throttler';

describe('AuthController', () => {
  let controller: AuthController;
  let authService: AuthService;

  const mockAuthService = {
    register: jest.fn(),
    login: jest.fn(),
    logout: jest.fn(),
  };

  const mockUser = {
    id: 'user-123',
    email: 'test@example.com',
    name: 'Test User',
    isPremium: false,
    plan: null,
    createdAt: new Date(),
  };

  const mockToken = 'jwt-token-123';

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      imports: [
        ThrottlerModule.forRoot([{
          ttl: 60000,
          limit: 10,
        }]),
      ],
      controllers: [AuthController],
      providers: [
        {
          provide: AuthService,
          useValue: mockAuthService,
        },
      ],
    })
      .overrideGuard(JwtAuthGuard)
      .useValue({ canActivate: () => true })
      .compile();

    controller = module.get<AuthController>(AuthController);
    authService = module.get<AuthService>(AuthService);

    jest.clearAllMocks();
  });

  it('should be defined', () => {
    expect(controller).toBeDefined();
  });

  describe('register', () => {
    const registerDto = {
      email: 'newuser@example.com',
      password: 'Password123!',
      name: 'New User',
    };

    it('should register a new user successfully', async () => {
      mockAuthService.register.mockResolvedValue({
        user: mockUser,
        token: mockToken,
      });

      const mockRequest = {
        ip: '127.0.0.1',
        socket: { remoteAddress: '127.0.0.1' },
      };

      const result = await controller.register(
        registerDto,
        'Mozilla/5.0',
        mockRequest as any
      );

      expect(result).toEqual({
        user: mockUser,
        token: mockToken,
      });
      expect(mockAuthService.register).toHaveBeenCalledWith(
        registerDto,
        'Mozilla/5.0',
        '127.0.0.1'
      );
    });

    it('should throw ConflictException if user already exists', async () => {
      mockAuthService.register.mockRejectedValue(
        new ConflictException('User with this email already exists')
      );

      await expect(
        controller.register(registerDto, 'Mozilla/5.0', {} as any)
      ).rejects.toThrow(ConflictException);
    });

    it('should handle missing user-agent', async () => {
      mockAuthService.register.mockResolvedValue({
        user: mockUser,
        token: mockToken,
      });

      const result = await controller.register(registerDto, undefined, {} as any);

      expect(result).toBeDefined();
      expect(mockAuthService.register).toHaveBeenCalledWith(
        registerDto,
        undefined,
        undefined
      );
    });

    it('should extract IP from socket.remoteAddress if req.ip is not available', async () => {
      mockAuthService.register.mockResolvedValue({
        user: mockUser,
        token: mockToken,
      });

      const mockRequest = {
        ip: undefined,
        socket: { remoteAddress: '192.168.1.1' },
      };

      await controller.register(registerDto, 'Mozilla/5.0', mockRequest as any);

      expect(mockAuthService.register).toHaveBeenCalledWith(
        registerDto,
        'Mozilla/5.0',
        '192.168.1.1'
      );
    });
  });

  describe('login', () => {
    const loginDto = {
      email: 'test@example.com',
      password: 'Password123!',
    };

    it('should login user successfully', async () => {
      mockAuthService.login.mockResolvedValue({
        user: mockUser,
        token: mockToken,
      });

      const mockRequest = {
        ip: '127.0.0.1',
        socket: { remoteAddress: '127.0.0.1' },
      };

      const result = await controller.login(
        loginDto,
        'Mozilla/5.0',
        mockRequest as any
      );

      expect(result).toEqual({
        user: mockUser,
        token: mockToken,
      });
      expect(mockAuthService.login).toHaveBeenCalledWith(
        loginDto,
        'Mozilla/5.0',
        '127.0.0.1'
      );
    });

    it('should throw UnauthorizedException for invalid credentials', async () => {
      mockAuthService.login.mockRejectedValue(
        new UnauthorizedException('Invalid credentials')
      );

      await expect(
        controller.login(loginDto, 'Mozilla/5.0', {} as any)
      ).rejects.toThrow(UnauthorizedException);
    });

    it('should handle missing request object', async () => {
      mockAuthService.login.mockResolvedValue({
        user: mockUser,
        token: mockToken,
      });

      const result = await controller.login(loginDto, 'Mozilla/5.0', undefined);

      expect(result).toBeDefined();
      expect(mockAuthService.login).toHaveBeenCalledWith(
        loginDto,
        'Mozilla/5.0',
        undefined
      );
    });

    it('should handle wrong password', async () => {
      mockAuthService.login.mockRejectedValue(
        new UnauthorizedException('Invalid credentials')
      );

      const wrongPasswordDto = {
        email: 'test@example.com',
        password: 'WrongPassword',
      };

      await expect(
        controller.login(wrongPasswordDto, 'Mozilla/5.0', {} as any)
      ).rejects.toThrow(UnauthorizedException);
    });

    it('should handle non-existent user', async () => {
      mockAuthService.login.mockRejectedValue(
        new UnauthorizedException('Invalid credentials')
      );

      const nonExistentDto = {
        email: 'nonexistent@example.com',
        password: 'Password123!',
      };

      await expect(
        controller.login(nonExistentDto, 'Mozilla/5.0', {} as any)
      ).rejects.toThrow(UnauthorizedException);
    });
  });

  describe('logout', () => {
    it('should logout user successfully', async () => {
      mockAuthService.logout.mockResolvedValue(undefined);

      const result = await controller.logout('Bearer jwt-token-123');

      expect(result).toEqual({ message: 'Logged out successfully' });
      expect(mockAuthService.logout).toHaveBeenCalledWith('jwt-token-123');
    });

    it('should handle logout without token', async () => {
      const result = await controller.logout(undefined as any);

      expect(result).toEqual({ message: 'Logged out successfully' });
      expect(mockAuthService.logout).not.toHaveBeenCalled();
    });

    it('should handle logout with empty authorization header', async () => {
      const result = await controller.logout('');

      expect(result).toEqual({ message: 'Logged out successfully' });
      expect(mockAuthService.logout).not.toHaveBeenCalled();
    });

    it('should strip Bearer prefix from token', async () => {
      mockAuthService.logout.mockResolvedValue(undefined);

      await controller.logout('Bearer my-secret-token');

      expect(mockAuthService.logout).toHaveBeenCalledWith('my-secret-token');
    });

    it('should handle token without Bearer prefix', async () => {
      mockAuthService.logout.mockResolvedValue(undefined);

      await controller.logout('raw-token-without-bearer');

      expect(mockAuthService.logout).toHaveBeenCalledWith('raw-token-without-bearer');
    });
  });

  describe('edge cases', () => {
    it('should handle special characters in email', async () => {
      const specialDto = {
        email: 'user+tag@example.com',
        password: 'Password123!',
        name: 'Special User',
      };

      mockAuthService.register.mockResolvedValue({
        user: { ...mockUser, email: specialDto.email },
        token: mockToken,
      });

      const result = await controller.register(specialDto, 'Mozilla/5.0', {} as any);

      expect(result.user.email).toBe('user+tag@example.com');
    });

    it('should handle unicode in name', async () => {
      const unicodeDto = {
        email: 'unicode@example.com',
        password: 'Password123!',
        name: 'Имя Пользователя 用户名',
      };

      mockAuthService.register.mockResolvedValue({
        user: { ...mockUser, name: unicodeDto.name },
        token: mockToken,
      });

      const result = await controller.register(unicodeDto, 'Mozilla/5.0', {} as any);

      expect(result.user.name).toBe('Имя Пользователя 用户名');
    });

    it('should handle IPv6 addresses', async () => {
      mockAuthService.login.mockResolvedValue({
        user: mockUser,
        token: mockToken,
      });

      const mockRequest = {
        ip: '::ffff:192.168.1.1',
        socket: { remoteAddress: '::1' },
      };

      await controller.login(
        { email: 'test@example.com', password: 'pass' },
        'Mozilla/5.0',
        mockRequest as any
      );

      expect(mockAuthService.login).toHaveBeenCalledWith(
        expect.any(Object),
        'Mozilla/5.0',
        '::ffff:192.168.1.1'
      );
    });
  });
});
