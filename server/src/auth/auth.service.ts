
import { Injectable, UnauthorizedException, ConflictException } from '@nestjs/common';
import { UsersService } from '../users/users.service';
import { SessionsService } from '../sessions/sessions.service';
import { JwtService } from '@nestjs/jwt';
import * as bcrypt from 'bcrypt';
import { LoginDto, RegisterDto } from '../common/dto';
import { User } from '@prisma/client';

@Injectable()
export class AuthService {
  constructor(
    private usersService: UsersService,
    private sessionsService: SessionsService,
    private jwtService: JwtService,
  ) {}

  async register(registerDto: RegisterDto, deviceInfo?: string, ipAddress?: string) {
    // Check if user exists
    const existing = await this.usersService.findOne(registerDto.email);
    if (existing) {
      throw new ConflictException('User with this email already exists');
    }

    // Hash password
    const hashedPassword = await bcrypt.hash(registerDto.password, 10);

    // Create user (isPremium defaults to false, computed from subscriptions)
    const user = await this.usersService.create({
      email: registerDto.email,
      name: registerDto.name,
      password: hashedPassword,
      preferences: {
        editorFontSize: 14,
        editorMinimap: false,
        editorTheme: 'vs-dark',
        editorLineNumbers: true,
        notifications: {
            emailDigest: true,
            newCourses: true
        }
      },
    });

    const token = this.generateToken(user);

    // Create session for single-device enforcement
    await this.sessionsService.createSession(user.id, token, deviceInfo, ipAddress);

    // Compute isPremium and plan from active subscriptions
    const isPremium = await this.usersService.isPremiumUser(user.id);
    const plan = await this.usersService.getActivePlan(user.id);

    // Transform to DTO
    return { user: this.transformUser(user, isPremium, plan), token };
  }

  async login(loginDto: LoginDto, deviceInfo?: string, ipAddress?: string) {
    // Use findOneForAuth to get password for verification
    const user = await this.usersService.findOneForAuth(loginDto.email);
    if (!user) {
      throw new UnauthorizedException('Invalid credentials');
    }

    const isMatch = await bcrypt.compare(loginDto.password, user.password);
    if (!isMatch) {
      throw new UnauthorizedException('Invalid credentials');
    }

    // Invalidate all existing sessions for single-device enforcement
    await this.sessionsService.invalidateUserSessions(user.id);

    const token = this.generateToken(user);

    // Create new session
    await this.sessionsService.createSession(user.id, token, deviceInfo, ipAddress);

    // Compute isPremium and plan from active subscriptions
    const isPremium = await this.usersService.isPremiumUser(user.id);
    const plan = await this.usersService.getActivePlan(user.id);

    // Transform to DTO
    return { user: this.transformUser(user, isPremium, plan), token };
  }

  /**
   * Logout - invalidate the current session
   */
  async logout(token: string): Promise<void> {
    const session = await this.sessionsService.validateSession(token);
    if (session) {
      await this.sessionsService.invalidateSession(session.id);
    }
  }

  private generateToken(user: User) {
    const payload = { email: user.email, sub: user.id };
    return this.jwtService.sign(payload);
  }

  /**
   * Shapes the DB entity into the Frontend DTO.
   * Overrides isPremium and plan with computed values from active subscriptions.
   */
  private transformUser(user: User, isPremium?: boolean, plan?: { name: string; expiresAt: string } | null) {
    const { password, ...result } = user;

    // Override with computed values if provided
    if (isPremium !== undefined) {
      result.isPremium = isPremium;
    }
    if (plan !== undefined) {
      result.plan = plan;
    }

    return result;
  }
}
