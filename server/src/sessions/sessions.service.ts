import { Injectable } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { Session } from '@prisma/client';

@Injectable()
export class SessionsService {
  constructor(private prisma: PrismaService) {}

  /**
   * Create a new session for a user
   * @param userId - User ID
   * @param token - JWT token
   * @param deviceInfo - Optional device information (browser, OS)
   * @param ipAddress - Optional IP address
   * @returns Created session
   */
  async createSession(
    userId: string,
    token: string,
    deviceInfo?: string,
    ipAddress?: string,
  ): Promise<Session> {
    // Session expires in 7 days
    const expiresAt = new Date();
    expiresAt.setDate(expiresAt.getDate() + 7);

    return this.prisma.session.create({
      data: {
        userId,
        token,
        deviceInfo,
        ipAddress,
        expiresAt,
        isActive: true,
      },
    });
  }

  /**
   * Validate a session by token
   * Returns the session if valid and active, null otherwise
   * @param token - JWT token
   * @returns Session or null
   */
  async validateSession(token: string): Promise<Session | null> {
    const session = await this.prisma.session.findUnique({
      where: { token },
    });

    if (!session) {
      return null;
    }

    // Check if session is active and not expired
    const now = new Date();
    if (!session.isActive || session.expiresAt < now) {
      return null;
    }

    return session;
  }

  /**
   * Invalidate all sessions for a user
   * Sets isActive to false for all user sessions
   * @param userId - User ID
   * @returns Number of sessions invalidated
   */
  async invalidateUserSessions(userId: string): Promise<number> {
    const result = await this.prisma.session.updateMany({
      where: {
        userId,
        isActive: true,
      },
      data: {
        isActive: false,
      },
    });

    return result.count;
  }

  /**
   * Invalidate a specific session
   * Sets isActive to false
   * @param sessionId - Session ID
   * @returns Updated session
   */
  async invalidateSession(sessionId: string): Promise<Session> {
    return this.prisma.session.update({
      where: { id: sessionId },
      data: { isActive: false },
    });
  }

  /**
   * Get all active sessions for a user
   * @param userId - User ID
   * @returns Array of active sessions
   */
  async getUserSessions(userId: string): Promise<Session[]> {
    const now = new Date();

    return this.prisma.session.findMany({
      where: {
        userId,
        isActive: true,
        expiresAt: {
          gte: now,
        },
      },
      orderBy: {
        lastActiveAt: 'desc',
      },
    });
  }

  /**
   * Update the last active timestamp for a session
   * @param token - JWT token
   * @returns Updated session or null if not found
   */
  async updateLastActive(token: string): Promise<Session | null> {
    try {
      return await this.prisma.session.update({
        where: { token },
        data: {
          lastActiveAt: new Date(),
        },
      });
    } catch (error) {
      // Session not found
      return null;
    }
  }
}
