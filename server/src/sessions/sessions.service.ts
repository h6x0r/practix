import { Injectable } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { Session, DeviceType } from '@prisma/client';
import * as crypto from 'crypto';

@Injectable()
export class SessionsService {
  constructor(private prisma: PrismaService) {}

  /**
   * Hash a token using SHA256
   * This is a one-way hash to protect tokens if DB is compromised
   */
  private hashToken(token: string): string {
    return crypto.createHash('sha256').update(token).digest('hex');
  }

  /**
   * Create a new session for a user
   * @param userId - User ID
   * @param token - JWT token (will be hashed before storage)
   * @param deviceType - Device type (MOBILE, DESKTOP, UNKNOWN)
   * @param deviceInfo - Optional device information (browser, OS)
   * @param ipAddress - Optional IP address
   * @returns Created session
   */
  async createSession(
    userId: string,
    token: string,
    deviceType: DeviceType = DeviceType.UNKNOWN,
    deviceInfo?: string,
    ipAddress?: string,
  ): Promise<Session> {
    // Session expires in 7 days
    const expiresAt = new Date();
    expiresAt.setDate(expiresAt.getDate() + 7);

    // Hash token before storing - protects against DB leaks
    const tokenHash = this.hashToken(token);

    return this.prisma.session.create({
      data: {
        userId,
        token: tokenHash, // Store hash, not plaintext
        deviceType,
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
   * @param token - JWT token (will be hashed for lookup)
   * @returns Session or null
   */
  async validateSession(token: string): Promise<Session | null> {
    // Hash the incoming token for comparison
    const tokenHash = this.hashToken(token);

    const session = await this.prisma.session.findUnique({
      where: { token: tokenHash },
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
   * Invalidate all sessions for a user (legacy - use invalidateUserSessionsByDevice)
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
   * Invalidate sessions for a user by device type
   * Allows 1 mobile + 1 desktop session simultaneously
   * @param userId - User ID
   * @param deviceType - Device type to invalidate
   * @returns Number of sessions invalidated
   */
  async invalidateUserSessionsByDevice(
    userId: string,
    deviceType: DeviceType,
  ): Promise<number> {
    const result = await this.prisma.session.updateMany({
      where: {
        userId,
        deviceType,
        isActive: true,
      },
      data: {
        isActive: false,
      },
    });

    return result.count;
  }

  /**
   * Get active session count by device type for a user
   * @param userId - User ID
   * @param deviceType - Device type to count
   * @returns Count of active sessions for that device type
   */
  async getActiveSessionCountByDevice(
    userId: string,
    deviceType: DeviceType,
  ): Promise<number> {
    return this.prisma.session.count({
      where: {
        userId,
        deviceType,
        isActive: true,
        expiresAt: { gte: new Date() },
      },
    });
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
   * @param token - JWT token (will be hashed for lookup)
   * @returns Updated session or null if not found
   */
  async updateLastActive(token: string): Promise<Session | null> {
    try {
      // Hash the incoming token for lookup
      const tokenHash = this.hashToken(token);

      return await this.prisma.session.update({
        where: { token: tokenHash },
        data: {
          lastActiveAt: new Date(),
        },
      });
    } catch (error) {
      // Session not found
      return null;
    }
  }

  /**
   * Clean up old invalidated or expired sessions
   * Should be called periodically (e.g., daily via cron)
   * @param olderThanDays - Delete sessions older than this many days (default: 30)
   * @returns Number of sessions deleted
   */
  async cleanupOldSessions(olderThanDays = 30): Promise<number> {
    const cutoffDate = new Date();
    cutoffDate.setDate(cutoffDate.getDate() - olderThanDays);

    const result = await this.prisma.session.deleteMany({
      where: {
        OR: [
          // Delete invalidated sessions older than cutoff
          {
            isActive: false,
            lastActiveAt: { lt: cutoffDate },
          },
          // Delete expired sessions older than cutoff
          {
            expiresAt: { lt: cutoffDate },
          },
        ],
      },
    });

    return result.count;
  }

  /**
   * Get session count per user (for monitoring/limiting)
   * @param userId - User ID
   * @returns Count of active sessions
   */
  async getActiveSessionCount(userId: string): Promise<number> {
    return this.prisma.session.count({
      where: {
        userId,
        isActive: true,
        expiresAt: { gte: new Date() },
      },
    });
  }
}
