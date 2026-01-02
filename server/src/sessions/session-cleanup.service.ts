import { Injectable, Logger } from '@nestjs/common';
import { Cron, CronExpression } from '@nestjs/schedule';
import { PrismaService } from '../prisma/prisma.service';

/**
 * Session Cleanup Service
 * Periodically removes expired and inactive sessions from the database
 */
@Injectable()
export class SessionCleanupService {
  private readonly logger = new Logger(SessionCleanupService.name);

  constructor(private prisma: PrismaService) {
    this.logger.log('Session cleanup service initialized');
  }

  /**
   * Clean up expired sessions every hour
   * Removes sessions that are either:
   * - Expired (expiresAt < now)
   * - Inactive (isActive = false)
   */
  @Cron(CronExpression.EVERY_HOUR)
  async cleanupExpiredSessions(): Promise<void> {
    const now = new Date();

    try {
      // Delete expired sessions
      const expiredResult = await this.prisma.session.deleteMany({
        where: {
          expiresAt: { lt: now },
        },
      });

      // Delete inactive sessions older than 24 hours
      const inactiveThreshold = new Date(now.getTime() - 24 * 60 * 60 * 1000);
      const inactiveResult = await this.prisma.session.deleteMany({
        where: {
          isActive: false,
          lastActiveAt: { lt: inactiveThreshold },
        },
      });

      const totalDeleted = expiredResult.count + inactiveResult.count;

      if (totalDeleted > 0) {
        this.logger.log(
          `Session cleanup: removed ${expiredResult.count} expired, ${inactiveResult.count} inactive sessions`,
        );
      }
    } catch (error) {
      this.logger.error('Session cleanup failed', error);
    }
  }

  /**
   * Deep cleanup: remove very old sessions (weekly)
   * Removes any session older than 30 days regardless of status
   */
  @Cron(CronExpression.EVERY_WEEK)
  async deepCleanup(): Promise<void> {
    const thirtyDaysAgo = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000);

    try {
      const result = await this.prisma.session.deleteMany({
        where: {
          createdAt: { lt: thirtyDaysAgo },
        },
      });

      if (result.count > 0) {
        this.logger.log(`Deep cleanup: removed ${result.count} sessions older than 30 days`);
      }
    } catch (error) {
      this.logger.error('Deep session cleanup failed', error);
    }
  }

  /**
   * Manual trigger for cleanup (for admin use)
   */
  async runCleanupNow(): Promise<{ expired: number; inactive: number; old: number }> {
    const now = new Date();
    const inactiveThreshold = new Date(now.getTime() - 24 * 60 * 60 * 1000);
    const thirtyDaysAgo = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);

    const expiredResult = await this.prisma.session.deleteMany({
      where: { expiresAt: { lt: now } },
    });

    const inactiveResult = await this.prisma.session.deleteMany({
      where: {
        isActive: false,
        lastActiveAt: { lt: inactiveThreshold },
      },
    });

    const oldResult = await this.prisma.session.deleteMany({
      where: { createdAt: { lt: thirtyDaysAgo } },
    });

    this.logger.log(
      `Manual cleanup: ${expiredResult.count} expired, ${inactiveResult.count} inactive, ${oldResult.count} old sessions removed`,
    );

    return {
      expired: expiredResult.count,
      inactive: inactiveResult.count,
      old: oldResult.count,
    };
  }

  /**
   * Get cleanup statistics
   */
  async getStats(): Promise<{
    totalSessions: number;
    activeSessions: number;
    expiredSessions: number;
    inactiveSessions: number;
  }> {
    const now = new Date();

    const [total, active, expired, inactive] = await Promise.all([
      this.prisma.session.count(),
      this.prisma.session.count({
        where: { isActive: true, expiresAt: { gt: now } },
      }),
      this.prisma.session.count({
        where: { expiresAt: { lt: now } },
      }),
      this.prisma.session.count({
        where: { isActive: false },
      }),
    ]);

    return {
      totalSessions: total,
      activeSessions: active,
      expiredSessions: expired,
      inactiveSessions: inactive,
    };
  }
}
