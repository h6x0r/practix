import { Injectable, NotFoundException } from "@nestjs/common";
import { PrismaService } from "../prisma/prisma.service";
import { AuditService } from "./audit/audit.service";

@Injectable()
export class AdminUsersService {
  constructor(
    private readonly prisma: PrismaService,
    private readonly auditService: AuditService,
  ) {}

  async searchUsers(query: string, limit = 20) {
    if (!query || query.length < 2) {
      return [];
    }

    const users = await this.prisma.user.findMany({
      where: {
        OR: [
          { email: { contains: query, mode: "insensitive" } },
          { name: { contains: query, mode: "insensitive" } },
        ],
      },
      select: {
        id: true,
        email: true,
        name: true,
        role: true,
        isPremium: true,
        isBanned: true,
        bannedAt: true,
        bannedReason: true,
        createdAt: true,
        lastActivityAt: true,
        _count: {
          select: {
            submissions: true,
            courses: true,
          },
        },
      },
      take: limit,
      orderBy: { createdAt: "desc" },
    });

    return users.map((user) => ({
      id: user.id,
      email: user.email,
      name: user.name,
      role: user.role,
      isPremium: user.isPremium,
      isBanned: user.isBanned,
      bannedAt: user.bannedAt,
      bannedReason: user.bannedReason,
      createdAt: user.createdAt,
      lastActivityAt: user.lastActivityAt,
      submissionsCount: user._count.submissions,
      coursesCount: user._count.courses,
    }));
  }

  /**
   * Get user by ID with full details
   */
  async getUserById(userId: string) {
    const user = await this.prisma.user.findUnique({
      where: { id: userId },
      select: {
        id: true,
        email: true,
        name: true,
        role: true,
        isPremium: true,
        isBanned: true,
        bannedAt: true,
        bannedReason: true,
        bannedBy: true,
        createdAt: true,
        lastActivityAt: true,
        xp: true,
        level: true,
        currentStreak: true,
        _count: {
          select: {
            submissions: true,
            courses: true,
            bugReports: true,
          },
        },
      },
    });

    if (!user) {
      return null;
    }

    return {
      ...user,
      submissionsCount: user._count.submissions,
      coursesCount: user._count.courses,
      bugReportsCount: user._count.bugReports,
    };
  }

  /**
   * Ban a user
   */
  async banUser(userId: string, reason: string, adminId: string) {
    const user = await this.prisma.user.findUnique({
      where: { id: userId },
      select: { id: true, role: true, isBanned: true },
    });

    if (!user) {
      throw new Error("User not found");
    }

    if (user.role === "ADMIN") {
      throw new Error("Cannot ban admin users");
    }

    if (user.isBanned) {
      throw new Error("User is already banned");
    }

    // Ban the user
    const updatedUser = await this.prisma.user.update({
      where: { id: userId },
      data: {
        isBanned: true,
        bannedAt: new Date(),
        bannedReason: reason,
        bannedBy: adminId,
      },
      select: {
        id: true,
        email: true,
        name: true,
        isBanned: true,
        bannedAt: true,
        bannedReason: true,
      },
    });

    // Invalidate all user sessions
    await this.prisma.session.updateMany({
      where: { userId, isActive: true },
      data: { isActive: false },
    });

    // Log audit event
    await this.auditService.log({
      adminId,
      action: "user_ban",
      entity: "user",
      entityId: userId,
      details: { reason, userEmail: updatedUser.email },
    });

    return updatedUser;
  }

  /**
   * Unban a user
   */
  async unbanUser(userId: string, adminId: string) {
    const user = await this.prisma.user.findUnique({
      where: { id: userId },
      select: { id: true, isBanned: true, email: true },
    });

    if (!user) {
      throw new Error("User not found");
    }

    if (!user.isBanned) {
      throw new Error("User is not banned");
    }

    const updatedUser = await this.prisma.user.update({
      where: { id: userId },
      data: {
        isBanned: false,
        bannedAt: null,
        bannedReason: null,
        bannedBy: null,
      },
      select: {
        id: true,
        email: true,
        name: true,
        isBanned: true,
      },
    });

    // Log audit event
    await this.auditService.log({
      adminId,
      action: "user_unban",
      entity: "user",
      entityId: userId,
      details: { userEmail: updatedUser.email },
    });

    return updatedUser;
  }

  /**
   * Get list of banned users
   */
  async getBannedUsers(limit = 50, offset = 0) {
    const [users, total] = await Promise.all([
      this.prisma.user.findMany({
        where: { isBanned: true },
        select: {
          id: true,
          email: true,
          name: true,
          bannedAt: true,
          bannedReason: true,
          bannedBy: true,
          createdAt: true,
        },
        orderBy: { bannedAt: "desc" },
        take: limit,
        skip: offset,
      }),
      this.prisma.user.count({ where: { isBanned: true } }),
    ]);

    return { users, total };
  }

}
