import { Injectable } from "@nestjs/common";
import { Prisma } from "@prisma/client";
import { PrismaService } from "../prisma/prisma.service";
import { CacheService } from "../cache/cache.service";

// XP rewards by difficulty
const XP_REWARDS: Record<string, number> = {
  easy: 25,
  medium: 50,
  hard: 100,
};

// Level thresholds (XP required for each level)
const LEVEL_THRESHOLDS = [
  0, // Level 1: 0 XP
  100, // Level 2: 100 XP
  250, // Level 3: 250 XP
  500, // Level 4: 500 XP
  1000, // Level 5: 1000 XP
  1750, // Level 6
  2750, // Level 7
  4000, // Level 8
  5500, // Level 9
  7500, // Level 10
  10000, // Level 11
  13000, // Level 12
  16500, // Level 13
  20500, // Level 14
  25000, // Level 15
  30000, // Level 16
  36000, // Level 17
  43000, // Level 18
  51000, // Level 19
  60000, // Level 20
  // Beyond level 20: +10000 per level
];

// Cache TTL for rank (5 minutes - frequently changing data)
const RANK_CACHE_TTL = 300;

@Injectable()
export class GamificationService {
  constructor(
    private prisma: PrismaService,
    private cache: CacheService,
  ) {}

  /**
   * Calculate level from XP
   * Beyond level 20: +10,000 XP per level
   */
  calculateLevel(xp: number): number {
    const maxDefinedLevel = LEVEL_THRESHOLDS.length; // 20
    const maxThreshold = LEVEL_THRESHOLDS[maxDefinedLevel - 1]; // 60000

    // Beyond level 20: each additional level requires 10,000 XP
    if (xp >= maxThreshold) {
      const beyondMax = xp - maxThreshold;
      return maxDefinedLevel + Math.floor(beyondMax / 10000);
    }

    // Find level within defined thresholds
    for (let i = maxDefinedLevel - 1; i >= 0; i--) {
      if (xp >= LEVEL_THRESHOLDS[i]) {
        return i + 1;
      }
    }

    return 1; // Default to level 1
  }

  /**
   * Get XP required for next level
   */
  getXpForNextLevel(level: number): number {
    if (level < LEVEL_THRESHOLDS.length) {
      return LEVEL_THRESHOLDS[level];
    }
    // Beyond defined levels
    return (
      LEVEL_THRESHOLDS[LEVEL_THRESHOLDS.length - 1] +
      (level - LEVEL_THRESHOLDS.length + 1) * 10000
    );
  }

  /**
   * Award XP for completing a task
   * Uses atomic increment to prevent race conditions
   */
  async awardTaskXp(
    userId: string,
    difficulty: string,
  ): Promise<{
    xpEarned: number;
    totalXp: number;
    level: number;
    leveledUp: boolean;
    newBadges: Array<{ slug: string; name: string; icon: string }>;
  }> {
    const xpEarned = XP_REWARDS[difficulty] || XP_REWARDS.easy;

    // Use transaction for atomic XP increment and level calculation
    const result = await this.prisma.$transaction(async (tx) => {
      // Get current user state
      const user = await tx.user.findUnique({
        where: { id: userId },
        select: {
          xp: true,
          level: true,
          currentStreak: true,
          maxStreak: true,
          lastActivityAt: true,
        },
      });

      const oldLevel = user?.level || 1;

      // Calculate streak
      const { newStreak, newMaxStreak } = this.calculateStreak(
        user?.currentStreak || 0,
        user?.maxStreak || 0,
        user?.lastActivityAt ?? null,
      );

      // Atomically increment XP and update other fields
      const updatedUser = await tx.user.update({
        where: { id: userId },
        data: {
          xp: { increment: xpEarned }, // Atomic increment!
          currentStreak: newStreak,
          maxStreak: newMaxStreak,
          lastActivityAt: new Date(),
        },
        select: { xp: true },
      });

      const newXp = updatedUser.xp;
      const newLevel = this.calculateLevel(newXp);

      // Update level if changed
      if (newLevel !== oldLevel) {
        await tx.user.update({
          where: { id: userId },
          data: { level: newLevel },
        });
      }

      return {
        oldLevel,
        newXp,
        newLevel,
        newStreak,
        newMaxStreak,
      };
    });

    // Invalidate rank cache since XP changed
    await this.invalidateRankCache(userId);

    // Check for new badges (outside transaction to avoid long lock)
    const newBadges = await this.checkAndAwardBadges(
      userId,
      result.newXp,
      result.newLevel,
      result.newStreak,
      result.newMaxStreak,
    );

    return {
      xpEarned,
      totalXp: result.newXp,
      level: result.newLevel,
      leveledUp: result.newLevel > result.oldLevel,
      newBadges,
    };
  }

  /**
   * Calculate streak based on last activity
   * Uses UTC dates to avoid timezone issues
   * Uses hour-based calculation with grace period for timezone tolerance
   */
  private calculateStreak(
    currentStreak: number,
    maxStreak: number,
    lastActivityAt: Date | null,
  ): { newStreak: number; newMaxStreak: number } {
    if (!lastActivityAt) {
      return { newStreak: 1, newMaxStreak: Math.max(1, maxStreak) };
    }

    const now = new Date();
    const lastActivity = new Date(lastActivityAt);

    // Calculate hours since last activity (more precise and timezone-agnostic)
    const hoursSinceLastActivity =
      (now.getTime() - lastActivity.getTime()) / (1000 * 60 * 60);

    // Use UTC dates for day comparison
    const todayUTC = new Date(
      Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()),
    );
    const lastActivityUTC = new Date(
      Date.UTC(
        lastActivity.getUTCFullYear(),
        lastActivity.getUTCMonth(),
        lastActivity.getUTCDate(),
      ),
    );

    const diffDays = Math.floor(
      (todayUTC.getTime() - lastActivityUTC.getTime()) / (1000 * 60 * 60 * 24),
    );

    if (diffDays === 0) {
      // Same UTC day - no streak change
      return { newStreak: currentStreak, newMaxStreak: maxStreak };
    } else if (diffDays === 1) {
      // Consecutive day - increment streak
      const newStreak = currentStreak + 1;
      return { newStreak, newMaxStreak: Math.max(newStreak, maxStreak) };
    } else if (diffDays === 2 && hoursSinceLastActivity <= 36) {
      // Grace period: if it's been less than 36 hours, still consider it consecutive
      // This helps users in edge timezone cases (e.g., activity at 11pm, next at 11am = 36h)
      const newStreak = currentStreak + 1;
      return { newStreak, newMaxStreak: Math.max(newStreak, maxStreak) };
    } else {
      // Streak broken - reset to 1
      return { newStreak: 1, newMaxStreak: maxStreak };
    }
  }

  /**
   * Check and award badges based on achievements
   * Uses transaction to ensure atomic badge award + XP reward
   */
  private async checkAndAwardBadges(
    userId: string,
    xp: number,
    level: number,
    streak: number,
    maxStreak: number,
  ): Promise<Array<{ slug: string; name: string; icon: string }>> {
    const badges = await this.prisma.badge.findMany();
    const userBadges = await this.prisma.userBadge.findMany({
      where: { userId },
      select: { badgeId: true },
    });
    const earnedBadgeIds = new Set(userBadges.map((ub) => ub.badgeId));

    const newBadges: Array<{ slug: string; name: string; icon: string }> = [];

    // Get task count for milestone badges
    const taskCount = await this.prisma.submission.count({
      where: { userId, status: "passed" },
    });

    for (const badge of badges) {
      if (earnedBadgeIds.has(badge.id)) continue;

      let earned = false;

      switch (badge.category) {
        case "milestone":
          earned = taskCount >= badge.requirement;
          break;
        case "streak":
          earned =
            streak >= badge.requirement || maxStreak >= badge.requirement;
          break;
        case "level":
          earned = level >= badge.requirement;
          break;
        case "xp":
          earned = xp >= badge.requirement;
          break;
      }

      if (earned) {
        // Use transaction to atomically create badge + award XP
        // Also use upsert to prevent duplicate badge creation from race conditions
        try {
          await this.prisma.$transaction(async (tx) => {
            // Check if badge already exists (race condition protection)
            const existingBadge = await tx.userBadge.findUnique({
              where: { userId_badgeId: { userId, badgeId: badge.id } },
            });

            if (existingBadge) {
              return; // Badge already awarded, skip
            }

            await tx.userBadge.create({
              data: { userId, badgeId: badge.id },
            });

            // Award bonus XP for badge if any
            if (badge.xpReward > 0) {
              await tx.user.update({
                where: { id: userId },
                data: { xp: { increment: badge.xpReward } },
              });
            }
          });

          newBadges.push({
            slug: badge.slug,
            name: badge.name,
            icon: badge.icon,
          });
        } catch (error: unknown) {
          // Only silently ignore unique constraint violations (race condition case)
          // P2002 = unique constraint violation in Prisma
          if (
            error instanceof Prisma.PrismaClientKnownRequestError &&
            error.code === "P2002"
          ) {
            // Race condition: badge was awarded by another concurrent request
            continue;
          }
          // Log unexpected errors but don't throw (don't block user experience)
          console.error(
            `Failed to award badge ${badge.slug} to user ${userId}:`,
            error,
          );
          continue;
        }
      }
    }

    return newBadges;
  }

  /**
   * Get user's gamification stats
   */
  async getUserStats(userId: string) {
    const user = await this.prisma.user.findUnique({
      where: { id: userId },
      select: {
        xp: true,
        level: true,
        currentStreak: true,
        maxStreak: true,
        badges: {
          include: { badge: true },
          orderBy: { earnedAt: "desc" },
        },
      },
    });

    if (!user) return null;

    const xpForCurrentLevel =
      user.level > 1 ? this.getXpForNextLevel(user.level - 1) : 0;
    const xpForNextLevel = this.getXpForNextLevel(user.level);
    const xpProgress = user.xp - xpForCurrentLevel;
    const xpNeeded = xpForNextLevel - xpForCurrentLevel;

    return {
      xp: user.xp,
      level: user.level,
      currentStreak: user.currentStreak,
      maxStreak: user.maxStreak,
      xpProgress,
      xpNeeded,
      progressPercent: Math.round((xpProgress / xpNeeded) * 100),
      badges: user.badges.map((ub) => ({
        ...ub.badge,
        earnedAt: ub.earnedAt,
      })),
    };
  }

  /**
   * Get leaderboard
   */
  async getLeaderboard(limit = 50) {
    const users = await this.prisma.user.findMany({
      orderBy: [{ xp: "desc" }, { level: "desc" }],
      take: limit,
      select: {
        id: true,
        name: true,
        avatarUrl: true,
        xp: true,
        level: true,
        currentStreak: true,
        _count: {
          select: {
            submissions: { where: { status: "passed" } },
          },
        },
      },
    });

    return users.map((user, index) => ({
      rank: index + 1,
      id: user.id,
      name: user.name,
      avatarUrl: user.avatarUrl,
      xp: user.xp,
      level: user.level,
      streak: user.currentStreak,
      tasksSolved: user._count.submissions,
    }));
  }

  /**
   * Get user's rank with caching and stampede protection
   * Uses getOrSet with distributed lock to prevent cache stampede
   */
  async getUserRank(userId: string): Promise<number> {
    const cacheKey = `rank:${userId}`;

    // Use getOrSet with lock to prevent cache stampede
    const rank = await this.cache.getOrSet<number>(
      cacheKey,
      RANK_CACHE_TTL,
      async () => {
        const user = await this.prisma.user.findUnique({
          where: { id: userId },
          select: { xp: true },
        });

        if (!user) return 0;

        const higherRanked = await this.prisma.user.count({
          where: { xp: { gt: user.xp } },
        });

        return higherRanked + 1;
      },
    );

    return rank ?? 0;
  }

  /**
   * Invalidate rank cache for a user
   * Called after XP changes to ensure fresh rank on next request
   */
  async invalidateRankCache(userId: string): Promise<void> {
    await this.cache.delete(`rank:${userId}`);
  }
}
