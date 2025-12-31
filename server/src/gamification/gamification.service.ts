import { Injectable } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';

// XP rewards by difficulty
const XP_REWARDS = {
  easy: 10,
  medium: 25,
  hard: 50,
  expert: 100,
};

// Level thresholds (XP required for each level)
const LEVEL_THRESHOLDS = [
  0,      // Level 1: 0 XP
  100,    // Level 2: 100 XP
  250,    // Level 3: 250 XP
  500,    // Level 4: 500 XP
  1000,   // Level 5: 1000 XP
  1750,   // Level 6
  2750,   // Level 7
  4000,   // Level 8
  5500,   // Level 9
  7500,   // Level 10
  10000,  // Level 11
  13000,  // Level 12
  16500,  // Level 13
  20500,  // Level 14
  25000,  // Level 15
  30000,  // Level 16
  36000,  // Level 17
  43000,  // Level 18
  51000,  // Level 19
  60000,  // Level 20
  // Beyond level 20: +10000 per level
];

@Injectable()
export class GamificationService {
  constructor(private prisma: PrismaService) {}

  /**
   * Calculate level from XP
   */
  calculateLevel(xp: number): number {
    for (let i = LEVEL_THRESHOLDS.length - 1; i >= 0; i--) {
      if (xp >= LEVEL_THRESHOLDS[i]) {
        return i + 1;
      }
    }
    // Beyond max threshold
    const beyondMax = xp - LEVEL_THRESHOLDS[LEVEL_THRESHOLDS.length - 1];
    return LEVEL_THRESHOLDS.length + Math.floor(beyondMax / 10000);
  }

  /**
   * Get XP required for next level
   */
  getXpForNextLevel(level: number): number {
    if (level < LEVEL_THRESHOLDS.length) {
      return LEVEL_THRESHOLDS[level];
    }
    // Beyond defined levels
    return LEVEL_THRESHOLDS[LEVEL_THRESHOLDS.length - 1] + (level - LEVEL_THRESHOLDS.length + 1) * 10000;
  }

  /**
   * Award XP for completing a task
   */
  async awardTaskXp(userId: string, difficulty: string): Promise<{
    xpEarned: number;
    totalXp: number;
    level: number;
    leveledUp: boolean;
    newBadges: Array<{ slug: string; name: string; icon: string }>;
  }> {
    const xpEarned = XP_REWARDS[difficulty] || XP_REWARDS.easy;

    const user = await this.prisma.user.findUnique({
      where: { id: userId },
      select: { xp: true, level: true, currentStreak: true, maxStreak: true, lastActivityAt: true },
    });

    const oldLevel = user?.level || 1;
    const oldXp = user?.xp || 0;
    const newXp = oldXp + xpEarned;
    const newLevel = this.calculateLevel(newXp);

    // Update streak
    const { newStreak, newMaxStreak } = this.calculateStreak(
      user?.currentStreak || 0,
      user?.maxStreak || 0,
      user?.lastActivityAt,
    );

    // Update user
    await this.prisma.user.update({
      where: { id: userId },
      data: {
        xp: newXp,
        level: newLevel,
        currentStreak: newStreak,
        maxStreak: newMaxStreak,
        lastActivityAt: new Date(),
      },
    });

    // Check for new badges
    const newBadges = await this.checkAndAwardBadges(userId, newXp, newLevel, newStreak, newMaxStreak);

    return {
      xpEarned,
      totalXp: newXp,
      level: newLevel,
      leveledUp: newLevel > oldLevel,
      newBadges,
    };
  }

  /**
   * Calculate streak based on last activity
   */
  private calculateStreak(
    currentStreak: number,
    maxStreak: number,
    lastActivityAt: Date | null,
  ): { newStreak: number; newMaxStreak: number } {
    const now = new Date();
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());

    if (!lastActivityAt) {
      return { newStreak: 1, newMaxStreak: Math.max(1, maxStreak) };
    }

    const lastActivity = new Date(lastActivityAt);
    const lastActivityDay = new Date(lastActivity.getFullYear(), lastActivity.getMonth(), lastActivity.getDate());

    const diffDays = Math.floor((today.getTime() - lastActivityDay.getTime()) / (1000 * 60 * 60 * 24));

    if (diffDays === 0) {
      // Same day - no streak change
      return { newStreak: currentStreak, newMaxStreak: maxStreak };
    } else if (diffDays === 1) {
      // Consecutive day - increment streak
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
    const earnedBadgeIds = new Set(userBadges.map(ub => ub.badgeId));

    const newBadges: Array<{ slug: string; name: string; icon: string }> = [];

    // Get task count for milestone badges
    const taskCount = await this.prisma.submission.count({
      where: { userId, status: 'passed' },
    });

    for (const badge of badges) {
      if (earnedBadgeIds.has(badge.id)) continue;

      let earned = false;

      switch (badge.category) {
        case 'milestone':
          earned = taskCount >= badge.requirement;
          break;
        case 'streak':
          earned = streak >= badge.requirement || maxStreak >= badge.requirement;
          break;
        case 'level':
          earned = level >= badge.requirement;
          break;
        case 'xp':
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

          newBadges.push({ slug: badge.slug, name: badge.name, icon: badge.icon });
        } catch (error) {
          // If transaction fails (e.g., duplicate key), just skip this badge
          // This handles race conditions gracefully
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
          orderBy: { earnedAt: 'desc' },
        },
      },
    });

    if (!user) return null;

    const xpForCurrentLevel = user.level > 1 ? this.getXpForNextLevel(user.level - 1) : 0;
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
      badges: user.badges.map(ub => ({
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
      orderBy: [{ xp: 'desc' }, { level: 'desc' }],
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
            submissions: { where: { status: 'passed' } },
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
   * Get user's rank
   */
  async getUserRank(userId: string): Promise<number> {
    const user = await this.prisma.user.findUnique({
      where: { id: userId },
      select: { xp: true },
    });

    if (!user) return 0;

    const higherRanked = await this.prisma.user.count({
      where: { xp: { gt: user.xp } },
    });

    return higherRanked + 1;
  }
}
