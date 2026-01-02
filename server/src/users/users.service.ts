
import { Injectable } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { CreateUserData, UserPreferences } from '../common/types';

export interface UserStats {
  totalSolved: number;
  totalSubmissions: number;
  hoursSpent: string;
  globalRank: number;
  skillPoints: number;
  currentStreak: number;
  maxStreak: number;
  weekThisWeek: number;
}

export interface DayActivity {
  name: string;
  date: string;
  solved: number;
  submissions: number;
}

@Injectable()
export class UsersService {
  constructor(private prisma: PrismaService) {}

  /**
   * Find user by email for authentication (includes password hash)
   * ONLY use this for login verification!
   */
  async findOneForAuth(email: string) {
    return this.prisma.user.findUnique({
      where: { email },
    });
  }

  /**
   * Find user by email (excludes password)
   * Safe for general use
   */
  async findOne(email: string) {
    return this.prisma.user.findUnique({
      where: { email },
      select: {
        id: true,
        email: true,
        name: true,
        avatarUrl: true,
        isPremium: true,
        plan: true,
        preferences: true,
        role: true,
        xp: true,
        level: true,
        currentStreak: true,
        maxStreak: true,
        lastActivityAt: true,
        createdAt: true,
        updatedAt: true,
        // password explicitly NOT selected
      },
    });
  }

  /**
   * Find user by ID (excludes password)
   * Safe for general use
   */
  async findById(id: string) {
    return this.prisma.user.findUnique({
      where: { id },
      select: {
        id: true,
        email: true,
        name: true,
        avatarUrl: true,
        isPremium: true,
        plan: true,
        preferences: true,
        role: true,
        xp: true,
        level: true,
        currentStreak: true,
        maxStreak: true,
        lastActivityAt: true,
        createdAt: true,
        updatedAt: true,
        // password explicitly NOT selected
      },
    });
  }

  async create(data: CreateUserData) {
    return this.prisma.user.create({
      data,
    });
  }

  async updatePreferences(userId: string, preferences: UserPreferences) {
    return this.prisma.user.update({
      where: { id: userId },
      data: { preferences },
    });
  }

  // NOTE: updatePlan method removed - isPremium should always be computed
  // from active subscriptions, not stored as a cached field.
  // Use SubscriptionsService to manage subscriptions.

  /**
   * Update user avatar URL
   * @param userId User ID
   * @param avatarUrl Either a URL or base64-encoded image data
   */
  async updateAvatar(userId: string, avatarUrl: string) {
    return this.prisma.user.update({
      where: { id: userId },
      data: { avatarUrl },
    });
  }

  /**
   * Get user statistics for Dashboard
   */
  async getUserStats(userId: string): Promise<UserStats> {
    // Get all user submissions
    const submissions = await this.prisma.submission.findMany({
      where: { userId },
      orderBy: { createdAt: 'desc' },
    });

    // Count unique passed tasks
    const passedTaskIds = new Set<string>();
    submissions.forEach(s => {
      if (s.status === 'passed') {
        passedTaskIds.add(s.taskId);
      }
    });
    const totalSolved = passedTaskIds.size;

    // Calculate hours spent (estimate: ~10 min per submission)
    const totalMinutes = submissions.length * 10;
    const hours = Math.floor(totalMinutes / 60);
    const hoursSpent = hours > 0 ? `${hours}h` : `${totalMinutes}m`;

    // Calculate skill points (100 per solved task)
    const skillPoints = totalSolved * 100;

    // Calculate streak
    const { currentStreak, maxStreak } = this.calculateStreak(submissions);

    // Calculate this week's solved count
    const weekStart = new Date();
    weekStart.setDate(weekStart.getDate() - weekStart.getDay());
    weekStart.setHours(0, 0, 0, 0);

    const thisWeekPassed = new Set<string>();
    submissions.forEach(s => {
      if (s.status === 'passed' && new Date(s.createdAt) >= weekStart) {
        thisWeekPassed.add(s.taskId);
      }
    });

    // Calculate global rank (simple: based on total solved tasks)
    const allUsersStats = await this.prisma.submission.groupBy({
      by: ['userId'],
      where: { status: 'passed' },
      _count: { taskId: true },
    });

    const sortedUsers = allUsersStats
      .map(u => ({ userId: u.userId, count: u._count.taskId }))
      .sort((a, b) => b.count - a.count);

    const userRankIndex = sortedUsers.findIndex(u => u.userId === userId);
    const globalRank = userRankIndex >= 0 ? userRankIndex + 1 : sortedUsers.length + 1;

    return {
      totalSolved,
      totalSubmissions: submissions.length,
      hoursSpent,
      globalRank,
      skillPoints,
      currentStreak,
      maxStreak,
      weekThisWeek: thisWeekPassed.size,
    };
  }

  /**
   * Get weekly activity for charts
   * @param days Number of days to fetch
   * @param offset Days offset from today (0 = today, 7 = one week ago)
   */
  async getWeeklyActivity(userId: string, days = 7, offset = 0): Promise<DayActivity[]> {
    const result: DayActivity[] = [];
    const dayNames = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

    // Calculate date range with offset
    const endDate = new Date();
    endDate.setDate(endDate.getDate() - offset);
    endDate.setHours(23, 59, 59, 999);

    const startDate = new Date(endDate);
    startDate.setDate(startDate.getDate() - days + 1);
    startDate.setHours(0, 0, 0, 0);

    const submissions = await this.prisma.submission.findMany({
      where: {
        userId,
        createdAt: {
          gte: startDate,
          lte: endDate,
        },
      },
    });

    // Group by date
    for (let i = 0; i < days; i++) {
      const date = new Date(startDate);
      date.setDate(startDate.getDate() + i);
      date.setHours(0, 0, 0, 0);

      const dateStr = date.toISOString().split('T')[0];
      const daySubmissions = submissions.filter(s =>
        s.createdAt.toISOString().split('T')[0] === dateStr
      );

      const solvedTaskIds = new Set<string>();
      daySubmissions.forEach(s => {
        if (s.status === 'passed') solvedTaskIds.add(s.taskId);
      });

      result.push({
        name: dayNames[date.getDay()],
        date: dateStr,
        solved: solvedTaskIds.size,
        submissions: daySubmissions.length,
      });
    }

    return result;
  }

  /**
   * Get yearly activity (for heatmap)
   */
  async getYearlyActivity(userId: string): Promise<{ date: string; count: number }[]> {
    const startDate = new Date();
    startDate.setFullYear(startDate.getFullYear() - 1);
    startDate.setHours(0, 0, 0, 0);

    const submissions = await this.prisma.submission.findMany({
      where: {
        userId,
        status: 'passed',
        createdAt: { gte: startDate },
      },
      select: { createdAt: true, taskId: true },
    });

    // Group by date
    const dateMap = new Map<string, Set<string>>();
    submissions.forEach(s => {
      const dateStr = s.createdAt.toISOString().split('T')[0];
      if (!dateMap.has(dateStr)) {
        dateMap.set(dateStr, new Set());
      }
      dateMap.get(dateStr)!.add(s.taskId);
    });

    return Array.from(dateMap.entries()).map(([date, taskIds]) => ({
      date,
      count: taskIds.size,
    }));
  }

  /**
   * Check if user has an active subscription (computed isPremium)
   * @param userId User ID
   * @returns true if user has any active subscription
   */
  async isPremiumUser(userId: string): Promise<boolean> {
    const activeSubscription = await this.prisma.subscription.findFirst({
      where: {
        userId,
        status: 'active',
        endDate: { gt: new Date() },
      },
    });
    return !!activeSubscription;
  }

  /**
   * Get user's active plan (if any)
   * @param userId User ID
   * @returns Plan details or null if no active subscription
   */
  async getActivePlan(userId: string): Promise<{ name: string; expiresAt: string } | null> {
    const subscription = await this.prisma.subscription.findFirst({
      where: {
        userId,
        status: 'active',
        endDate: { gt: new Date() },
      },
      include: { plan: true },
      orderBy: { endDate: 'desc' },
    });

    if (!subscription) return null;

    return {
      name: subscription.plan.name,
      expiresAt: subscription.endDate.toISOString(),
    };
  }

  /**
   * Calculate current and max streak from submissions
   */
  private calculateStreak(submissions: { status: string; createdAt: Date }[]): {
    currentStreak: number;
    maxStreak: number;
  } {
    if (submissions.length === 0) {
      return { currentStreak: 0, maxStreak: 0 };
    }

    // Get unique dates with passed submissions
    const passedDates = new Set<string>();
    submissions.forEach(s => {
      if (s.status === 'passed') {
        passedDates.add(s.createdAt.toISOString().split('T')[0]);
      }
    });

    if (passedDates.size === 0) {
      return { currentStreak: 0, maxStreak: 0 };
    }

    // Sort dates
    const sortedDates = Array.from(passedDates).sort();

    // Calculate max streak
    let maxStreak = 1;
    let currentRun = 1;

    for (let i = 1; i < sortedDates.length; i++) {
      const prevDate = new Date(sortedDates[i - 1]);
      const currDate = new Date(sortedDates[i]);
      const diffDays = Math.floor((currDate.getTime() - prevDate.getTime()) / (1000 * 60 * 60 * 24));

      if (diffDays === 1) {
        currentRun++;
        maxStreak = Math.max(maxStreak, currentRun);
      } else {
        currentRun = 1;
      }
    }

    // Calculate current streak (from today backwards)
    const today = new Date().toISOString().split('T')[0];
    const yesterday = new Date(Date.now() - 86400000).toISOString().split('T')[0];

    let currentStreak = 0;

    // Check if user has activity today or yesterday
    if (passedDates.has(today) || passedDates.has(yesterday)) {
      const startDate = passedDates.has(today) ? today : yesterday;
      let checkDate = new Date(startDate);

      while (passedDates.has(checkDate.toISOString().split('T')[0])) {
        currentStreak++;
        checkDate.setDate(checkDate.getDate() - 1);
      }
    }

    return { currentStreak, maxStreak };
  }
}
