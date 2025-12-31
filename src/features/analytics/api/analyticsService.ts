
import { api } from '@/lib/api';

export interface WeeklyStats {
  name: string;
  date: string;
  tasks: number;
  submissions: number;
}

export interface YearlyContribution {
  date: string;
  count: number;
  intensity: number;
}

export interface AnalyticsSummary {
  totalSubmissions: number;
  totalSolved: number;
  currentStreak: number;
  maxStreak: number;
  completionRate: number;
  avgRuntime: string;
  totalXP: number;
  weekSolvedCount: number;
}

/**
 * Analytics Service - Connected to Real Backend API
 *
 * Endpoints:
 * - GET /users/me/activity?days=7&offset=X - Weekly activity
 * - GET /users/me/activity/yearly - Yearly heatmap data
 * - GET /users/me/stats - User stats summary
 */
export const analyticsService = {
  /**
   * Get weekly stats for bar chart
   * @param weekOffset 0 = current week, 1 = last week, etc.
   */
  getWeeklyStats: async (weekOffset: number): Promise<WeeklyStats[]> => {
    // Calculate date range based on offset
    const endDate = new Date();
    endDate.setDate(endDate.getDate() - (weekOffset * 7));

    // Get 7 days of data
    const data = await api.get<{ name: string; date: string; solved: number; submissions: number }[]>(
      `/users/me/activity?days=7&offset=${weekOffset * 7}`
    );

    // Transform to match expected format
    return data.map(d => ({
      name: d.name,
      date: d.date,
      tasks: d.solved,
      submissions: d.submissions,
    }));
  },

  /**
   * Get yearly contributions for heatmap
   */
  getYearlyContributions: async (): Promise<YearlyContribution[]> => {
    const data = await api.get<{ date: string; count: number }[]>('/users/me/activity/yearly');

    // Fill in missing dates with zeros and calculate intensity
    const result: YearlyContribution[] = [];
    const dateMap = new Map(data.map(d => [d.date, d.count]));

    // Generate last 365 days
    for (let i = 364; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      const dateStr = date.toISOString().split('T')[0];
      const count = dateMap.get(dateStr) || 0;

      // Calculate intensity (0-4 scale)
      let intensity = 0;
      if (count > 0) intensity = 1;
      if (count >= 3) intensity = 2;
      if (count >= 5) intensity = 3;
      if (count >= 8) intensity = 4;

      result.push({ date: dateStr, count, intensity });
    }

    return result;
  },

  /**
   * Get summary statistics
   */
  getSummary: async (): Promise<AnalyticsSummary> => {
    const stats = await api.get<{
      totalSolved: number;
      totalSubmissions: number;
      hoursSpent: string;
      globalRank: number;
      skillPoints: number;
      currentStreak: number;
      maxStreak: number;
      weekThisWeek: number;
    }>('/users/me/stats');

    // Calculate completion rate (solved / total unique tasks attempted)
    const completionRate = stats.totalSubmissions > 0
      ? Math.round((stats.totalSolved / stats.totalSubmissions) * 100)
      : 0;

    return {
      totalSubmissions: stats.totalSubmissions,
      totalSolved: stats.totalSolved,
      currentStreak: stats.currentStreak,
      maxStreak: stats.maxStreak,
      completionRate,
      avgRuntime: '~15ms', // TODO: Calculate from submissions
      totalXP: stats.skillPoints,
      weekSolvedCount: stats.weekThisWeek,
    };
  },
};
