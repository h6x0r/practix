
import { api } from '@/lib/api';

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

/**
 * Dashboard Service - Connected to Real Backend API
 *
 * Endpoints:
 * - GET /users/me/stats - User statistics
 * - GET /users/me/activity?days=7 - Weekly activity chart data
 * - GET /users/me/activity/yearly - Yearly activity for heatmap
 */
export const dashboardService = {
  /**
   * Get user statistics for dashboard cards
   */
  getStats: async (): Promise<UserStats> => {
    return api.get<UserStats>('/users/me/stats');
  },

  /**
   * Get weekly activity for charts
   * @param days Number of days (default 7)
   */
  getWeeklyActivity: async (days = 7): Promise<DayActivity[]> => {
    return api.get<DayActivity[]>(`/users/me/activity?days=${days}`);
  },

  /**
   * Get yearly activity for heatmap
   */
  getYearlyActivity: async (): Promise<{ date: string; count: number }[]> => {
    return api.get<{ date: string; count: number }[]>('/users/me/activity/yearly');
  },
};
