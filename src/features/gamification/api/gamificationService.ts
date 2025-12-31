import { api } from '@/lib/api';

export interface Badge {
  id: string;
  slug: string;
  name: string;
  description: string;
  icon: string;
  category: string;
  requirement: number;
  xpReward: number;
  earnedAt?: string;
  translations?: {
    ru?: { name: string; description: string };
    uz?: { name: string; description: string };
  };
}

export interface UserGamificationStats {
  xp: number;
  level: number;
  currentStreak: number;
  maxStreak: number;
  xpProgress: number;
  xpNeeded: number;
  progressPercent: number;
  badges: Badge[];
}

export interface LeaderboardEntry {
  rank: number;
  id: string;
  name: string;
  avatarUrl: string | null;
  xp: number;
  level: number;
  streak: number;
  tasksSolved: number;
}

export interface GamificationReward {
  xpEarned?: number;
  totalXp?: number;
  level?: number;
  leveledUp?: boolean;
  newBadges?: Array<{ slug: string; name: string; icon: string }>;
}

export const gamificationService = {
  /**
   * Get current user's gamification stats
   */
  getMyStats: async (): Promise<UserGamificationStats> => {
    return api.get<UserGamificationStats>('/gamification/me');
  },

  /**
   * Get current user's rank
   */
  getMyRank: async (): Promise<{ rank: number }> => {
    return api.get<{ rank: number }>('/gamification/me/rank');
  },

  /**
   * Get leaderboard
   */
  getLeaderboard: async (limit = 50): Promise<LeaderboardEntry[]> => {
    return api.get<LeaderboardEntry[]>(`/gamification/leaderboard?limit=${limit}`);
  },
};
