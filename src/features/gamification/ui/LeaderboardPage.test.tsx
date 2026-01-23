import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import React from 'react';
import { MemoryRouter } from 'react-router-dom';
import LeaderboardPage from './LeaderboardPage';
import { AuthContext } from '@/components/Layout';

// Mock dependencies
vi.mock('../api/gamificationService', () => ({
  gamificationService: {
    getLeaderboard: vi.fn(),
    getMyStats: vi.fn(),
    getMyRank: vi.fn(),
  },
}));

vi.mock('@/contexts/LanguageContext', () => ({
  useUITranslation: () => ({
    tUI: (key: string) => {
      const translations: Record<string, string> = {
        'leaderboard.title': 'Leaderboard',
        'leaderboard.subtitle': 'Top coders ranked by XP',
        'leaderboard.loginRequired': 'Sign in to see leaderboard',
        'leaderboard.loginRequiredDesc': 'Join the competition',
        'leaderboard.level': 'Level',
        'leaderboard.xp': 'XP',
        'leaderboard.streak': 'Streak',
        'leaderboard.tasksSolved': 'Tasks',
        'common.loading': 'Loading...',
      };
      return translations[key] || key;
    },
  }),
}));

vi.mock('@/lib/logger', () => ({
  createLogger: () => ({
    info: vi.fn(),
    error: vi.fn(),
    warn: vi.fn(),
    debug: vi.fn(),
  }),
}));

vi.mock('@/components/Toast', () => ({
  useToast: () => ({
    showToast: vi.fn(),
  }),
}));

import { gamificationService } from '../api/gamificationService';

describe('LeaderboardPage', () => {
  const mockUser = { id: 'user-1', email: 'test@example.com', name: 'Test User' };

  const mockLeaderboard = [
    { id: '1', name: 'Alice', level: 10, xp: 8500, streak: 15, tasksSolved: 120, rank: 1 },
    { id: '2', name: 'Bob', level: 9, xp: 7200, streak: 8, tasksSolved: 95, rank: 2 },
    { id: '3', name: 'Charlie', level: 8, xp: 5800, streak: 5, tasksSolved: 78, rank: 3 },
  ];

  const mockMyStats = {
    level: 6,
    xp: 2500,
    streak: 3,
    maxStreak: 10,
    tasksSolved: 45,
    badges: [],
  };

  const mockMyRank = { rank: 42 };

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(gamificationService.getLeaderboard).mockResolvedValue(mockLeaderboard);
    vi.mocked(gamificationService.getMyStats).mockResolvedValue(mockMyStats);
    vi.mocked(gamificationService.getMyRank).mockResolvedValue(mockMyRank);
  });

  const renderWithAuth = (user: typeof mockUser | null) => {
    return render(
      <AuthContext.Provider value={{ user, setUser: vi.fn() }}>
        <MemoryRouter>
          <LeaderboardPage />
        </MemoryRouter>
      </AuthContext.Provider>
    );
  };

  describe('loading state', () => {
    it('should show loading state initially', () => {
      renderWithAuth(mockUser);

      expect(screen.getByText('Loading...')).toBeInTheDocument();
    });
  });

  describe('authenticated user', () => {
    it('should load and display leaderboard', async () => {
      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText('Leaderboard')).toBeInTheDocument();
      });

      expect(gamificationService.getLeaderboard).toHaveBeenCalledWith(50);
    });

    it('should display top players', async () => {
      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText('Alice')).toBeInTheDocument();
      });

      expect(screen.getByText('Bob')).toBeInTheDocument();
      expect(screen.getByText('Charlie')).toBeInTheDocument();
    });

    it('should load user stats', async () => {
      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(gamificationService.getMyStats).toHaveBeenCalled();
      });

      expect(gamificationService.getMyRank).toHaveBeenCalled();
    });

    it('should display user rank', async () => {
      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText('#42')).toBeInTheDocument();
      });
    });

    it('should display user name', async () => {
      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText('Test User')).toBeInTheDocument();
      });
    });

    it('should handle API error gracefully', async () => {
      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
      vi.mocked(gamificationService.getLeaderboard).mockRejectedValue(new Error('API Error'));

      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.queryByText('Loading...')).not.toBeInTheDocument();
      });

      consoleSpy.mockRestore();
    });
  });

  describe('unauthenticated user', () => {
    it('should show auth required overlay', async () => {
      vi.mocked(gamificationService.getMyStats).mockResolvedValue(null as any);
      vi.mocked(gamificationService.getMyRank).mockResolvedValue(null as any);

      renderWithAuth(null);

      await waitFor(() => {
        expect(screen.getByText('Sign in to see leaderboard')).toBeInTheDocument();
      });
    });

    it('should still load public leaderboard', async () => {
      vi.mocked(gamificationService.getMyStats).mockResolvedValue(null as any);
      vi.mocked(gamificationService.getMyRank).mockResolvedValue(null as any);

      renderWithAuth(null);

      await waitFor(() => {
        expect(gamificationService.getLeaderboard).toHaveBeenCalled();
      });
    });

    it('should not load personal stats', async () => {
      vi.mocked(gamificationService.getMyStats).mockResolvedValue(null as any);
      vi.mocked(gamificationService.getMyRank).mockResolvedValue(null as any);

      renderWithAuth(null);

      await waitFor(() => {
        expect(screen.queryByText('Loading...')).not.toBeInTheDocument();
      });

      // getMyStats should be called with null result
      expect(gamificationService.getMyStats).not.toHaveBeenCalled();
    });
  });
});
