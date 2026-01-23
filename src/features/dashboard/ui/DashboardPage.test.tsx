import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import React from 'react';
import { MemoryRouter } from 'react-router-dom';
import DashboardPage from './DashboardPage';
import { AuthContext } from '@/components/Layout';

// Mock dependencies
vi.mock('@/features/tasks/api/taskService', () => ({
  taskService: {
    getRecentSubmissions: vi.fn(),
  },
}));

vi.mock('../api/dashboardService', () => ({
  dashboardService: {
    getStats: vi.fn(),
    getWeeklyActivity: vi.fn(),
  },
}));

vi.mock('@/contexts/LanguageContext', () => ({
  useUITranslation: () => ({
    tUI: (key: string) => {
      const translations: Record<string, string> = {
        'dashboard.pageTitle': 'Dashboard',
        'dashboard.pageSubtitle': 'Track your progress',
        'dashboard.currentStreak': 'Current Streak',
        'dashboard.totalSolved': 'Total Solved',
        'dashboard.thisWeek': 'this week',
        'dashboard.hoursSpent': 'Hours Spent',
        'dashboard.submissions': 'submissions',
        'dashboard.globalRank': 'Global Rank',
        'dashboard.topPercent': 'Top',
        'dashboard.skillPoints': 'Skill Points',
        'dashboard.maxStreak': 'Max Streak',
        'dashboard.days': 'days',
        'dashboard.hours': 'h',
        'dashboard.minutes': 'm',
        'dashboard.loginRequired': 'Sign in to view your dashboard',
        'dashboard.loginRequiredDesc': 'Track your progress',
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

import { taskService } from '@/features/tasks/api/taskService';
import { dashboardService } from '../api/dashboardService';

describe('DashboardPage', () => {
  const mockUser = { id: 'user-1', email: 'test@example.com', name: 'Test User' };

  const mockStats = {
    totalSolved: 42,
    totalSubmissions: 150,
    totalMinutes: 1200,
    globalRank: 100,
    topPercent: 15,
    skillPoints: 2500,
    currentStreak: 7,
    maxStreak: 14,
    weekThisWeek: 5,
  };

  const mockActivity = [
    { date: '2025-01-10', name: 'Fri', solved: 3, submissions: 5 },
    { date: '2025-01-11', name: 'Sat', solved: 2, submissions: 4 },
  ];

  const mockSubmissions = [
    { id: 'sub-1', status: 'passed', taskId: 'task-1', createdAt: '2025-01-16T10:00:00Z' },
    { id: 'sub-2', status: 'failed', taskId: 'task-2', createdAt: '2025-01-16T09:00:00Z' },
  ];

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(taskService.getRecentSubmissions).mockResolvedValue(mockSubmissions);
    vi.mocked(dashboardService.getStats).mockResolvedValue(mockStats);
    vi.mocked(dashboardService.getWeeklyActivity).mockResolvedValue(mockActivity);
  });

  const renderWithAuth = (user: typeof mockUser | null) => {
    return render(
      <AuthContext.Provider value={{ user, setUser: vi.fn() }}>
        <MemoryRouter>
          <DashboardPage />
        </MemoryRouter>
      </AuthContext.Provider>
    );
  };

  describe('authenticated user', () => {
    it('should show loading state initially', () => {
      renderWithAuth(mockUser);

      expect(screen.getByText('Loading...')).toBeInTheDocument();
    });

    it('should load and display dashboard data', async () => {
      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText('Dashboard')).toBeInTheDocument();
      });

      expect(taskService.getRecentSubmissions).toHaveBeenCalledWith(15);
      expect(dashboardService.getStats).toHaveBeenCalled();
      expect(dashboardService.getWeeklyActivity).toHaveBeenCalledWith(7);
    });

    it('should display user stats', async () => {
      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText('42')).toBeInTheDocument(); // totalSolved
      });

      expect(screen.getByText('#100')).toBeInTheDocument(); // globalRank
      expect(screen.getByText('2.5k')).toBeInTheDocument(); // skillPoints formatted
    });

    it('should display current streak', async () => {
      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText(/🔥 7 days/)).toBeInTheDocument();
      });
    });

    it('should format hours correctly', async () => {
      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.getByText('20h')).toBeInTheDocument(); // 1200 minutes = 20 hours
      });
    });

    it('should handle API error gracefully', async () => {
      vi.mocked(dashboardService.getStats).mockRejectedValue(new Error('API Error'));

      renderWithAuth(mockUser);

      // Should still render without crashing
      await waitFor(() => {
        expect(screen.queryByText('Loading...')).not.toBeInTheDocument();
      });
    });
  });

  describe('unauthenticated user', () => {
    it('should show auth required overlay', () => {
      renderWithAuth(null);

      expect(screen.getByText('Sign in to view your dashboard')).toBeInTheDocument();
    });

    it('should not fetch data when not authenticated', () => {
      renderWithAuth(null);

      expect(taskService.getRecentSubmissions).not.toHaveBeenCalled();
      expect(dashboardService.getStats).not.toHaveBeenCalled();
    });
  });
});
