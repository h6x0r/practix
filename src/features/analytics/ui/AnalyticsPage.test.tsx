import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import React from 'react';
import { MemoryRouter } from 'react-router-dom';
import AnalyticsPage from './AnalyticsPage';
import { AuthContext } from '@/components/Layout';

// Mock dependencies
vi.mock('../api/analyticsService', () => ({
  analyticsService: {
    getWeeklyStats: vi.fn(),
    getYearlyContributions: vi.fn(),
    getSummary: vi.fn(),
  },
}));

vi.mock('@/contexts/LanguageContext', () => ({
  useUITranslation: () => ({
    tUI: (key: string) => {
      const translations: Record<string, string> = {
        'analytics.title': 'Analytics',
        'analytics.subtitle': 'Your coding journey',
        'analytics.loginRequired': 'Sign in to view analytics',
        'analytics.loginRequiredDesc': 'Track your progress over time',
        'analytics.weeklyActivity': 'Weekly Activity',
        'analytics.yearlyContributions': 'Yearly Contributions',
        'analytics.totalSolved': 'Total Solved',
        'analytics.totalSubmissions': 'Total Submissions',
        'analytics.currentStreak': 'Current Streak',
        'analytics.maxStreak': 'Max Streak',
        'analytics.loading': 'Loading...',
        'analytics.day.mon': 'Mon',
        'analytics.day.tue': 'Tue',
        'analytics.day.wed': 'Wed',
        'analytics.day.thu': 'Thu',
        'analytics.day.fri': 'Fri',
        'analytics.day.sat': 'Sat',
        'analytics.day.sun': 'Sun',
        'analytics.completionRate': 'Completion Rate',
        'analytics.avgRuntime': 'Avg Runtime',
        'analytics.totalXP': 'Total XP',
        'analytics.less': 'Less',
        'analytics.more': 'More',
        'analytics.youAreOnFire': 'You are on fire!',
        'analytics.tasksSolvedThisWeek': '{count} tasks solved this week',
        'analytics.historicalData': 'Historical data for {dateRange}',
        'analytics.noActivityThisWeek': 'No activity this week',
        'analytics.noActivity': 'No activity',
        'analytics.tasksCompleted': '{count} tasks completed',
      };
      return translations[key] || key;
    },
  }),
  useLanguage: () => ({
    language: 'en',
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

import { analyticsService } from '../api/analyticsService';

describe('AnalyticsPage', () => {
  const mockUser = { id: 'user-1', email: 'test@example.com', name: 'Test User' };

  const mockWeeklyStats = [
    { name: 'Mon', date: '2025-01-13', tasks: 3, submissions: 5 },
    { name: 'Tue', date: '2025-01-14', tasks: 2, submissions: 4 },
    { name: 'Wed', date: '2025-01-15', tasks: 4, submissions: 8 },
  ];

  const mockYearlyContributions = Array.from({ length: 365 }, (_, i) => ({
    date: new Date(Date.now() - i * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
    count: Math.floor(Math.random() * 5),
    intensity: Math.floor(Math.random() * 5) as 0 | 1 | 2 | 3 | 4,
  }));

  const mockSummary = {
    totalSolved: 150,
    totalSubmissions: 450,
    currentStreak: 7,
    maxStreak: 21,
    totalXP: 5000,
    weekSolvedCount: 12,
    completionRate: 75,
  };

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(analyticsService.getWeeklyStats).mockResolvedValue(mockWeeklyStats);
    vi.mocked(analyticsService.getYearlyContributions).mockResolvedValue(mockYearlyContributions);
    vi.mocked(analyticsService.getSummary).mockResolvedValue(mockSummary);
  });

  const renderWithAuth = (user: typeof mockUser | null) => {
    return render(
      <AuthContext.Provider value={{ user, setUser: vi.fn() }}>
        <MemoryRouter>
          <AnalyticsPage />
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
    it('should load and display analytics data', async () => {
      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.queryByText('Loading...')).not.toBeInTheDocument();
      });

      expect(analyticsService.getWeeklyStats).toHaveBeenCalledWith(0);
      expect(analyticsService.getYearlyContributions).toHaveBeenCalled();
      expect(analyticsService.getSummary).toHaveBeenCalled();
    });

    it('should display summary stats', async () => {
      renderWithAuth(mockUser);

      await waitFor(() => {
        // totalSubmissions is shown as 450
        expect(screen.getByText('450')).toBeInTheDocument();
      });

      // Also check currentStreak and maxStreak
      expect(screen.getByText('7')).toBeInTheDocument(); // currentStreak
      expect(screen.getByText('21')).toBeInTheDocument(); // maxStreak
    });

    it('should handle API error gracefully', async () => {
      vi.mocked(analyticsService.getWeeklyStats).mockRejectedValue(new Error('API Error'));

      renderWithAuth(mockUser);

      await waitFor(() => {
        expect(screen.queryByText('Loading...')).not.toBeInTheDocument();
      });
    });
  });

  describe('unauthenticated user', () => {
    it('should show auth required overlay', () => {
      renderWithAuth(null);

      expect(screen.getByText('Sign in to view analytics')).toBeInTheDocument();
    });

    it('should not fetch data when not authenticated', () => {
      renderWithAuth(null);

      expect(analyticsService.getWeeklyStats).not.toHaveBeenCalled();
      expect(analyticsService.getYearlyContributions).not.toHaveBeenCalled();
      expect(analyticsService.getSummary).not.toHaveBeenCalled();
    });
  });
});
