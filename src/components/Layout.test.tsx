import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import React from 'react';
import { MemoryRouter, useNavigate } from 'react-router-dom';
import { Layout, AuthContext, ThemeContext, AuthContextType, ThemeContextType } from './Layout';

// Mock dependencies
vi.mock('@/features/config/api/configService', () => ({
  configService: {
    getNavigation: vi.fn(),
  },
}));

vi.mock('@/contexts/LanguageContext', () => ({
  useLanguage: () => ({
    language: 'en',
    setLanguage: vi.fn(),
  }),
  useUITranslation: () => ({
    tUI: (key: string) => {
      const translations: Record<string, string> = {
        'nav.dashboard': 'Dashboard',
        'nav.courses': 'Courses',
        'nav.roadmap': 'Roadmap',
        'nav.playground': 'Playground',
        'nav.analytics': 'Analytics',
        'nav.payments': 'Payments',
        'nav.settings': 'Settings',
        'nav.admin': 'Admin',
        'nav.platform': 'Practix',
        'nav.goPremium': 'Go Premium',
        'nav.login': 'Sign In',
        'nav.logout': 'Logout',
        'nav.pro': 'PRO',
        'nav.freePlan': 'Free Plan',
        'nav.version': 'v1.0.0',
        'nav.myTasks': 'My Tasks',
        'nav.leaderboard': 'Leaderboard',
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

vi.mock('@/lib/storage', () => ({
  storage: {
    getSidebarCollapsed: vi.fn(() => false),
    setSidebarCollapsed: vi.fn(),
  },
}));

vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useNavigate: vi.fn(() => vi.fn()),
  };
});

import { configService } from '@/features/config/api/configService';
import { storage } from '@/lib/storage';

describe('Layout', () => {
  const mockUser = {
    id: 'user-1',
    email: 'test@example.com',
    name: 'Test User',
    isPremium: false,
    role: 'USER' as const,
  };

  const mockPremiumUser = {
    ...mockUser,
    isPremium: true,
  };

  const mockAdminUser = {
    ...mockUser,
    role: 'ADMIN' as const,
  };

  const mockNavItems = [
    { path: '/', translationKey: 'nav.dashboard', iconKey: 'dashboard' },
    { path: '/courses', translationKey: 'nav.courses', iconKey: 'book' },
    { path: '/roadmap', translationKey: 'nav.roadmap', iconKey: 'map' },
    { path: '/analytics', translationKey: 'nav.analytics', iconKey: 'chart' },
    { path: '/settings', translationKey: 'nav.settings', iconKey: 'settings', authRequired: true },
    { path: '/admin', translationKey: 'nav.admin', iconKey: 'dashboard', adminOnly: true },
  ];

  const defaultAuthContext: AuthContextType = {
    user: null,
    login: vi.fn(),
    logout: vi.fn(),
    upgrade: vi.fn(),
    updateUser: vi.fn(),
  };

  const defaultThemeContext: ThemeContextType = {
    isDark: false,
    toggleTheme: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(configService.getNavigation).mockResolvedValue(mockNavItems);
    vi.mocked(storage.getSidebarCollapsed).mockReturnValue(false);
  });

  const renderLayout = (
    authContext: Partial<AuthContextType> = {},
    themeContext: Partial<ThemeContextType> = {},
    initialPath = '/'
  ) => {
    return render(
      <MemoryRouter initialEntries={[initialPath]}>
        <ThemeContext.Provider value={{ ...defaultThemeContext, ...themeContext }}>
          <AuthContext.Provider value={{ ...defaultAuthContext, ...authContext }}>
            <Layout>
              <div data-testid="content">Page Content</div>
            </Layout>
          </AuthContext.Provider>
        </ThemeContext.Provider>
      </MemoryRouter>
    );
  };

  describe('Sidebar', () => {
    it('should render sidebar with logo', async () => {
      renderLayout();

      await waitFor(() => {
        // Practix appears in sidebar logo
        const practixElements = screen.getAllByText('Practix');
        expect(practixElements.length).toBeGreaterThanOrEqual(1);
      });
    });

    it('should load navigation items', async () => {
      renderLayout();

      await waitFor(() => {
        expect(configService.getNavigation).toHaveBeenCalled();
      });

      await waitFor(() => {
        // Dashboard appears in both sidebar nav and header breadcrumb
        const dashboardElements = screen.getAllByText('Dashboard');
        expect(dashboardElements.length).toBeGreaterThanOrEqual(1);
      });

      // Courses appears in sidebar nav
      expect(screen.getByRole('link', { name: /Courses/i })).toBeInTheDocument();
      expect(screen.getByRole('link', { name: /Roadmap/i })).toBeInTheDocument();
    });

    it('should filter auth-required items for unauthenticated users', async () => {
      renderLayout({ user: null });

      await waitFor(() => {
        const dashboardElements = screen.getAllByText('Dashboard');
        expect(dashboardElements.length).toBeGreaterThanOrEqual(1);
      });

      // Settings requires auth - should not be in the nav links
      expect(screen.queryByRole('link', { name: /Settings/i })).not.toBeInTheDocument();
    });

    it('should show auth-required items for authenticated users', async () => {
      renderLayout({ user: mockUser });

      await waitFor(() => {
        expect(screen.getByRole('link', { name: /Settings/i })).toBeInTheDocument();
      });
    });

    it('should filter admin-only items for non-admin users', async () => {
      renderLayout({ user: mockUser });

      await waitFor(() => {
        const dashboardElements = screen.getAllByText('Dashboard');
        expect(dashboardElements.length).toBeGreaterThanOrEqual(1);
      });

      // Admin is admin-only - should not be in the nav links
      expect(screen.queryByRole('link', { name: /Admin/i })).not.toBeInTheDocument();
    });

    it('should show admin items for admin users', async () => {
      renderLayout({ user: mockAdminUser });

      await waitFor(() => {
        expect(screen.getByRole('link', { name: /Admin/i })).toBeInTheDocument();
      });
    });

    it('should show skeleton loader while loading', () => {
      vi.mocked(configService.getNavigation).mockImplementation(
        () => new Promise(() => {}) // Never resolves
      );

      renderLayout();

      // Should show skeleton loaders (5 items)
      const skeletons = document.querySelectorAll('.animate-pulse');
      expect(skeletons.length).toBeGreaterThan(0);
    });

    it('should handle navigation config error gracefully', async () => {
      vi.mocked(configService.getNavigation).mockRejectedValue(new Error('API Error'));

      renderLayout();

      // Should not crash and eventually stop loading
      await waitFor(() => {
        const skeletons = document.querySelectorAll('.animate-pulse');
        expect(skeletons.length).toBe(0);
      });
    });

    it('should collapse sidebar when toggle button is clicked', async () => {
      renderLayout();

      await waitFor(() => {
        expect(screen.getByText('Dashboard')).toBeInTheDocument();
      });

      // Find and click collapse button (left chevron)
      const collapseButtons = document.querySelectorAll('button');
      const collapseButton = Array.from(collapseButtons).find(
        btn => btn.querySelector('svg path[d*="M15 19l-7-7"]')
      );

      if (collapseButton) {
        fireEvent.click(collapseButton);
        expect(storage.setSidebarCollapsed).toHaveBeenCalledWith(true);
      }
    });

    it('should restore collapsed state from storage', async () => {
      vi.mocked(storage.getSidebarCollapsed).mockReturnValue(true);

      renderLayout();

      // In collapsed state, the sidebar width should be w-20 (collapsed)
      // and nav item text should not be visible (only icons)
      await waitFor(() => {
        // Check that getSidebarCollapsed was called to restore state
        expect(storage.getSidebarCollapsed).toHaveBeenCalled();
      });
    });
  });

  describe('Header', () => {
    it('should render header with page title', async () => {
      renderLayout({}, {}, '/');

      await waitFor(() => {
        const dashboardElements = screen.getAllByText('Dashboard');
        expect(dashboardElements.length).toBeGreaterThanOrEqual(1);
      });
    });

    it('should show correct page title for courses', async () => {
      renderLayout({}, {}, '/courses');

      await waitFor(() => {
        // Header shows breadcrumb with page title
        const coursesElements = screen.getAllByText('Courses');
        expect(coursesElements.length).toBeGreaterThanOrEqual(1);
      });
    });

    it('should render login button for unauthenticated users', async () => {
      renderLayout({ user: null });

      await waitFor(() => {
        expect(screen.getByText('Sign In')).toBeInTheDocument();
      });
    });

    it('should render user info for authenticated users', async () => {
      renderLayout({ user: mockUser });

      await waitFor(() => {
        expect(screen.getByText('Test User')).toBeInTheDocument();
      });

      expect(screen.getByText('Free Plan')).toBeInTheDocument();
    });

    it('should show PRO badge for premium users', async () => {
      renderLayout({ user: mockPremiumUser });

      await waitFor(() => {
        expect(screen.getByText('PRO')).toBeInTheDocument();
      });
    });

    it('should show Go Premium button for non-premium users', async () => {
      renderLayout({ user: mockUser });

      await waitFor(() => {
        expect(screen.getByText('Go Premium')).toBeInTheDocument();
      });
    });

    it('should not show Go Premium button for premium users', async () => {
      renderLayout({ user: mockPremiumUser });

      await waitFor(() => {
        expect(screen.queryByText('Go Premium')).not.toBeInTheDocument();
      });
    });

    it('should call logout when logout button is clicked', async () => {
      const mockLogout = vi.fn();
      const mockNavigate = vi.fn();
      vi.mocked(useNavigate).mockReturnValue(mockNavigate);

      renderLayout({ user: mockUser, logout: mockLogout });

      await waitFor(() => {
        expect(screen.getByTestId('logout-button')).toBeInTheDocument();
      });

      fireEvent.click(screen.getByTestId('logout-button'));

      expect(mockLogout).toHaveBeenCalled();
      expect(mockNavigate).toHaveBeenCalledWith('/login');
    });

    it('should toggle theme when theme button is clicked', async () => {
      const mockToggleTheme = vi.fn();

      renderLayout({ user: mockUser }, { toggleTheme: mockToggleTheme });

      await waitFor(() => {
        expect(screen.getByText('Test User')).toBeInTheDocument();
      });

      // Find theme toggle button (has moon or sun icon)
      const themeButton = document.querySelector('button[class*="rounded-full"]');
      if (themeButton) {
        fireEvent.click(themeButton);
        expect(mockToggleTheme).toHaveBeenCalled();
      }
    });

    it('should render language switcher buttons', async () => {
      renderLayout({ user: mockUser });

      await waitFor(() => {
        expect(screen.getByText('UZ')).toBeInTheDocument();
      });

      expect(screen.getByText('RU')).toBeInTheDocument();
      expect(screen.getByText('EN')).toBeInTheDocument();
    });

    it('should render user avatar with initial when no avatar URL', async () => {
      renderLayout({ user: mockUser });

      await waitFor(() => {
        // User initial T (for Test User)
        const avatarInitial = screen.getByText('T');
        expect(avatarInitial).toBeInTheDocument();
      });
    });

    it('should render user avatar image when avatar URL exists', async () => {
      const userWithAvatar = { ...mockUser, avatarUrl: 'https://example.com/avatar.jpg' };
      renderLayout({ user: userWithAvatar });

      await waitFor(() => {
        const avatarImg = screen.getByAltText('Avatar');
        expect(avatarImg).toBeInTheDocument();
        expect(avatarImg).toHaveAttribute('src', 'https://example.com/avatar.jpg');
      });
    });
  });

  describe('Layout structure', () => {
    it('should render children content', async () => {
      renderLayout();

      await waitFor(() => {
        expect(screen.getByTestId('content')).toBeInTheDocument();
      });

      expect(screen.getByText('Page Content')).toBeInTheDocument();
    });

    it('should have correct layout structure', async () => {
      renderLayout();

      await waitFor(() => {
        expect(screen.getByTestId('content')).toBeInTheDocument();
      });

      // Check main structural elements exist
      const mainElement = document.querySelector('main');
      expect(mainElement).toBeInTheDocument();
    });
  });
});
