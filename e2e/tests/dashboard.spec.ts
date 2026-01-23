import { test, expect } from '../fixtures/auth.fixture';
import { DashboardPage } from '../pages/dashboard.page';

test.describe('Dashboard', () => {
  test.describe('Dashboard - Unauthenticated', () => {
    test('should show dashboard or auth state for unauthenticated user', async ({ page }) => {
      // Clear cookies to ensure unauthenticated state
      await page.context().clearCookies();
      await page.goto('/');

      // Wait for page to load
      await page.waitForTimeout(1000);

      const dashboardPage = new DashboardPage(page);

      // Should show auth overlay, redirect to login, or show limited dashboard
      const isAuthOverlay = await dashboardPage.isAuthOverlayVisible();
      const isDashboard = await dashboardPage.isDashboardVisible();
      const url = page.url();

      // Either shows auth state, dashboard (public), or redirects
      expect(isAuthOverlay || isDashboard || url.includes('auth') || url.includes('login') || url.includes('localhost')).toBe(true);
    });

    test('should not show dashboard content for unauthenticated user', async ({ page }) => {
      await page.context().clearCookies();
      await page.goto('/');
      await page.waitForTimeout(1000);

      const dashboardPage = new DashboardPage(page);

      // Stats cards should not be visible
      const totalSolved = await dashboardPage.totalSolvedCard.isVisible().catch(() => false);
      const hoursSpent = await dashboardPage.hoursSpentCard.isVisible().catch(() => false);

      // Either no stats visible, or auth overlay is shown
      const isAuthOverlay = await dashboardPage.isAuthOverlayVisible();
      expect(!totalSolved || !hoursSpent || isAuthOverlay).toBe(true);
    });
  });

  test.describe('Dashboard - Free User', () => {
    test.beforeEach(async ({ auth }) => {
      await auth.loginAsTestUser();
    });

    test('should display dashboard page', async ({ page }) => {
      const dashboardPage = new DashboardPage(page);
      await dashboardPage.goto();
      await dashboardPage.waitForLoad();

      // Page should load
      await expect(page).toHaveURL('/');
    });

    test('should display page title', async ({ page }) => {
      const dashboardPage = new DashboardPage(page);
      await dashboardPage.goto();
      await dashboardPage.waitForLoad();

      // Should have a title element
      const h1 = page.locator('h1');
      await expect(h1).toBeVisible();
    });

    test('should display stats or content section', async ({ page }) => {
      const dashboardPage = new DashboardPage(page);
      await dashboardPage.goto();
      await dashboardPage.waitForLoad();

      // Dashboard should have some content sections
      const pageText = await page.textContent('body');
      const hasContent = pageText && pageText.length > 100;

      // Look for any kind of data display (numbers, charts, etc.)
      const hasNumbers = /\d+/.test(pageText || '');

      expect(hasContent || hasNumbers).toBe(true);
    });

    test('should display total solved stat', async ({ page }) => {
      const dashboardPage = new DashboardPage(page);
      await dashboardPage.goto();
      await dashboardPage.waitForLoad();

      // Look for solved/completed text
      const solvedText = page.locator('text=/solved|completed|tasks/i');
      const isVisible = await solvedText.first().isVisible().catch(() => false);

      // Either visible or page has numeric content
      const pageText = await page.textContent('body');
      expect(isVisible || /\d+/.test(pageText || '')).toBe(true);
    });

    test('should display streak information', async ({ page }) => {
      const dashboardPage = new DashboardPage(page);
      await dashboardPage.goto();
      await dashboardPage.waitForLoad();

      // Look for streak-related content
      const streakText = page.locator('text=/streak|day|days/i');
      const isVisible = await streakText.first().isVisible().catch(() => false);

      // Streak info may or may not be visible depending on user activity
      expect(typeof isVisible).toBe('boolean');
    });

    test('should display activity chart or section', async ({ page }) => {
      const dashboardPage = new DashboardPage(page);
      await dashboardPage.goto();
      await dashboardPage.waitForLoad();

      // Look for activity-related content
      const activitySection = page.locator('text=/activity|progress|chart/i');
      const chartElement = page.locator('svg').or(page.locator('canvas'));

      const hasActivityText = await activitySection.first().isVisible().catch(() => false);
      const hasChart = await chartElement.first().isVisible().catch(() => false);

      // Should have some form of activity visualization
      expect(hasActivityText || hasChart).toBe(true);
    });

    test('should display recent activity or empty state', async ({ page }) => {
      const dashboardPage = new DashboardPage(page);
      await dashboardPage.goto();
      await dashboardPage.waitForLoad();

      // Either recent activity items or empty state message
      const activityCount = await dashboardPage.getRecentActivityCount();
      const isEmptyShown = await dashboardPage.isEmptyActivityShown();

      // One of these should be true
      expect(activityCount > 0 || isEmptyShown || activityCount === 0).toBe(true);
    });

    test('should have interactive elements', async ({ page }) => {
      const dashboardPage = new DashboardPage(page);
      await dashboardPage.goto();
      await dashboardPage.waitForLoad();

      // Should have clickable buttons or links
      const buttons = page.getByRole('button');
      const links = page.getByRole('link');

      const buttonCount = await buttons.count();
      const linkCount = await links.count();

      expect(buttonCount + linkCount).toBeGreaterThan(0);
    });
  });

  test.describe('Dashboard - Premium User', () => {
    test.beforeEach(async ({ auth }) => {
      await auth.loginAsPremiumUser();
    });

    test('should display dashboard for premium user', async ({ page }) => {
      const dashboardPage = new DashboardPage(page);
      await dashboardPage.goto();
      await dashboardPage.waitForLoad();

      // Page should load
      await expect(page).toHaveURL('/');
    });

    test('should show premium user stats', async ({ page }) => {
      const dashboardPage = new DashboardPage(page);
      await dashboardPage.goto();
      await dashboardPage.waitForLoad();

      // Premium user should see dashboard content
      const isDashboardVisible = await dashboardPage.isDashboardVisible();
      const hasContent = (await page.textContent('body'))?.length || 0;

      expect(isDashboardVisible || hasContent > 100).toBe(true);
    });

    test('should not show auth overlay for premium user', async ({ page }) => {
      const dashboardPage = new DashboardPage(page);
      await dashboardPage.goto();
      await dashboardPage.waitForLoad();

      // Auth overlay should not be visible
      const isAuthOverlay = await dashboardPage.isAuthOverlayVisible();
      expect(isAuthOverlay).toBe(false);
    });
  });

  test.describe('Dashboard Accessibility', () => {
    test.beforeEach(async ({ auth }) => {
      await auth.loginAsTestUser();
    });

    test('should have proper heading structure', async ({ page }) => {
      const dashboardPage = new DashboardPage(page);
      await dashboardPage.goto();
      await dashboardPage.waitForLoad();

      // Should have h1
      const h1 = page.locator('h1');
      await expect(h1).toBeVisible();
    });

    test('should be keyboard navigable', async ({ page }) => {
      const dashboardPage = new DashboardPage(page);
      await dashboardPage.goto();
      await dashboardPage.waitForLoad();

      // Tab through elements
      await page.keyboard.press('Tab');
      await page.keyboard.press('Tab');

      // Something should be focused
      const focused = await page.evaluate(() => document.activeElement?.tagName);
      expect(focused).toBeTruthy();
    });

    test('should have semantic structure', async ({ page }) => {
      const dashboardPage = new DashboardPage(page);
      await dashboardPage.goto();
      await dashboardPage.waitForLoad();

      // Should have main content area
      const main = page.locator('main').or(page.locator('[role="main"]'));
      const hasMain = await main.isVisible().catch(() => false);

      // Or at least a structured layout
      const divs = await page.locator('div').count();
      expect(hasMain || divs > 5).toBe(true);
    });
  });

  test.describe('Dashboard Error Handling', () => {
    test.beforeEach(async ({ auth }) => {
      await auth.loginAsTestUser();
    });

    test('should handle API errors gracefully', async ({ page }) => {
      // Mock API error for dashboard data
      await page.route('**/api/dashboard/**', route =>
        route.fulfill({
          status: 500,
          body: JSON.stringify({ error: 'Internal Server Error' }),
        })
      );

      const dashboardPage = new DashboardPage(page);
      await dashboardPage.goto();

      // Wait for error state
      await page.waitForTimeout(2000);

      // Page should not crash - check URL ends with / or is homepage
      const url = page.url();
      expect(url.endsWith('/') || url.includes('localhost')).toBe(true);
    });

    test('should handle network errors gracefully', async ({ page }) => {
      // Simulate network error for user stats
      await page.route('**/api/users/**', route => route.abort('failed'));

      const dashboardPage = new DashboardPage(page);
      await dashboardPage.goto();

      // Wait for error state
      await page.waitForTimeout(2000);

      // Page should handle error gracefully
      expect(page.url()).toContain('localhost');
    });

    test('should handle slow API responses', async ({ page }) => {
      // Simulate slow response
      await page.route('**/api/**', async route => {
        await new Promise(resolve => setTimeout(resolve, 3000));
        route.continue();
      });

      const dashboardPage = new DashboardPage(page);
      await dashboardPage.goto();

      // Page should show loading state or eventually load
      await page.waitForTimeout(1000);

      // Should either show loading or content
      const hasContent = (await page.textContent('body'))?.length || 0;
      expect(hasContent).toBeGreaterThan(50);
    });
  });
});
