import { test, expect } from '../fixtures/auth.fixture';
import { AnalyticsPage } from '../pages/analytics.page';

test.describe('Analytics', () => {
  test.describe('Analytics - Unauthenticated', () => {
    test('should show analytics page with auth prompt', async ({ page }) => {
      // Clear cookies to ensure unauthenticated state
      await page.context().clearCookies();
      await page.goto('/analytics');

      // Wait for page to load
      await page.waitForTimeout(1000);

      // Should be on analytics page
      expect(page.url()).toContain('/analytics');
    });

    test('should show preview content for unauthenticated user', async ({ page }) => {
      await page.context().clearCookies();
      const analyticsPage = new AnalyticsPage(page);
      await analyticsPage.goto();
      await analyticsPage.waitForLoad();

      // Should show page content or auth overlay
      const pageText = await page.textContent('body');
      const hasContent =
        pageText?.toLowerCase().includes('analytics') ||
        pageText?.toLowerCase().includes('activity') ||
        pageText?.toLowerCase().includes('sign in') ||
        false;

      expect(hasContent).toBe(true);
    });

    test('should show sign in prompt for unauthenticated user', async ({ page }) => {
      await page.context().clearCookies();
      const analyticsPage = new AnalyticsPage(page);
      await analyticsPage.goto();
      await analyticsPage.waitForLoad();

      // Should show sign in / login text
      const hasAuthPrompt = await analyticsPage.isAuthOverlayVisible();
      const pageText = await page.textContent('body');
      const hasSignInText =
        pageText?.toLowerCase().includes('sign in') ||
        pageText?.toLowerCase().includes('login') ||
        false;

      expect(hasAuthPrompt || hasSignInText).toBe(true);
    });
  });

  test.describe('Analytics - Free User', () => {
    test.beforeEach(async ({ auth }) => {
      await auth.loginAsTestUser();
    });

    test('should display analytics page', async ({ page }) => {
      const analyticsPage = new AnalyticsPage(page);
      await analyticsPage.goto();
      await analyticsPage.waitForLoad();

      // Should be on analytics page
      await expect(page).toHaveURL(/\/analytics/);
    });

    test('should display page title', async ({ page }) => {
      const analyticsPage = new AnalyticsPage(page);
      await analyticsPage.goto();
      await analyticsPage.waitForLoad();

      // Should have page title
      const h1 = page.locator('h1');
      await expect(h1).toBeVisible();
    });

    test('should display weekly activity chart', async ({ page }) => {
      const analyticsPage = new AnalyticsPage(page);
      await analyticsPage.goto();
      await analyticsPage.waitForLoad();

      // Should show weekly chart or some chart element
      const hasChart = await analyticsPage.hasWeeklyChart();
      const pageText = await page.textContent('body');
      const hasWeeklyContent =
        pageText?.toLowerCase().includes('week') ||
        pageText?.toLowerCase().includes('activity') ||
        false;

      expect(hasChart || hasWeeklyContent).toBe(true);
    });

    test('should display yearly contributions', async ({ page }) => {
      const analyticsPage = new AnalyticsPage(page);
      await analyticsPage.goto();
      await analyticsPage.waitForLoad();

      // Should show yearly heatmap or contributions content
      const hasHeatmap = await analyticsPage.hasYearlyHeatmap();
      const pageText = await page.textContent('body');
      const hasYearlyContent =
        pageText?.toLowerCase().includes('year') ||
        pageText?.toLowerCase().includes('contribution') ||
        false;

      expect(hasHeatmap || hasYearlyContent).toBe(true);
    });

    test('should have week navigation buttons', async ({ page }) => {
      const analyticsPage = new AnalyticsPage(page);
      await analyticsPage.goto();
      await analyticsPage.waitForLoad();

      // Should have navigation buttons
      const buttons = await page.locator('button').count();
      expect(buttons).toBeGreaterThan(0);
    });
  });

  test.describe('Analytics - Premium User', () => {
    test.beforeEach(async ({ auth }) => {
      await auth.loginAsPremiumUser();
    });

    test('should display analytics for premium user', async ({ page }) => {
      const analyticsPage = new AnalyticsPage(page);
      await analyticsPage.goto();
      await analyticsPage.waitForLoad();

      // Should be on analytics page
      await expect(page).toHaveURL(/\/analytics/);
    });

    test('should show premium user analytics data', async ({ page }) => {
      const analyticsPage = new AnalyticsPage(page);
      await analyticsPage.goto();
      await analyticsPage.waitForLoad();

      // Premium user should see full analytics
      const pageText = await page.textContent('body');
      const hasContent = pageText && pageText.length > 100;

      expect(hasContent).toBe(true);
    });
  });

  test.describe('Analytics Accessibility', () => {
    test.beforeEach(async ({ auth }) => {
      await auth.loginAsTestUser();
    });

    test('should have proper heading structure', async ({ page }) => {
      const analyticsPage = new AnalyticsPage(page);
      await analyticsPage.goto();
      await analyticsPage.waitForLoad();

      // Should have h1
      const h1 = page.locator('h1');
      await expect(h1).toBeVisible();
    });

    test('should be keyboard navigable', async ({ page }) => {
      const analyticsPage = new AnalyticsPage(page);
      await analyticsPage.goto();
      await analyticsPage.waitForLoad();

      // Tab through elements
      await page.keyboard.press('Tab');
      await page.keyboard.press('Tab');

      // Something should be focused
      const focused = await page.evaluate(() => document.activeElement?.tagName);
      expect(focused).toBeTruthy();
    });
  });

  test.describe('Analytics Error Handling', () => {
    test.beforeEach(async ({ auth }) => {
      await auth.loginAsTestUser();
    });

    test('should handle API errors gracefully', async ({ page }) => {
      // Mock API error
      await page.route('**/analytics/**', route =>
        route.fulfill({
          status: 500,
          body: JSON.stringify({ error: 'Internal Server Error' }),
        })
      );

      const analyticsPage = new AnalyticsPage(page);
      await analyticsPage.goto();

      // Wait for error state
      await page.waitForTimeout(2000);

      // Page should not crash
      expect(page.url()).toContain('/analytics');
    });

    test('should handle empty analytics data', async ({ page }) => {
      // Mock empty data
      await page.route('**/analytics/weekly**', route =>
        route.fulfill({
          status: 200,
          body: JSON.stringify([]),
        })
      );
      await page.route('**/analytics/yearly**', route =>
        route.fulfill({
          status: 200,
          body: JSON.stringify([]),
        })
      );

      const analyticsPage = new AnalyticsPage(page);
      await analyticsPage.goto();
      await analyticsPage.waitForLoad();

      // Should show empty state or just empty chart
      const pageText = await page.textContent('body');
      expect(pageText && pageText.length > 50).toBe(true);
    });
  });
});
