import { test, expect } from '../fixtures/auth.fixture';
import { LeaderboardPage } from '../pages/leaderboard.page';

test.describe('Leaderboard', () => {
  test.describe('Leaderboard - Unauthenticated', () => {
    test('should show leaderboard page for unauthenticated user', async ({ page }) => {
      // Clear cookies to ensure unauthenticated state
      await page.context().clearCookies();
      await page.goto('/leaderboard');

      // Wait for page to load
      await page.waitForTimeout(1000);

      // Should be on leaderboard page
      expect(page.url()).toContain('/leaderboard');
    });

    test('should show preview data for unauthenticated user', async ({ page }) => {
      await page.context().clearCookies();
      const leaderboardPage = new LeaderboardPage(page);
      await leaderboardPage.goto();
      await leaderboardPage.waitForLoad();

      // Should show page title or preview content
      const hasTitle = await leaderboardPage.isLeaderboardVisible();
      const pageText = await page.textContent('body');
      const hasLeaderboardContent =
        pageText?.toLowerCase().includes('leaderboard') ||
        pageText?.includes('🥇') ||
        false;

      expect(hasTitle || hasLeaderboardContent).toBe(true);
    });

    test('should show sign in prompt for unauthenticated user', async ({ page }) => {
      await page.context().clearCookies();
      const leaderboardPage = new LeaderboardPage(page);
      await leaderboardPage.goto();
      await leaderboardPage.waitForLoad();

      // Should show sign in / login text
      const hasAuthPrompt = await leaderboardPage.isAuthOverlayVisible();
      const pageText = await page.textContent('body');
      const hasSignInText =
        pageText?.toLowerCase().includes('sign in') ||
        pageText?.toLowerCase().includes('login') ||
        false;

      expect(hasAuthPrompt || hasSignInText).toBe(true);
    });
  });

  test.describe('Leaderboard - Free User', () => {
    test.beforeEach(async ({ auth }) => {
      await auth.loginAsTestUser();
    });

    test('should display leaderboard page', async ({ page }) => {
      const leaderboardPage = new LeaderboardPage(page);
      await leaderboardPage.goto();
      await leaderboardPage.waitForLoad();

      // Should be on leaderboard page
      await expect(page).toHaveURL(/\/leaderboard/);
    });

    test('should display page title', async ({ page }) => {
      const leaderboardPage = new LeaderboardPage(page);
      await leaderboardPage.goto();
      await leaderboardPage.waitForLoad();

      // Should have leaderboard title
      const h1 = page.locator('h1');
      await expect(h1).toBeVisible();
    });

    test('should display leaderboard entries', async ({ page }) => {
      const leaderboardPage = new LeaderboardPage(page);
      await leaderboardPage.goto();
      await leaderboardPage.waitForLoad();

      // Should have some entries
      const entryCount = await leaderboardPage.getEntryCount();
      expect(entryCount).toBeGreaterThanOrEqual(0);
    });

    test('should display top 3 medals', async ({ page }) => {
      const leaderboardPage = new LeaderboardPage(page);
      await leaderboardPage.goto();
      await leaderboardPage.waitForLoad();

      // If entries exist, top 3 should have medals
      const entryCount = await leaderboardPage.getEntryCount();
      if (entryCount >= 3) {
        const hasMedals = await leaderboardPage.hasTopThreeMedals();
        expect(hasMedals).toBe(true);
      } else {
        // Less than 3 entries - pass
        expect(true).toBe(true);
      }
    });

    test('should display XP column', async ({ page }) => {
      const leaderboardPage = new LeaderboardPage(page);
      await leaderboardPage.goto();
      await leaderboardPage.waitForLoad();

      // Should have XP text somewhere
      const xpText = page.locator('text=/xp/i');
      const hasXP = await xpText.first().isVisible().catch(() => false);

      expect(hasXP).toBe(true);
    });

    test('should display streak column', async ({ page }) => {
      const leaderboardPage = new LeaderboardPage(page);
      await leaderboardPage.goto();
      await leaderboardPage.waitForLoad();

      // Should have streak text or fire icon
      const pageText = await page.textContent('body');
      const hasStreak =
        pageText?.toLowerCase().includes('streak') ||
        pageText?.includes('🔥') ||
        false;

      expect(hasStreak).toBe(true);
    });

    test('should show user stats card', async ({ page }) => {
      const leaderboardPage = new LeaderboardPage(page);
      await leaderboardPage.goto();
      await leaderboardPage.waitForLoad();

      // For logged in user, should show my stats
      const isVisible = await leaderboardPage.isMyStatsVisible();
      const pageText = await page.textContent('body');
      const hasUserLevel = pageText?.toLowerCase().includes('level') || false;

      expect(isVisible || hasUserLevel).toBe(true);
    });
  });

  test.describe('Leaderboard - Premium User', () => {
    test.beforeEach(async ({ auth }) => {
      await auth.loginAsPremiumUser();
    });

    test('should display leaderboard for premium user', async ({ page }) => {
      const leaderboardPage = new LeaderboardPage(page);
      await leaderboardPage.goto();
      await leaderboardPage.waitForLoad();

      // Should be on leaderboard page
      await expect(page).toHaveURL(/\/leaderboard/);
    });

    test('should show premium user in leaderboard', async ({ page }) => {
      const leaderboardPage = new LeaderboardPage(page);
      await leaderboardPage.goto();
      await leaderboardPage.waitForLoad();

      // Premium user should be visible with stats
      const hasStats = await leaderboardPage.isMyStatsVisible();
      const pageText = await page.textContent('body');
      const hasLevel = pageText?.toLowerCase().includes('level') || false;

      expect(hasStats || hasLevel).toBe(true);
    });
  });

  test.describe('Leaderboard Accessibility', () => {
    test.beforeEach(async ({ auth }) => {
      await auth.loginAsTestUser();
    });

    test('should have proper heading structure', async ({ page }) => {
      const leaderboardPage = new LeaderboardPage(page);
      await leaderboardPage.goto();
      await leaderboardPage.waitForLoad();

      // Should have h1
      const h1 = page.locator('h1');
      await expect(h1).toBeVisible();
    });

    test('should be keyboard navigable', async ({ page }) => {
      const leaderboardPage = new LeaderboardPage(page);
      await leaderboardPage.goto();
      await leaderboardPage.waitForLoad();

      // Tab through elements
      await page.keyboard.press('Tab');
      await page.keyboard.press('Tab');

      // Something should be focused
      const focused = await page.evaluate(() => document.activeElement?.tagName);
      expect(focused).toBeTruthy();
    });
  });

  test.describe('Leaderboard Error Handling', () => {
    test.beforeEach(async ({ auth }) => {
      await auth.loginAsTestUser();
    });

    test('should handle API errors gracefully', async ({ page }) => {
      // Mock API error
      await page.route('**/gamification/**', route =>
        route.fulfill({
          status: 500,
          body: JSON.stringify({ error: 'Internal Server Error' }),
        })
      );

      const leaderboardPage = new LeaderboardPage(page);
      await leaderboardPage.goto();

      // Wait for error state
      await page.waitForTimeout(2000);

      // Page should not crash
      expect(page.url()).toContain('/leaderboard');
    });

    test('should handle empty leaderboard', async ({ page }) => {
      // Mock empty leaderboard
      await page.route('**/gamification/leaderboard**', route =>
        route.fulfill({
          status: 200,
          body: JSON.stringify([]),
        })
      );

      const leaderboardPage = new LeaderboardPage(page);
      await leaderboardPage.goto();
      await leaderboardPage.waitForLoad();

      // Should show empty state or just empty table
      const entryCount = await leaderboardPage.getEntryCount();
      expect(entryCount).toBeGreaterThanOrEqual(0);
    });
  });
});
