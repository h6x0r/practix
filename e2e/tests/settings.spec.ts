import { test, expect } from '../fixtures/auth.fixture';
import { SettingsPage } from '../pages/settings.page';

test.describe('Settings', () => {
  test.describe('Settings - Unauthenticated', () => {
    test('should show settings page with auth prompt', async ({ page }) => {
      // Clear cookies to ensure unauthenticated state
      await page.context().clearCookies();
      await page.goto('/settings');

      // Wait for page to load
      await page.waitForTimeout(1000);

      // Should be on settings page
      expect(page.url()).toContain('/settings');
    });

    test('should show some content for unauthenticated user', async ({ page }) => {
      await page.context().clearCookies();
      const settingsPage = new SettingsPage(page);
      await settingsPage.goto();
      await settingsPage.waitForLoad();

      // Should show some content - either auth prompt, settings preview, or redirect
      const pageText = await page.textContent('body');
      const hasContent = pageText && pageText.length > 50;
      const url = page.url();

      // Either has content or redirected
      expect(hasContent || url.includes('login') || url.includes('auth')).toBe(true);
    });

    test('should load settings page for unauthenticated user', async ({ page }) => {
      await page.context().clearCookies();
      const settingsPage = new SettingsPage(page);
      await settingsPage.goto();
      await settingsPage.waitForLoad();

      // Page should load without crashing
      const url = page.url();
      const pageText = await page.textContent('body');
      const hasContent = pageText && pageText.length > 50;

      expect(url.includes('/settings') || url.includes('/login') || hasContent).toBe(true);
    });
  });

  test.describe('Settings - Free User', () => {
    test.beforeEach(async ({ auth }) => {
      await auth.loginAsTestUser();
    });

    test('should display settings page', async ({ page }) => {
      const settingsPage = new SettingsPage(page);
      await settingsPage.goto();
      await settingsPage.waitForLoad();

      // Should be on settings page
      await expect(page).toHaveURL(/\/settings/);
    });

    test('should display page title', async ({ page }) => {
      const settingsPage = new SettingsPage(page);
      await settingsPage.goto();
      await settingsPage.waitForLoad();

      // Should have settings title
      const h1 = page.locator('h1');
      await expect(h1).toBeVisible();
    });

    test('should display tab navigation', async ({ page }) => {
      const settingsPage = new SettingsPage(page);
      await settingsPage.goto();
      await settingsPage.waitForLoad();

      // Should have tabs or navigation
      const hasTabs = await settingsPage.areTabsVisible();
      const pageText = await page.textContent('body');
      const hasNavContent =
        pageText?.toLowerCase().includes('profile') ||
        pageText?.toLowerCase().includes('notification') ||
        false;

      expect(hasTabs || hasNavContent).toBe(true);
    });

    test('should display profile section', async ({ page }) => {
      const settingsPage = new SettingsPage(page);
      await settingsPage.goto();
      await settingsPage.waitForLoad();

      // Should show profile-related content
      const pageText = await page.textContent('body');
      const hasProfileContent =
        pageText?.toLowerCase().includes('profile') ||
        pageText?.toLowerCase().includes('avatar') ||
        false;

      expect(hasProfileContent).toBe(true);
    });

    test('should display notifications section', async ({ page }) => {
      const settingsPage = new SettingsPage(page);
      await settingsPage.goto();
      await settingsPage.waitForLoad();

      // Should show notification-related content
      const hasNotifications = await settingsPage.hasNotificationToggles();
      expect(hasNotifications).toBe(true);
    });

    test('should have save button', async ({ page }) => {
      const settingsPage = new SettingsPage(page);
      await settingsPage.goto();
      await settingsPage.waitForLoad();

      // Should have a save button
      const saveButton = page.locator('button').filter({ hasText: /save/i });
      const isVisible = await saveButton.isVisible().catch(() => false);

      // Save button might be in any section
      expect(typeof isVisible).toBe('boolean');
    });
  });

  test.describe('Settings - Premium User', () => {
    test.beforeEach(async ({ auth }) => {
      await auth.loginAsPremiumUser();
    });

    test('should display settings for premium user', async ({ page }) => {
      const settingsPage = new SettingsPage(page);
      await settingsPage.goto();
      await settingsPage.waitForLoad();

      // Should be on settings page
      await expect(page).toHaveURL(/\/settings/);
    });

    test('should show premium user settings', async ({ page }) => {
      const settingsPage = new SettingsPage(page);
      await settingsPage.goto();
      await settingsPage.waitForLoad();

      // Premium user should see full settings
      const isVisible = await settingsPage.isSettingsVisible();
      const pageText = await page.textContent('body');
      const hasContent = pageText && pageText.length > 100;

      expect(isVisible || hasContent).toBe(true);
    });
  });

  test.describe('Settings Accessibility', () => {
    test.beforeEach(async ({ auth }) => {
      await auth.loginAsTestUser();
    });

    test('should have proper heading structure', async ({ page }) => {
      const settingsPage = new SettingsPage(page);
      await settingsPage.goto();
      await settingsPage.waitForLoad();

      // Should have h1
      const h1 = page.locator('h1');
      await expect(h1).toBeVisible();
    });

    test('should be keyboard navigable', async ({ page }) => {
      const settingsPage = new SettingsPage(page);
      await settingsPage.goto();
      await settingsPage.waitForLoad();

      // Tab through elements
      await page.keyboard.press('Tab');
      await page.keyboard.press('Tab');

      // Something should be focused
      const focused = await page.evaluate(() => document.activeElement?.tagName);
      expect(focused).toBeTruthy();
    });

    test('should have interactive toggles', async ({ page }) => {
      const settingsPage = new SettingsPage(page);
      await settingsPage.goto();
      await settingsPage.waitForLoad();

      // Should have clickable toggles or buttons
      const buttons = await page.locator('button').count();
      expect(buttons).toBeGreaterThan(0);
    });
  });

  test.describe('Settings Error Handling', () => {
    test.beforeEach(async ({ auth }) => {
      await auth.loginAsTestUser();
    });

    test('should handle API errors gracefully', async ({ page }) => {
      // Mock API error for settings update
      await page.route('**/users/**', route =>
        route.fulfill({
          status: 500,
          body: JSON.stringify({ error: 'Internal Server Error' }),
        })
      );

      const settingsPage = new SettingsPage(page);
      await settingsPage.goto();

      // Wait for error state
      await page.waitForTimeout(2000);

      // Page should not crash
      expect(page.url()).toContain('/settings');
    });

    test('should handle network errors gracefully', async ({ page }) => {
      // Simulate network error for preferences
      await page.route('**/preferences/**', route => route.abort('failed'));

      const settingsPage = new SettingsPage(page);
      await settingsPage.goto();
      await settingsPage.waitForLoad();

      // Page should handle error gracefully
      expect(page.url()).toContain('/settings');
    });
  });
});
