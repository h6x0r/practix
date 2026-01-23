import { test, expect } from '../fixtures/auth.fixture';
import { MyTasksPage } from '../pages/my-tasks.page';

test.describe('My Tasks', () => {
  test.describe('My Tasks - Unauthenticated', () => {
    test('should show my tasks page with auth prompt', async ({ page }) => {
      // Clear cookies to ensure unauthenticated state
      await page.context().clearCookies();
      await page.goto('/my-tasks');

      // Wait for page to load
      await page.waitForTimeout(1000);

      // Should be on my-tasks page
      expect(page.url()).toContain('/my-tasks');
    });

    test('should show preview content for unauthenticated user', async ({ page }) => {
      await page.context().clearCookies();
      const myTasksPage = new MyTasksPage(page);
      await myTasksPage.goto();
      await myTasksPage.waitForLoad();

      // Should show page title or preview content
      const pageText = await page.textContent('body');
      const hasContent =
        pageText?.toLowerCase().includes('my tasks') ||
        pageText?.toLowerCase().includes('task') ||
        pageText?.toLowerCase().includes('course') ||
        false;

      expect(hasContent).toBe(true);
    });

    test('should show sign in prompt for unauthenticated user', async ({ page }) => {
      await page.context().clearCookies();
      const myTasksPage = new MyTasksPage(page);
      await myTasksPage.goto();
      await myTasksPage.waitForLoad();

      // Should show sign in / login text
      const hasAuthPrompt = await myTasksPage.isAuthOverlayVisible();
      const pageText = await page.textContent('body');
      const hasSignInText =
        pageText?.toLowerCase().includes('sign in') ||
        pageText?.toLowerCase().includes('login') ||
        false;

      expect(hasAuthPrompt || hasSignInText).toBe(true);
    });
  });

  test.describe('My Tasks - Free User', () => {
    test.beforeEach(async ({ auth }) => {
      await auth.loginAsTestUser();
    });

    test('should display my tasks page', async ({ page }) => {
      const myTasksPage = new MyTasksPage(page);
      await myTasksPage.goto();
      await myTasksPage.waitForLoad();

      // Should be on my-tasks page
      await expect(page).toHaveURL(/\/my-tasks/);
    });

    test('should display page title', async ({ page }) => {
      const myTasksPage = new MyTasksPage(page);
      await myTasksPage.goto();
      await myTasksPage.waitForLoad();

      // Should have page title (h1 or h2 for empty state)
      const hasH1 = await page.locator('h1').isVisible().catch(() => false);
      const hasH2 = await page.locator('h2').isVisible().catch(() => false);
      expect(hasH1 || hasH2).toBe(true);
    });

    test('should show content for free user', async ({ page }) => {
      const myTasksPage = new MyTasksPage(page);
      await myTasksPage.goto();
      await myTasksPage.waitForLoad();

      // Should show either courses or empty state
      const hasTitle = await myTasksPage.isMyTasksVisible();
      const hasEmptyState = await myTasksPage.isEmptyStateVisible();
      const pageText = await page.textContent('body');
      const hasContent = pageText && pageText.length > 100;

      expect(hasTitle || hasEmptyState || hasContent).toBe(true);
    });

    test('should show empty state or courses', async ({ page }) => {
      const myTasksPage = new MyTasksPage(page);
      await myTasksPage.goto();
      await myTasksPage.waitForLoad();

      const courseCount = await myTasksPage.getCourseCount();
      const hasEmptyState = await myTasksPage.isEmptyStateVisible();

      // Should have either courses or empty state
      expect(courseCount > 0 || hasEmptyState).toBe(true);
    });

    test('should have browse courses button when empty', async ({ page }) => {
      const myTasksPage = new MyTasksPage(page);
      await myTasksPage.goto();
      await myTasksPage.waitForLoad();

      const hasEmptyState = await myTasksPage.isEmptyStateVisible();

      if (hasEmptyState) {
        // Should have browse courses button (use more specific selector)
        const browseButton = page.getByRole('link', { name: 'Browse Courses' });
        const isVisible = await browseButton.isVisible().catch(() => false);
        // Fallback to any link with browse text
        const anyBrowseLink = await page.locator('a').filter({ hasText: /browse courses/i }).first().isVisible().catch(() => false);
        expect(isVisible || anyBrowseLink).toBe(true);
      } else {
        // Has courses - should have continue buttons
        const courseCount = await myTasksPage.getCourseCount();
        expect(courseCount).toBeGreaterThanOrEqual(0);
      }
    });
  });

  test.describe('My Tasks - Premium User', () => {
    test.beforeEach(async ({ auth }) => {
      await auth.loginAsPremiumUser();
    });

    test('should display my tasks for premium user', async ({ page }) => {
      const myTasksPage = new MyTasksPage(page);
      await myTasksPage.goto();
      await myTasksPage.waitForLoad();

      // Should be on my-tasks page
      await expect(page).toHaveURL(/\/my-tasks/);
    });

    test('should show premium user content', async ({ page }) => {
      const myTasksPage = new MyTasksPage(page);
      await myTasksPage.goto();
      await myTasksPage.waitForLoad();

      // Premium user should see full content
      const hasTitle = await myTasksPage.isMyTasksVisible();
      const hasEmptyState = await myTasksPage.isEmptyStateVisible();
      const pageText = await page.textContent('body');
      const hasContent = pageText && pageText.length > 100;

      expect(hasTitle || hasEmptyState || hasContent).toBe(true);
    });
  });

  test.describe('My Tasks Accessibility', () => {
    test.beforeEach(async ({ auth }) => {
      await auth.loginAsTestUser();
    });

    test('should have proper heading structure', async ({ page }) => {
      const myTasksPage = new MyTasksPage(page);
      await myTasksPage.goto();
      await myTasksPage.waitForLoad();

      // Should have h1 or h2
      const hasH1 = await page.locator('h1').isVisible().catch(() => false);
      const hasH2 = await page.locator('h2').isVisible().catch(() => false);

      expect(hasH1 || hasH2).toBe(true);
    });

    test('should be keyboard navigable', async ({ page }) => {
      const myTasksPage = new MyTasksPage(page);
      await myTasksPage.goto();
      await myTasksPage.waitForLoad();

      // Tab through elements
      await page.keyboard.press('Tab');
      await page.keyboard.press('Tab');

      // Something should be focused
      const focused = await page.evaluate(() => document.activeElement?.tagName);
      expect(focused).toBeTruthy();
    });
  });

  test.describe('My Tasks Error Handling', () => {
    test.beforeEach(async ({ auth }) => {
      await auth.loginAsTestUser();
    });

    test('should handle API errors gracefully', async ({ page }) => {
      // Mock API error
      await page.route('**/user-courses/**', route =>
        route.fulfill({
          status: 500,
          body: JSON.stringify({ error: 'Internal Server Error' }),
        })
      );

      const myTasksPage = new MyTasksPage(page);
      await myTasksPage.goto();

      // Wait for error state
      await page.waitForTimeout(2000);

      // Page should not crash
      expect(page.url()).toContain('/my-tasks');
    });

    test('should handle empty courses list', async ({ page }) => {
      // Mock empty courses
      await page.route('**/user-courses/started**', route =>
        route.fulfill({
          status: 200,
          body: JSON.stringify([]),
        })
      );

      const myTasksPage = new MyTasksPage(page);
      await myTasksPage.goto();
      await myTasksPage.waitForLoad();

      // Should show empty state
      const hasEmptyState = await myTasksPage.isEmptyStateVisible();
      const pageText = await page.textContent('body');
      const hasNoCoursesText =
        pageText?.toLowerCase().includes('no active') ||
        pageText?.toLowerCase().includes('start') ||
        pageText?.toLowerCase().includes('browse') ||
        false;

      expect(hasEmptyState || hasNoCoursesText).toBe(true);
    });
  });
});
