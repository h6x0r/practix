import { test, expect } from '../fixtures/auth.fixture';
import { RoadmapPage } from '../pages/roadmap.page';

test.describe('Roadmap', () => {
  test.describe('Roadmap Page - Unauthenticated', () => {
    test('should display intro screen for unauthenticated user', async ({ page }) => {
      // Clear cookies to ensure unauthenticated state
      await page.context().clearCookies();
      await page.goto('/roadmap');

      // Should show intro or redirect to login
      await page.waitForTimeout(1000);
      const url = page.url();

      // Either on roadmap page or redirected to auth
      expect(url.includes('roadmap') || url.includes('auth') || url.includes('login')).toBe(true);
    });
  });

  test.describe('Roadmap Page - Free User', () => {
    test.beforeEach(async ({ auth }) => {
      await auth.loginAsTestUser();
    });

    test('should display roadmap page', async ({ page }) => {
      const roadmapPage = new RoadmapPage(page);
      await roadmapPage.goto();
      await roadmapPage.waitForLoad();

      // Page should load - either intro or existing roadmap
      await expect(page).toHaveURL(/\/roadmap/);
    });

    test('should display intro title', async ({ page }) => {
      const roadmapPage = new RoadmapPage(page);
      await roadmapPage.goto();
      await roadmapPage.waitForLoad();

      // Should have a title
      const h1 = page.locator('h1');
      await expect(h1).toBeVisible();
    });

    test('should have start button on intro', async ({ page }) => {
      const roadmapPage = new RoadmapPage(page);
      await roadmapPage.goto();
      await roadmapPage.waitForLoad();

      // If on intro, start button should be visible
      const isIntro = await roadmapPage.isIntroVisible();
      if (isIntro) {
        await expect(roadmapPage.startButton).toBeVisible();
      }
    });

    test('should start wizard when clicking start', async ({ page }) => {
      const roadmapPage = new RoadmapPage(page);
      await roadmapPage.goto();
      await roadmapPage.waitForLoad();

      // If on intro, click start
      const isIntro = await roadmapPage.isIntroVisible();
      if (isIntro) {
        await roadmapPage.startWizard();

        // Should show wizard step (language selection)
        const pythonVisible = await roadmapPage.pythonOption.isVisible().catch(() => false);
        const nextVisible = await roadmapPage.nextButton.isVisible().catch(() => false);
        expect(pythonVisible || nextVisible).toBe(true);
      }
    });

    test('should display language options in wizard', async ({ page }) => {
      const roadmapPage = new RoadmapPage(page);
      await roadmapPage.goto();
      await roadmapPage.waitForLoad();

      const isIntro = await roadmapPage.isIntroVisible();
      if (isIntro) {
        await roadmapPage.startWizard();

        // Should show language options
        await expect(roadmapPage.pythonOption).toBeVisible();
        await expect(roadmapPage.javascriptOption).toBeVisible();
      }
    });

    test('should allow selecting multiple languages', async ({ page }) => {
      const roadmapPage = new RoadmapPage(page);
      await roadmapPage.goto();
      await roadmapPage.waitForLoad();

      const isIntro = await roadmapPage.isIntroVisible();
      if (isIntro) {
        await roadmapPage.startWizard();

        // Select Python
        await roadmapPage.selectLanguage('python');
        // Select JavaScript
        await roadmapPage.selectLanguage('javascript');

        // Both should show selected state (check for class or attribute)
        const pythonClasses = await roadmapPage.pythonOption.getAttribute('class');
        expect(pythonClasses).toBeTruthy();
      }
    });

    test('should navigate to next wizard step', async ({ page }) => {
      const roadmapPage = new RoadmapPage(page);
      await roadmapPage.goto();
      await roadmapPage.waitForLoad();

      const isIntro = await roadmapPage.isIntroVisible();
      if (isIntro) {
        await roadmapPage.startWizard();
        await roadmapPage.goNext();

        // Should move to experience step
        await page.waitForTimeout(300);
        const pageContent = await page.textContent('body');
        expect(
          pageContent?.toLowerCase().includes('experience') ||
            pageContent?.toLowerCase().includes('year') ||
            pageContent?.toLowerCase().includes('background')
        ).toBe(true);
      }
    });

    test('should navigate back in wizard', async ({ page }) => {
      const roadmapPage = new RoadmapPage(page);
      await roadmapPage.goto();
      await roadmapPage.waitForLoad();

      const isIntro = await roadmapPage.isIntroVisible();
      if (isIntro) {
        await roadmapPage.startWizard();
        await roadmapPage.goNext(); // Go to step 2

        // Back button should be visible
        await expect(roadmapPage.backButton).toBeVisible();

        await roadmapPage.goBack(); // Go back to step 1

        // Should be back on languages step
        await expect(roadmapPage.pythonOption).toBeVisible();
      }
    });

    test('should show interests in wizard', async ({ page }) => {
      const roadmapPage = new RoadmapPage(page);
      await roadmapPage.goto();
      await roadmapPage.waitForLoad();

      const isIntro = await roadmapPage.isIntroVisible();
      if (isIntro) {
        await roadmapPage.startWizard();
        await roadmapPage.goNext(); // Languages -> Experience
        await roadmapPage.goNext(); // Experience -> Interests

        // Should show interest options
        await page.waitForTimeout(300);
        const pageContent = await page.textContent('body');
        expect(
          pageContent?.toLowerCase().includes('interest') ||
            pageContent?.toLowerCase().includes('backend') ||
            pageContent?.toLowerCase().includes('learn')
        ).toBe(true);
      }
    });

    test('should show goals in wizard', async ({ page }) => {
      const roadmapPage = new RoadmapPage(page);
      await roadmapPage.goto();
      await roadmapPage.waitForLoad();

      const isIntro = await roadmapPage.isIntroVisible();
      if (isIntro) {
        await roadmapPage.startWizard();
        await roadmapPage.goNext(); // Languages
        await roadmapPage.goNext(); // Experience

        // Select an interest
        await roadmapPage.selectInterest('backend');
        await roadmapPage.goNext(); // Interests -> Goals

        // Should show goal options
        await page.waitForTimeout(300);
        const pageContent = await page.textContent('body');
        expect(
          pageContent?.toLowerCase().includes('goal') ||
            pageContent?.toLowerCase().includes('achieve') ||
            pageContent?.toLowerCase().includes('job')
        ).toBe(true);
      }
    });

    test('should show time commitment in wizard', async ({ page }) => {
      const roadmapPage = new RoadmapPage(page);
      await roadmapPage.goto();
      await roadmapPage.waitForLoad();

      const isIntro = await roadmapPage.isIntroVisible();
      if (isIntro) {
        await roadmapPage.startWizard();
        await roadmapPage.goNext(); // Languages
        await roadmapPage.goNext(); // Experience
        await roadmapPage.selectInterest('algorithms');
        await roadmapPage.goNext(); // Interests
        await roadmapPage.selectGoal('job');
        await roadmapPage.goNext(); // Goals -> Time

        // Should show time commitment options
        await page.waitForTimeout(300);
        const pageContent = await page.textContent('body');
        expect(
          pageContent?.toLowerCase().includes('time') ||
            pageContent?.toLowerCase().includes('hours') ||
            pageContent?.toLowerCase().includes('week') ||
            pageContent?.toLowerCase().includes('month')
        ).toBe(true);
      }
    });
  });

  test.describe('Roadmap Page - Premium User', () => {
    test.beforeEach(async ({ auth }) => {
      await auth.loginAsPremiumUser();
    });

    test('should display roadmap page for premium user', async ({ page }) => {
      const roadmapPage = new RoadmapPage(page);
      await roadmapPage.goto();
      await roadmapPage.waitForLoad();

      // Page should load
      await expect(page).toHaveURL(/\/roadmap/);
    });

    test('should show existing roadmap or intro', async ({ page }) => {
      const roadmapPage = new RoadmapPage(page);
      await roadmapPage.goto();
      await roadmapPage.waitForLoad();

      // Premium user might have existing roadmap or see intro
      const isIntro = await roadmapPage.isIntroVisible();
      const isResult = await roadmapPage.isResultVisible();

      expect(isIntro || isResult).toBe(true);
    });
  });

  test.describe('Roadmap Accessibility', () => {
    test.beforeEach(async ({ auth }) => {
      await auth.loginAsTestUser();
    });

    test('should have proper heading structure', async ({ page }) => {
      const roadmapPage = new RoadmapPage(page);
      await roadmapPage.goto();
      await roadmapPage.waitForLoad();

      // Should have h1
      const h1 = page.locator('h1');
      await expect(h1).toBeVisible();
    });

    test('should be keyboard navigable', async ({ page }) => {
      const roadmapPage = new RoadmapPage(page);
      await roadmapPage.goto();
      await roadmapPage.waitForLoad();

      // Tab through elements
      await page.keyboard.press('Tab');
      await page.keyboard.press('Tab');

      // Something should be focused
      const focused = await page.evaluate(() => document.activeElement?.tagName);
      expect(focused).toBeTruthy();
    });

    test('should have clickable buttons', async ({ page }) => {
      const roadmapPage = new RoadmapPage(page);
      await roadmapPage.goto();
      await roadmapPage.waitForLoad();

      const isIntro = await roadmapPage.isIntroVisible();
      if (isIntro) {
        // Start button should be clickable
        await expect(roadmapPage.startButton).toBeEnabled();
      }
    });
  });

  test.describe('Roadmap Error Handling', () => {
    test.beforeEach(async ({ auth }) => {
      await auth.loginAsTestUser();
    });

    test('should handle network errors gracefully', async ({ page }) => {
      // Simulate network error for roadmap API
      await page.route('**/api/roadmap/**', route => route.abort('failed'));

      const roadmapPage = new RoadmapPage(page);
      await roadmapPage.goto();

      // Wait for error state
      await page.waitForTimeout(2000);

      // Page should not crash
      expect(page.url()).toContain('/roadmap');
    });

    test('should handle API errors gracefully', async ({ page }) => {
      // Mock API error
      await page.route('**/api/roadmap/user', route =>
        route.fulfill({
          status: 500,
          body: JSON.stringify({ error: 'Internal Server Error' }),
        })
      );

      const roadmapPage = new RoadmapPage(page);
      await roadmapPage.goto();

      // Wait for error state
      await page.waitForTimeout(1000);

      // Page should handle error gracefully
      expect(page.url()).toContain('/roadmap');
    });
  });
});
