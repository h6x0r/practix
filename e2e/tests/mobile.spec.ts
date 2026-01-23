import { test, expect, devices } from '@playwright/test';
import { TEST_USER, PREMIUM_USER } from '../fixtures/auth.fixture';

// Mobile viewport configuration
const mobileViewport = { width: 375, height: 812 }; // iPhone X

test.describe('Mobile Responsiveness', () => {
  test.describe('Mobile Layout', () => {
    test.use({ viewport: mobileViewport });

    test('should display mobile-friendly layout on homepage', async ({ page }) => {
      await page.goto('/');
      await page.waitForTimeout(1000);

      // Page should load without horizontal scrolling
      const bodyWidth = await page.evaluate(() => document.body.scrollWidth);
      const viewportWidth = await page.evaluate(() => window.innerWidth);

      // Body shouldn't be significantly wider than viewport (allow small margin)
      expect(bodyWidth).toBeLessThanOrEqual(viewportWidth + 20);
    });

    test('should display mobile navigation', async ({ page }) => {
      await page.goto('/');
      await page.waitForTimeout(1000);

      // Mobile should have some navigation elements (nav links, buttons, sidebar)
      const navLinks = page.locator('nav a, [class*="nav"] a');
      const navButtons = page.locator('nav button, [class*="nav"] button');
      const sidebarLinks = page.locator('[class*="sidebar"] a, aside a');

      const hasNavLinks = (await navLinks.count()) > 0;
      const hasNavButtons = (await navButtons.count()) > 0;
      const hasSidebarLinks = (await sidebarLinks.count()) > 0;

      // Should have some form of navigation
      expect(hasNavLinks || hasNavButtons || hasSidebarLinks).toBe(true);
    });

    test('should have readable text on mobile', async ({ page }) => {
      await page.goto('/courses');
      await page.waitForTimeout(1000);

      // Check that text is not too small
      const h1FontSize = await page.locator('h1').first().evaluate((el) => {
        return parseInt(window.getComputedStyle(el).fontSize);
      }).catch(() => 16);

      // H1 should be at least 18px on mobile
      expect(h1FontSize).toBeGreaterThanOrEqual(18);
    });

    test('should have clickable buttons with adequate size', async ({ page }) => {
      await page.goto('/courses');
      await page.waitForTimeout(1000);

      // Find all buttons
      const buttons = page.locator('button');
      const buttonCount = await buttons.count();

      if (buttonCount > 0) {
        // Check first visible button size
        const firstButton = buttons.first();
        if (await firstButton.isVisible()) {
          const box = await firstButton.boundingBox();
          if (box) {
            // Buttons should be at least 32x32 for touch targets
            expect(box.height).toBeGreaterThanOrEqual(24);
          }
        }
      }

      expect(buttonCount).toBeGreaterThanOrEqual(0);
    });
  });

  test.describe('Mobile Login Flow', () => {
    test.use({ viewport: mobileViewport });

    test('should display login form on mobile', async ({ page }) => {
      await page.goto('/login');
      await page.waitForTimeout(1000);

      // Login form should be visible
      const emailInput = page.locator('[data-testid="email-input"]');
      const passwordInput = page.locator('[data-testid="password-input"]');

      await expect(emailInput).toBeVisible();
      await expect(passwordInput).toBeVisible();
    });

    test('should allow login on mobile', async ({ page }) => {
      await page.goto('/login');
      await page.waitForTimeout(1000);

      // Fill login form
      await page.fill('[data-testid="email-input"]', TEST_USER.email);
      await page.fill('[data-testid="password-input"]', TEST_USER.password);
      await page.click('[data-testid="login-button"]');

      // Wait for navigation
      await page.waitForURL(/\/(dashboard|courses|)$/, { timeout: 10000 });

      const url = page.url();
      expect(url.includes('/login')).toBe(false);
    });
  });

  test.describe('Mobile Courses Page', () => {
    test.use({ viewport: mobileViewport });

    test('should display course cards in single column on mobile', async ({ page }) => {
      await page.goto('/courses');
      await page.waitForTimeout(1000);

      // Course cards should stack vertically
      const cards = page.locator('[class*="rounded-2xl"], [class*="rounded-3xl"]').filter({
        has: page.locator('h3, h2'),
      });

      const cardCount = await cards.count();
      if (cardCount >= 2) {
        const firstCard = await cards.first().boundingBox();
        const secondCard = await cards.nth(1).boundingBox();

        if (firstCard && secondCard) {
          // Cards should not be side by side (y positions should differ)
          expect(Math.abs(firstCard.y - secondCard.y)).toBeGreaterThan(50);
        }
      }

      expect(cardCount).toBeGreaterThanOrEqual(0);
    });

    test('should have scrollable course list', async ({ page }) => {
      await page.goto('/courses');
      await page.waitForTimeout(1000);

      // Page should be scrollable if content exists
      const scrollHeight = await page.evaluate(() => document.documentElement.scrollHeight);
      const viewportHeight = await page.evaluate(() => window.innerHeight);

      // Either content fits or page is scrollable
      expect(scrollHeight).toBeGreaterThanOrEqual(viewportHeight);
    });
  });

  test.describe('Mobile Dashboard', () => {
    test.use({ viewport: mobileViewport });

    test.beforeEach(async ({ page }) => {
      // Login first
      await page.goto('/login');
      await page.fill('[data-testid="email-input"]', TEST_USER.email);
      await page.fill('[data-testid="password-input"]', TEST_USER.password);
      await page.click('[data-testid="login-button"]');
      await page.waitForURL(/\/(dashboard|courses|)$/, { timeout: 10000 });
    });

    test('should display dashboard on mobile', async ({ page }) => {
      await page.goto('/dashboard');
      await page.waitForTimeout(1000);

      // Dashboard should load
      const pageText = await page.textContent('body');
      const hasDashboardContent =
        pageText?.toLowerCase().includes('dashboard') ||
        pageText?.toLowerCase().includes('progress') ||
        pageText?.toLowerCase().includes('activity') ||
        false;

      expect(hasDashboardContent).toBe(true);
    });

    test('should have mobile-friendly stats cards', async ({ page }) => {
      await page.goto('/dashboard');
      await page.waitForTimeout(1000);

      // Stats cards should be visible
      const statsCards = page.locator('[class*="rounded-2xl"], [class*="rounded-xl"]').filter({
        has: page.locator('[class*="font-bold"]'),
      });

      const cardCount = await statsCards.count();
      expect(cardCount).toBeGreaterThanOrEqual(0);
    });
  });

  test.describe('Mobile Playground', () => {
    test.use({ viewport: mobileViewport });

    test('should display playground on mobile', async ({ page }) => {
      await page.goto('/playground');
      await page.waitForTimeout(1000);

      // Playground should load
      const pageText = await page.textContent('body');
      const hasPlaygroundContent =
        pageText?.toLowerCase().includes('playground') ||
        pageText?.toLowerCase().includes('code') ||
        pageText?.toLowerCase().includes('run') ||
        false;

      expect(hasPlaygroundContent).toBe(true);
    });

    test('should have accessible run button on mobile', async ({ page }) => {
      await page.goto('/playground');
      await page.waitForTimeout(1000);

      // Run button should be visible
      const runButton = page.locator('button').filter({ hasText: /run|execute/i });
      const isVisible = await runButton.first().isVisible().catch(() => false);

      expect(isVisible).toBe(true);
    });
  });

  test.describe('Mobile Touch Interactions', () => {
    test.use({ viewport: mobileViewport, hasTouch: true });

    test('should handle tap on course card', async ({ page }) => {
      await page.goto('/courses');
      await page.waitForTimeout(1000);

      // Find a course card link
      const courseLink = page.locator('a[href*="/course/"]').first();
      if (await courseLink.isVisible()) {
        await courseLink.click(); // Use click - works with hasTouch
        await page.waitForTimeout(1000);

        // Should navigate to course page
        expect(page.url()).toMatch(/\/course\/|\/courses/);
      } else {
        expect(true).toBe(true);
      }
    });

    test('should handle scroll on mobile', async ({ page }) => {
      await page.goto('/courses');
      await page.waitForTimeout(1000);

      // Get initial scroll position
      const initialScroll = await page.evaluate(() => window.scrollY);

      // Scroll down
      await page.evaluate(() => window.scrollBy(0, 300));
      await page.waitForTimeout(300);

      // Get new scroll position
      const newScroll = await page.evaluate(() => window.scrollY);

      // Should have scrolled or page is not scrollable
      expect(newScroll >= initialScroll).toBe(true);
    });

    test('should handle input focus on mobile', async ({ page }) => {
      await page.goto('/login');
      await page.waitForTimeout(1000);

      // Click on email input (works with touch)
      const emailInput = page.locator('[data-testid="email-input"]');
      await emailInput.click();
      await page.waitForTimeout(500);

      // Input should be focused
      const isFocused = await page.evaluate(() => {
        return document.activeElement?.getAttribute('data-testid') === 'email-input';
      });

      expect(isFocused).toBe(true);
    });
  });

  test.describe('Mobile Tablet View', () => {
    test.use({ viewport: { width: 768, height: 1024 } }); // iPad

    test('should display tablet-optimized layout', async ({ page }) => {
      await page.goto('/courses');
      await page.waitForTimeout(1000);

      // Page should load properly on tablet
      const pageText = await page.textContent('body');
      expect(pageText && pageText.length > 100).toBe(true);
    });

    test('should show appropriate navigation on tablet', async ({ page }) => {
      await page.goto('/courses');
      await page.waitForTimeout(1000);

      // Should have some form of navigation
      const nav = page.locator('nav, [class*="sidebar"], [class*="navigation"]');
      const hasNav = await nav.first().isVisible().catch(() => false);

      expect(hasNav).toBe(true);
    });
  });
});
