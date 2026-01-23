import { Page, Locator } from '@playwright/test';

/**
 * Page Object for Dashboard page
 */
export class DashboardPage {
  readonly page: Page;

  // Page elements
  readonly pageTitle: Locator;
  readonly streakBadge: Locator;

  // Stats cards
  readonly totalSolvedCard: Locator;
  readonly hoursSpentCard: Locator;
  readonly globalRankCard: Locator;
  readonly skillPointsCard: Locator;

  // Activity chart
  readonly activityChart: Locator;
  readonly chartPeriodSelector: Locator;

  // Recent activity
  readonly recentActivitySection: Locator;
  readonly recentActivityItems: Locator;
  readonly emptyActivityMessage: Locator;

  // Auth overlay (for unauthenticated)
  readonly authOverlay: Locator;

  // Loading
  readonly loadingSpinner: Locator;

  constructor(page: Page) {
    this.page = page;

    // Page elements
    this.pageTitle = page.locator('h1').filter({ hasText: /dashboard/i });
    this.streakBadge = page.locator('[class*="streak"]').or(page.locator('text=/\\d+\\s*days?/i'));

    // Stats cards
    this.totalSolvedCard = page.locator('[class*="stat"]').filter({ hasText: /solved|completed/i });
    this.hoursSpentCard = page.locator('[class*="stat"]').filter({ hasText: /hours|time/i });
    this.globalRankCard = page.locator('[class*="stat"]').filter({ hasText: /rank/i });
    this.skillPointsCard = page.locator('[class*="stat"]').filter({ hasText: /points|skill/i });

    // Activity chart
    this.activityChart = page.locator('[class*="chart"]').or(page.locator('svg').filter({ has: page.locator('path') }));
    this.chartPeriodSelector = page.locator('select').or(page.getByRole('combobox'));

    // Recent activity
    this.recentActivitySection = page.locator('h2').filter({ hasText: /recent|activity/i }).locator('..');
    this.recentActivityItems = page.locator('[class*="activity-item"]').or(page.locator('[data-testid="activity-item"]'));
    this.emptyActivityMessage = page.locator('text=/no.*activity|start.*learning/i');

    // Auth overlay
    this.authOverlay = page.locator('[class*="overlay"]').filter({ hasText: /sign in|login/i });

    // Loading
    this.loadingSpinner = page.locator('text=/loading/i').or(page.locator('[class*="animate-pulse"]'));
  }

  /**
   * Navigate to dashboard page
   */
  async goto() {
    await this.page.goto('/');
  }

  /**
   * Wait for page to load
   */
  async waitForLoad() {
    await Promise.race([
      this.pageTitle.waitFor({ state: 'visible', timeout: 15000 }),
      this.authOverlay.waitFor({ state: 'visible', timeout: 15000 }),
      this.loadingSpinner.waitFor({ state: 'hidden', timeout: 15000 }),
    ]).catch(() => {});
    await this.page.waitForTimeout(500);
  }

  /**
   * Check if auth overlay is visible
   */
  async isAuthOverlayVisible(): Promise<boolean> {
    return await this.authOverlay.isVisible().catch(() => false);
  }

  /**
   * Check if dashboard content is visible
   */
  async isDashboardVisible(): Promise<boolean> {
    return await this.pageTitle.isVisible().catch(() => false);
  }

  /**
   * Get streak value
   */
  async getStreakValue(): Promise<string | null> {
    const text = await this.streakBadge.textContent().catch(() => null);
    return text;
  }

  /**
   * Change chart period
   */
  async changeChartPeriod(days: number) {
    if (await this.chartPeriodSelector.isVisible()) {
      await this.chartPeriodSelector.selectOption({ label: new RegExp(`${days}`, 'i') });
    }
  }

  /**
   * Check if activity chart is visible
   */
  async isChartVisible(): Promise<boolean> {
    return await this.activityChart.isVisible().catch(() => false);
  }

  /**
   * Get recent activity count
   */
  async getRecentActivityCount(): Promise<number> {
    return await this.recentActivityItems.count();
  }

  /**
   * Check if empty activity message is shown
   */
  async isEmptyActivityShown(): Promise<boolean> {
    return await this.emptyActivityMessage.isVisible().catch(() => false);
  }
}
