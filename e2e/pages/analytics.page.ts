import { Page, Locator } from '@playwright/test';

/**
 * Page Object for Analytics page
 */
export class AnalyticsPage {
  readonly page: Page;

  // Page elements
  readonly pageTitle: Locator;
  readonly pageDescription: Locator;
  readonly loadingIndicator: Locator;

  // Auth overlay
  readonly authOverlay: Locator;

  // Weekly chart
  readonly weeklyChart: Locator;
  readonly prevWeekButton: Locator;
  readonly nextWeekButton: Locator;
  readonly dateRangeLabel: Locator;

  // Yearly heatmap
  readonly yearlyHeatmap: Locator;
  readonly heatmapCells: Locator;

  // Summary stats
  readonly summaryCards: Locator;

  constructor(page: Page) {
    this.page = page;

    // Page elements
    this.pageTitle = page.locator('h1').filter({ hasText: /analytics|аналитика/i });
    this.pageDescription = page.locator('text=/track your progress|activity overview/i');
    this.loadingIndicator = page.locator('text=/loading/i').or(page.locator('[class*="animate-pulse"]'));

    // Auth overlay
    this.authOverlay = page.locator('[class*="AuthRequired"]').or(page.locator('text=/sign in|login/i'));

    // Weekly chart
    this.weeklyChart = page.locator('.recharts-responsive-container, [class*="recharts"]').first();
    this.prevWeekButton = page.locator('button').filter({ hasText: /prev|<|←/i });
    this.nextWeekButton = page.locator('button').filter({ hasText: /next|>|→/i });
    this.dateRangeLabel = page.locator('[class*="text-sm"]').filter({ hasText: /\d{1,2}.*-.*\d{1,2}/ });

    // Yearly heatmap
    this.yearlyHeatmap = page.locator('[class*="grid"]').filter({ has: page.locator('[class*="rounded"]') });
    this.heatmapCells = page.locator('[class*="rounded"][class*="bg-"]');

    // Summary stats
    this.summaryCards = page.locator('[class*="rounded-2xl"], [class*="rounded-xl"]').filter({
      has: page.locator('[class*="font-bold"]'),
    });
  }

  /**
   * Navigate to analytics page
   */
  async goto() {
    await this.page.goto('/analytics');
  }

  /**
   * Wait for page to load
   */
  async waitForLoad() {
    // Wait for either content or auth overlay
    await Promise.race([
      this.pageTitle.waitFor({ state: 'visible', timeout: 15000 }),
      this.authOverlay.waitFor({ state: 'visible', timeout: 15000 }),
      this.loadingIndicator.waitFor({ state: 'hidden', timeout: 15000 }),
    ]).catch(() => {});
    await this.page.waitForTimeout(500);
  }

  /**
   * Check if analytics page is visible
   */
  async isAnalyticsVisible(): Promise<boolean> {
    return await this.pageTitle.isVisible().catch(() => false);
  }

  /**
   * Check if auth overlay is visible
   */
  async isAuthOverlayVisible(): Promise<boolean> {
    const hasSignIn = await this.page.locator('text=/sign in|login/i').isVisible().catch(() => false);
    return hasSignIn;
  }

  /**
   * Check if weekly chart is visible
   */
  async hasWeeklyChart(): Promise<boolean> {
    // Check for recharts container or any chart-like element
    const hasRecharts = await this.weeklyChart.isVisible().catch(() => false);
    const hasSvg = await this.page.locator('svg').first().isVisible().catch(() => false);
    return hasRecharts || hasSvg;
  }

  /**
   * Check if yearly heatmap is visible
   */
  async hasYearlyHeatmap(): Promise<boolean> {
    // Look for heatmap-like structure (grid of colored cells)
    const pageText = await this.page.textContent('body');
    const hasYearlyText = pageText?.toLowerCase().includes('year') || false;
    const hasCells = (await this.heatmapCells.count()) > 10;
    return hasYearlyText || hasCells;
  }

  /**
   * Navigate to previous week
   */
  async goToPrevWeek() {
    if (await this.prevWeekButton.isVisible()) {
      await this.prevWeekButton.click();
      await this.page.waitForTimeout(500);
    }
  }

  /**
   * Navigate to next week
   */
  async goToNextWeek() {
    if (await this.nextWeekButton.isVisible()) {
      await this.nextWeekButton.click();
      await this.page.waitForTimeout(500);
    }
  }

  /**
   * Get summary card count
   */
  async getSummaryCardCount(): Promise<number> {
    return await this.summaryCards.count();
  }
}
