import { Page, Locator } from '@playwright/test';

/**
 * Page Object for Leaderboard page
 */
export class LeaderboardPage {
  readonly page: Page;

  // Page elements
  readonly pageTitle: Locator;
  readonly loadingIndicator: Locator;

  // Auth overlay
  readonly authOverlay: Locator;

  // My stats card (for logged in users)
  readonly myStatsCard: Locator;
  readonly myRank: Locator;
  readonly myLevel: Locator;
  readonly myStreak: Locator;
  readonly progressBar: Locator;

  // Leaderboard table
  readonly leaderboardTable: Locator;
  readonly leaderboardEntries: Locator;
  readonly topThree: Locator;

  // Empty state
  readonly emptyState: Locator;

  constructor(page: Page) {
    this.page = page;

    // Page elements
    this.pageTitle = page.locator('h1').filter({ hasText: /leaderboard/i });
    this.loadingIndicator = page.locator('text=/loading/i').or(page.locator('[class*="animate-pulse"]'));

    // Auth overlay
    this.authOverlay = page.locator('[class*="AuthRequired"]').or(page.locator('text=/sign in|login/i'));

    // My stats card
    this.myStatsCard = page.locator('[class*="gradient"]').filter({ hasText: /level|xp/i }).first();
    this.myRank = page.locator('text=/#\\d+/').first();
    this.myLevel = page.locator('text=/level \\d+/i').first();
    this.myStreak = page.locator('[class*="fire"]').or(page.locator('text=/\\d+ day/i'));
    this.progressBar = page.locator('[class*="progress"]').or(page.locator('[class*="h-2"][class*="bg"]'));

    // Leaderboard table
    this.leaderboardTable = page.locator('[class*="rounded-2xl"]').filter({ hasText: /top|coders/i });
    this.leaderboardEntries = page.locator('[class*="divide-y"]').locator('> div');
    this.topThree = page.locator('text=/🥇|🥈|🥉/');

    // Empty state
    this.emptyState = page.locator('text=/no entries|be the first/i');
  }

  /**
   * Navigate to leaderboard page
   */
  async goto() {
    await this.page.goto('/leaderboard');
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
   * Check if leaderboard is visible
   */
  async isLeaderboardVisible(): Promise<boolean> {
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
   * Check if my stats card is visible
   */
  async isMyStatsVisible(): Promise<boolean> {
    // Look for the gradient card with user stats
    const hasGradientCard = await this.page.locator('[class*="gradient"]').filter({ has: this.page.locator('text=/level/i') }).isVisible().catch(() => false);
    return hasGradientCard;
  }

  /**
   * Get leaderboard entry count
   */
  async getEntryCount(): Promise<number> {
    // Count rows in the leaderboard table
    const entries = this.page.locator('[class*="divide-y"] > div').filter({ has: this.page.locator('[class*="rounded-full"]') });
    return await entries.count();
  }

  /**
   * Check if top 3 medals are shown
   */
  async hasTopThreeMedals(): Promise<boolean> {
    const goldMedal = await this.page.locator('text=🥇').isVisible().catch(() => false);
    const silverMedal = await this.page.locator('text=🥈').isVisible().catch(() => false);
    const bronzeMedal = await this.page.locator('text=🥉').isVisible().catch(() => false);

    return goldMedal || silverMedal || bronzeMedal;
  }

  /**
   * Check if "You" label is visible (current user highlighted)
   */
  async isCurrentUserHighlighted(): Promise<boolean> {
    const youLabel = await this.page.locator('text=/you/i').isVisible().catch(() => false);
    const highlightedRow = await this.page.locator('[class*="brand-50"]').or(this.page.locator('[class*="brand-900"]')).isVisible().catch(() => false);

    return youLabel || highlightedRow;
  }

  /**
   * Get first entry name
   */
  async getFirstEntryName(): Promise<string | null> {
    const firstEntry = this.page.locator('[class*="divide-y"] > div').first();
    const nameElement = firstEntry.locator('[class*="font-medium"]').first();
    return await nameElement.textContent().catch(() => null);
  }
}
