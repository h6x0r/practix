import { Page, Locator } from '@playwright/test';

/**
 * Page Object for Admin Dashboard page
 */
export class AdminPage {
  readonly page: Page;

  // Page elements
  readonly pageTitle: Locator;
  readonly loadingIndicator: Locator;

  // Access denied elements
  readonly loginRequired: Locator;
  readonly accessDenied: Locator;

  // Stats cards
  readonly totalUsersCard: Locator;
  readonly newUsersCard: Locator;
  readonly activeUsersCard: Locator;
  readonly totalSubmissionsCard: Locator;

  // Subscription stats
  readonly activeSubscriptionsCard: Locator;
  readonly subscriptionsByPlanSection: Locator;
  readonly monthlyRevenueCard: Locator;

  // Charts and tables
  readonly submissionsByStatusSection: Locator;
  readonly submissionsByDayChart: Locator;
  readonly coursePopularityTable: Locator;
  readonly topCoursesTable: Locator;
  readonly hardestTasksTable: Locator;

  constructor(page: Page) {
    this.page = page;

    // Page elements
    this.pageTitle = page.locator('h1').filter({ hasText: /admin|dashboard/i });
    this.loadingIndicator = page.locator('text=/loading/i').or(page.locator('[class*="animate-pulse"]'));

    // Access denied elements
    this.loginRequired = page.locator('text=/login.*required|sign in/i');
    this.accessDenied = page.locator('text=/access denied|not authorized|redirect/i');

    // Stats cards - look for labels and values
    this.totalUsersCard = page.locator('text=/total.*user/i').locator('..').or(page.locator('[class*="bg-blue-500"]').locator('..'));
    this.newUsersCard = page.locator('text=/new.*user/i').locator('..');
    this.activeUsersCard = page.locator('text=/active.*user/i').locator('..');
    this.totalSubmissionsCard = page.locator('text=/total.*submission/i').locator('..');

    // Subscription stats
    this.activeSubscriptionsCard = page.locator('text=/active.*subscription/i').locator('..');
    this.subscriptionsByPlanSection = page.locator('text=/subscription.*plan/i').locator('..').locator('..');
    this.monthlyRevenueCard = page.locator('text=/monthly.*revenue/i').locator('..');

    // Charts and tables
    this.submissionsByStatusSection = page.locator('text=/submission.*status/i').locator('..');
    this.submissionsByDayChart = page.locator('text=/submission.*day/i').locator('..').locator('..');
    this.coursePopularityTable = page.locator('text=/course.*popularity/i').locator('..');
    this.topCoursesTable = page.locator('text=/top.*course/i').locator('..');
    this.hardestTasksTable = page.locator('text=/hardest.*task/i').locator('..');
  }

  /**
   * Navigate to admin page
   */
  async goto() {
    await this.page.goto('/admin');
  }

  /**
   * Wait for page to load
   */
  async waitForLoad() {
    // Wait for either content to load or access denied
    await Promise.race([
      this.pageTitle.waitFor({ state: 'visible', timeout: 15000 }),
      this.loginRequired.waitFor({ state: 'visible', timeout: 15000 }),
      this.loadingIndicator.waitFor({ state: 'hidden', timeout: 15000 }),
    ]).catch(() => {});
    await this.page.waitForTimeout(500);
  }

  /**
   * Check if admin dashboard is visible
   */
  async isAdminDashboardVisible(): Promise<boolean> {
    // Check for admin-specific content
    const pageText = await this.page.textContent('body');
    return pageText?.toLowerCase().includes('admin') || false;
  }

  /**
   * Check if access is denied (non-admin user)
   */
  async isAccessDenied(): Promise<boolean> {
    const url = this.page.url();
    // Check if redirected away from /admin
    if (!url.includes('/admin')) {
      return true;
    }

    const hasLoginRequired = await this.loginRequired.isVisible().catch(() => false);
    return hasLoginRequired;
  }

  /**
   * Check if loading
   */
  async isLoading(): Promise<boolean> {
    return await this.loadingIndicator.isVisible().catch(() => false);
  }

  /**
   * Get total users count from card
   */
  async getTotalUsersCount(): Promise<string | null> {
    const text = await this.totalUsersCard.textContent().catch(() => null);
    if (!text) return null;

    // Extract number
    const match = text.match(/\d[\d,]*/);
    return match ? match[0] : null;
  }

  /**
   * Get active subscriptions count
   */
  async getActiveSubscriptionsCount(): Promise<string | null> {
    const text = await this.activeSubscriptionsCard.textContent().catch(() => null);
    if (!text) return null;

    const match = text.match(/\d[\d,]*/);
    return match ? match[0] : null;
  }

  /**
   * Check if charts are visible
   */
  async areChartsVisible(): Promise<boolean> {
    // Look for SVG elements (Recharts) or chart containers
    const svgElements = await this.page.locator('svg').count();
    const chartContainers = await this.page.locator('[class*="recharts"]').count();

    return svgElements > 0 || chartContainers > 0;
  }

  /**
   * Check if tables are visible
   */
  async areTablesVisible(): Promise<boolean> {
    const courseTable = await this.coursePopularityTable.isVisible().catch(() => false);
    const topTable = await this.topCoursesTable.isVisible().catch(() => false);
    const hardestTable = await this.hardestTasksTable.isVisible().catch(() => false);

    return courseTable || topTable || hardestTable;
  }

  /**
   * Get stats cards count
   */
  async getStatsCardsCount(): Promise<number> {
    // Look for stat card containers - they have specific class patterns
    const cards = this.page.locator('[class*="rounded-2xl"][class*="border"]').filter({
      has: this.page.locator('[class*="text-3xl"]'),
    });
    return await cards.count();
  }
}
