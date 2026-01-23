import { Page, Locator } from '@playwright/test';

/**
 * Page Object for My Tasks page
 */
export class MyTasksPage {
  readonly page: Page;

  // Page elements
  readonly pageTitle: Locator;
  readonly pageDescription: Locator;
  readonly loadingIndicator: Locator;

  // Auth overlay
  readonly authOverlay: Locator;

  // Empty state
  readonly emptyStateTitle: Locator;
  readonly emptyStateDescription: Locator;
  readonly browseCoursesButton: Locator;

  // Course cards
  readonly courseCards: Locator;
  readonly progressBars: Locator;
  readonly continueButtons: Locator;

  constructor(page: Page) {
    this.page = page;

    // Page elements
    this.pageTitle = page.locator('h1').filter({ hasText: /my tasks|мои задачи/i });
    this.pageDescription = page.locator('text=/continue where you left off|продолжайте там/i');
    this.loadingIndicator = page.locator('text=/loading/i').or(page.locator('[class*="animate-pulse"]'));

    // Auth overlay
    this.authOverlay = page.locator('[class*="AuthRequired"]').or(page.locator('text=/sign in|login/i'));

    // Empty state
    this.emptyStateTitle = page.locator('h2').filter({ hasText: /no active courses|нет активных/i });
    this.emptyStateDescription = page.locator('text=/start a course|начните курс/i');
    this.browseCoursesButton = page.locator('a').filter({ hasText: /browse courses|выбрать курс/i });

    // Course cards
    this.courseCards = page.locator('[class*="rounded-3xl"]').filter({ has: page.locator('[class*="progress"]') });
    this.progressBars = page.locator('[class*="bg-gradient-to-r"][style*="width"]');
    this.continueButtons = page.locator('a').filter({ hasText: /continue|продолжить/i });
  }

  /**
   * Navigate to my tasks page
   */
  async goto() {
    await this.page.goto('/my-tasks');
  }

  /**
   * Wait for page to load
   */
  async waitForLoad() {
    // Wait for either content or auth overlay
    await Promise.race([
      this.pageTitle.waitFor({ state: 'visible', timeout: 15000 }),
      this.authOverlay.waitFor({ state: 'visible', timeout: 15000 }),
      this.emptyStateTitle.waitFor({ state: 'visible', timeout: 15000 }),
      this.loadingIndicator.waitFor({ state: 'hidden', timeout: 15000 }),
    ]).catch(() => {});
    await this.page.waitForTimeout(500);
  }

  /**
   * Check if my tasks page is visible
   */
  async isMyTasksVisible(): Promise<boolean> {
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
   * Check if empty state is visible
   */
  async isEmptyStateVisible(): Promise<boolean> {
    return await this.emptyStateTitle.isVisible().catch(() => false);
  }

  /**
   * Get course card count
   */
  async getCourseCount(): Promise<number> {
    // Count course cards with continue buttons
    const cards = this.page.locator('[class*="rounded-3xl"]').filter({
      has: this.page.locator('a').filter({ hasText: /continue|продолжить/i }),
    });
    return await cards.count();
  }

  /**
   * Click browse courses button (from empty state)
   */
  async clickBrowseCourses() {
    await this.browseCoursesButton.click();
    await this.page.waitForURL(/\/courses/);
  }

  /**
   * Click continue on first course
   */
  async clickContinueFirstCourse() {
    await this.continueButtons.first().click();
    await this.page.waitForURL(/\/course\//);
  }

  /**
   * Get first course title
   */
  async getFirstCourseTitle(): Promise<string | null> {
    const firstCard = this.page.locator('[class*="rounded-3xl"]').first();
    const title = firstCard.locator('h3').first();
    return await title.textContent().catch(() => null);
  }

  /**
   * Check if progress bars are visible
   */
  async hasProgressBars(): Promise<boolean> {
    const progressText = await this.page.locator('text=/progress|%/i').first().isVisible().catch(() => false);
    return progressText;
  }
}
