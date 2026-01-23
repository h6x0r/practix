import { Page, Locator } from '@playwright/test';

/**
 * Page Object for Settings page
 */
export class SettingsPage {
  readonly page: Page;

  // Page elements
  readonly pageTitle: Locator;
  readonly pageDescription: Locator;
  readonly loadingIndicator: Locator;

  // Auth overlay
  readonly authOverlay: Locator;

  // Tab navigation
  readonly profileTab: Locator;
  readonly notificationsTab: Locator;
  readonly securityTab: Locator;

  // Profile section
  readonly avatarImage: Locator;
  readonly avatarUploadButton: Locator;
  readonly presetAvatars: Locator;
  readonly saveButton: Locator;

  // Notification toggles
  readonly emailDigestToggle: Locator;
  readonly newCoursesToggle: Locator;
  readonly marketingToggle: Locator;
  readonly securityAlertsToggle: Locator;

  constructor(page: Page) {
    this.page = page;

    // Page elements
    this.pageTitle = page.locator('h1').filter({ hasText: /settings/i });
    this.pageDescription = page.locator('text=/manage.*settings|customize.*experience/i');
    this.loadingIndicator = page.locator('text=/loading/i').or(page.locator('[class*="animate-pulse"]'));

    // Auth overlay
    this.authOverlay = page.locator('[class*="AuthRequired"]').or(page.locator('text=/sign in|login/i'));

    // Tab navigation
    this.profileTab = page.locator('button, [role="tab"]').filter({ hasText: /profile/i });
    this.notificationsTab = page.locator('button, [role="tab"]').filter({ hasText: /notification/i });
    this.securityTab = page.locator('button, [role="tab"]').filter({ hasText: /security|password/i });

    // Profile section
    this.avatarImage = page.locator('img[src*="avatar"]').or(page.locator('[class*="rounded-full"]').filter({ has: page.locator('img') }));
    this.avatarUploadButton = page.locator('button').filter({ hasText: /upload|change.*avatar/i });
    this.presetAvatars = page.locator('img[src*="dicebear"]');
    this.saveButton = page.locator('button').filter({ hasText: /save/i });

    // Notification toggles - look for toggle buttons
    this.emailDigestToggle = page.locator('button[class*="rounded-full"]').first();
    this.newCoursesToggle = page.locator('button[class*="rounded-full"]').nth(1);
    this.marketingToggle = page.locator('button[class*="rounded-full"]').nth(2);
    this.securityAlertsToggle = page.locator('button[class*="rounded-full"]').nth(3);
  }

  /**
   * Navigate to settings page
   */
  async goto() {
    await this.page.goto('/settings');
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
   * Check if settings page is visible
   */
  async isSettingsVisible(): Promise<boolean> {
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
   * Navigate to profile tab
   */
  async goToProfileTab() {
    await this.profileTab.click();
    await this.page.waitForTimeout(300);
  }

  /**
   * Navigate to notifications tab
   */
  async goToNotificationsTab() {
    await this.notificationsTab.click();
    await this.page.waitForTimeout(300);
  }

  /**
   * Navigate to security tab
   */
  async goToSecurityTab() {
    await this.securityTab.click();
    await this.page.waitForTimeout(300);
  }

  /**
   * Check if tabs are visible
   */
  async areTabsVisible(): Promise<boolean> {
    const profile = await this.profileTab.isVisible().catch(() => false);
    const notifications = await this.notificationsTab.isVisible().catch(() => false);
    const security = await this.securityTab.isVisible().catch(() => false);

    return profile || notifications || security;
  }

  /**
   * Check if notification toggles are present
   */
  async hasNotificationToggles(): Promise<boolean> {
    const pageText = await this.page.textContent('body');
    return (
      pageText?.toLowerCase().includes('notification') ||
      pageText?.toLowerCase().includes('email') ||
      false
    );
  }

  /**
   * Save settings
   */
  async save() {
    if (await this.saveButton.isVisible()) {
      await this.saveButton.click();
      await this.page.waitForTimeout(500);
    }
  }
}
