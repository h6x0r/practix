import { Page, Locator, expect } from '@playwright/test';

/**
 * Page Object for Payment pages
 */
export class PaymentsPage {
  readonly page: Page;

  // Page elements
  readonly pageTitle: Locator;
  readonly subscriptionPlans: Locator;
  readonly globalPremiumPlan: Locator;
  readonly coursePremiumPlan: Locator;
  readonly roadmapCreditsPlan: Locator;

  // Tabs
  readonly subscribeTab: Locator;
  readonly purchasesTab: Locator;
  readonly historyTab: Locator;

  // Provider selection
  readonly paymeButton: Locator;
  readonly clickButton: Locator;

  // Checkout
  readonly checkoutButton: Locator;

  // Payment history
  readonly historyTable: Locator;
  readonly historyEmpty: Locator;
  readonly historyRows: Locator;

  // Status
  readonly subscriptionActive: Locator;
  readonly currentPlanName: Locator;

  // Loading
  readonly loadingSpinner: Locator;
  readonly errorMessage: Locator;

  constructor(page: Page) {
    this.page = page;

    // Page elements
    this.pageTitle = page.getByTestId('payments-title');
    this.subscriptionPlans = page.getByTestId('subscription-plans');
    this.globalPremiumPlan = page.getByTestId('plan-global-premium');
    this.coursePremiumPlan = page.getByTestId('plan-course-premium');
    this.roadmapCreditsPlan = page.getByTestId('plan-roadmap-credits');

    // Tabs
    this.subscribeTab = page.getByTestId('subscribe-tab');
    this.purchasesTab = page.getByTestId('purchases-tab');
    this.historyTab = page.getByTestId('history-tab');

    // Provider buttons
    this.paymeButton = page.getByTestId('provider-payme');
    this.clickButton = page.getByTestId('provider-click');

    // Checkout
    this.checkoutButton = page.getByTestId('checkout-button');

    // History
    this.historyTable = page.getByTestId('payment-history-table');
    this.historyEmpty = page.getByTestId('history-empty');
    this.historyRows = page.getByTestId('payment-history-row');

    // Status
    this.subscriptionActive = page.getByTestId('subscription-active');
    this.currentPlanName = page.getByTestId('current-plan-name');

    // Loading
    this.loadingSpinner = page.getByTestId('loading-spinner');
    this.errorMessage = page.getByTestId('error-message');
  }

  /**
   * Navigate to payments page
   */
  async goto() {
    await this.page.goto('/payments');
  }

  /**
   * Wait for page to load (loading spinner disappears)
   */
  async waitForLoad() {
    // Wait for either loading to finish or content to appear
    await Promise.race([
      this.loadingSpinner.waitFor({ state: 'hidden', timeout: 15000 }),
      this.pageTitle.waitFor({ state: 'visible', timeout: 15000 }),
    ]).catch(() => {
      // Ignore timeout - page may have loaded differently
    });
    // Give extra time for content to render
    await this.page.waitForTimeout(500);
  }

  /**
   * Select a subscription plan
   */
  async selectPlan(planType: 'global' | 'course' | 'roadmap') {
    switch (planType) {
      case 'global':
        if (await this.globalPremiumPlan.isVisible()) {
          await this.globalPremiumPlan.click();
        }
        break;
      case 'course':
        if (await this.coursePremiumPlan.isVisible()) {
          await this.coursePremiumPlan.click();
        }
        break;
      case 'roadmap':
        // First go to purchases tab
        await this.purchasesTab.click();
        await this.page.waitForTimeout(300);
        if (await this.roadmapCreditsPlan.isVisible()) {
          await this.roadmapCreditsPlan.click();
        }
        break;
    }
  }

  /**
   * Select payment provider
   */
  async selectProvider(provider: 'payme' | 'click') {
    if (provider === 'payme') {
      if (await this.paymeButton.isVisible()) {
        await this.paymeButton.click();
      }
    } else {
      if (await this.clickButton.isVisible()) {
        await this.clickButton.click();
      }
    }
  }

  /**
   * Go to payment history tab
   */
  async goToHistory() {
    await this.historyTab.click();
    await this.page.waitForTimeout(300);
  }

  /**
   * Go to purchases tab
   */
  async goToPurchases() {
    await this.purchasesTab.click();
    await this.page.waitForTimeout(300);
  }

  /**
   * Initiate checkout
   */
  async initiateCheckout() {
    await this.checkoutButton.click();
  }

  /**
   * Check if user has active subscription
   */
  async hasActiveSubscription(): Promise<boolean> {
    return (await this.subscriptionActive.count()) > 0;
  }

  /**
   * Get current plan name
   */
  async getCurrentPlanName(): Promise<string | null> {
    if ((await this.currentPlanName.count()) > 0) {
      return await this.currentPlanName.textContent();
    }
    return null;
  }

  /**
   * Get payment history items count
   */
  async getHistoryCount(): Promise<number> {
    await this.goToHistory();
    await this.page.waitForTimeout(300);

    // Check if empty state is shown
    if (await this.historyEmpty.isVisible()) {
      return 0;
    }

    return await this.historyRows.count();
  }

  /**
   * Check if Global Premium plan is selected
   */
  async isGlobalPlanSelected(): Promise<boolean> {
    const selected = await this.globalPremiumPlan.getAttribute('data-selected');
    return selected === 'true';
  }

  /**
   * Check if a provider is configured
   */
  async isProviderConfigured(provider: 'payme' | 'click'): Promise<boolean> {
    const button = provider === 'payme' ? this.paymeButton : this.clickButton;
    if (!(await button.isVisible())) return false;
    return !(await button.isDisabled());
  }
}
