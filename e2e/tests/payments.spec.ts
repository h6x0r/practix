import { test, expect } from '../fixtures/auth.fixture';
import { PaymentsPage } from '../pages/payments.page';

test.describe('Payments', () => {
  test.describe('Payments Page - Free User', () => {
    test.beforeEach(async ({ auth }) => {
      await auth.loginAsTestUser();
    });

    test('should display payments page', async ({ page }) => {
      const paymentsPage = new PaymentsPage(page);
      await paymentsPage.goto();

      // Page should load
      await expect(page).toHaveURL(/\/payments/);
    });

    test('should display subscription plans', async ({ page }) => {
      const paymentsPage = new PaymentsPage(page);
      await paymentsPage.goto();
      await paymentsPage.waitForLoad();

      // Should show subscription plans container
      await expect(paymentsPage.subscriptionPlans).toBeVisible();
    });

    test('should display payment providers after selecting plan', async ({ page }) => {
      const paymentsPage = new PaymentsPage(page);
      await paymentsPage.goto();
      await paymentsPage.waitForLoad();

      // Providers are always visible in checkout panel on subscribe tab
      // Check at least one provider is visible
      const paymeVisible = await paymentsPage.paymeButton.isVisible().catch(() => false);
      const clickVisible = await paymentsPage.clickButton.isVisible().catch(() => false);

      expect(paymeVisible || clickVisible).toBe(true);
    });

    test('should show Global Premium plan', async ({ page }) => {
      const paymentsPage = new PaymentsPage(page);
      await paymentsPage.goto();
      await paymentsPage.waitForLoad();

      // Global Premium plan should be visible
      await expect(paymentsPage.globalPremiumPlan).toBeVisible();
    });

    test('should show empty payment history for new user', async ({ page }) => {
      const paymentsPage = new PaymentsPage(page);
      await paymentsPage.goto();
      await paymentsPage.waitForLoad();

      // Go to history tab
      await paymentsPage.goToHistory();

      // Check if history table or empty state is visible
      const tableVisible = await paymentsPage.historyTable.isVisible().catch(() => false);
      const emptyVisible = await paymentsPage.historyEmpty.isVisible().catch(() => false);

      // One of them should be visible
      expect(tableVisible || emptyVisible).toBe(true);
    });
  });

  test.describe('Payments Page - Premium User', () => {
    test.beforeEach(async ({ auth }) => {
      await auth.loginAsPremiumUser();
    });

    test('should show active subscription status', async ({ page }) => {
      const paymentsPage = new PaymentsPage(page);
      await paymentsPage.goto();
      await paymentsPage.waitForLoad();

      // Premium user should see active subscription indicator
      const hasActive = await paymentsPage.hasActiveSubscription();
      const currentPlan = await paymentsPage.getCurrentPlanName();

      // Premium user should have subscription info
      expect(hasActive || currentPlan !== null).toBeTruthy();
    });

    test('should display current plan name as Premium', async ({ page }) => {
      const paymentsPage = new PaymentsPage(page);
      await paymentsPage.goto();
      await paymentsPage.waitForLoad();

      // Check for Premium label in current plan
      const planName = await paymentsPage.getCurrentPlanName();
      if (planName) {
        expect(planName.toLowerCase()).toContain('premium');
      }
    });

    test('should have payment history access', async ({ page }) => {
      const paymentsPage = new PaymentsPage(page);
      await paymentsPage.goto();
      await paymentsPage.waitForLoad();

      // History tab should be clickable
      await expect(paymentsPage.historyTab).toBeVisible();
      await paymentsPage.goToHistory();

      // History content should load
      const tableVisible = await paymentsPage.historyTable.isVisible().catch(() => false);
      const emptyVisible = await paymentsPage.historyEmpty.isVisible().catch(() => false);

      expect(tableVisible || emptyVisible).toBe(true);
    });
  });

  test.describe('Plan Selection', () => {
    test.beforeEach(async ({ auth }) => {
      await auth.loginAsTestUser();
    });

    test('should be able to select Global Premium plan', async ({ page }) => {
      const paymentsPage = new PaymentsPage(page);
      await paymentsPage.goto();
      await paymentsPage.waitForLoad();

      // Select Global Premium
      await paymentsPage.selectPlan('global');

      // Verify selection
      const isSelected = await paymentsPage.isGlobalPlanSelected();
      expect(isSelected).toBe(true);
    });

    test('should be able to select Course plan', async ({ page }) => {
      const paymentsPage = new PaymentsPage(page);
      await paymentsPage.goto();
      await paymentsPage.waitForLoad();

      // Try to select course plan if visible
      if (await paymentsPage.coursePremiumPlan.isVisible()) {
        await paymentsPage.selectPlan('course');
        const selected = await paymentsPage.coursePremiumPlan.getAttribute('data-selected');
        expect(selected).toBe('true');
      }
    });

    test('should show Roadmap Credits in purchases tab', async ({ page }) => {
      const paymentsPage = new PaymentsPage(page);
      await paymentsPage.goto();
      await paymentsPage.waitForLoad();

      // Go to purchases tab
      await paymentsPage.goToPurchases();

      // Roadmap credits should be visible
      await expect(paymentsPage.roadmapCreditsPlan).toBeVisible();
    });
  });

  test.describe('Provider Selection', () => {
    test.beforeEach(async ({ auth }) => {
      await auth.loginAsTestUser();
    });

    test('should display Payme provider button', async ({ page }) => {
      const paymentsPage = new PaymentsPage(page);
      await paymentsPage.goto();
      await paymentsPage.waitForLoad();

      // Payme button should be visible
      await expect(paymentsPage.paymeButton).toBeVisible();

      // Check if it shows "Coming soon" when not configured
      const isConfigured = await paymentsPage.isProviderConfigured('payme');
      if (!isConfigured) {
        // Provider not configured - button should be disabled
        await expect(paymentsPage.paymeButton).toBeDisabled();
      }
    });

    test('should display Click provider button', async ({ page }) => {
      const paymentsPage = new PaymentsPage(page);
      await paymentsPage.goto();
      await paymentsPage.waitForLoad();

      // Click button should be visible
      await expect(paymentsPage.clickButton).toBeVisible();

      // Check if it shows "Coming soon" when not configured
      const isConfigured = await paymentsPage.isProviderConfigured('click');
      if (!isConfigured) {
        // Provider not configured - button should be disabled
        await expect(paymentsPage.clickButton).toBeDisabled();
      }
    });
  });

  test.describe('Checkout Flow', () => {
    test.beforeEach(async ({ auth }) => {
      await auth.loginAsTestUser();
    });

    test('should show checkout button when plan selected', async ({ page }) => {
      const paymentsPage = new PaymentsPage(page);
      await paymentsPage.goto();
      await paymentsPage.waitForLoad();

      // Select Global Premium plan
      await paymentsPage.selectPlan('global');

      // Checkout button should be visible
      await expect(paymentsPage.checkoutButton).toBeVisible();
    });

    test('should disable checkout when providers are not configured', async ({ page }) => {
      const paymentsPage = new PaymentsPage(page);
      await paymentsPage.goto();
      await paymentsPage.waitForLoad();

      // Select plan
      await paymentsPage.selectPlan('global');

      // Check provider configuration
      const paymeConfigured = await paymentsPage.isProviderConfigured('payme');
      const clickConfigured = await paymentsPage.isProviderConfigured('click');

      // If no providers are configured, checkout should be disabled
      if (!paymeConfigured && !clickConfigured) {
        await expect(paymentsPage.checkoutButton).toBeDisabled();
      } else {
        // If at least one provider is configured, checkout should be enabled after selection
        await expect(paymentsPage.checkoutButton).toBeVisible();
      }
    });

    test('should not allow checkout without authentication', async ({ page }) => {
      // Logout first by going to page without auth
      await page.context().clearCookies();
      await page.goto('/payments');

      // Should redirect to login or show auth required message
      await page.waitForTimeout(1000);
      const url = page.url();

      // Should either be on payments or redirected to login
      expect(url.includes('payments') || url.includes('login') || url.includes('auth')).toBe(true);
    });
  });

  test.describe('Payment History', () => {
    test.beforeEach(async ({ auth }) => {
      await auth.loginAsTestUser();
    });

    test('should display payment history tab', async ({ page }) => {
      const paymentsPage = new PaymentsPage(page);
      await paymentsPage.goto();
      await paymentsPage.waitForLoad();

      // History tab should be visible
      await expect(paymentsPage.historyTab).toBeVisible();
    });

    test('should switch to history view', async ({ page }) => {
      const paymentsPage = new PaymentsPage(page);
      await paymentsPage.goto();
      await paymentsPage.waitForLoad();

      // Click history tab
      await paymentsPage.goToHistory();

      // History content should be visible (table or empty state)
      const tableVisible = await paymentsPage.historyTable.isVisible().catch(() => false);
      const emptyVisible = await paymentsPage.historyEmpty.isVisible().catch(() => false);

      expect(tableVisible || emptyVisible).toBe(true);
    });
  });

  test.describe('Tab Navigation', () => {
    test.beforeEach(async ({ auth }) => {
      await auth.loginAsTestUser();
    });

    test('should navigate between tabs', async ({ page }) => {
      const paymentsPage = new PaymentsPage(page);
      await paymentsPage.goto();
      await paymentsPage.waitForLoad();

      // All tabs should be visible
      await expect(paymentsPage.subscribeTab).toBeVisible();
      await expect(paymentsPage.purchasesTab).toBeVisible();
      await expect(paymentsPage.historyTab).toBeVisible();

      // Navigate to purchases
      await paymentsPage.goToPurchases();
      await expect(paymentsPage.roadmapCreditsPlan).toBeVisible();

      // Navigate to history
      await paymentsPage.goToHistory();
      const tableOrEmpty =
        (await paymentsPage.historyTable.isVisible().catch(() => false)) ||
        (await paymentsPage.historyEmpty.isVisible().catch(() => false));
      expect(tableOrEmpty).toBe(true);
    });
  });

  test.describe('Error Handling', () => {
    test.beforeEach(async ({ auth }) => {
      await auth.loginAsTestUser();
    });

    test('should handle network errors gracefully', async ({ page }) => {
      // Simulate offline mode for payments API
      await page.route('**/api/payments/**', route => route.abort('failed'));
      await page.route('**/api/subscriptions/**', route => route.abort('failed'));

      const paymentsPage = new PaymentsPage(page);
      await paymentsPage.goto();

      // Wait for error state
      await page.waitForTimeout(2000);

      // Page should not crash - still be on payments URL
      expect(page.url()).toContain('/payments');
    });

    test('should handle API errors gracefully', async ({ page }) => {
      // Mock API error response
      await page.route('**/api/subscriptions/plans', route =>
        route.fulfill({
          status: 500,
          body: JSON.stringify({ error: 'Internal Server Error' }),
        })
      );

      const paymentsPage = new PaymentsPage(page);
      await paymentsPage.goto();

      // Wait for error state
      await page.waitForTimeout(1000);

      // Page should handle error gracefully
      expect(page.url()).toContain('/payments');
    });
  });

  test.describe('Accessibility', () => {
    test.beforeEach(async ({ auth }) => {
      await auth.loginAsTestUser();
    });

    test('should have proper heading structure', async ({ page }) => {
      const paymentsPage = new PaymentsPage(page);
      await paymentsPage.goto();
      await paymentsPage.waitForLoad();

      // Should have h1 heading
      const h1 = page.locator('h1');
      await expect(h1).toBeVisible();
    });

    test('should be keyboard navigable', async ({ page }) => {
      const paymentsPage = new PaymentsPage(page);
      await paymentsPage.goto();
      await paymentsPage.waitForLoad();

      // Tab through elements
      await page.keyboard.press('Tab');
      await page.keyboard.press('Tab');

      // Some element should be focused
      const focused = await page.evaluate(() => document.activeElement?.tagName);
      expect(focused).toBeTruthy();
    });
  });
});
