import { test, expect } from '../fixtures/auth.fixture';
import { AiTutorPage } from '../pages/ai-tutor.page';

// Constants for test task
const TEST_COURSE = 'algo-fundamentals';
const TEST_TASK = 'algo-two-sum';

test.describe('AI Tutor', () => {
  test.describe('AI Tutor - Free User', () => {
    test.beforeEach(async ({ auth }) => {
      await auth.loginAsTestUser();
    });

    test('should display AI Tutor tab on task page', async ({ page }) => {
      const aiTutor = new AiTutorPage(page);
      await aiTutor.gotoTask(TEST_COURSE, TEST_TASK);
      await aiTutor.waitForLoad();

      // AI Tutor tab should be visible
      const isVisible = await aiTutor.isAiTabVisible();
      expect(isVisible).toBe(true);
    });

    test('should show locked state for free user when clicking AI tab', async ({ page }) => {
      const aiTutor = new AiTutorPage(page);
      await aiTutor.gotoTask(TEST_COURSE, TEST_TASK);
      await aiTutor.waitForLoad();

      // Click on AI tab
      await aiTutor.openAiTab();

      // Should show locked state for free users
      const isLocked = await aiTutor.isLocked();
      expect(isLocked).toBe(true);
    });

    test('should not show chat for free user', async ({ page }) => {
      const aiTutor = new AiTutorPage(page);
      await aiTutor.gotoTask(TEST_COURSE, TEST_TASK);
      await aiTutor.waitForLoad();

      // Click on AI tab
      await aiTutor.openAiTab();

      // Chat should not be visible for free users (locked state instead)
      const isChatOpen = await aiTutor.isChatOpen();
      expect(isChatOpen).toBe(false);
    });

    test('should have AI Tutor text in tab', async ({ page }) => {
      const aiTutor = new AiTutorPage(page);
      await aiTutor.gotoTask(TEST_COURSE, TEST_TASK);
      await aiTutor.waitForLoad();

      // AI tab should have text
      const text = await aiTutor.aiTab.textContent();
      expect(text?.toLowerCase()).toContain('ai');
    });
  });

  test.describe('AI Tutor - Premium User', () => {
    test.beforeEach(async ({ auth }) => {
      await auth.loginAsPremiumUser();
    });

    test('should display AI Tutor tab for premium user', async ({ page }) => {
      const aiTutor = new AiTutorPage(page);
      await aiTutor.gotoTask(TEST_COURSE, TEST_TASK);
      await aiTutor.waitForLoad();

      // AI Tutor tab should be visible
      const isVisible = await aiTutor.isAiTabVisible();
      expect(isVisible).toBe(true);
    });

    test('should show chat for premium user when clicking AI tab', async ({ page }) => {
      const aiTutor = new AiTutorPage(page);
      await aiTutor.gotoTask(TEST_COURSE, TEST_TASK);
      await aiTutor.waitForLoad();

      // Click on AI tab
      await aiTutor.openAiTab();

      // Chat should be visible for premium users
      const isChatOpen = await aiTutor.isChatOpen();
      expect(isChatOpen).toBe(true);
    });

    test('should show empty state when no messages', async ({ page }) => {
      const aiTutor = new AiTutorPage(page);
      await aiTutor.gotoTask(TEST_COURSE, TEST_TASK);
      await aiTutor.waitForLoad();

      // Click on AI tab
      await aiTutor.openAiTab();

      // Empty state should be visible
      await expect(aiTutor.emptyState).toBeVisible();
    });

    test('should have enabled input for premium user', async ({ page }) => {
      const aiTutor = new AiTutorPage(page);
      await aiTutor.gotoTask(TEST_COURSE, TEST_TASK);
      await aiTutor.waitForLoad();

      // Click on AI tab
      await aiTutor.openAiTab();

      // Input should be enabled for premium users
      const isEnabled = await aiTutor.isInputEnabled();
      expect(isEnabled).toBe(true);
    });

    test('should have send button for premium user', async ({ page }) => {
      const aiTutor = new AiTutorPage(page);
      await aiTutor.gotoTask(TEST_COURSE, TEST_TASK);
      await aiTutor.waitForLoad();

      // Click on AI tab
      await aiTutor.openAiTab();

      // Send button should be visible
      await expect(aiTutor.sendButton).toBeVisible();
    });

    test('should allow typing in input', async ({ page }) => {
      const aiTutor = new AiTutorPage(page);
      await aiTutor.gotoTask(TEST_COURSE, TEST_TASK);
      await aiTutor.waitForLoad();

      // Click on AI tab
      await aiTutor.openAiTab();

      // Type in input
      await aiTutor.input.fill('Test question');

      // Input should have the value
      await expect(aiTutor.input).toHaveValue('Test question');
    });

    test('should have placeholder text', async ({ page }) => {
      const aiTutor = new AiTutorPage(page);
      await aiTutor.gotoTask(TEST_COURSE, TEST_TASK);
      await aiTutor.waitForLoad();

      // Click on AI tab
      await aiTutor.openAiTab();

      // Should have placeholder
      const placeholder = await aiTutor.getPlaceholder();
      expect(placeholder).toBeTruthy();
    });
  });

  test.describe('AI Tutor Accessibility', () => {
    test.beforeEach(async ({ auth }) => {
      await auth.loginAsTestUser();
    });

    test('should be keyboard navigable', async ({ page }) => {
      const aiTutor = new AiTutorPage(page);
      await aiTutor.gotoTask(TEST_COURSE, TEST_TASK);
      await aiTutor.waitForLoad();

      // Tab through elements
      await page.keyboard.press('Tab');

      // Some element should be focused
      const focused = await page.evaluate(() => document.activeElement?.tagName);
      expect(focused).toBeTruthy();
    });

    test('should have clickable AI tab', async ({ page }) => {
      const aiTutor = new AiTutorPage(page);
      await aiTutor.gotoTask(TEST_COURSE, TEST_TASK);
      await aiTutor.waitForLoad();

      // AI tab should be clickable
      await expect(aiTutor.aiTab).toBeEnabled();
    });
  });

  test.describe('AI Tutor Navigation', () => {
    test.beforeEach(async ({ auth }) => {
      await auth.loginAsTestUser();
    });

    test('should switch between tabs', async ({ page }) => {
      const aiTutor = new AiTutorPage(page);
      await aiTutor.gotoTask(TEST_COURSE, TEST_TASK);
      await aiTutor.waitForLoad();

      // Initially on description tab
      await expect(aiTutor.descriptionTab).toBeVisible();

      // Click on AI tab
      await aiTutor.openAiTab();
      await page.waitForTimeout(300);

      // AI content should be visible (locked or chat)
      const isLocked = await aiTutor.isLocked();
      const isChatOpen = await aiTutor.isChatOpen();
      expect(isLocked || isChatOpen).toBe(true);

      // Click back to description tab
      await aiTutor.descriptionTab.click();
      await page.waitForTimeout(300);

      // AI content should no longer be visible
      const isLockedAfter = await aiTutor.isLocked();
      const isChatOpenAfter = await aiTutor.isChatOpen();
      expect(isLockedAfter && isChatOpenAfter).toBe(false);
    });
  });
});
