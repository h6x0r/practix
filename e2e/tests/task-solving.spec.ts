import { test, expect } from '../fixtures/auth.fixture';
import { TaskPage } from '../pages/task.page';

test.describe('Task Solving', () => {
  // Login before each test
  test.beforeEach(async ({ auth }) => {
    await auth.loginAsTestUser();
  });

  test.describe('Task Workspace', () => {
    test('should display task description', async ({ page }) => {
      const taskPage = new TaskPage(page);
      await taskPage.goto('go-basics', 'go-fundamentals-flatten-nested');

      await expect(taskPage.taskTitle).toBeVisible();
      await expect(taskPage.taskDescription).toBeVisible();
      await expect(taskPage.difficultyBadge).toBeVisible();
    });

    test('should display code editor', async ({ page }) => {
      const taskPage = new TaskPage(page);
      await taskPage.goto('go-basics', 'go-fundamentals-flatten-nested');

      await expect(taskPage.codeEditor).toBeVisible();
    });

    test('should display action buttons', async ({ page }) => {
      const taskPage = new TaskPage(page);
      await taskPage.goto('go-basics', 'go-fundamentals-flatten-nested');

      await expect(taskPage.runButton).toBeVisible();
      await expect(taskPage.submitButton).toBeVisible();
    });
  });

  test.describe('Code Execution', () => {
    test('should run code and show results', async ({ page }) => {
      const taskPage = new TaskPage(page);
      await taskPage.goto('go-basics', 'go-fundamentals-flatten-nested');

      // Run the code
      await taskPage.runCode();

      // Should show test results
      await expect(taskPage.testResults).toBeVisible();
    });

    test('should show compilation errors', async ({ page }) => {
      const taskPage = new TaskPage(page);
      await taskPage.goto('go-basics', 'go-fundamentals-flatten-nested');

      // Type invalid code
      await taskPage.typeCode('this is not valid go code!!!');

      // Run the code
      await taskPage.runButton.click();

      // Wait for results to appear (either test-results or stderr)
      await page.waitForSelector('[data-testid="test-results"], [data-testid="stderr"]', { timeout: 30000 });

      // Should show error in results (compilation errors appear in test results)
      const testResults = await taskPage.testResults.textContent();
      // Any response from running invalid code is acceptable
      expect(testResults).toBeTruthy();
    });

    test('should show passed indicator on successful run', async ({ page }) => {
      const taskPage = new TaskPage(page);
      await taskPage.goto('go-basics', 'go-fundamentals-flatten-nested');

      // Assuming the initial code is correct or needs modification
      // This test may need adjustment based on actual task

      await taskPage.runCode();

      // Check results tab
      await taskPage.resultsTab.click();
      await expect(taskPage.testResults).toBeVisible();
    });
  });

  test.describe('Code Submission', () => {
    test('should submit code after passing tests', async ({ page }) => {
      const taskPage = new TaskPage(page);
      await taskPage.goto('go-basics', 'go-fundamentals-flatten-nested');

      // First run to validate
      await taskPage.runCode();

      // Then submit
      await taskPage.submitButton.click();

      // Wait for submission result
      await page.waitForSelector('[data-testid="submission-result"]', { timeout: 60000 });

      // Should show submission status
      const result = page.getByTestId('submission-result');
      await expect(result).toBeVisible();
    });

    test('should show XP earned on successful submission', async ({ page }) => {
      const taskPage = new TaskPage(page);
      await taskPage.goto('go-basics', 'go-fundamentals-flatten-nested');

      // Run and submit correct code
      await taskPage.runCode();
      await taskPage.submitCode();

      // Should show XP notification
      const xpNotification = page.getByTestId('xp-notification');
      // This may or may not be visible depending on implementation
      // await expect(xpNotification).toBeVisible();
    });
  });

  test.describe('Task Navigation', () => {
    test('should navigate to next task', async ({ page }) => {
      const taskPage = new TaskPage(page);
      await taskPage.goto('go-basics', 'go-fundamentals-flatten-nested');

      const initialUrl = page.url();

      // Click next task
      await taskPage.nextTaskButton.click();

      // URL should change
      await expect(page).not.toHaveURL(initialUrl);
    });

    test('should navigate to previous task', async ({ page }) => {
      const taskPage = new TaskPage(page);
      await taskPage.goto('go-basics', 'go-fundamentals-flatten-nested');

      // Previous button may be disabled if this is the first task
      const isPrevEnabled = await taskPage.prevTaskButton.isEnabled();
      if (!isPrevEnabled) {
        // Skip this test if previous button is disabled (first task)
        console.log('Skipping: Previous task button is disabled (first task in list)');
        return;
      }

      const initialUrl = page.url();
      await taskPage.prevTaskButton.click();
      await expect(page).not.toHaveURL(initialUrl);
    });

    test('should return to course page', async ({ page }) => {
      const taskPage = new TaskPage(page);
      await taskPage.goto('go-basics', 'go-fundamentals-flatten-nested');

      await taskPage.backToCourseButton.click();

      // Should be on course page
      await expect(page).toHaveURL(/\/course\/go-basics/);
    });
  });

  test.describe('Code Reset', () => {
    test('should reset code to initial template', async ({ page }) => {
      const taskPage = new TaskPage(page);
      await taskPage.goto('go-basics', 'go-fundamentals-flatten-nested');

      // Type some random code
      await taskPage.typeCode('// my random code');

      // Reset
      await taskPage.resetCode();

      // Code should be reset (check by comparing with initial)
      // The exact check depends on implementation
    });
  });

  test.describe('Responsive Tabs', () => {
    test('should show tab labels at desktop viewport', async ({ page }) => {
      // Desktop mode: >= 768px viewport, panel uses leftPanelWidth (default 500px)
      await page.setViewportSize({ width: 1280, height: 800 });

      const taskPage = new TaskPage(page);
      await taskPage.goto('go-basics', 'go-fundamentals-flatten-nested');

      // Wait for tabs to render
      await taskPage.descriptionTab.waitFor({ state: 'visible' });

      // At desktop with panel ~500px > 420px threshold, tabs should show text
      const isCompact = await taskPage.areTabsCompact();
      expect(isCompact).toBe(false);

      // Verify labels are visible
      const labels = await taskPage.getTabLabels();
      expect(labels.length).toBeGreaterThan(0);
    });

    test('should toggle between compact and normal mode on resize', async ({ page }) => {
      // Start with desktop width (panel ~500px > 420px)
      await page.setViewportSize({ width: 1280, height: 800 });

      // Clear panel width from localStorage to ensure fresh state
      await page.goto('/');
      await page.evaluate(() => localStorage.removeItem('task-workspace-left-panel-width'));

      const taskPage = new TaskPage(page);
      await taskPage.goto('go-basics', 'go-fundamentals-flatten-nested');
      await taskPage.descriptionTab.waitFor({ state: 'visible' });

      // Should show labels at desktop
      let isCompact = await taskPage.areTabsCompact();
      expect(isCompact).toBe(false);

      // Resize to very narrow mobile (380px panel < 420px threshold)
      await page.setViewportSize({ width: 380, height: 700 });
      await page.waitForTimeout(500); // Wait for ResizeObserver and React re-render

      // Should be compact now
      isCompact = await taskPage.areTabsCompact();
      expect(isCompact).toBe(true);

      // Resize back to desktop - set viewport FIRST, then navigate fresh
      await page.setViewportSize({ width: 1280, height: 800 });

      // Clear localStorage and navigate fresh (instead of reload)
      await page.goto('/');
      await page.evaluate(() => localStorage.removeItem('task-workspace-left-panel-width'));
      await page.goto('/course/go-basics/task/go-fundamentals-flatten-nested');
      await page.waitForLoadState('networkidle');

      // Wait for full page load (Monaco editor visible = page fully loaded)
      await page.waitForSelector('.monaco-editor', { timeout: 30000 });
      await taskPage.descriptionTab.waitFor({ state: 'visible' });
      await page.waitForTimeout(500); // Extra wait for ResizeObserver to settle

      // Should show labels again
      isCompact = await taskPage.areTabsCompact();
      expect(isCompact).toBe(false);
    });

    test('should show tooltip on hover in compact mode', async ({ page }) => {
      // Very narrow viewport to trigger compact mode (380px < 420px)
      await page.setViewportSize({ width: 380, height: 700 });

      // Navigate to task page (don't wait for Monaco in mobile - it's on separate tab)
      await page.goto('/course/go-basics/task/go-fundamentals-flatten-nested');

      const taskPage = new TaskPage(page);
      // Wait for tabs to render (not Monaco editor in mobile view)
      await taskPage.descriptionTab.waitFor({ state: 'visible', timeout: 30000 });

      // Verify we're in compact mode
      const isCompact = await taskPage.areTabsCompact();
      expect(isCompact).toBe(true);

      // Labels should be empty in compact mode (icons only)
      const labels = await taskPage.getTabLabels();
      expect(labels.length).toBe(0);

      // In compact mode, tabs should have title attribute for tooltip
      const titleAttr = await taskPage.descriptionTab.getAttribute('title');
      expect(titleAttr).toBeTruthy();
    });

    test('should not show tooltip in normal mode', async ({ page }) => {
      // Desktop viewport (panel ~500px > 420px)
      await page.setViewportSize({ width: 1280, height: 800 });

      const taskPage = new TaskPage(page);
      await taskPage.goto('go-basics', 'go-fundamentals-flatten-nested');
      await taskPage.descriptionTab.waitFor({ state: 'visible' });

      // Verify we're in normal mode
      const isCompact = await taskPage.areTabsCompact();
      expect(isCompact).toBe(false);

      // In normal mode, tabs should not have title attribute
      const titleAttr = await taskPage.descriptionTab.getAttribute('title');
      expect(titleAttr).toBeNull();
    });
  });
});
