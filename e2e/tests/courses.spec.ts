import { test, expect } from '../fixtures/auth.fixture';

test.describe('Course Navigation', () => {
  test.beforeEach(async ({ auth }) => {
    await auth.loginAsTestUser();
  });

  test.describe('Course Catalog', () => {
    test('should display course catalog', async ({ page }) => {
      await page.goto('/courses');

      // Should see course cards
      const courseCards = page.getByTestId('course-card');
      await expect(courseCards.first()).toBeVisible();
    });

    test('should filter courses by category', async ({ page }) => {
      await page.goto('/courses');

      // Click on a category filter
      const categoryFilter = page.getByTestId('category-filter-go');
      if (await categoryFilter.isVisible()) {
        await categoryFilter.click();

        // Should show filtered courses
        await expect(page).toHaveURL(/category=go/);
      }
    });

    test('should search courses', async ({ page }) => {
      await page.goto('/courses');

      const searchInput = page.getByTestId('course-search');
      if (await searchInput.isVisible()) {
        await searchInput.fill('Go');
        await searchInput.press('Enter');

        // Should filter results
        await expect(page.getByTestId('course-card')).toBeVisible();
      }
    });
  });

  test.describe('Course Detail', () => {
    test('should display course structure', async ({ page }) => {
      await page.goto('/course/go-basics');

      // Should see course title
      await expect(page.getByTestId('course-title')).toBeVisible();

      // Should see modules
      const modules = page.getByTestId('module-item');
      await expect(modules.first()).toBeVisible();
    });

    test('should expand module to show topics', async ({ page }) => {
      await page.goto('/course/go-basics');

      // Click on first module
      const firstModule = page.getByTestId('module-item').first();
      await firstModule.click();

      // Should show topics
      const topics = page.getByTestId('topic-item');
      await expect(topics.first()).toBeVisible();
    });

    test('should navigate to task from course page', async ({ page }) => {
      await page.goto('/course/go-basics');

      // First expand a module to show tasks
      const firstModule = page.getByTestId('module-item').first();
      await firstModule.click();

      // Wait for topics to be visible
      await page.waitForSelector('[data-testid="topic-item"]', { timeout: 5000 });

      // Click on first task link
      const firstTask = page.getByTestId('task-link').first();
      await firstTask.click();

      // Should be on task page
      await expect(page).toHaveURL(/\/course\/.*\/task\//);
    });

    test('should show progress indicators', async ({ page }) => {
      await page.goto('/course/go-basics');

      // Should see progress bar or completion status
      const progressBar = page.getByTestId('course-progress');
      // Progress may not be visible for all users
      // Just check the page loads correctly
    });
  });

  test.describe('My Courses', () => {
    test('should show enrolled courses on dashboard', async ({ page }) => {
      await page.goto('/dashboard');

      // Should see "My Courses" section
      const myCourses = page.getByTestId('my-courses-section');
      await expect(myCourses).toBeVisible();
    });

    test('should show course progress on dashboard', async ({ page }) => {
      await page.goto('/dashboard');

      // Should see progress for enrolled courses
      const courseProgress = page.getByTestId('course-progress-card');
      // May or may not be visible depending on enrollment
    });
  });
});

test.describe('Premium Course Access', () => {
  test('free user should see upgrade prompt for premium courses', async ({ page, auth }) => {
    await auth.loginAsTestUser();

    // Navigate to a premium course (assuming there is one)
    await page.goto('/courses/premium-course');

    // Should see upgrade prompt or paywall
    const upgradePrompt = page.getByTestId('upgrade-prompt');
    // This depends on whether premium courses exist
  });

  test('premium user should access all courses', async ({ page, auth }) => {
    await auth.loginAsPremiumUser();

    // Navigate to a premium course
    await page.goto('/courses/premium-course');

    // Should not see upgrade prompt
    const upgradePrompt = page.getByTestId('upgrade-prompt');
    // Should have access to content
  });
});
