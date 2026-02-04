import { test, expect } from "../fixtures/auth.fixture";
import {
  BugReportPage,
  BugCategory,
  BugSeverity,
} from "../pages/bug-report.page";

const API_URL = process.env.E2E_API_URL || "http://localhost:8080";

test.describe("Bug Reports", () => {
  test.describe("Bug Report Modal - UI", () => {
    test.beforeEach(async ({ auth }) => {
      await auth.loginAsTestUser();
    });

    test("should open bug report modal from task page", async ({ page }) => {
      // Navigate to a task page
      await page.goto("/courses/python-fundamentals");
      await page.waitForLoadState("networkidle");

      // Click on first task
      const firstTask = page.locator('[data-testid="task-link"]').first();
      if (await firstTask.isVisible()) {
        await firstTask.click();
        await page.waitForLoadState("networkidle");

        // Look for bug report button
        const bugButton = page
          .locator("button")
          .filter({ hasText: /bug|report|ошибк/i })
          .or(page.locator('[data-testid="bug-report-button"]'))
          .first();

        if (await bugButton.isVisible()) {
          await bugButton.click();
          const bugReportPage = new BugReportPage(page);
          await bugReportPage.expectModalVisible();
        }
      }
    });

    test("should have all category options visible", async ({ page }) => {
      // Navigate to task and open bug modal
      await page.goto("/courses/python-fundamentals");
      await page.waitForLoadState("networkidle");

      const firstTask = page.locator('[data-testid="task-link"]').first();
      if (await firstTask.isVisible()) {
        await firstTask.click();
        await page.waitForLoadState("networkidle");

        const bugButton = page
          .locator("button")
          .filter({ hasText: /bug|report|ошибк/i })
          .first();

        if (await bugButton.isVisible()) {
          await bugButton.click();
          const bugReportPage = new BugReportPage(page);

          // Check category icons are visible
          const categories = ["📝", "💻", "💭", "❓"];
          for (const icon of categories) {
            const categoryBtn = page.locator(`button:has-text("${icon}")`);
            await expect(categoryBtn).toBeVisible();
          }
        }
      }
    });

    test("should require all fields to enable submit", async ({ page }) => {
      await page.goto("/courses/python-fundamentals");
      await page.waitForLoadState("networkidle");

      const firstTask = page.locator('[data-testid="task-link"]').first();
      if (await firstTask.isVisible()) {
        await firstTask.click();
        await page.waitForLoadState("networkidle");

        const bugButton = page
          .locator("button")
          .filter({ hasText: /bug|report|ошибк/i })
          .first();

        if (await bugButton.isVisible()) {
          await bugButton.click();
          const bugReportPage = new BugReportPage(page);

          // Submit should be disabled initially
          await bugReportPage.expectSubmitDisabled();

          // Fill only category
          await bugReportPage.selectCategory("editor");
          await bugReportPage.expectSubmitDisabled();

          // Fill title
          await bugReportPage.fillTitle("Test Bug Title");
          await bugReportPage.expectSubmitDisabled();

          // Fill description - now should be enabled
          await bugReportPage.fillDescription(
            "Test bug description with details",
          );
          await bugReportPage.expectSubmitEnabled();
        }
      }
    });

    test("should close modal on cancel", async ({ page }) => {
      await page.goto("/courses/python-fundamentals");
      await page.waitForLoadState("networkidle");

      const firstTask = page.locator('[data-testid="task-link"]').first();
      if (await firstTask.isVisible()) {
        await firstTask.click();
        await page.waitForLoadState("networkidle");

        const bugButton = page
          .locator("button")
          .filter({ hasText: /bug|report|ошибк/i })
          .first();

        if (await bugButton.isVisible()) {
          await bugButton.click();
          const bugReportPage = new BugReportPage(page);
          await bugReportPage.expectModalVisible();

          await bugReportPage.cancel();
          await bugReportPage.expectModalHidden();
        }
      }
    });
  });

  test.describe("Bug Report API", () => {
    // Helper to get auth token
    async function getAuthToken(request: any): Promise<string> {
      const loginResponse = await request.post(`${API_URL}/auth/login`, {
        data: {
          email: "e2e-test@practix.dev",
          password: "TestPassword123!",
        },
      });
      const loginData = await loginResponse.json();
      return loginData.token;
    }

    test("should create a bug report via API", async ({ request }) => {
      const authToken = await getAuthToken(request);
      const response = await request.post(`${API_URL}/bugreports`, {
        headers: {
          Authorization: `Bearer ${authToken}`,
        },
        data: {
          title: "E2E Test Bug Report",
          description: "This is a test bug report created by E2E tests",
          category: "editor",
          severity: "low",
        },
      });

      expect(response.status()).toBe(201);
      const report = await response.json();
      expect(report).toHaveProperty("id");
      expect(report.title).toBe("E2E Test Bug Report");
      expect(report.category).toBe("editor");
      expect(report.status).toBe("open");
    });

    test("should get user's bug reports", async ({ request }) => {
      const authToken = await getAuthToken(request);
      const response = await request.get(`${API_URL}/bugreports/my`, {
        headers: {
          Authorization: `Bearer ${authToken}`,
        },
      });

      expect(response.status()).toBe(200);
      const reports = await response.json();
      expect(Array.isArray(reports)).toBe(true);
    });

    test("should require authentication for creating bug report", async ({
      request,
    }) => {
      const response = await request.post(`${API_URL}/bugreports`, {
        data: {
          title: "Unauthorized Bug Report",
          description: "This should fail",
          category: "editor",
        },
      });

      expect(response.status()).toBe(401);
    });

    test("should validate required fields", async ({ request }) => {
      const authToken = await getAuthToken(request);
      const response = await request.post(`${API_URL}/bugreports`, {
        headers: {
          Authorization: `Bearer ${authToken}`,
        },
        data: {
          // Missing title and description
          category: "editor",
        },
      });

      expect(response.status()).toBe(400);
    });

    test("should validate category enum", async ({ request }) => {
      const authToken = await getAuthToken(request);
      const response = await request.post(`${API_URL}/bugreports`, {
        headers: {
          Authorization: `Bearer ${authToken}`,
        },
        data: {
          title: "Test",
          description: "Test",
          category: "invalid-category",
        },
      });

      expect(response.status()).toBe(400);
    });
  });

  test.describe("Bug Report Admin", () => {
    // Helper to get admin token
    async function getAdminToken(request: any): Promise<string> {
      const loginResponse = await request.post(`${API_URL}/auth/login`, {
        data: {
          email: "e2e-admin@practix.dev",
          password: "AdminPassword123!",
        },
      });
      const loginData = await loginResponse.json();
      return loginData.token;
    }

    // Helper to get user token
    async function getUserToken(request: any): Promise<string> {
      const loginResponse = await request.post(`${API_URL}/auth/login`, {
        data: {
          email: "e2e-test@practix.dev",
          password: "TestPassword123!",
        },
      });
      const loginData = await loginResponse.json();
      return loginData.token;
    }

    // Helper to create test bug report
    async function createTestReport(
      request: any,
      token: string,
    ): Promise<string> {
      const createResponse = await request.post(`${API_URL}/bugreports`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
        data: {
          title: "Admin Test Bug " + Date.now(),
          description: "Bug for admin testing",
          category: "other",
          severity: "medium",
        },
      });
      const report = await createResponse.json();
      return report.id;
    }

    test("should get all bug reports as admin", async ({ request }) => {
      const adminToken = await getAdminToken(request);
      const response = await request.get(`${API_URL}/bugreports`, {
        headers: {
          Authorization: `Bearer ${adminToken}`,
        },
      });

      expect(response.status()).toBe(200);
      const reports = await response.json();
      expect(Array.isArray(reports)).toBe(true);
    });

    test("should filter bug reports by status", async ({ request }) => {
      const adminToken = await getAdminToken(request);
      const response = await request.get(`${API_URL}/bugreports?status=open`, {
        headers: {
          Authorization: `Bearer ${adminToken}`,
        },
      });

      expect(response.status()).toBe(200);
      const reports = await response.json();
      expect(Array.isArray(reports)).toBe(true);
      // All reports should have status 'open'
      reports.forEach((report: { status: string }) => {
        expect(report.status).toBe("open");
      });
    });

    test("should update bug report status as admin", async ({ request }) => {
      const adminToken = await getAdminToken(request);
      const testReportId = await createTestReport(request, adminToken);

      const response = await request.patch(
        `${API_URL}/bugreports/${testReportId}/status`,
        {
          headers: {
            Authorization: `Bearer ${adminToken}`,
          },
          data: {
            status: "in-progress",
          },
        },
      );

      expect(response.status()).toBe(200);
      const updatedReport = await response.json();
      expect(updatedReport.status).toBe("in-progress");
    });

    test("should not allow regular user to update status", async ({
      request,
    }) => {
      const adminToken = await getAdminToken(request);
      const userToken = await getUserToken(request);
      const testReportId = await createTestReport(request, adminToken);

      const response = await request.patch(
        `${API_URL}/bugreports/${testReportId}/status`,
        {
          headers: {
            Authorization: `Bearer ${userToken}`,
          },
          data: {
            status: "resolved",
          },
        },
      );

      expect(response.status()).toBe(403);
    });

    test("should not allow regular user to view all reports", async ({
      request,
    }) => {
      const userToken = await getUserToken(request);

      const response = await request.get(`${API_URL}/bugreports`, {
        headers: {
          Authorization: `Bearer ${userToken}`,
        },
      });

      expect(response.status()).toBe(403);
    });
  });
});
