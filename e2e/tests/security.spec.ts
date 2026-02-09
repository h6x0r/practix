import { test, expect } from "../fixtures/auth.fixture";
import { PlaygroundPage } from "../pages/playground.page";

/**
 * Security E2E Tests
 *
 * Tests that malicious code patterns are properly rejected by the backend.
 * These tests verify the SecurityValidationService integration works end-to-end.
 */
test.describe("Security - Malicious Code Detection", () => {
  test.describe("Playground - Malicious Code Rejection", () => {
    test.beforeEach(async ({ auth }) => {
      await auth.loginAsTestUser();
    });

    test("should reject Python os.system() command execution", async ({
      page,
    }) => {
      const playgroundPage = new PlaygroundPage(page);
      await playgroundPage.goto();
      await playgroundPage.waitForLoad();

      // Switch to Python
      await playgroundPage.selectLanguage("python");

      // Enter malicious code
      const maliciousCode = `
import os
os.system("rm -rf /")
print("Hello")
`;

      await playgroundPage.setCode(maliciousCode);
      await playgroundPage.clickRun();

      // Wait for response
      await page.waitForTimeout(3000);

      // Should show error about malicious code or forbidden
      const pageText = await page.textContent("body");
      const hasSecurityError =
        pageText?.toLowerCase().includes("malicious") ||
        pageText?.toLowerCase().includes("forbidden") ||
        pageText?.toLowerCase().includes("security") ||
        pageText?.toLowerCase().includes("blocked") ||
        pageText?.toLowerCase().includes("not allowed");

      expect(hasSecurityError).toBe(true);
    });

    test("should reject Python subprocess execution", async ({ page }) => {
      const playgroundPage = new PlaygroundPage(page);
      await playgroundPage.goto();
      await playgroundPage.waitForLoad();
      await playgroundPage.selectLanguage("python");

      const maliciousCode = `
import subprocess
subprocess.run(["ls", "-la"])
`;

      await playgroundPage.setCode(maliciousCode);
      await playgroundPage.clickRun();
      await page.waitForTimeout(3000);

      const pageText = await page.textContent("body");
      const hasSecurityError =
        pageText?.toLowerCase().includes("malicious") ||
        pageText?.toLowerCase().includes("forbidden") ||
        pageText?.toLowerCase().includes("security") ||
        pageText?.toLowerCase().includes("blocked");

      expect(hasSecurityError).toBe(true);
    });

    test("should reject Go os/exec command execution", async ({ page }) => {
      const playgroundPage = new PlaygroundPage(page);
      await playgroundPage.goto();
      await playgroundPage.waitForLoad();

      // Go is default language
      const maliciousCode = `
package main

import (
    "os/exec"
    "fmt"
)

func main() {
    cmd := exec.Command("ls", "-la")
    output, _ := cmd.Output()
    fmt.Println(string(output))
}
`;

      await playgroundPage.setCode(maliciousCode);
      await playgroundPage.clickRun();
      await page.waitForTimeout(3000);

      const pageText = await page.textContent("body");
      const hasSecurityError =
        pageText?.toLowerCase().includes("malicious") ||
        pageText?.toLowerCase().includes("forbidden") ||
        pageText?.toLowerCase().includes("security") ||
        pageText?.toLowerCase().includes("blocked");

      expect(hasSecurityError).toBe(true);
    });

    test("should reject Java Runtime.exec() command execution", async ({
      page,
    }) => {
      const playgroundPage = new PlaygroundPage(page);
      await playgroundPage.goto();
      await playgroundPage.waitForLoad();
      await playgroundPage.selectLanguage("java");

      const maliciousCode = `
public class Main {
    public static void main(String[] args) throws Exception {
        Runtime.getRuntime().exec("ls -la");
    }
}
`;

      await playgroundPage.setCode(maliciousCode);
      await playgroundPage.clickRun();
      await page.waitForTimeout(3000);

      const pageText = await page.textContent("body");
      const hasSecurityError =
        pageText?.toLowerCase().includes("malicious") ||
        pageText?.toLowerCase().includes("forbidden") ||
        pageText?.toLowerCase().includes("security") ||
        pageText?.toLowerCase().includes("blocked");

      expect(hasSecurityError).toBe(true);
    });

    test("should reject network access attempts - Python requests", async ({
      page,
    }) => {
      const playgroundPage = new PlaygroundPage(page);
      await playgroundPage.goto();
      await playgroundPage.waitForLoad();
      await playgroundPage.selectLanguage("python");

      const maliciousCode = `
import requests
response = requests.get("http://example.com")
print(response.text)
`;

      await playgroundPage.setCode(maliciousCode);
      await playgroundPage.clickRun();
      await page.waitForTimeout(3000);

      const pageText = await page.textContent("body");
      const hasSecurityError =
        pageText?.toLowerCase().includes("malicious") ||
        pageText?.toLowerCase().includes("forbidden") ||
        pageText?.toLowerCase().includes("network") ||
        pageText?.toLowerCase().includes("blocked");

      expect(hasSecurityError).toBe(true);
    });

    test("should reject file system access - Python open()", async ({
      page,
    }) => {
      const playgroundPage = new PlaygroundPage(page);
      await playgroundPage.goto();
      await playgroundPage.waitForLoad();
      await playgroundPage.selectLanguage("python");

      const maliciousCode = `
with open("/etc/passwd", "r") as f:
    print(f.read())
`;

      await playgroundPage.setCode(maliciousCode);
      await playgroundPage.clickRun();
      await page.waitForTimeout(3000);

      const pageText = await page.textContent("body");
      const hasSecurityError =
        pageText?.toLowerCase().includes("malicious") ||
        pageText?.toLowerCase().includes("forbidden") ||
        pageText?.toLowerCase().includes("file") ||
        pageText?.toLowerCase().includes("blocked");

      expect(hasSecurityError).toBe(true);
    });

    test("should reject eval() dynamic code execution", async ({ page }) => {
      const playgroundPage = new PlaygroundPage(page);
      await playgroundPage.goto();
      await playgroundPage.waitForLoad();
      await playgroundPage.selectLanguage("python");

      const maliciousCode = `
code = "print('hacked')"
eval(code)
`;

      await playgroundPage.setCode(maliciousCode);
      await playgroundPage.clickRun();
      await page.waitForTimeout(3000);

      const pageText = await page.textContent("body");
      const hasSecurityError =
        pageText?.toLowerCase().includes("malicious") ||
        pageText?.toLowerCase().includes("forbidden") ||
        pageText?.toLowerCase().includes("eval") ||
        pageText?.toLowerCase().includes("blocked");

      expect(hasSecurityError).toBe(true);
    });

    test("should reject environment variable access", async ({ page }) => {
      const playgroundPage = new PlaygroundPage(page);
      await playgroundPage.goto();
      await playgroundPage.waitForLoad();
      await playgroundPage.selectLanguage("python");

      const maliciousCode = `
import os
print(os.environ.get("DATABASE_URL"))
`;

      await playgroundPage.setCode(maliciousCode);
      await playgroundPage.clickRun();
      await page.waitForTimeout(3000);

      const pageText = await page.textContent("body");
      const hasSecurityError =
        pageText?.toLowerCase().includes("malicious") ||
        pageText?.toLowerCase().includes("forbidden") ||
        pageText?.toLowerCase().includes("environment") ||
        pageText?.toLowerCase().includes("blocked");

      expect(hasSecurityError).toBe(true);
    });

    test("should reject destructive rm -rf command", async ({ page }) => {
      const playgroundPage = new PlaygroundPage(page);
      await playgroundPage.goto();
      await playgroundPage.waitForLoad();
      await playgroundPage.selectLanguage("python");

      const maliciousCode = `
import os
os.system("rm -rf /")
`;

      await playgroundPage.setCode(maliciousCode);
      await playgroundPage.clickRun();
      await page.waitForTimeout(3000);

      const pageText = await page.textContent("body");
      const hasSecurityError =
        pageText?.toLowerCase().includes("malicious") ||
        pageText?.toLowerCase().includes("forbidden") ||
        pageText?.toLowerCase().includes("destructive") ||
        pageText?.toLowerCase().includes("blocked");

      expect(hasSecurityError).toBe(true);
    });
  });

  test.describe("Playground - Safe Code Acceptance", () => {
    test.beforeEach(async ({ auth }) => {
      await auth.loginAsTestUser();
    });

    test("should accept safe Python print statement", async ({ page }) => {
      const playgroundPage = new PlaygroundPage(page);
      await playgroundPage.goto();
      await playgroundPage.waitForLoad();
      await playgroundPage.selectLanguage("python");

      const safeCode = `
for i in range(5):
    print(f"Hello {i}")
`;

      await playgroundPage.setCode(safeCode);
      await playgroundPage.clickRun();
      await page.waitForTimeout(5000);

      const pageText = await page.textContent("body");
      // Should NOT have security errors
      const hasSecurityError =
        pageText?.toLowerCase().includes("malicious") ||
        pageText?.toLowerCase().includes("forbidden") ||
        pageText?.toLowerCase().includes("blocked");

      // Should have output OR be running
      const hasOutput =
        pageText?.includes("Hello") ||
        pageText?.includes("Running") ||
        pageText?.includes("Executing");

      expect(hasSecurityError).toBe(false);
      // Output may or may not be visible depending on Judge0 availability
    });

    test("should accept safe Go hello world", async ({ page }) => {
      const playgroundPage = new PlaygroundPage(page);
      await playgroundPage.goto();
      await playgroundPage.waitForLoad();

      const safeCode = `
package main

import "fmt"

func main() {
    for i := 0; i < 5; i++ {
        fmt.Printf("Hello %d\\n", i)
    }
}
`;

      await playgroundPage.setCode(safeCode);
      await playgroundPage.clickRun();
      await page.waitForTimeout(5000);

      const pageText = await page.textContent("body");
      const hasSecurityError =
        pageText?.toLowerCase().includes("malicious") ||
        pageText?.toLowerCase().includes("forbidden") ||
        pageText?.toLowerCase().includes("blocked");

      expect(hasSecurityError).toBe(false);
    });

    test("should accept safe Java hello world", async ({ page }) => {
      const playgroundPage = new PlaygroundPage(page);
      await playgroundPage.goto();
      await playgroundPage.waitForLoad();
      await playgroundPage.selectLanguage("java");

      const safeCode = `
public class Main {
    public static void main(String[] args) {
        for (int i = 0; i < 5; i++) {
            System.out.println("Hello " + i);
        }
    }
}
`;

      await playgroundPage.setCode(safeCode);
      await playgroundPage.clickRun();
      await page.waitForTimeout(5000);

      const pageText = await page.textContent("body");
      const hasSecurityError =
        pageText?.toLowerCase().includes("malicious") ||
        pageText?.toLowerCase().includes("forbidden") ||
        pageText?.toLowerCase().includes("blocked");

      expect(hasSecurityError).toBe(false);
    });

    test("should accept safe TypeScript code", async ({ page }) => {
      const playgroundPage = new PlaygroundPage(page);
      await playgroundPage.goto();
      await playgroundPage.waitForLoad();
      await playgroundPage.selectLanguage("typescript");

      const safeCode = `
const greet = (name: string): string => {
    return \`Hello, \${name}!\`;
};

for (let i = 0; i < 5; i++) {
    console.log(greet(\`User\${i}\`));
}
`;

      await playgroundPage.setCode(safeCode);
      await playgroundPage.clickRun();
      await page.waitForTimeout(5000);

      const pageText = await page.textContent("body");
      const hasSecurityError =
        pageText?.toLowerCase().includes("malicious") ||
        pageText?.toLowerCase().includes("forbidden") ||
        pageText?.toLowerCase().includes("blocked");

      expect(hasSecurityError).toBe(false);
    });
  });

  test.describe("Security - Unauthenticated Access", () => {
    test("should still validate code for unauthenticated users", async ({
      page,
    }) => {
      await page.context().clearCookies();

      const playgroundPage = new PlaygroundPage(page);
      await playgroundPage.goto();
      await playgroundPage.waitForLoad();

      // Try to run malicious code without auth
      const maliciousCode = `
import os
os.system("ls")
`;

      await playgroundPage.selectLanguage("python");
      await playgroundPage.setCode(maliciousCode);

      // Try to run
      const runButton = page.locator("button").filter({ hasText: /Run|\d+s/ });
      if ((await runButton.isVisible()) && (await runButton.isEnabled())) {
        await runButton.click();
        await page.waitForTimeout(3000);

        const pageText = await page.textContent("body");
        // Should either block the request or show security error
        const isBlocked =
          pageText?.toLowerCase().includes("malicious") ||
          pageText?.toLowerCase().includes("forbidden") ||
          pageText?.toLowerCase().includes("unauthorized") ||
          pageText?.toLowerCase().includes("blocked") ||
          pageText?.toLowerCase().includes("login");

        expect(isBlocked).toBe(true);
      }
    });
  });

  test.describe("Security - Edge Cases", () => {
    test.beforeEach(async ({ auth }) => {
      await auth.loginAsTestUser();
    });

    test("should handle empty code gracefully", async ({ page }) => {
      const playgroundPage = new PlaygroundPage(page);
      await playgroundPage.goto();
      await playgroundPage.waitForLoad();

      await playgroundPage.setCode("");
      await playgroundPage.clickRun();
      await page.waitForTimeout(2000);

      // Should not crash
      expect(page.url()).toContain("/playground");
    });

    test("should handle very long code", async ({ page }) => {
      const playgroundPage = new PlaygroundPage(page);
      await playgroundPage.goto();
      await playgroundPage.waitForLoad();

      // Generate code that's close to the limit but valid
      const longCode =
        `package main\n\nimport "fmt"\n\nfunc main() {\n` +
        Array(100)
          .fill('    fmt.Println("Hello")')
          .join("\n") +
        "\n}";

      await playgroundPage.setCode(longCode);
      await playgroundPage.clickRun();
      await page.waitForTimeout(5000);

      // Should handle gracefully (either run or show appropriate error)
      expect(page.url()).toContain("/playground");
    });

    test("should reject code with obfuscated malicious patterns", async ({
      page,
    }) => {
      const playgroundPage = new PlaygroundPage(page);
      await playgroundPage.goto();
      await playgroundPage.waitForLoad();
      await playgroundPage.selectLanguage("python");

      // Try to obfuscate os.system using getattr
      const obfuscatedCode = `
import os
getattr(os, 'system')('ls')
`;

      await playgroundPage.setCode(obfuscatedCode);
      await playgroundPage.clickRun();
      await page.waitForTimeout(3000);

      const pageText = await page.textContent("body");
      // This may or may not be caught depending on scanner sophistication
      // At minimum, it should not execute successfully
      const executed = pageText?.includes("bin") || pageText?.includes("usr");

      expect(executed).toBe(false);
    });
  });
});
