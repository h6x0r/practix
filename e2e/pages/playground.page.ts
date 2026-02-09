import { Page, Locator } from "@playwright/test";

/**
 * Page Object for Playground page
 */
export class PlaygroundPage {
  readonly page: Page;

  // Editor elements
  readonly editorContainer: Locator;
  readonly codeEditor: Locator;

  // Language selector
  readonly languageTab: Locator;
  readonly languageDropdown: Locator;
  readonly goOption: Locator;
  readonly javaOption: Locator;
  readonly pythonOption: Locator;
  readonly typescriptOption: Locator;

  // Action buttons
  readonly runButton: Locator;
  readonly resetButton: Locator;

  // Output panel
  readonly outputPanel: Locator;
  readonly outputContent: Locator;
  readonly outputHeader: Locator;
  readonly outputTime: Locator;

  // Status indicators
  readonly statusIndicator: Locator;
  readonly rateLimitIndicator: Locator;
  readonly cooldownTimer: Locator;

  // Loading states
  readonly runningIndicator: Locator;
  readonly editorLoading: Locator;

  // Auto-save popup
  readonly autoSavePopup: Locator;
  readonly autoSaveCloseButton: Locator;

  constructor(page: Page) {
    this.page = page;

    // Editor elements
    this.editorContainer = page.locator(".monaco-editor").first();
    this.codeEditor = page
      .locator("[data-keybinding-context]")
      .or(page.locator(".monaco-editor textarea"));

    // Language selector - the tab button that opens dropdown
    this.languageTab = page
      .locator("button")
      .filter({ hasText: /main\.(go|java|py|ts)/ });
    this.languageDropdown = page
      .locator('[class*="shadow-lg"]')
      .filter({
        has: page
          .locator("button")
          .filter({ hasText: /Go|Java|Python|TypeScript/ }),
      });
    // Language options in dropdown - look for span with language name inside button
    this.goOption = page
      .locator("button span")
      .filter({ hasText: "Go" })
      .locator("..");
    this.javaOption = page
      .locator("button span")
      .filter({ hasText: "Java" })
      .locator("..");
    this.pythonOption = page
      .locator("button span")
      .filter({ hasText: "Python" })
      .locator("..");
    this.typescriptOption = page
      .locator("button span")
      .filter({ hasText: "TypeScript" })
      .locator("..");

    // Action buttons
    this.runButton = page
      .locator("button")
      .filter({ hasText: /Run|Running|\d+s/ });
    this.resetButton = page
      .locator('button[title="Reset"]')
      .or(
        page
          .locator("button")
          .filter({ has: page.locator('svg[class*="refresh"]') }),
      );

    // Output panel
    this.outputPanel = page
      .locator("span")
      .filter({ hasText: /^Output$/ })
      .locator("..")
      .locator("..");
    this.outputContent = page
      .locator("pre")
      .or(page.locator("text=Run code to see output"));
    this.outputHeader = page
      .locator("span.uppercase")
      .filter({ hasText: "Output" });
    this.outputTime = page.locator("text=/\\d+\\.?\\d*s/");

    // Status indicators
    this.statusIndicator = page.locator("text=/Ready|Mock/");
    this.rateLimitIndicator = page.locator("text=/Premium|\\d+s limit/");
    this.cooldownTimer = page.locator("button").filter({ hasText: /^\d+s$/ });

    // Loading states
    this.runningIndicator = page
      .locator("text=Running")
      .or(page.locator("text=Executing"));
    this.editorLoading = page.locator("text=Loading editor");

    // Auto-save popup
    this.autoSavePopup = page
      .locator('[class*="fixed"]')
      .filter({ hasText: /auto|cloud|save/i });
    this.autoSaveCloseButton = page
      .locator("button")
      .filter({ hasText: /got it|понял|tushundim/i });
  }

  /**
   * Navigate to playground page
   */
  async goto() {
    await this.page.goto("/playground");
  }

  /**
   * Wait for page to load
   */
  async waitForLoad() {
    // Wait for editor to be ready
    await this.page.waitForLoadState("networkidle");
    await this.page.waitForTimeout(1000); // Give Monaco time to initialize
  }

  /**
   * Wait for editor to be ready
   */
  async waitForEditorReady() {
    await this.editorContainer
      .waitFor({ state: "visible", timeout: 15000 })
      .catch(() => {});
    await this.editorLoading
      .waitFor({ state: "hidden", timeout: 15000 })
      .catch(() => {});
  }

  /**
   * Check if editor is visible
   */
  async isEditorVisible(): Promise<boolean> {
    return await this.editorContainer.isVisible().catch(() => false);
  }

  /**
   * Open language dropdown
   */
  async openLanguageDropdown() {
    await this.languageTab.click();
    await this.page.waitForTimeout(200);
  }

  /**
   * Select a language
   */
  async selectLanguage(language: "go" | "java" | "python" | "typescript") {
    await this.openLanguageDropdown();

    const optionMap = {
      go: this.goOption,
      java: this.javaOption,
      python: this.pythonOption,
      typescript: this.typescriptOption,
    };

    await optionMap[language].click();
    await this.page.waitForTimeout(300);
  }

  /**
   * Get current language from tab
   */
  async getCurrentLanguage(): Promise<string | null> {
    const tabText = await this.languageTab.textContent().catch(() => null);
    if (!tabText) return null;

    if (tabText.includes(".go")) return "go";
    if (tabText.includes(".java")) return "java";
    if (tabText.includes(".py")) return "python";
    if (tabText.includes(".ts")) return "typescript";
    return null;
  }

  /**
   * Type code into editor
   */
  async typeCode(code: string) {
    // Click on editor to focus
    await this.editorContainer.click();
    // Clear existing code with Cmd/Ctrl+A and then type
    await this.page.keyboard.press("Meta+a");
    await this.page.keyboard.type(code, { delay: 10 });
  }

  /**
   * Set code in editor using Monaco API
   * More reliable than typeCode for complex code
   */
  async setCode(code: string) {
    // Wait for editor to be ready
    await this.waitForEditorReady();

    // Use Monaco editor API to set code directly
    await this.page.evaluate((newCode) => {
      // Access Monaco editor instance
      const editor = (window as any).monaco?.editor?.getEditors?.()?.[0];
      if (editor) {
        editor.setValue(newCode);
        return true;
      }

      // Fallback: try to find editor in DOM
      const monacoContainer = document.querySelector(".monaco-editor");
      if (monacoContainer) {
        const model = (
          monacoContainer as any
        )?.__vue__?.$refs?.editor?.getModel?.();
        if (model) {
          model.setValue(newCode);
          return true;
        }
      }

      return false;
    }, code);

    // Small delay to ensure code is set
    await this.page.waitForTimeout(300);
  }

  /**
   * Run code
   */
  async runCode() {
    await this.runButton.click();
  }

  /**
   * Click run button (alias for runCode)
   */
  async clickRun() {
    // Wait for button to be enabled (not on cooldown)
    const button = this.page.locator("button").filter({ hasText: /^Run$/ });

    // Try the specific Run button first
    if (await button.isVisible().catch(() => false)) {
      if (await button.isEnabled().catch(() => false)) {
        await button.click();
        return;
      }
    }

    // Fallback to runButton which may show cooldown timer
    if (await this.runButton.isEnabled().catch(() => false)) {
      await this.runButton.click();
    }
  }

  /**
   * Reset code
   */
  async resetCode() {
    await this.resetButton.click();
    await this.page.waitForTimeout(200);
  }

  /**
   * Check if run button is enabled
   */
  async isRunButtonEnabled(): Promise<boolean> {
    return !(await this.runButton.isDisabled());
  }

  /**
   * Wait for execution to complete
   */
  async waitForExecution(timeout: number = 30000) {
    // Wait for running indicator to appear
    await this.runningIndicator
      .waitFor({ state: "visible", timeout: 5000 })
      .catch(() => {});
    // Wait for running indicator to disappear (execution complete)
    await this.runningIndicator.waitFor({ state: "hidden", timeout });
  }

  /**
   * Check if code is running
   */
  async isRunning(): Promise<boolean> {
    return await this.runningIndicator.isVisible().catch(() => false);
  }

  /**
   * Check if on cooldown
   */
  async isOnCooldown(): Promise<boolean> {
    const buttonText = await this.runButton.textContent().catch(() => "");
    return /^\d+s$/.test(buttonText?.trim() || "");
  }

  /**
   * Get output text
   */
  async getOutput(): Promise<string | null> {
    // Try to get text from pre element (actual output)
    const preElements = this.page.locator("pre");
    const count = await preElements.count();

    for (let i = 0; i < count; i++) {
      const text = await preElements
        .nth(i)
        .textContent()
        .catch(() => null);
      if (text && text.trim().length > 0) {
        return text;
      }
    }

    return null;
  }

  /**
   * Check if output contains error
   */
  async hasError(): Promise<boolean> {
    const errorPre = this.page
      .locator('pre[class*="red"]')
      .or(this.page.locator('[class*="red-"]').locator("pre"));
    return await errorPre.isVisible().catch(() => false);
  }

  /**
   * Check if judge is ready
   */
  async isJudgeReady(): Promise<boolean> {
    const statusText = await this.statusIndicator.textContent().catch(() => "");
    return statusText === "Ready";
  }

  /**
   * Check if auto-save popup is visible
   */
  async isAutoSavePopupVisible(): Promise<boolean> {
    return await this.autoSavePopup.isVisible().catch(() => false);
  }

  /**
   * Close auto-save popup
   */
  async closeAutoSavePopup() {
    if (await this.isAutoSavePopupVisible()) {
      await this.autoSaveCloseButton.click();
    }
  }
}
