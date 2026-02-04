import { Page, Locator, expect } from "@playwright/test";
import { BasePage } from "./base.page";

export type BugCategory =
  | "description"
  | "solution"
  | "editor"
  | "hints"
  | "ai-tutor"
  | "other";
export type BugSeverity = "low" | "medium" | "high";

export class BugReportPage extends BasePage {
  // Modal elements
  readonly modal: Locator;
  readonly closeButton: Locator;
  readonly submitButton: Locator;
  readonly cancelButton: Locator;

  // Form fields
  readonly titleInput: Locator;
  readonly descriptionTextarea: Locator;

  constructor(page: Page) {
    super(page);
    this.modal = page.locator('[class*="fixed"][class*="z-50"]').filter({
      has: page.locator('text="Report a Bug"').or(page.locator('text="Сообщить об ошибке"')),
    });
    this.closeButton = this.modal.locator('button').filter({ has: page.locator('svg') }).first();
    this.submitButton = this.modal.locator('button').filter({ hasText: /submit|отправить/i });
    this.cancelButton = this.modal.locator('button').filter({ hasText: /cancel|отмена/i });
    this.titleInput = this.modal.locator('input[type="text"]');
    this.descriptionTextarea = this.modal.locator("textarea");
  }

  async isModalVisible(): Promise<boolean> {
    try {
      await this.modal.waitFor({ state: "visible", timeout: 3000 });
      return true;
    } catch {
      return false;
    }
  }

  async selectCategory(category: BugCategory): Promise<void> {
    const categoryIcons: Record<BugCategory, string> = {
      description: "📝",
      solution: "💡",
      editor: "💻",
      hints: "💭",
      "ai-tutor": "🤖",
      other: "❓",
    };

    const icon = categoryIcons[category];
    const categoryButton = this.modal.locator(`button:has-text("${icon}")`);
    await categoryButton.click();
  }

  async selectSeverity(severity: BugSeverity): Promise<void> {
    const radio = this.modal.locator(`input[type="radio"][value="${severity}"]`);
    await radio.check();
  }

  async fillTitle(title: string): Promise<void> {
    await this.titleInput.fill(title);
  }

  async fillDescription(description: string): Promise<void> {
    await this.descriptionTextarea.fill(description);
  }

  async submit(): Promise<void> {
    await this.submitButton.click();
  }

  async cancel(): Promise<void> {
    await this.cancelButton.click();
  }

  async close(): Promise<void> {
    await this.closeButton.click();
  }

  async fillAndSubmitReport(data: {
    category: BugCategory;
    severity?: BugSeverity;
    title: string;
    description: string;
  }): Promise<void> {
    await this.selectCategory(data.category);
    if (data.severity) {
      await this.selectSeverity(data.severity);
    }
    await this.fillTitle(data.title);
    await this.fillDescription(data.description);
    await this.submit();
  }

  async expectModalVisible(): Promise<void> {
    await expect(this.modal).toBeVisible();
  }

  async expectModalHidden(): Promise<void> {
    await expect(this.modal).toBeHidden();
  }

  async expectSubmitDisabled(): Promise<void> {
    await expect(this.submitButton).toBeDisabled();
  }

  async expectSubmitEnabled(): Promise<void> {
    await expect(this.submitButton).toBeEnabled();
  }
}
