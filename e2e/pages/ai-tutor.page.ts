import { Page, Locator } from '@playwright/test';

/**
 * Page Object for AI Tutor tab in Task Workspace
 */
export class AiTutorPage {
  readonly page: Page;

  // Tab elements
  readonly aiTab: Locator;
  readonly descriptionTab: Locator;

  // Chat container (visible when AI tab is active and user has access)
  readonly chatContainer: Locator;
  readonly messagesContainer: Locator;
  readonly emptyState: Locator;

  // Input elements
  readonly input: Locator;
  readonly sendButton: Locator;
  readonly loadingIndicator: Locator;

  // Locked state (visible when user doesn't have premium)
  readonly lockedState: Locator;

  // Messages
  readonly userMessages: Locator;
  readonly assistantMessages: Locator;

  constructor(page: Page) {
    this.page = page;

    // Tab elements
    this.aiTab = page.getByTestId('ai-tutor-toggle');
    this.descriptionTab = page.getByTestId('description-tab');

    // Chat container elements
    this.chatContainer = page.getByTestId('ai-tutor-chat');
    this.messagesContainer = page.getByTestId('ai-tutor-messages');
    this.emptyState = page.getByTestId('ai-tutor-empty');

    // Input elements
    this.input = page.getByTestId('ai-tutor-input');
    this.sendButton = page.getByTestId('ai-tutor-send');
    this.loadingIndicator = page.getByTestId('ai-tutor-loading');

    // Locked state
    this.lockedState = page.getByTestId('ai-tutor-locked');

    // Messages
    this.userMessages = page.locator('[data-testid^="ai-tutor-message-user"]');
    this.assistantMessages = page.locator('[data-testid^="ai-tutor-message-assistant"]');
  }

  /**
   * Navigate to a task page (where AI Tutor is available)
   */
  async gotoTask(courseSlug: string, taskSlug: string) {
    await this.page.goto(`/course/${courseSlug}/task/${taskSlug}`);
  }

  /**
   * Wait for task page to load
   */
  async waitForLoad() {
    // Wait for the AI tab to be visible (indicates task page loaded)
    await this.aiTab.waitFor({ state: 'visible', timeout: 15000 }).catch(() => {});
    await this.page.waitForTimeout(500);
  }

  /**
   * Click on AI Tutor tab to open it
   */
  async openAiTab() {
    await this.aiTab.click();
    await this.page.waitForTimeout(300);
  }

  /**
   * Check if AI Tutor tab is visible
   */
  async isAiTabVisible(): Promise<boolean> {
    return await this.aiTab.isVisible().catch(() => false);
  }

  /**
   * Check if AI Tutor chat is open (tab is selected and chat visible)
   */
  async isChatOpen(): Promise<boolean> {
    return await this.chatContainer.isVisible().catch(() => false);
  }

  /**
   * Check if locked state is visible (user doesn't have premium)
   */
  async isLocked(): Promise<boolean> {
    return await this.lockedState.isVisible().catch(() => false);
  }

  /**
   * Check if input is enabled (premium user)
   */
  async isInputEnabled(): Promise<boolean> {
    const isVisible = await this.input.isVisible().catch(() => false);
    if (!isVisible) return false;
    return !(await this.input.isDisabled());
  }

  /**
   * Send a message to AI Tutor
   */
  async sendMessage(message: string) {
    await this.input.fill(message);
    await this.sendButton.click();
  }

  /**
   * Get the number of user messages
   */
  async getUserMessageCount(): Promise<number> {
    return await this.userMessages.count();
  }

  /**
   * Get the number of assistant messages
   */
  async getAssistantMessageCount(): Promise<number> {
    return await this.assistantMessages.count();
  }

  /**
   * Wait for AI response
   */
  async waitForResponse(timeout: number = 30000) {
    // Wait for loading to appear and disappear
    await this.loadingIndicator.waitFor({ state: 'visible', timeout: 5000 }).catch(() => {});
    await this.loadingIndicator.waitFor({ state: 'hidden', timeout });
  }

  /**
   * Check if loading indicator is visible
   */
  async isLoading(): Promise<boolean> {
    return await this.loadingIndicator.isVisible().catch(() => false);
  }

  /**
   * Get placeholder text
   */
  async getPlaceholder(): Promise<string | null> {
    return await this.input.getAttribute('placeholder');
  }
}
