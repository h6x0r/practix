import { Page, Locator } from '@playwright/test';

/**
 * Page Object for Roadmap page
 */
export class RoadmapPage {
  readonly page: Page;

  // Intro Section
  readonly introTitle: Locator;
  readonly startButton: Locator;

  // Wizard Navigation
  readonly wizardContainer: Locator;
  readonly backButton: Locator;
  readonly nextButton: Locator;
  readonly wizardProgress: Locator;

  // Step 1: Languages
  readonly languageOptions: Locator;
  readonly pythonOption: Locator;
  readonly javascriptOption: Locator;
  readonly javaOption: Locator;
  readonly goOption: Locator;

  // Step 2: Experience
  readonly experienceOptions: Locator;

  // Step 3: Interests
  readonly interestOptions: Locator;
  readonly backendInterest: Locator;
  readonly aiMlInterest: Locator;
  readonly algorithmsInterest: Locator;

  // Step 4: Goal
  readonly goalOptions: Locator;
  readonly findJobGoal: Locator;
  readonly seniorGoal: Locator;
  readonly startupGoal: Locator;
  readonly masterSkillGoal: Locator;

  // Step 5: Time
  readonly hoursOptions: Locator;
  readonly monthsOptions: Locator;

  // Generating
  readonly generatingContainer: Locator;
  readonly generatingProgress: Locator;
  readonly generatingText: Locator;

  // Variants
  readonly variantsContainer: Locator;
  readonly variantCards: Locator;
  readonly selectVariantButtons: Locator;

  // Result
  readonly resultContainer: Locator;
  readonly roadmapTitle: Locator;
  readonly roadmapMilestones: Locator;
  readonly regenerateButton: Locator;

  // Error
  readonly errorMessage: Locator;

  // Loading
  readonly loadingSpinner: Locator;

  constructor(page: Page) {
    this.page = page;

    // Intro
    this.introTitle = page.locator('h1').filter({ hasText: /roadmap|learning path/i });
    this.startButton = page.getByRole('button', { name: /start|begin|create/i });

    // Wizard
    this.wizardContainer = page.locator('[data-testid="wizard-container"]');
    this.backButton = page.getByRole('button', { name: /back|previous/i });
    this.nextButton = page.getByRole('button', { name: /next|continue|generate/i });
    this.wizardProgress = page.locator('[data-testid="wizard-progress"]');

    // Languages
    this.languageOptions = page.locator('[data-testid="language-option"]');
    this.pythonOption = page.locator('button', { hasText: 'Python' });
    this.javascriptOption = page.locator('button', { hasText: 'JavaScript' });
    this.javaOption = page.locator('button', { hasText: /^Java$/ });
    this.goOption = page.locator('button', { hasText: /^Go$/ });

    // Experience
    this.experienceOptions = page.locator('[data-testid="experience-option"]');

    // Interests
    this.interestOptions = page.locator('[data-testid="interest-option"]');
    this.backendInterest = page.locator('button', { hasText: /backend/i });
    this.aiMlInterest = page.locator('button', { hasText: /AI|Machine Learning/i });
    this.algorithmsInterest = page.locator('button', { hasText: /algorithm/i });

    // Goals
    this.goalOptions = page.locator('[data-testid="goal-option"]');
    this.findJobGoal = page.locator('button', { hasText: /job|career/i });
    this.seniorGoal = page.locator('button', { hasText: /senior/i });
    this.startupGoal = page.locator('button', { hasText: /startup/i });
    this.masterSkillGoal = page.locator('button', { hasText: /master|skill/i });

    // Time
    this.hoursOptions = page.locator('[data-testid="hours-option"]');
    this.monthsOptions = page.locator('[data-testid="months-option"]');

    // Generating
    this.generatingContainer = page.locator('[data-testid="generating-container"]');
    this.generatingProgress = page.locator('[data-testid="generating-progress"]');
    this.generatingText = page.locator('[data-testid="generating-text"]');

    // Variants
    this.variantsContainer = page.locator('[data-testid="variants-container"]');
    this.variantCards = page.locator('[data-testid="variant-card"]');
    this.selectVariantButtons = page.locator('[data-testid="select-variant"]');

    // Result
    this.resultContainer = page.locator('[data-testid="result-container"]');
    this.roadmapTitle = page.locator('[data-testid="roadmap-title"]');
    this.roadmapMilestones = page.locator('[data-testid="roadmap-milestone"]');
    this.regenerateButton = page.getByRole('button', { name: /regenerate|new/i });

    // Error and Loading
    this.errorMessage = page.locator('[data-testid="error-message"]');
    this.loadingSpinner = page.locator('[data-testid="loading-spinner"]');
  }

  /**
   * Navigate to roadmap page
   */
  async goto() {
    await this.page.goto('/roadmap');
  }

  /**
   * Wait for page to load
   */
  async waitForLoad() {
    // Wait for either intro or result to be visible
    await Promise.race([
      this.introTitle.waitFor({ state: 'visible', timeout: 15000 }),
      this.resultContainer.waitFor({ state: 'visible', timeout: 15000 }),
      this.loadingSpinner.waitFor({ state: 'hidden', timeout: 15000 }),
    ]).catch(() => {});
    await this.page.waitForTimeout(500);
  }

  /**
   * Start the wizard
   */
  async startWizard() {
    await this.startButton.click();
    await this.page.waitForTimeout(300);
  }

  /**
   * Go to next wizard step
   */
  async goNext() {
    await this.nextButton.click();
    await this.page.waitForTimeout(300);
  }

  /**
   * Go to previous wizard step
   */
  async goBack() {
    await this.backButton.click();
    await this.page.waitForTimeout(300);
  }

  /**
   * Select a language
   */
  async selectLanguage(language: 'python' | 'javascript' | 'java' | 'go') {
    const optionMap = {
      python: this.pythonOption,
      javascript: this.javascriptOption,
      java: this.javaOption,
      go: this.goOption,
    };
    await optionMap[language].click();
  }

  /**
   * Select experience level (by index 0-4)
   */
  async selectExperience(index: number) {
    const options = this.page.locator('[class*="cursor-pointer"]').filter({ hasText: /experience|year|beginner|junior|senior/i });
    const option = options.nth(index);
    if (await option.isVisible()) {
      await option.click();
    }
  }

  /**
   * Select an interest
   */
  async selectInterest(interest: 'backend' | 'ai-ml' | 'algorithms') {
    const optionMap = {
      backend: this.backendInterest,
      'ai-ml': this.aiMlInterest,
      algorithms: this.algorithmsInterest,
    };
    await optionMap[interest].click();
  }

  /**
   * Select a goal
   */
  async selectGoal(goal: 'job' | 'senior' | 'startup' | 'master') {
    const optionMap = {
      job: this.findJobGoal,
      senior: this.seniorGoal,
      startup: this.startupGoal,
      master: this.masterSkillGoal,
    };
    await optionMap[goal].click();
  }

  /**
   * Check if intro screen is visible
   */
  async isIntroVisible(): Promise<boolean> {
    return await this.startButton.isVisible().catch(() => false);
  }

  /**
   * Check if roadmap result is visible
   */
  async isResultVisible(): Promise<boolean> {
    return await this.resultContainer.isVisible().catch(() => false);
  }

  /**
   * Check if variants are visible
   */
  async areVariantsVisible(): Promise<boolean> {
    return await this.variantsContainer.isVisible().catch(() => false);
  }

  /**
   * Get variant count
   */
  async getVariantCount(): Promise<number> {
    return await this.variantCards.count();
  }

  /**
   * Select a variant by index
   */
  async selectVariant(index: number) {
    const button = this.selectVariantButtons.nth(index);
    if (await button.isVisible()) {
      await button.click();
    }
  }

  /**
   * Wait for generation to complete
   */
  async waitForGeneration(timeout: number = 60000) {
    await this.generatingContainer.waitFor({ state: 'visible', timeout: 10000 }).catch(() => {});
    await this.variantsContainer.waitFor({ state: 'visible', timeout }).catch(() => {});
  }

  /**
   * Check if generating is in progress
   */
  async isGenerating(): Promise<boolean> {
    return await this.generatingContainer.isVisible().catch(() => false);
  }
}
