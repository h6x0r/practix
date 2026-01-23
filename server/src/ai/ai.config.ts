/**
 * AI Service Configuration
 * Centralized configuration for AI tutor and prompt evaluation
 */

// Default AI model (can be overridden via AI_MODEL_NAME env var)
export const DEFAULT_AI_MODEL = 'gemini-2.0-flash';

// Daily request limits by subscription type
export const AI_DAILY_LIMITS = {
  FREE: 5,                    // No subscription
  COURSE_SUBSCRIPTION: 30,    // Course subscription
  GLOBAL_PREMIUM: 100,        // Global premium subscription
  PROMPT_ENGINEERING: 100,    // Special limit for PE course
};

// Response configuration
export const AI_RESPONSE_CONFIG = {
  MAX_RESPONSE_WORDS: 250,
  MAX_CODE_LINES: 3,
};
