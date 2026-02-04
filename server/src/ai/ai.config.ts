/**
 * AI Service Configuration
 * Centralized configuration for AI tutor and prompt evaluation
 */

// Default AI model (can be overridden via AI_MODEL_NAME env var)
export const DEFAULT_AI_MODEL = "gemini-2.0-flash";

// Daily request limits by subscription type
export const AI_DAILY_LIMITS = {
  FREE: 5, // No subscription (authenticated users only)
  COURSE_SUBSCRIPTION: 30, // Course subscription
  GLOBAL_PREMIUM: 100, // Global premium subscription
  PROMPT_ENGINEERING: 100, // Special limit for PE course
} as const;

// Prompt Engineering course slug for special limit detection
export const PROMPT_ENGINEERING_COURSE_SLUG = "prompt-engineering";

// Response configuration
export const AI_RESPONSE_CONFIG = {
  MAX_RESPONSE_WORDS: 250,
  MAX_CODE_LINES: 3,
};

// Rate limit tiers for UI display
export type AiLimitTier = "free" | "course" | "global" | "prompt_engineering";
