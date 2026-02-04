import { api, isAbortError } from "@/lib/api";
import { createLogger } from "@/lib/logger";

const log = createLogger("AiTutor");

// AI limit tiers
export type AiLimitTier = "free" | "course" | "global" | "prompt_engineering";

// AI limit info response
export interface AiLimitInfo {
  tier: AiLimitTier;
  limit: number;
  used: number;
  remaining: number;
}

interface AiResponse {
  answer: string;
  remaining: number;
  limit: number;
  tier: AiLimitTier;
}

interface AskOptions {
  signal?: AbortSignal;
}

/**
 * Get current AI usage limits for the authenticated user
 */
export const getAiLimits = async (taskId?: string): Promise<AiLimitInfo> => {
  const params = taskId ? `?taskId=${taskId}` : "";
  return api.get<AiLimitInfo>(`/ai/limits${params}`);
};

/**
 * Ask AI Tutor for help with a coding task
 * Returns both the answer and updated limit info
 */
export const askAiTutor = async (
  taskId: string,
  taskTitle: string,
  userCode: string,
  question: string,
  language: string,
  uiLanguage: string = "en",
  options?: AskOptions,
): Promise<AiResponse> => {
  try {
    const data = await api.post<AiResponse>(
      "/ai/tutor",
      {
        taskId,
        taskTitle,
        userCode,
        question,
        language,
        uiLanguage,
      },
      { signal: options?.signal },
    );

    return data;
  } catch (error: unknown) {
    // Re-throw abort errors to allow proper handling upstream
    if (isAbortError(error)) {
      throw error;
    }
    log.error("AI Tutor request failed", error);
    if (
      error &&
      typeof error === "object" &&
      "status" in error &&
      error.status === 403
    ) {
      throw new Error(
        "You have reached your daily AI limit. Please upgrade to Premium or come back tomorrow!",
      );
    }
    throw new Error(
      "Sorry, I'm having trouble connecting to the AI brain right now. Please try again later.",
    );
  }
};

/**
 * Helper to get tier display name
 */
export const getTierDisplayName = (tier: AiLimitTier): string => {
  const names: Record<AiLimitTier, string> = {
    free: "Free",
    course: "Course Subscription",
    global: "Global Premium",
    prompt_engineering: "Prompt Engineering",
  };
  return names[tier] || tier;
};
