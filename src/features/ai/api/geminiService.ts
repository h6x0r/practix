import { api, isAbortError } from '@/lib/api';
import { createLogger } from '@/lib/logger';

const log = createLogger('AiTutor');

interface AiResponse {
  answer: string;
  remaining?: number;
}

interface AskOptions {
  signal?: AbortSignal;
}

export const askAiTutor = async (
  taskId: string,
  taskTitle: string,
  userCode: string,
  question: string,
  language: string,
  uiLanguage: string = 'en',
  options?: AskOptions
): Promise<string> => {
  try {
    const data = await api.post<AiResponse>('/ai/tutor', {
        taskId,
        taskTitle,
        userCode,
        question,
        language,
        uiLanguage
    }, { signal: options?.signal });

    return data.answer;
  } catch (error: unknown) {
    // Re-throw abort errors to allow proper handling upstream
    if (isAbortError(error)) {
      throw error;
    }
    log.error('AI Tutor request failed', error);
    if (error && typeof error === 'object' && 'status' in error && error.status === 403) {
        return "You have reached your daily AI limit. Please upgrade to Premium or come back tomorrow!";
    }
    return "Sorry, I'm having trouble connecting to the AI brain right now. Please try again later.";
  }
};