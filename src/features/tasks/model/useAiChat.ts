import { useState, useRef, useEffect, useCallback } from "react";
import { AiChatMessage, Task } from "@/types";
import { askAiTutor, AiLimitInfo } from "../../ai/api/geminiService";
import { isAbortError } from "@/lib/api";

interface AiChatState {
  question: string;
  setQuestion: (q: string) => void;
  chat: AiChatMessage[];
  isLoading: boolean;
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;
  askAi: () => Promise<void>;
  limitInfo: AiLimitInfo | null;
}

export const useAiChat = (
  task: Task | null,
  code: string,
  langLabel: string,
  uiLanguage: string = "en",
): AiChatState => {
  const [question, setQuestion] = useState("");
  const [chat, setChat] = useState<AiChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isOpen, setIsOpen] = useState(true);
  const [limitInfo, setLimitInfo] = useState<AiLimitInfo | null>(null);

  // Track current request for cancellation
  const abortControllerRef = useRef<AbortController | null>(null);
  const isMountedRef = useRef(true);

  // Cleanup on unmount
  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      // Abort any pending request on unmount
      abortControllerRef.current?.abort();
    };
  }, []);

  const askAi = useCallback(async () => {
    if (!question.trim() || !task) return;

    // Abort previous request if still pending
    abortControllerRef.current?.abort();
    const controller = new AbortController();
    abortControllerRef.current = controller;

    const userMsg: AiChatMessage = { role: "user", text: question };
    setChat((prev) => [...prev, userMsg]);
    setQuestion("");
    setIsLoading(true);

    try {
      const response = await askAiTutor(
        task.id,
        task.title,
        code,
        userMsg.text,
        langLabel,
        uiLanguage,
        { signal: controller.signal },
      );

      // Only update state if still mounted and not aborted
      if (isMountedRef.current && !controller.signal.aborted) {
        setChat((prev) => [...prev, { role: "model", text: response.answer }]);
        // Update limit info from response
        setLimitInfo({
          tier: response.tier,
          limit: response.limit,
          used: response.limit - response.remaining,
          remaining: response.remaining,
        });
      }
    } catch (e) {
      // Don't update state for aborted requests
      if (isAbortError(e)) {
        return;
      }
      if (isMountedRef.current) {
        const errorMessage =
          e instanceof Error
            ? e.message
            : "Connection error. Please try again.";
        setChat((prev) => [...prev, { role: "model", text: errorMessage }]);
      }
    } finally {
      if (isMountedRef.current && !controller.signal.aborted) {
        setIsLoading(false);
      }
    }
  }, [question, task, code, langLabel, uiLanguage]);

  return {
    question,
    setQuestion,
    chat,
    isLoading,
    isOpen,
    setIsOpen,
    askAi,
    limitInfo,
  };
};
