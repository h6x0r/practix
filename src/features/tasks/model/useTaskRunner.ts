import { useState, useEffect, useRef, useCallback } from "react";
import { Task, Submission } from "@/types";
import { taskService, RunTestsResult } from "../api/taskService";
import { storage } from "@/lib/storage";
import { createLogger } from "@/lib/logger";
import { isAbortError } from "@/lib/api";
import { useCooldown } from "./useCooldown";
import {
  detectTaskLanguage,
  sanitizeRunResult,
  sanitizeSubmission,
} from "./taskRunner.utils";

// Re-export for backward compatibility
export { detectTaskLanguage } from "./taskRunner.utils";

const log = createLogger("TaskRunner");

export const useTaskRunner = (task: Task | null, courseId?: string) => {
  const [code, setCodeState] = useState("");
  const [activeTab, setActiveTab] = useState<"editor" | "history">("editor");
  const [submissions, setSubmissions] = useState<Submission[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isLoadingSubmissions, setIsLoadingSubmissions] = useState(false);
  const [runResult, setRunResult] = useState<RunTestsResult | null>(null);
  const [isRunResultsOpen, setIsRunResultsOpen] = useState(false);

  const debounceRef = useRef<NodeJS.Timeout | null>(null);
  const isMountedRef = useRef(true);
  const submissionsAbortRef = useRef<AbortController | null>(null);
  const runResultAbortRef = useRef<AbortController | null>(null);
  const runAbortRef = useRef<AbortController | null>(null);
  const submitAbortRef = useRef<AbortController | null>(null);

  const { cooldownRemaining, startCooldown, isRateLimited } = useCooldown();

  // Initialize code when task loads
  useEffect(() => {
    if (task) {
      const savedCode = storage.getTaskCode(task.slug);
      setCodeState(savedCode !== null ? savedCode : task.initialCode);
    }
  }, [task]);

  // Load submissions for the task
  useEffect(() => {
    if (!task?.id) {
      setSubmissions([]);
      return;
    }

    submissionsAbortRef.current?.abort();
    const controller = new AbortController();
    submissionsAbortRef.current = controller;

    setIsLoadingSubmissions(true);
    taskService
      .getTaskSubmissions(task.id, { signal: controller.signal })
      .then((subs) => {
        if (isMountedRef.current && !controller.signal.aborted) {
          setSubmissions(subs);
        }
      })
      .catch((error) => {
        if (isAbortError(error)) return;
        log.warn("Failed to load submissions", error);
        if (isMountedRef.current) setSubmissions([]);
      })
      .finally(() => {
        if (isMountedRef.current && !controller.signal.aborted) {
          setIsLoadingSubmissions(false);
        }
      });
  }, [task?.id]);

  // Load persisted run result for the task
  useEffect(() => {
    if (!task?.id) {
      setRunResult(null);
      return;
    }

    runResultAbortRef.current?.abort();
    const controller = new AbortController();
    runResultAbortRef.current = controller;

    taskService
      .getRunResult(task.id, { signal: controller.signal })
      .then((result) => {
        if (isMountedRef.current && !controller.signal.aborted && result) {
          setRunResult(result);
          log.info("Loaded persisted run result for task", task.slug);
        }
      })
      .catch((error) => {
        if (isAbortError(error)) return;
        log.warn("Failed to load run result", error);
      });
  }, [task?.id]);

  // Save code to localStorage with debounce
  const setCode = useCallback(
    (newCode: string) => {
      setCodeState(newCode);
      if (debounceRef.current) clearTimeout(debounceRef.current);
      if (task) {
        debounceRef.current = setTimeout(() => {
          storage.setTaskCode(task.slug, newCode);
        }, 500);
      }
    },
    [task],
  );

  // Cleanup on unmount
  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      if (debounceRef.current) clearTimeout(debounceRef.current);
      submissionsAbortRef.current?.abort();
      runResultAbortRef.current?.abort();
      runAbortRef.current?.abort();
      submitAbortRef.current?.abort();
    };
  }, []);

  /** Run quick tests (5 tests) — opens bottom panel, doesn't save to DB */
  const runQuickTests = useCallback(async () => {
    if (!task || isRateLimited()) {
      if (task && isRateLimited()) {
        log.warn("Rate limited - please wait before running again");
      }
      return;
    }

    startCooldown();
    runAbortRef.current?.abort();
    const controller = new AbortController();
    runAbortRef.current = controller;

    setIsRunning(true);
    setRunResult(null);
    setIsRunResultsOpen(true);

    const language = detectTaskLanguage(task, courseId);

    try {
      const result = await taskService.runTests(code, task.id, language, {
        signal: controller.signal,
      });
      if (isMountedRef.current && !controller.signal.aborted) {
        setRunResult(sanitizeRunResult(result, 5));
        setIsRunning(false);
      }
    } catch (e) {
      if (isAbortError(e)) return;
      if (isMountedRef.current) {
        setIsRunning(false);
        log.error("runQuickTests failed", e);
        setRunResult(createErrorRunResult());
      }
    }
  }, [task, code, courseId, isRateLimited, startCooldown]);

  /** Submit code for full evaluation (all tests) — saves to DB */
  const submitCode = useCallback(async () => {
    if (!task || isRateLimited()) {
      if (task && isRateLimited()) {
        log.warn("Rate limited - please wait before submitting again");
      }
      return;
    }

    startCooldown();
    submitAbortRef.current?.abort();
    const controller = new AbortController();
    submitAbortRef.current = controller;

    setIsSubmitting(true);
    setIsRunResultsOpen(false);
    setActiveTab("history");

    const language = detectTaskLanguage(task, courseId);

    try {
      const newSub = await taskService.submitCode(code, task.id, language, {
        signal: controller.signal,
      });
      if (isMountedRef.current && !controller.signal.aborted) {
        setSubmissions((prev) => [sanitizeSubmission(newSub), ...prev]);
        setIsSubmitting(false);
      }
    } catch (e) {
      if (isAbortError(e)) return;
      if (isMountedRef.current) {
        setIsSubmitting(false);
        log.error("submitCode failed", e);
        setSubmissions((prev) => [createErrorSubmission(code), ...prev]);
      }
    }
  }, [task, code, courseId, isRateLimited, startCooldown]);

  return {
    code,
    setCode,
    activeTab,
    setActiveTab,
    submissions,
    isRunning,
    isSubmitting,
    isLoadingSubmissions,
    runQuickTests,
    submitCode,
    runResult,
    isRunResultsOpen,
    closeRunResults: () => setIsRunResultsOpen(false),
    cooldownRemaining,
  };
};

function createErrorRunResult(): RunTestsResult {
  return {
    status: "error",
    testsPassed: 0,
    testsTotal: 5,
    testCases: [],
    runtime: "-",
    message: "",
  };
}

function createErrorSubmission(code: string): Submission {
  return {
    id: `error-${Date.now()}`,
    status: "error",
    score: 0,
    runtime: "-",
    createdAt: new Date().toISOString(),
    code,
    message: "",
    testsTotal: 10,
    testsPassed: 0,
  };
}
