import { describe, it, expect, beforeEach, vi } from "vitest";
import { askAiTutor, getAiLimits, getTierDisplayName } from "./geminiService";

vi.mock("@/lib/api", () => ({
  api: {
    post: vi.fn(),
    get: vi.fn(),
  },
  isAbortError: (error: unknown) =>
    error instanceof Error && error.name === "AbortError",
}));

vi.mock("@/lib/logger", () => ({
  createLogger: () => ({
    info: vi.fn(),
    error: vi.fn(),
    warn: vi.fn(),
    debug: vi.fn(),
  }),
}));

import { api } from "@/lib/api";

describe("askAiTutor", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should return full AI response on success", async () => {
    const mockResponse = {
      answer: "Here is a hint: try using a loop.",
      remaining: 95,
      limit: 100,
      tier: "global" as const,
    };
    vi.mocked(api.post).mockResolvedValueOnce(mockResponse);

    const result = await askAiTutor(
      "task-123",
      "Hello World",
      'print("hello")',
      "How do I fix this?",
      "python",
      "en",
    );

    expect(api.post).toHaveBeenCalledWith(
      "/ai/tutor",
      {
        taskId: "task-123",
        taskTitle: "Hello World",
        userCode: 'print("hello")',
        question: "How do I fix this?",
        language: "python",
        uiLanguage: "en",
      },
      { signal: undefined },
    );
    expect(result).toEqual(mockResponse);
    expect(result.answer).toBe("Here is a hint: try using a loop.");
    expect(result.tier).toBe("global");
  });

  it("should use default uiLanguage when not provided", async () => {
    vi.mocked(api.post).mockResolvedValueOnce({
      answer: "Response",
      remaining: 90,
      limit: 100,
      tier: "global",
    });

    await askAiTutor("task-123", "Task", "code", "question", "python");

    expect(api.post).toHaveBeenCalledWith(
      "/ai/tutor",
      expect.objectContaining({
        uiLanguage: "en",
      }),
      expect.any(Object),
    );
  });

  it("should pass abort signal when provided", async () => {
    const controller = new AbortController();
    vi.mocked(api.post).mockResolvedValueOnce({
      answer: "Response",
      remaining: 90,
      limit: 100,
      tier: "global",
    });

    await askAiTutor("task-123", "Task", "code", "question", "python", "en", {
      signal: controller.signal,
    });

    expect(api.post).toHaveBeenCalledWith("/ai/tutor", expect.any(Object), {
      signal: controller.signal,
    });
  });

  it("should re-throw abort errors", async () => {
    const abortError = new Error("Aborted");
    abortError.name = "AbortError";
    vi.mocked(api.post).mockRejectedValueOnce(abortError);

    await expect(
      askAiTutor("task-123", "Task", "code", "question", "python"),
    ).rejects.toThrow("Aborted");
  });

  it("should throw error with limit message on 403 error", async () => {
    const error = { status: 403, message: "Forbidden" };
    vi.mocked(api.post).mockRejectedValueOnce(error);

    await expect(
      askAiTutor("task-123", "Task", "code", "question", "python"),
    ).rejects.toThrow("daily AI limit");
  });

  it("should throw error with generic message on other errors", async () => {
    vi.mocked(api.post).mockRejectedValueOnce(new Error("Network error"));

    await expect(
      askAiTutor("task-123", "Task", "code", "question", "python"),
    ).rejects.toThrow("trouble connecting");
  });

  it("should handle 500 errors gracefully", async () => {
    const error = { status: 500, message: "Internal Server Error" };
    vi.mocked(api.post).mockRejectedValueOnce(error);

    await expect(
      askAiTutor("task-123", "Task", "code", "question", "python"),
    ).rejects.toThrow("trouble connecting");
  });
});

describe("getAiLimits", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should fetch AI limits without taskId", async () => {
    const mockLimits = {
      tier: "free" as const,
      limit: 5,
      used: 2,
      remaining: 3,
    };
    vi.mocked(api.get).mockResolvedValueOnce(mockLimits);

    const result = await getAiLimits();

    expect(api.get).toHaveBeenCalledWith("/ai/limits");
    expect(result).toEqual(mockLimits);
  });

  it("should fetch AI limits with taskId", async () => {
    const mockLimits = {
      tier: "course" as const,
      limit: 30,
      used: 10,
      remaining: 20,
    };
    vi.mocked(api.get).mockResolvedValueOnce(mockLimits);

    const result = await getAiLimits("task-123");

    expect(api.get).toHaveBeenCalledWith("/ai/limits?taskId=task-123");
    expect(result).toEqual(mockLimits);
  });
});

describe("getTierDisplayName", () => {
  it("should return correct display names", () => {
    expect(getTierDisplayName("free")).toBe("Free");
    expect(getTierDisplayName("course")).toBe("Course Subscription");
    expect(getTierDisplayName("global")).toBe("Global Premium");
    expect(getTierDisplayName("prompt_engineering")).toBe("Prompt Engineering");
  });

  it("should return tier string for unknown tiers", () => {
    expect(getTierDisplayName("unknown" as any)).toBe("unknown");
  });
});
