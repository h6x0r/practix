import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { usePlaygroundStorage } from "./usePlaygroundStorage";

// Mock localStorage
const mockStorage: Record<string, string> = {};
const localStorageMock = {
  getItem: vi.fn((key: string) => mockStorage[key] ?? null),
  setItem: vi.fn((key: string, value: string) => {
    mockStorage[key] = value;
  }),
  removeItem: vi.fn((key: string) => {
    delete mockStorage[key];
  }),
  clear: vi.fn(),
  length: 0,
  key: vi.fn(),
};

Object.defineProperty(window, "localStorage", { value: localStorageMock });

vi.mock("@/lib/logger", () => ({
  createLogger: () => ({
    info: vi.fn(),
    debug: vi.fn(),
    error: vi.fn(),
  }),
}));

describe("usePlaygroundStorage", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    Object.keys(mockStorage).forEach((key) => delete mockStorage[key]);
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("should return null savedState initially when no data stored", () => {
    const { result } = renderHook(() => usePlaygroundStorage());

    expect(result.current.savedState).toBeNull();
    expect(result.current.lastSavedAt).toBeNull();
  });

  it("should load saved state from storage on mount", () => {
    const savedData = {
      code: 'console.log("test")',
      language: "typescript",
      savedAt: "2024-01-15T10:00:00.000Z",
    };
    mockStorage["playground_state"] = JSON.stringify(savedData);

    const { result } = renderHook(() => usePlaygroundStorage());

    expect(result.current.savedState).toEqual(savedData);
    expect(result.current.lastSavedAt).toEqual(new Date(savedData.savedAt));
  });

  it("should save state with debounce", async () => {
    const { result } = renderHook(() => usePlaygroundStorage());

    act(() => {
      result.current.saveState('console.log("hello")', "typescript");
    });

    // Should not save immediately
    expect(mockStorage["playground_state"]).toBeUndefined();

    // Fast-forward debounce timer
    act(() => {
      vi.advanceTimersByTime(1000);
    });

    // Now it should be saved
    expect(mockStorage["playground_state"]).toBeDefined();
    const saved = JSON.parse(mockStorage["playground_state"]);
    expect(saved.code).toBe('console.log("hello")');
    expect(saved.language).toBe("typescript");
  });

  it("should debounce multiple rapid saves", () => {
    const { result } = renderHook(() => usePlaygroundStorage());

    act(() => {
      result.current.saveState("code1", "go");
    });

    act(() => {
      vi.advanceTimersByTime(500);
    });

    act(() => {
      result.current.saveState("code2", "go");
    });

    act(() => {
      vi.advanceTimersByTime(500);
    });

    act(() => {
      result.current.saveState("code3", "go");
    });

    // Only after full debounce from last save
    act(() => {
      vi.advanceTimersByTime(1000);
    });

    const saved = JSON.parse(mockStorage["playground_state"]);
    expect(saved.code).toBe("code3");
  });

  it("should clear state", () => {
    mockStorage["playground_state"] = JSON.stringify({
      code: "test",
      language: "go",
      savedAt: new Date().toISOString(),
    });

    const { result } = renderHook(() => usePlaygroundStorage());

    act(() => {
      result.current.clearState();
    });

    expect(result.current.savedState).toBeNull();
    expect(result.current.lastSavedAt).toBeNull();
  });

  it("should update lastSavedAt after save", () => {
    const { result } = renderHook(() => usePlaygroundStorage());

    expect(result.current.lastSavedAt).toBeNull();

    act(() => {
      result.current.saveState("test code", "python");
    });

    act(() => {
      vi.advanceTimersByTime(1000);
    });

    expect(result.current.lastSavedAt).not.toBeNull();
  });
});
