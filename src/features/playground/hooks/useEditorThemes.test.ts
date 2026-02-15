import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import {
  useEditorThemes,
  EDITOR_THEMES,
  defineAllThemes,
} from "./useEditorThemes";

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

describe("useEditorThemes", () => {
  beforeEach(() => {
    Object.keys(mockStorage).forEach((key) => delete mockStorage[key]);
    vi.clearAllMocks();
  });

  describe("theme selection", () => {
    it("should return practix-dark as default theme when isDark is true", () => {
      const { result } = renderHook(() => useEditorThemes(true));

      expect(result.current.currentTheme).toBe("practix-dark");
    });

    it("should return light as default theme when isDark is false", () => {
      const { result } = renderHook(() => useEditorThemes(false));

      expect(result.current.currentTheme).toBe("light");
    });

    it("should use stored theme preference", () => {
      mockStorage["playground_editor_theme"] = "dracula";

      const { result } = renderHook(() => useEditorThemes(true));

      expect(result.current.currentTheme).toBe("dracula");
    });

    it("should change theme and persist to storage", () => {
      const { result } = renderHook(() => useEditorThemes(true));

      act(() => {
        result.current.setTheme("one-dark");
      });

      expect(result.current.currentTheme).toBe("one-dark");
      expect(mockStorage["playground_editor_theme"]).toBe("one-dark");
    });
  });

  describe("theme utilities", () => {
    it("should return all available themes", () => {
      const { result } = renderHook(() => useEditorThemes(true));

      const themes = result.current.getAvailableThemes();

      expect(themes).toEqual(EDITOR_THEMES);
      expect(themes.length).toBeGreaterThan(0);
    });

    it("should return current theme info", () => {
      mockStorage["playground_editor_theme"] = "dracula";

      const { result } = renderHook(() => useEditorThemes(true));

      const themeInfo = result.current.getCurrentThemeInfo();

      expect(themeInfo.id).toBe("dracula");
      expect(themeInfo.name).toBe("Dracula");
      expect(themeInfo.type).toBe("dark");
    });
  });
});

describe("EDITOR_THEMES", () => {
  it("should have both dark and light themes", () => {
    const darkThemes = EDITOR_THEMES.filter((t) => t.type === "dark");
    const lightThemes = EDITOR_THEMES.filter((t) => t.type === "light");

    expect(darkThemes.length).toBeGreaterThan(0);
    expect(lightThemes.length).toBeGreaterThan(0);
  });

  it("should have unique theme ids", () => {
    const ids = EDITOR_THEMES.map((t) => t.id);
    const uniqueIds = new Set(ids);

    expect(uniqueIds.size).toBe(ids.length);
  });

  it("should include practix-dark and light themes", () => {
    expect(EDITOR_THEMES.find((t) => t.id === "practix-dark")).toBeDefined();
    expect(EDITOR_THEMES.find((t) => t.id === "light")).toBeDefined();
  });
});

describe("defineAllThemes", () => {
  it("should define all custom themes on Monaco", () => {
    const mockMonaco = {
      editor: {
        defineTheme: vi.fn(),
      },
    };

    defineAllThemes(mockMonaco as any);

    // Should define custom themes (practix-dark, one-dark, github-dark, dracula, github-light)
    expect(mockMonaco.editor.defineTheme).toHaveBeenCalledTimes(5);

    // Check that practix-dark is defined
    expect(mockMonaco.editor.defineTheme).toHaveBeenCalledWith(
      "practix-dark",
      expect.objectContaining({
        base: "vs-dark",
        inherit: true,
      }),
    );
  });
});
