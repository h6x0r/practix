import { useCallback, useState, useEffect } from "react";
import type { Monaco } from "@monaco-editor/react";
import { storage } from "@/lib/storage";

const THEME_STORAGE_KEY = "playground_editor_theme";

export interface EditorTheme {
  id: string;
  name: string;
  type: "dark" | "light";
}

export const EDITOR_THEMES: EditorTheme[] = [
  { id: "practix-dark", name: "Practix Dark", type: "dark" },
  { id: "vs-dark", name: "VS Dark", type: "dark" },
  { id: "one-dark", name: "One Dark", type: "dark" },
  { id: "github-dark", name: "GitHub Dark", type: "dark" },
  { id: "dracula", name: "Dracula", type: "dark" },
  { id: "light", name: "Light", type: "light" },
  { id: "github-light", name: "GitHub Light", type: "light" },
];

export function defineAllThemes(monaco: Monaco) {
  // Practix Dark (default)
  monaco.editor.defineTheme("practix-dark", {
    base: "vs-dark",
    inherit: true,
    rules: [],
    colors: {
      "editor.background": "#0d1117",
      "editor.lineHighlightBackground": "#161b22",
      "editorLineNumber.foreground": "#484f58",
      "editorLineNumber.activeForeground": "#8b949e",
      "editor.selectionBackground": "#264f78",
      "editorCursor.foreground": "#c9d1d9",
    },
  });

  // One Dark
  monaco.editor.defineTheme("one-dark", {
    base: "vs-dark",
    inherit: true,
    rules: [
      { token: "comment", foreground: "5c6370", fontStyle: "italic" },
      { token: "keyword", foreground: "c678dd" },
      { token: "string", foreground: "98c379" },
      { token: "number", foreground: "d19a66" },
      { token: "type", foreground: "e5c07b" },
    ],
    colors: {
      "editor.background": "#282c34",
      "editor.foreground": "#abb2bf",
      "editor.lineHighlightBackground": "#2c313c",
      "editorLineNumber.foreground": "#495162",
      "editorCursor.foreground": "#528bff",
    },
  });

  // GitHub Dark
  monaco.editor.defineTheme("github-dark", {
    base: "vs-dark",
    inherit: true,
    rules: [
      { token: "comment", foreground: "8b949e", fontStyle: "italic" },
      { token: "keyword", foreground: "ff7b72" },
      { token: "string", foreground: "a5d6ff" },
      { token: "number", foreground: "79c0ff" },
      { token: "type", foreground: "ffa657" },
    ],
    colors: {
      "editor.background": "#0d1117",
      "editor.foreground": "#c9d1d9",
      "editor.lineHighlightBackground": "#161b22",
      "editorLineNumber.foreground": "#6e7681",
      "editorCursor.foreground": "#c9d1d9",
    },
  });

  // Dracula
  monaco.editor.defineTheme("dracula", {
    base: "vs-dark",
    inherit: true,
    rules: [
      { token: "comment", foreground: "6272a4", fontStyle: "italic" },
      { token: "keyword", foreground: "ff79c6" },
      { token: "string", foreground: "f1fa8c" },
      { token: "number", foreground: "bd93f9" },
      { token: "type", foreground: "8be9fd", fontStyle: "italic" },
    ],
    colors: {
      "editor.background": "#282a36",
      "editor.foreground": "#f8f8f2",
      "editor.lineHighlightBackground": "#44475a",
      "editorLineNumber.foreground": "#6272a4",
      "editorCursor.foreground": "#f8f8f2",
    },
  });

  // GitHub Light
  monaco.editor.defineTheme("github-light", {
    base: "vs",
    inherit: true,
    rules: [
      { token: "comment", foreground: "6a737d", fontStyle: "italic" },
      { token: "keyword", foreground: "d73a49" },
      { token: "string", foreground: "032f62" },
      { token: "number", foreground: "005cc5" },
      { token: "type", foreground: "6f42c1" },
    ],
    colors: {
      "editor.background": "#ffffff",
      "editor.foreground": "#24292e",
      "editor.lineHighlightBackground": "#f6f8fa",
      "editorLineNumber.foreground": "#6a737d",
      "editorCursor.foreground": "#24292e",
    },
  });
}

export function useEditorThemes(isDark: boolean) {
  const [currentTheme, setCurrentTheme] = useState<string>(() => {
    const stored = storage.getItem(THEME_STORAGE_KEY);
    if (stored) return stored;
    return isDark ? "practix-dark" : "light";
  });

  // Update theme when system theme changes (only if using default)
  useEffect(() => {
    const stored = storage.getItem(THEME_STORAGE_KEY);
    if (!stored) {
      setCurrentTheme(isDark ? "practix-dark" : "light");
    }
  }, [isDark]);

  const setTheme = useCallback((themeId: string) => {
    setCurrentTheme(themeId);
    storage.setItem(THEME_STORAGE_KEY, themeId);
  }, []);

  const getAvailableThemes = useCallback(() => {
    return EDITOR_THEMES;
  }, []);

  const getCurrentThemeInfo = useCallback(() => {
    return EDITOR_THEMES.find((t) => t.id === currentTheme) || EDITOR_THEMES[0];
  }, [currentTheme]);

  return {
    currentTheme,
    setTheme,
    getAvailableThemes,
    getCurrentThemeInfo,
  };
}
