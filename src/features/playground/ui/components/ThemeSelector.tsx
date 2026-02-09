import React, { useState, useRef, useEffect } from "react";
import { EDITOR_THEMES, EditorTheme } from "../../hooks/useEditorThemes";
import { useUITranslation } from "@/contexts/LanguageContext";

interface ThemeSelectorProps {
  currentTheme: string;
  onThemeChange: (themeId: string) => void;
}

export const ThemeSelector: React.FC<ThemeSelectorProps> = ({
  currentTheme,
  onThemeChange,
}) => {
  const { tUI } = useUITranslation();
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const currentThemeInfo =
    EDITOR_THEMES.find((t) => t.id === currentTheme) || EDITOR_THEMES[0];

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(e.target as Node)
      ) {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    }

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isOpen]);

  const darkThemes = EDITOR_THEMES.filter((t) => t.type === "dark");
  const lightThemes = EDITOR_THEMES.filter((t) => t.type === "light");

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-1.5 px-2 py-1.5 text-xs text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-[#21262d] rounded transition-colors"
        title={tUI("playground.theme")}
      >
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01"
          />
        </svg>
        <span className="hidden sm:inline">{currentThemeInfo.name}</span>
      </button>

      {isOpen && (
        <div className="absolute top-full right-0 mt-1 bg-white dark:bg-[#161b22] rounded-lg shadow-lg z-50 min-w-[180px] py-2 border border-gray-200 dark:border-[#21262d]">
          {/* Dark Themes */}
          <div className="px-3 py-1 text-[10px] font-semibold text-gray-400 uppercase tracking-wider">
            {tUI("playground.darkThemes")}
          </div>
          {darkThemes.map((theme) => (
            <ThemeOption
              key={theme.id}
              theme={theme}
              isSelected={currentTheme === theme.id}
              onSelect={() => {
                onThemeChange(theme.id);
                setIsOpen(false);
              }}
            />
          ))}

          {/* Divider */}
          <div className="my-2 border-t border-gray-200 dark:border-[#21262d]" />

          {/* Light Themes */}
          <div className="px-3 py-1 text-[10px] font-semibold text-gray-400 uppercase tracking-wider">
            {tUI("playground.lightThemes")}
          </div>
          {lightThemes.map((theme) => (
            <ThemeOption
              key={theme.id}
              theme={theme}
              isSelected={currentTheme === theme.id}
              onSelect={() => {
                onThemeChange(theme.id);
                setIsOpen(false);
              }}
            />
          ))}
        </div>
      )}
    </div>
  );
};

interface ThemeOptionProps {
  theme: EditorTheme;
  isSelected: boolean;
  onSelect: () => void;
}

const ThemeOption: React.FC<ThemeOptionProps> = ({
  theme,
  isSelected,
  onSelect,
}) => {
  return (
    <button
      onClick={onSelect}
      className={`w-full text-left px-3 py-1.5 text-xs hover:bg-gray-100 dark:hover:bg-[#21262d] transition-colors flex items-center gap-2 ${
        isSelected ? "bg-gray-100 dark:bg-[#21262d]" : ""
      }`}
    >
      <div
        className={`w-3 h-3 rounded-sm ${
          theme.type === "dark" ? "bg-gray-800" : "bg-gray-200"
        } border border-gray-300 dark:border-gray-600`}
      />
      <span className="text-gray-700 dark:text-gray-300">{theme.name}</span>
      {isSelected && (
        <svg
          className="w-3 h-3 ml-auto text-brand-500"
          fill="currentColor"
          viewBox="0 0 20 20"
        >
          <path
            fillRule="evenodd"
            d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
            clipRule="evenodd"
          />
        </svg>
      )}
    </button>
  );
};

export default ThemeSelector;
