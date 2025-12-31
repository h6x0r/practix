
import React, { useState, useRef, useEffect, useContext } from 'react';
import { IconSettings } from '@/components/Icons';
import { AuthContext } from '@/components/Layout';
import { authService } from '@/features/auth/api/authService';
import { useUITranslation } from '@/contexts/LanguageContext';

// Popular coding fonts (Google Fonts - all visually distinct)
const FONT_OPTIONS = [
  { id: 'jetbrains', label: 'JetBrains Mono', value: "'JetBrains Mono', monospace" },
  { id: 'fira', label: 'Fira Code', value: "'Fira Code', monospace" },
  { id: 'source', label: 'Source Code Pro', value: "'Source Code Pro', monospace" },
  { id: 'ibm', label: 'IBM Plex Mono', value: "'IBM Plex Mono', monospace" },
  { id: 'roboto', label: 'Roboto Mono', value: "'Roboto Mono', monospace" },
  { id: 'ubuntu', label: 'Ubuntu Mono', value: "'Ubuntu Mono', monospace" },
  { id: 'space', label: 'Space Mono', value: "'Space Mono', monospace" },
  { id: 'courier', label: 'Courier New', value: "'Courier New', monospace" },
];

interface EditorSettingsDropdownProps {
  onSettingsChange?: () => void;
}

export const EditorSettingsDropdown = ({ onSettingsChange }: EditorSettingsDropdownProps) => {
  const { user, updateUser } = useContext(AuthContext);
  const { tUI } = useUITranslation();
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Local state for immediate UI feedback
  const [settings, setSettings] = useState({
    fontSize: user?.preferences?.editorFontSize ?? 14,
    fontFamily: user?.preferences?.editorFontFamily ?? "'JetBrains Mono', monospace",
    minimap: user?.preferences?.editorMinimap ?? false,
    vimMode: user?.preferences?.editorVimMode ?? false,
    lineNumbers: user?.preferences?.editorLineNumbers ?? true,
  });

  // Sync with user preferences
  useEffect(() => {
    if (user?.preferences) {
      setSettings({
        fontSize: user.preferences.editorFontSize ?? 14,
        fontFamily: user.preferences.editorFontFamily ?? "'JetBrains Mono', monospace",
        minimap: user.preferences.editorMinimap ?? false,
        vimMode: user.preferences.editorVimMode ?? false,
        lineNumbers: user.preferences.editorLineNumbers ?? true,
      });
    }
  }, [user?.preferences]);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Close on Escape key
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setIsOpen(false);
      }
    };
    if (isOpen) {
      document.addEventListener('keydown', handleKeyDown);
      return () => document.removeEventListener('keydown', handleKeyDown);
    }
  }, [isOpen]);

  const updateSetting = async <K extends keyof typeof settings>(key: K, value: typeof settings[K]) => {
    // Update local state immediately for instant feedback
    const newSettings = { ...settings, [key]: value };
    setSettings(newSettings);

    // Persist to backend
    if (user) {
      try {
        const updatedPrefs = {
          editorFontSize: newSettings.fontSize,
          editorFontFamily: newSettings.fontFamily,
          editorMinimap: newSettings.minimap,
          editorVimMode: newSettings.vimMode,
          editorLineNumbers: newSettings.lineNumbers,
          editorTheme: user.preferences?.editorTheme ?? 'vs-dark',
          notifications: user.preferences?.notifications ?? {
            emailDigest: true,
            newCourses: true,
            marketing: false,
            securityAlerts: true
          }
        };
        const updatedUser = await authService.updatePreferences(updatedPrefs);
        updateUser(updatedUser);
        onSettingsChange?.();
      } catch (e) {
        // Revert on error
        setSettings(settings);
      }
    }
  };

  const adjustFontSize = (delta: number) => {
    const newSize = Math.max(10, Math.min(24, settings.fontSize + delta));
    updateSetting('fontSize', newSize);
  };

  const Toggle = ({ checked, onChange }: { checked: boolean; onChange: (v: boolean) => void }) => (
    <button
      onClick={() => onChange(!checked)}
      className={`w-9 h-5 flex items-center rounded-full transition-colors duration-200 ${
        checked ? 'bg-brand-600' : 'bg-gray-600'
      }`}
    >
      <div
        className={`w-3.5 h-3.5 bg-white rounded-full shadow-md transform transition-transform duration-200 ${
          checked ? 'translate-x-5' : 'translate-x-1'
        }`}
      ></div>
    </button>
  );

  // Get current font label
  const currentFontLabel = FONT_OPTIONS.find(f => f.value === settings.fontFamily)?.label || 'JetBrains Mono';

  return (
    <div className="relative" ref={dropdownRef}>
      {/* Gear Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`w-8 h-8 flex items-center justify-center rounded-md transition-colors ${
          isOpen
            ? 'bg-gray-200 dark:bg-[#3d3d3d] text-gray-900 dark:text-white'
            : 'text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-[#333]'
        }`}
        title={tUI('editor.settings')}
      >
        <IconSettings className="w-4 h-4" />
      </button>

      {/* Dropdown Menu */}
      {isOpen && (
        <div className="absolute right-0 top-full mt-2 w-72 bg-white dark:bg-[#2d2d2d] rounded-xl shadow-xl border border-gray-200 dark:border-[#3d3d3d] z-50 overflow-hidden">
          {/* Header */}
          <div className="px-4 py-3 border-b border-gray-200 dark:border-[#3d3d3d]">
            <h4 className="text-sm font-bold text-gray-900 dark:text-white">{tUI('editor.editorSettings')}</h4>
          </div>

          {/* Settings */}
          <div className="p-3 space-y-4">
            {/* Font Size with +/- buttons */}
            <div>
              <div className="text-[10px] font-bold text-gray-500 uppercase mb-2">{tUI('settings.fontSize')}</div>
              <div className="flex items-center justify-between bg-gray-50 dark:bg-[#1e1e1e] rounded-lg p-1 border border-gray-200 dark:border-[#3d3d3d]">
                <button
                  onClick={() => adjustFontSize(-1)}
                  className="w-10 h-8 flex items-center justify-center text-gray-500 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white hover:bg-gray-200 dark:hover:bg-[#333] rounded transition-colors text-lg font-bold"
                >
                  −
                </button>
                <div className="flex-1 text-center">
                  <span className="text-sm font-mono font-bold text-gray-900 dark:text-white">{settings.fontSize}</span>
                  <span className="text-xs text-gray-500 ml-1">px</span>
                </div>
                <button
                  onClick={() => adjustFontSize(1)}
                  className="w-10 h-8 flex items-center justify-center text-gray-500 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white hover:bg-gray-200 dark:hover:bg-[#333] rounded transition-colors text-lg font-bold"
                >
                  +
                </button>
              </div>
              {/* Quick size presets */}
              <div className="flex gap-1.5 mt-2">
                {[12, 14, 16, 18, 20].map(size => (
                  <button
                    key={size}
                    onClick={() => updateSetting('fontSize', size)}
                    className={`flex-1 py-1.5 rounded text-xs font-medium transition-colors ${
                      settings.fontSize === size
                        ? 'bg-brand-600 text-white'
                        : 'bg-gray-50 dark:bg-[#1e1e1e] text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-200 dark:hover:bg-[#333] border border-gray-200 dark:border-[#3d3d3d]'
                    }`}
                  >
                    {size}
                  </button>
                ))}
              </div>
            </div>

            {/* Divider */}
            <div className="border-t border-gray-200 dark:border-[#3d3d3d]"></div>

            {/* Font Family */}
            <div>
              <div className="text-[10px] font-bold text-gray-500 uppercase mb-2">{tUI('editor.fontFamily')}</div>
              <div className="grid grid-cols-2 gap-1.5">
                {FONT_OPTIONS.map(font => (
                  <button
                    key={font.id}
                    onClick={() => updateSetting('fontFamily', font.value)}
                    className={`px-2 py-2 rounded-lg text-xs transition-colors text-left ${
                      settings.fontFamily === font.value
                        ? 'bg-brand-600/20 text-brand-400 border border-brand-500/30'
                        : 'bg-gray-50 dark:bg-[#1e1e1e] text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-200 dark:hover:bg-[#333] border border-gray-200 dark:border-[#3d3d3d]'
                    }`}
                    style={{ fontFamily: font.value }}
                  >
                    {font.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Divider */}
            <div className="border-t border-gray-200 dark:border-[#3d3d3d]"></div>

            {/* Toggles */}
            <div className="space-y-3">
              {/* Minimap */}
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-xs font-medium text-gray-700 dark:text-gray-300">{tUI('settings.minimap')}</div>
                  <div className="text-[10px] text-gray-500">{tUI('editor.minimapHint')}</div>
                </div>
                <Toggle checked={settings.minimap} onChange={(v) => updateSetting('minimap', v)} />
              </div>

              {/* Vim Mode */}
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-xs font-medium text-gray-700 dark:text-gray-300">{tUI('settings.vimBindings')}</div>
                  <div className="text-[10px] text-gray-500">{tUI('editor.vimHint')}</div>
                </div>
                <Toggle checked={settings.vimMode} onChange={(v) => updateSetting('vimMode', v)} />
              </div>

              {/* Line Numbers */}
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-xs font-medium text-gray-700 dark:text-gray-300">{tUI('settings.lineNumbers')}</div>
                  <div className="text-[10px] text-gray-500">{tUI('editor.lineNumbersHint')}</div>
                </div>
                <Toggle checked={settings.lineNumbers} onChange={(v) => updateSetting('lineNumbers', v)} />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
