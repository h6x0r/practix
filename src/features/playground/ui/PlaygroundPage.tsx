import React, { useState, useEffect, useContext, useCallback, useRef } from 'react';
import Editor, { loader, Monaco } from '@monaco-editor/react';
import type { editor } from 'monaco-editor';
import { playgroundService, ExecutionResult, isRateLimitError, extractRateLimitInfo, RateLimitInfo } from '../api/playgroundService';
import { AuthContext, ThemeContext } from '@/components/Layout';
import { IconPlay, IconRefresh, IconChevronDown } from '@/components/Icons';
import { TimerStopwatch } from '../../tasks/ui/components/TimerStopwatch';
import { EditorSettingsDropdown } from '../../tasks/ui/components/EditorSettingsDropdown';
import { useUITranslation } from '@/contexts/LanguageContext';
import { createLogger } from '@/lib/logger';
import { ApiError } from '@/lib/api';

const log = createLogger('Playground');

loader.config({
  paths: { vs: 'https://cdn.jsdelivr.net/npm/monaco-editor@0.46.0/min/vs' },
});

// Define custom dark theme matching the task details editor
const defineCustomTheme = (monaco: Monaco) => {
  monaco.editor.defineTheme('practix-dark', {
    base: 'vs-dark',
    inherit: true,
    rules: [],
    colors: {
      'editor.background': '#0d1117',
      'editor.lineHighlightBackground': '#161b22',
      'editorLineNumber.foreground': '#484f58',
      'editorLineNumber.activeForeground': '#8b949e',
      'editor.selectionBackground': '#264f78',
      'editorCursor.foreground': '#c9d1d9',
    }
  });
};

const CODE_TEMPLATES: Record<string, string> = {
  go: `package main

import "fmt"

func main() {
    for i := 1; i <= 5; i++ {
        fmt.Printf("Iteration %d: Hello from Go!\\n", i)
    }
}
`,
  java: `public class Main {
    public static void main(String[] args) {
        for (int i = 1; i <= 5; i++) {
            System.out.println("Iteration " + i + ": Hello from Java!");
        }
    }
}
`,
  python: `for i in range(1, 6):
    print(f"Iteration {i}: Hello from Python!")
`,
  typescript: `for (let i: number = 1; i <= 5; i++) {
    console.log(\`Iteration \${i}: Hello from TypeScript!\`);
}
`,
};

const LANGUAGE_COLORS: Record<string, string> = {
  go: 'text-cyan-400',
  java: 'text-orange-400',
  python: 'text-green-400',
  typescript: 'text-blue-400',
};

const LANGUAGE_DISPLAY: Record<string, string> = {
  go: 'Go',
  java: 'Java',
  python: 'Python',
  typescript: 'TypeScript',
};

const FILE_EXTENSIONS: Record<string, string> = {
  go: '.go',
  java: '.java',
  python: '.py',
  typescript: '.ts',
};

const PlaygroundPage = () => {
  const { user } = useContext(AuthContext);
  const { isDark } = useContext(ThemeContext);
  const { tUI } = useUITranslation();
  const [language, setLanguage] = useState('go');
  const [code, setCode] = useState(CODE_TEMPLATES.go);
  const [output, setOutput] = useState<ExecutionResult | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [showLangDropdown, setShowLangDropdown] = useState(false);
  const [judgeAvailable, setJudgeAvailable] = useState(true);
  const [showSavePopup, setShowSavePopup] = useState(false);
  const editorRef = useRef<editor.IStandaloneCodeEditor | null>(null);

  // Rate limiting state
  const [cooldownSeconds, setCooldownSeconds] = useState(0);
  const [rateLimitInfo, setRateLimitInfo] = useState<RateLimitInfo | null>(null);
  const cooldownIntervalRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    playgroundService.getJudgeStatus()
      .then((status) => {
        setJudgeAvailable(status.available);
      })
      .catch((error) => {
        log.error('Failed to get judge status', error);
        setJudgeAvailable(false);
      });

    // Fetch rate limit info
    playgroundService.getRateLimitInfo()
      .then(setRateLimitInfo)
      .catch((error) => {
        log.error('Failed to get rate limit info', error);
      });
  }, []);

  // Cleanup cooldown interval on unmount
  useEffect(() => {
    return () => {
      if (cooldownIntervalRef.current) {
        clearInterval(cooldownIntervalRef.current);
      }
    };
  }, []);

  // Start cooldown timer
  const startCooldown = useCallback((seconds: number) => {
    setCooldownSeconds(seconds);

    if (cooldownIntervalRef.current) {
      clearInterval(cooldownIntervalRef.current);
    }

    cooldownIntervalRef.current = setInterval(() => {
      setCooldownSeconds((prev) => {
        if (prev <= 1) {
          if (cooldownIntervalRef.current) {
            clearInterval(cooldownIntervalRef.current);
            cooldownIntervalRef.current = null;
          }
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
  }, []);

  // Handle editor mount to add Cmd+S / Ctrl+S keybinding
  const handleEditorMount = useCallback((editorInstance: editor.IStandaloneCodeEditor, monaco: Monaco) => {
    editorRef.current = editorInstance;

    // Override Cmd+S / Ctrl+S to show popup instead of browser save
    editorInstance.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, () => {
      setShowSavePopup(true);
    });
  }, []);

  // Also handle Cmd+S / Ctrl+S at document level to prevent browser save dialog
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 's') {
        e.preventDefault();
        setShowSavePopup(true);
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, []);

  const handleLanguageChange = useCallback((newLang: string) => {
    setLanguage(newLang);
    setCode(CODE_TEMPLATES[newLang] || '// Start coding...');
    setOutput(null);
    setShowLangDropdown(false);
  }, []);

  const handleRun = async () => {
    // Don't run if on cooldown
    if (cooldownSeconds > 0) {
      return;
    }

    setIsRunning(true);
    setOutput(null);

    try {
      const result = await playgroundService.runCode(code, language);
      setOutput(result);

      // Start cooldown after successful run
      if (rateLimitInfo) {
        startCooldown(rateLimitInfo.rateLimitSeconds);
      }
    } catch (error: unknown) {
      // Handle rate limit error specially
      if (error instanceof ApiError && error.status === 429) {
        const rateLimitData = extractRateLimitInfo(error);
        startCooldown(rateLimitData.retryAfter);

        setOutput({
          status: 'error',
          statusId: 429,
          description: 'Rate Limited',
          stdout: '',
          stderr: '',
          compileOutput: '',
          time: '0',
          memory: 0,
          exitCode: null,
          message: rateLimitData.message,
        });
      } else {
        const errorMessage = error instanceof Error ? error.message : 'Failed to execute code';
        setOutput({
          status: 'error',
          statusId: 13,
          description: 'Error',
          stdout: '',
          stderr: errorMessage,
          compileOutput: '',
          time: '0',
          memory: 0,
          exitCode: null,
          message: errorMessage,
        });
      }
    } finally {
      setIsRunning(false);
    }
  };

  const handleReset = () => {
    setCode(CODE_TEMPLATES[language] || '');
    setOutput(null);
  };

  const editorOptions = {
    minimap: { enabled: user?.preferences?.editorMinimap ?? false },
    fontSize: user?.preferences?.editorFontSize || 14,
    lineNumbers: (user?.preferences?.editorLineNumbers !== false ? 'on' : 'off') as 'on' | 'off',
    scrollBeyondLastLine: false,
    automaticLayout: true,
    padding: { top: 16, bottom: 16 },
    fontFamily: user?.preferences?.editorFontFamily || "'JetBrains Mono', 'Courier New', monospace",
    lineHeight: 24,
    renderLineHighlight: 'all' as const,
    hideCursorInOverviewRuler: true,
    overviewRulerBorder: false,
  };


  return (
    <div className="h-[calc(100vh-7rem)] flex bg-white dark:bg-[#0d1117] rounded-xl overflow-hidden shadow-sm dark:shadow-none border border-gray-200 dark:border-gray-800">
      {/* Editor Section */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* File Tabs - matching task details style */}
        <div className="flex h-[41px] bg-gray-50 dark:bg-[#161b22] border-b border-gray-200 dark:border-[#21262d] select-none flex-shrink-0">
          {/* File Tab with Language Selector */}
          <div className="relative">
            <button
              onClick={() => setShowLangDropdown(!showLangDropdown)}
              className={`px-4 py-2.5 text-xs flex items-center gap-2 border-t-2 transition-colors bg-white dark:bg-[#0d1117] text-gray-900 dark:text-white border-brand-500`}
            >
              <span className={LANGUAGE_COLORS[language]}>
                {LANGUAGE_DISPLAY[language]?.charAt(0) || language.charAt(0).toUpperCase()}
              </span>
              main{FILE_EXTENSIONS[language]}
              <IconChevronDown className={`w-3 h-3 text-gray-400 transition-transform ${showLangDropdown ? 'rotate-180' : ''}`} />
            </button>

            {showLangDropdown && (
              <div className="absolute top-full left-0 mt-0 bg-white dark:bg-[#161b22] rounded-b-lg shadow-lg z-50 min-w-[160px] py-1 border border-gray-200 dark:border-[#21262d] border-t-0">
                {Object.keys(CODE_TEMPLATES).map((lang) => (
                  <button
                    key={lang}
                    onClick={() => handleLanguageChange(lang)}
                    className={`w-full text-left px-4 py-2 text-xs hover:bg-gray-100 dark:hover:bg-[#21262d] transition-colors flex items-center gap-2 ${
                      language === lang ? 'bg-gray-100 dark:bg-[#21262d]' : ''
                    }`}
                  >
                    <span className={LANGUAGE_COLORS[lang]}>
                      {LANGUAGE_DISPLAY[lang]?.charAt(0) || lang.charAt(0).toUpperCase()}
                    </span>
                    <span className="text-gray-700 dark:text-gray-300">{LANGUAGE_DISPLAY[lang]}</span>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Status indicator */}
          <div className="flex items-center gap-3 px-3 text-xs text-gray-400">
            <div className="flex items-center gap-1.5">
              <div className={`w-1.5 h-1.5 rounded-full ${judgeAvailable ? 'bg-green-500' : 'bg-yellow-500'}`} />
              {judgeAvailable ? 'Ready' : 'Mock'}
            </div>

            {/* Rate limit indicator */}
            {rateLimitInfo && (
              <div className="flex items-center gap-1.5" title={rateLimitInfo.isPremium ? 'Premium: 5s cooldown' : 'Free: 10s cooldown'}>
                {rateLimitInfo.isPremium ? (
                  <span className="px-1.5 py-0.5 bg-gradient-to-r from-amber-500/20 to-orange-500/20 text-amber-500 rounded text-[10px] font-medium">
                    Premium
                  </span>
                ) : (
                  <span className="text-gray-500">
                    {rateLimitInfo.rateLimitSeconds}s limit
                  </span>
                )}
              </div>
            )}
          </div>

          {/* Spacer */}
          <div className="flex-1" />

          {/* Tools */}
          <div className="flex items-center gap-1 px-2">
            <TimerStopwatch />
            <EditorSettingsDropdown />
          </div>
        </div>

        {/* Editor */}
        <div className="flex-1 relative bg-gray-50 dark:bg-[#0d1117]">
          <Editor
            height="100%"
            language={language === 'cpp' ? 'cpp' : language}
            theme={isDark ? 'practix-dark' : 'light'}
            beforeMount={defineCustomTheme}
            onMount={handleEditorMount}
            value={code}
            onChange={(value) => setCode(value || '')}
            options={editorOptions}
            loading={
              <div className="flex h-full items-center justify-center text-gray-500 text-sm bg-gray-50 dark:bg-[#0d1117]">
                <span className="animate-pulse">Loading editor...</span>
              </div>
            }
          />
        </div>
      </div>

      {/* Output Panel */}
      <div className="w-[380px] flex-shrink-0 flex flex-col border-l border-gray-200 dark:border-[#21262d] bg-white dark:bg-[#0d1117]">
        {/* Output Header */}
        <div className="flex h-[41px] items-center justify-between px-4 bg-gray-50 dark:bg-[#161b22] border-b border-gray-200 dark:border-[#21262d]">
          <div className="flex items-center gap-3">
            <span className="text-xs font-medium text-gray-600 dark:text-gray-400 uppercase tracking-wide">Output</span>
            {output && output.time !== '0' && (
              <span className="text-xs text-gray-400 font-mono">{output.time}s</span>
            )}
          </div>

          <div className="flex items-center gap-1">
            <button
              onClick={handleReset}
              className="p-1.5 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-gray-200 dark:hover:bg-[#21262d] rounded transition-colors"
              title="Reset"
            >
              <IconRefresh className="w-3.5 h-3.5" />
            </button>

            <button
              onClick={handleRun}
              disabled={isRunning || cooldownSeconds > 0}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-semibold text-xs transition-all ${
                isRunning || cooldownSeconds > 0
                  ? 'bg-gray-200 dark:bg-[#21262d] text-gray-400 cursor-not-allowed'
                  : 'bg-green-500 hover:bg-green-600 text-white'
              }`}
              title={cooldownSeconds > 0 ? `Wait ${cooldownSeconds}s` : undefined}
            >
              {isRunning ? (
                <>
                  <svg className="w-3 h-3 animate-spin" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  <span>Running</span>
                </>
              ) : cooldownSeconds > 0 ? (
                <>
                  <svg className="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="10" />
                    <polyline points="12 6 12 12 16 14" />
                  </svg>
                  <span>{cooldownSeconds}s</span>
                </>
              ) : (
                <>
                  <IconPlay className="w-3 h-3" />
                  <span>Run</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Output Content */}
        <div className="flex-1 overflow-auto p-4 bg-gray-50 dark:bg-[#0d1117]">
          {!output && !isRunning && (
            <div className="h-full flex items-center justify-center text-gray-300 dark:text-gray-600 text-sm">
              Run code to see output
            </div>
          )}

          {isRunning && (
            <div className="h-full flex items-center justify-center">
              <div className="flex items-center gap-2 text-gray-400 text-sm">
                <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                <span>Executing...</span>
              </div>
            </div>
          )}

          {output && !isRunning && (
            <div className="space-y-3">
              {/* Compile Error */}
              {output.compileOutput && (
                <pre className="p-3 bg-red-50 dark:bg-red-900/20 rounded-lg text-red-600 dark:text-red-400 text-xs font-mono whitespace-pre-wrap overflow-x-auto border border-red-200 dark:border-red-900/30">
                  {output.compileOutput}
                </pre>
              )}

              {/* Stdout */}
              {output.stdout && (
                <pre className="p-3 bg-white dark:bg-[#161b22] rounded-lg text-gray-800 dark:text-gray-200 text-sm font-mono whitespace-pre-wrap overflow-x-auto border border-gray-200 dark:border-[#21262d]">
                  {output.stdout}
                </pre>
              )}

              {/* Stderr */}
              {output.stderr && (
                <pre className="p-3 bg-red-50 dark:bg-red-900/20 rounded-lg text-red-600 dark:text-red-400 text-xs font-mono whitespace-pre-wrap overflow-x-auto border border-red-200 dark:border-red-900/30">
                  {output.stderr}
                </pre>
              )}

              {/* Message */}
              {output.message && !output.stderr && !output.compileOutput && (
                <pre className="p-3 bg-white dark:bg-[#161b22] rounded-lg text-gray-600 dark:text-gray-400 text-sm font-mono whitespace-pre-wrap border border-gray-200 dark:border-[#21262d]">
                  {output.message}
                </pre>
              )}

              {/* No output */}
              {!output.stdout && !output.stderr && !output.compileOutput && !output.message && output.status === 'passed' && (
                <div className="text-gray-400 text-sm">
                  No output
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Auto-save popup when user presses Cmd+S / Ctrl+S */}
      {showSavePopup && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm animate-in fade-in duration-200">
          <div className="bg-white dark:bg-[#161b22] rounded-2xl shadow-2xl border border-gray-200 dark:border-gray-700 p-6 max-w-sm mx-4 transform animate-in zoom-in-95 duration-200">
            <div className="text-center">
              <div className="text-4xl mb-4">☁️</div>
              <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-2">
                {tUI('editor.autoSaveTitle')}
              </h3>
              <p className="text-gray-600 dark:text-gray-400 text-sm mb-5">
                {tUI('editor.autoSaveMessage')}
              </p>
              <button
                onClick={() => setShowSavePopup(false)}
                className="w-full px-4 py-2.5 bg-gradient-to-r from-brand-600 to-purple-600 hover:from-brand-500 hover:to-purple-500 text-white font-bold rounded-xl shadow-lg shadow-brand-500/25 transition-all"
              >
                {tUI('editor.gotIt')}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PlaygroundPage;
