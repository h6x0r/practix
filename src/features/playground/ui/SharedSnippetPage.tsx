import React, { useState, useEffect, useContext, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import Editor, { loader } from '@monaco-editor/react';
import { snippetsService, Snippet } from '../api/snippetsService';
import { ThemeContext } from '@/components/Layout';
import { useEditorThemes, defineAllThemes } from '../hooks/useEditorThemes';
import { useUITranslation } from '@/contexts/LanguageContext';

loader.config({
  paths: { vs: 'https://cdn.jsdelivr.net/npm/monaco-editor@0.46.0/min/vs' },
});

const LANGUAGE_DISPLAY: Record<string, string> = {
  go: 'Go',
  java: 'Java',
  python: 'Python',
  typescript: 'TypeScript',
  javascript: 'JavaScript',
  c: 'C',
  cpp: 'C++',
  rust: 'Rust',
};

const LANGUAGE_COLORS: Record<string, string> = {
  go: 'text-cyan-400',
  java: 'text-orange-400',
  python: 'text-green-400',
  typescript: 'text-blue-400',
  javascript: 'text-yellow-400',
  c: 'text-gray-400',
  cpp: 'text-pink-400',
  rust: 'text-orange-500',
};

const SharedSnippetPage: React.FC = () => {
  const { shortId } = useParams<{ shortId: string }>();
  const navigate = useNavigate();
  const { isDark } = useContext(ThemeContext);
  const { tUI } = useUITranslation();
  const { currentTheme } = useEditorThemes(isDark);

  const [snippet, setSnippet] = useState<Snippet | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!shortId) return;

    const fetchSnippet = async () => {
      try {
        const data = await snippetsService.getByShortId(shortId);
        setSnippet(data);
      } catch {
        setError('Snippet not found');
      } finally {
        setLoading(false);
      }
    };

    fetchSnippet();
  }, [shortId]);

  const handleFork = useCallback(() => {
    if (!snippet) return;
    // Store snippet in sessionStorage and navigate to playground
    sessionStorage.setItem(
      'playground_fork',
      JSON.stringify({
        code: snippet.code,
        language: snippet.language,
      }),
    );
    navigate('/playground');
  }, [snippet, navigate]);

  if (loading) {
    return (
      <div className="min-h-[calc(100vh-7rem)] flex items-center justify-center bg-white dark:bg-[#0d1117]">
        <div className="flex flex-col items-center gap-4">
          <div className="w-8 h-8 border-2 border-brand-500 border-t-transparent rounded-full animate-spin" />
          <span className="text-gray-400 text-sm">Loading snippet...</span>
        </div>
      </div>
    );
  }

  if (error || !snippet) {
    return (
      <div className="min-h-[calc(100vh-7rem)] flex items-center justify-center bg-white dark:bg-[#0d1117]">
        <div className="text-center">
          <div className="text-6xl mb-4">🔗</div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
            Snippet Not Found
          </h1>
          <p className="text-gray-500 dark:text-gray-400 mb-6">
            This snippet may have been deleted or expired.
          </p>
          <button
            onClick={() => navigate('/playground')}
            className="px-6 py-2.5 bg-gradient-to-r from-brand-600 to-purple-600 hover:from-brand-500 hover:to-purple-500 text-white font-bold rounded-xl transition-all"
          >
            Go to Playground
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="h-[calc(100vh-7rem)] flex flex-col bg-white dark:bg-[#0d1117] rounded-xl overflow-hidden shadow-sm dark:shadow-none border border-gray-200 dark:border-gray-800">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 bg-gray-50 dark:bg-[#161b22] border-b border-gray-200 dark:border-[#21262d]">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span
              className={`text-lg font-bold ${LANGUAGE_COLORS[snippet.language] || 'text-gray-400'}`}
            >
              {LANGUAGE_DISPLAY[snippet.language] || snippet.language}
            </span>
            {snippet.title && (
              <>
                <span className="text-gray-400">•</span>
                <span className="text-gray-700 dark:text-gray-300 font-medium">
                  {snippet.title}
                </span>
              </>
            )}
          </div>

          <div className="flex items-center gap-3 text-xs text-gray-400">
            <span className="flex items-center gap-1">
              <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 20 20">
                <path d="M10 12a2 2 0 100-4 2 2 0 000 4z" />
                <path
                  fillRule="evenodd"
                  d="M.458 10C1.732 5.943 5.522 3 10 3s8.268 2.943 9.542 7c-1.274 4.057-5.064 7-9.542 7S1.732 14.057.458 10zM14 10a4 4 0 11-8 0 4 4 0 018 0z"
                  clipRule="evenodd"
                />
              </svg>
              {snippet.viewCount} {tUI('playground.views')}
            </span>
            <span>
              {new Date(snippet.createdAt).toLocaleDateString()}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleFork}
            className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-brand-600 to-purple-600 hover:from-brand-500 hover:to-purple-500 text-white font-bold rounded-lg text-sm transition-all"
            data-testid="fork-button"
          >
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
              <path
                fillRule="evenodd"
                d="M7.707 3.293a1 1 0 010 1.414L5.414 7H11a7 7 0 017 7v2a1 1 0 11-2 0v-2a5 5 0 00-5-5H5.414l2.293 2.293a1 1 0 11-1.414 1.414l-4-4a1 1 0 010-1.414l4-4a1 1 0 011.414 0z"
                clipRule="evenodd"
              />
            </svg>
            {tUI('playground.forkToEdit')}
          </button>
        </div>
      </div>

      {/* Editor (Read-only) */}
      <div className="flex-1 relative">
        <Editor
          height="100%"
          language={snippet.language}
          theme={currentTheme}
          beforeMount={defineAllThemes}
          value={snippet.code}
          options={{
            readOnly: true,
            minimap: { enabled: false },
            fontSize: 14,
            lineNumbers: 'on',
            scrollBeyondLastLine: false,
            automaticLayout: true,
            padding: { top: 16, bottom: 16 },
            fontFamily: "'JetBrains Mono', 'Courier New', monospace",
            lineHeight: 24,
            renderLineHighlight: 'all',
          }}
          loading={
            <div className="flex h-full items-center justify-center text-gray-500 text-sm bg-gray-50 dark:bg-[#0d1117]">
              <span className="animate-pulse">Loading editor...</span>
            </div>
          }
        />

        {/* Read-only badge */}
        <div className="absolute top-4 right-4 px-3 py-1 bg-gray-100 dark:bg-gray-800 text-gray-500 dark:text-gray-400 text-xs font-medium rounded-full">
          Read-only
        </div>
      </div>
    </div>
  );
};

export default SharedSnippetPage;
