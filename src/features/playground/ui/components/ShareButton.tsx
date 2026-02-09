import React, { useState, useCallback } from 'react';
import { snippetsService } from '../../api/snippetsService';
import { useUITranslation } from '@/contexts/LanguageContext';

interface ShareButtonProps {
  code: string;
  language: string;
}

export const ShareButton: React.FC<ShareButtonProps> = ({ code, language }) => {
  const { tUI } = useUITranslation();
  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [shareUrl, setShareUrl] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [title, setTitle] = useState('');

  const handleShare = useCallback(async () => {
    if (!code.trim()) {
      setError(tUI('playground.emptyCode'));
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const result = await snippetsService.create({
        code,
        language,
        title: title.trim() || undefined,
      });
      setShareUrl(snippetsService.getShareUrl(result.shortId));
    } catch (err) {
      setError(tUI('playground.shareError'));
    } finally {
      setIsLoading(false);
    }
  }, [code, language, title, tUI]);

  const handleCopy = useCallback(async () => {
    if (!shareUrl) return;

    try {
      await navigator.clipboard.writeText(shareUrl);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback for older browsers
      const textarea = document.createElement('textarea');
      textarea.value = shareUrl;
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      document.body.removeChild(textarea);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  }, [shareUrl]);

  const handleClose = useCallback(() => {
    setIsOpen(false);
    setShareUrl(null);
    setError(null);
    setTitle('');
    setCopied(false);
  }, []);

  return (
    <>
      <button
        onClick={() => setIsOpen(true)}
        className="p-1.5 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-gray-200 dark:hover:bg-[#21262d] rounded transition-colors"
        title={tUI('playground.share')}
        data-testid="share-button"
      >
        <svg
          className="w-4 h-4"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z"
          />
        </svg>
      </button>

      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm animate-in fade-in duration-200">
          <div className="bg-white dark:bg-[#161b22] rounded-2xl shadow-2xl border border-gray-200 dark:border-gray-700 p-6 max-w-md w-full mx-4 transform animate-in zoom-in-95 duration-200">
            {!shareUrl ? (
              <>
                <div className="text-center mb-6">
                  <div className="text-4xl mb-3">🔗</div>
                  <h3 className="text-lg font-bold text-gray-900 dark:text-white">
                    {tUI('playground.shareCode')}
                  </h3>
                  <p className="text-gray-500 dark:text-gray-400 text-sm mt-1">
                    {tUI('playground.shareDescription')}
                  </p>
                </div>

                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    {tUI('playground.snippetTitle')}
                  </label>
                  <input
                    type="text"
                    value={title}
                    onChange={(e) => setTitle(e.target.value)}
                    placeholder={tUI('playground.snippetTitlePlaceholder')}
                    maxLength={100}
                    className="w-full px-3 py-2 bg-gray-50 dark:bg-[#0d1117] border border-gray-200 dark:border-gray-700 rounded-lg text-gray-900 dark:text-white placeholder-gray-400 focus:ring-2 focus:ring-brand-500 focus:border-transparent transition-all"
                    data-testid="snippet-title-input"
                  />
                </div>

                {error && (
                  <div className="mb-4 p-3 bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 text-sm rounded-lg">
                    {error}
                  </div>
                )}

                <div className="flex gap-3">
                  <button
                    onClick={handleClose}
                    className="flex-1 px-4 py-2.5 bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300 font-medium rounded-xl transition-all"
                  >
                    {tUI('common.cancel')}
                  </button>
                  <button
                    onClick={handleShare}
                    disabled={isLoading}
                    className="flex-1 px-4 py-2.5 bg-gradient-to-r from-brand-600 to-purple-600 hover:from-brand-500 hover:to-purple-500 text-white font-bold rounded-xl shadow-lg shadow-brand-500/25 transition-all disabled:opacity-50"
                    data-testid="create-share-link"
                  >
                    {isLoading ? (
                      <span className="flex items-center justify-center gap-2">
                        <svg
                          className="w-4 h-4 animate-spin"
                          viewBox="0 0 24 24"
                        >
                          <circle
                            className="opacity-25"
                            cx="12"
                            cy="12"
                            r="10"
                            stroke="currentColor"
                            strokeWidth="4"
                            fill="none"
                          />
                          <path
                            className="opacity-75"
                            fill="currentColor"
                            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                          />
                        </svg>
                        {tUI('common.loading')}
                      </span>
                    ) : (
                      tUI('playground.createLink')
                    )}
                  </button>
                </div>
              </>
            ) : (
              <>
                <div className="text-center mb-6">
                  <div className="text-4xl mb-3">✅</div>
                  <h3 className="text-lg font-bold text-gray-900 dark:text-white">
                    {tUI('playground.linkReady')}
                  </h3>
                </div>

                <div className="mb-4">
                  <div className="flex items-center gap-2 p-3 bg-gray-50 dark:bg-[#0d1117] border border-gray-200 dark:border-gray-700 rounded-lg">
                    <input
                      type="text"
                      value={shareUrl}
                      readOnly
                      className="flex-1 bg-transparent text-gray-900 dark:text-white text-sm font-mono outline-none"
                      data-testid="share-url-input"
                    />
                    <button
                      onClick={handleCopy}
                      className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                        copied
                          ? 'bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400'
                          : 'bg-brand-100 dark:bg-brand-900/30 text-brand-600 dark:text-brand-400 hover:bg-brand-200 dark:hover:bg-brand-900/50'
                      }`}
                      data-testid="copy-url-button"
                    >
                      {copied ? tUI('common.copied') : tUI('common.copy')}
                    </button>
                  </div>
                </div>

                <button
                  onClick={handleClose}
                  className="w-full px-4 py-2.5 bg-gradient-to-r from-brand-600 to-purple-600 hover:from-brand-500 hover:to-purple-500 text-white font-bold rounded-xl shadow-lg shadow-brand-500/25 transition-all"
                >
                  {tUI('common.done')}
                </button>
              </>
            )}
          </div>
        </div>
      )}
    </>
  );
};
