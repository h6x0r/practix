import React, { useState, memo } from 'react';
import { IconChevronDown } from '@/components/Icons';
import { useUITranslation } from '@/contexts/LanguageContext';

interface HintsPanelProps {
  hint1?: string;
  hint2?: string;
}

export const HintsPanel = memo(({ hint1, hint2 }: HintsPanelProps) => {
  const { tUI } = useUITranslation();
  const [openHint, setOpenHint] = useState<1 | 2 | null>(null);

  if (!hint1 && !hint2) return null;

  const toggleHint = (hint: 1 | 2) => {
    setOpenHint(openHint === hint ? null : hint);
  };

  return (
    <div className="mt-6 space-y-2">
      <h4 className="text-xs uppercase font-bold text-gray-400 mb-3 tracking-wider">{tUI('task.hints')}</h4>

      {hint1 && (
        <div className="border border-gray-200 dark:border-dark-border rounded-lg overflow-hidden">
          <button
            onClick={() => toggleHint(1)}
            className="w-full flex items-center justify-between px-4 py-3 bg-gray-50 dark:bg-dark-bg hover:bg-gray-100 dark:hover:bg-dark-surface transition-colors"
          >
            <span className="flex items-center gap-2 text-sm font-medium text-gray-700 dark:text-gray-300">
              <span className="w-5 h-5 bg-amber-100 dark:bg-amber-900/30 text-amber-600 dark:text-amber-400 rounded flex items-center justify-center text-xs font-bold">1</span>
              {tUI('task.hint1')}
            </span>
            <IconChevronDown className={`w-4 h-4 text-gray-400 transition-transform ${openHint === 1 ? 'rotate-180' : ''}`} />
          </button>
          {openHint === 1 && (
            <div className="px-4 py-3 bg-white dark:bg-dark-surface border-t border-gray-200 dark:border-dark-border">
              <p className="text-sm text-gray-600 dark:text-gray-400">{hint1}</p>
            </div>
          )}
        </div>
      )}

      {hint2 && (
        <div className="border border-gray-200 dark:border-dark-border rounded-lg overflow-hidden">
          <button
            onClick={() => toggleHint(2)}
            className="w-full flex items-center justify-between px-4 py-3 bg-gray-50 dark:bg-dark-bg hover:bg-gray-100 dark:hover:bg-dark-surface transition-colors"
          >
            <span className="flex items-center gap-2 text-sm font-medium text-gray-700 dark:text-gray-300">
              <span className="w-5 h-5 bg-orange-100 dark:bg-orange-900/30 text-orange-600 dark:text-orange-400 rounded flex items-center justify-center text-xs font-bold">2</span>
              {tUI('task.hint2')}
            </span>
            <IconChevronDown className={`w-4 h-4 text-gray-400 transition-transform ${openHint === 2 ? 'rotate-180' : ''}`} />
          </button>
          {openHint === 2 && (
            <div className="px-4 py-3 bg-white dark:bg-dark-surface border-t border-gray-200 dark:border-dark-border">
              <p className="text-sm text-gray-600 dark:text-gray-400">{hint2}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
});
