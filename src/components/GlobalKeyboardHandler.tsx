import { useEffect } from 'react';
import { useToast } from './Toast';
import { useUITranslation } from '@/contexts/LanguageContext';

/**
 * Global keyboard handler component that intercepts common shortcuts
 * and provides user-friendly feedback across the entire platform.
 */
export const GlobalKeyboardHandler = () => {
  const { showToast } = useToast();
  const { tUI } = useUITranslation();

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Cmd+S / Ctrl+S - Show auto-save toast
      if ((e.metaKey || e.ctrlKey) && e.key === 's') {
        e.preventDefault();
        showToast(tUI('editor.autoSaveMessage'), 'info');
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [showToast, tUI]);

  // This component doesn't render anything - it just handles keyboard events
  return null;
};
