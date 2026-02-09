import { useState, useEffect, useCallback, useRef } from "react";
import { storage } from "@/lib/storage";
import { createLogger } from "@/lib/logger";

const log = createLogger("PlaygroundStorage");

const STORAGE_KEY = "playground_state";
const AUTOSAVE_DELAY = 1000; // 1 second debounce

export interface PlaygroundState {
  code: string;
  language: string;
  savedAt: string;
}

export interface UsePlaygroundStorageReturn {
  savedState: PlaygroundState | null;
  saveState: (code: string, language: string) => void;
  clearState: () => void;
  lastSavedAt: Date | null;
}

export function usePlaygroundStorage(): UsePlaygroundStorageReturn {
  const [savedState, setSavedState] = useState<PlaygroundState | null>(null);
  const [lastSavedAt, setLastSavedAt] = useState<Date | null>(null);
  const saveTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Load saved state on mount
  useEffect(() => {
    try {
      const stored = storage.getItem(STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored) as PlaygroundState;
        setSavedState(parsed);
        setLastSavedAt(new Date(parsed.savedAt));
        log.info("Loaded playground state", { language: parsed.language });
      }
    } catch (error) {
      log.error("Failed to load playground state", error);
    }
  }, []);

  // Debounced save function
  const saveState = useCallback((code: string, language: string) => {
    if (saveTimeoutRef.current) {
      clearTimeout(saveTimeoutRef.current);
    }

    saveTimeoutRef.current = setTimeout(() => {
      try {
        const state: PlaygroundState = {
          code,
          language,
          savedAt: new Date().toISOString(),
        };
        storage.setItem(STORAGE_KEY, JSON.stringify(state));
        setSavedState(state);
        setLastSavedAt(new Date(state.savedAt));
        log.debug("Saved playground state", { language });
      } catch (error) {
        log.error("Failed to save playground state", error);
      }
    }, AUTOSAVE_DELAY);
  }, []);

  // Clear saved state
  const clearState = useCallback(() => {
    try {
      storage.removeItem(STORAGE_KEY);
      setSavedState(null);
      setLastSavedAt(null);
      log.info("Cleared playground state");
    } catch (error) {
      log.error("Failed to clear playground state", error);
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (saveTimeoutRef.current) {
        clearTimeout(saveTimeoutRef.current);
      }
    };
  }, []);

  return {
    savedState,
    saveState,
    clearState,
    lastSavedAt,
  };
}
