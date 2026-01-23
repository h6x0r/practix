import { useState, useCallback, useEffect, useRef } from 'react';

interface UseResizablePanelOptions {
  /** Storage key for persisting width */
  storageKey: string;
  /** Default width in pixels */
  defaultWidth: number;
  /** Minimum width in pixels */
  minWidth: number;
  /** Maximum width in pixels (absolute limit) */
  maxWidth: number;
  /** Maximum width as ratio of viewport (0-1), e.g. 0.6 = 60% of window width */
  maxWidthRatio?: number;
}

interface UseResizablePanelReturn {
  /** Current width of the panel */
  width: number;
  /** Effective maximum width (considers both maxWidth and viewport ratio) */
  effectiveMaxWidth: number;
  /** Whether the panel is being resized */
  isResizing: boolean;
  /** Start resize handler - attach to onMouseDown of resize handle */
  startResize: (e: React.MouseEvent) => void;
  /** Reset width to default */
  resetWidth: () => void;
}

/**
 * Calculate effective max width based on absolute limit and viewport ratio
 */
const getEffectiveMaxWidth = (maxWidth: number, maxWidthRatio?: number): number => {
  if (typeof window === 'undefined') return maxWidth;

  if (maxWidthRatio && maxWidthRatio > 0 && maxWidthRatio <= 1) {
    const viewportMax = Math.floor(window.innerWidth * maxWidthRatio);
    return Math.min(maxWidth, viewportMax);
  }
  return maxWidth;
};

/**
 * Hook for creating resizable panel functionality with drag-to-resize,
 * localStorage persistence, and dynamic viewport-based max width.
 */
export function useResizablePanel({
  storageKey,
  defaultWidth,
  minWidth,
  maxWidth,
  maxWidthRatio,
}: UseResizablePanelOptions): UseResizablePanelReturn {
  // Track effective max width (responsive to viewport)
  const [effectiveMaxWidth, setEffectiveMaxWidth] = useState(() =>
    getEffectiveMaxWidth(maxWidth, maxWidthRatio)
  );

  // Initialize from localStorage or default
  const [width, setWidth] = useState(() => {
    const currentMax = getEffectiveMaxWidth(maxWidth, maxWidthRatio);
    try {
      const stored = localStorage.getItem(storageKey);
      if (stored) {
        const parsed = parseInt(stored, 10);
        if (!isNaN(parsed) && parsed >= minWidth && parsed <= currentMax) {
          return parsed;
        }
      }
    } catch {
      // localStorage not available
    }
    return Math.min(defaultWidth, currentMax);
  });

  const [isResizing, setIsResizing] = useState(false);
  const startXRef = useRef(0);
  const startWidthRef = useRef(0);

  // Save to localStorage when width changes
  useEffect(() => {
    try {
      localStorage.setItem(storageKey, String(width));
    } catch {
      // localStorage not available
    }
  }, [storageKey, width]);

  const startResize = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizing(true);
    startXRef.current = e.clientX;
    startWidthRef.current = width;
  }, [width]);

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!isResizing) return;

    const deltaX = e.clientX - startXRef.current;
    const newWidth = Math.min(effectiveMaxWidth, Math.max(minWidth, startWidthRef.current + deltaX));
    setWidth(newWidth);
  }, [isResizing, minWidth, effectiveMaxWidth]);

  const handleMouseUp = useCallback(() => {
    setIsResizing(false);
  }, []);

  // Add/remove global event listeners during resize
  useEffect(() => {
    if (isResizing) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      // Prevent text selection during resize
      document.body.style.userSelect = 'none';
      document.body.style.cursor = 'col-resize';
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.style.userSelect = '';
      document.body.style.cursor = '';
    };
  }, [isResizing, handleMouseMove, handleMouseUp]);

  // Handle window resize - recalculate effective max width and adjust current width
  useEffect(() => {
    const handleWindowResize = () => {
      const newEffectiveMax = getEffectiveMaxWidth(maxWidth, maxWidthRatio);
      setEffectiveMaxWidth(newEffectiveMax);
      setWidth(prev => Math.min(newEffectiveMax, Math.max(minWidth, prev)));
    };

    window.addEventListener('resize', handleWindowResize);
    return () => window.removeEventListener('resize', handleWindowResize);
  }, [minWidth, maxWidth, maxWidthRatio]);

  const resetWidth = useCallback(() => {
    setWidth(defaultWidth);
  }, [defaultWidth]);

  return {
    width,
    effectiveMaxWidth,
    isResizing,
    startResize,
    resetWidth,
  };
}
