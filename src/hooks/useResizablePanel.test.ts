import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useResizablePanel } from './useResizablePanel';

describe('useResizablePanel', () => {
  const defaultOptions = {
    storageKey: 'test-panel-width',
    defaultWidth: 400,
    minWidth: 200,
    maxWidth: 800,
  };

  beforeEach(() => {
    vi.stubGlobal('localStorage', {
      getItem: vi.fn(),
      setItem: vi.fn(),
      removeItem: vi.fn(),
    });

    vi.stubGlobal('innerWidth', 1920);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  describe('initial state', () => {
    it('should use default width when no saved value', () => {
      vi.mocked(localStorage.getItem).mockReturnValue(null);

      const { result } = renderHook(() => useResizablePanel(defaultOptions));

      expect(result.current.width).toBe(400);
    });

    it('should use saved width from localStorage', () => {
      vi.mocked(localStorage.getItem).mockReturnValue('500');

      const { result } = renderHook(() => useResizablePanel(defaultOptions));

      expect(result.current.width).toBe(500);
    });

    it('should clamp saved width to minWidth', () => {
      vi.mocked(localStorage.getItem).mockReturnValue('100'); // Below minWidth

      const { result } = renderHook(() => useResizablePanel(defaultOptions));

      // Falls back to default when out of range
      expect(result.current.width).toBe(400);
    });

    it('should clamp saved width to maxWidth', () => {
      vi.mocked(localStorage.getItem).mockReturnValue('1000'); // Above maxWidth

      const { result } = renderHook(() => useResizablePanel(defaultOptions));

      // Falls back to default when out of range
      expect(result.current.width).toBe(400);
    });

    it('should handle invalid localStorage value', () => {
      vi.mocked(localStorage.getItem).mockReturnValue('not-a-number');

      const { result } = renderHook(() => useResizablePanel(defaultOptions));

      expect(result.current.width).toBe(400);
    });

    it('should not be resizing initially', () => {
      const { result } = renderHook(() => useResizablePanel(defaultOptions));

      expect(result.current.isResizing).toBe(false);
    });
  });

  describe('effectiveMaxWidth', () => {
    it('should use maxWidth when no ratio provided', () => {
      const { result } = renderHook(() => useResizablePanel(defaultOptions));

      expect(result.current.effectiveMaxWidth).toBe(800);
    });

    it('should calculate effective max based on viewport ratio', () => {
      vi.stubGlobal('innerWidth', 1000);

      const { result } = renderHook(() =>
        useResizablePanel({
          ...defaultOptions,
          maxWidthRatio: 0.5, // 50% of viewport
        })
      );

      // 50% of 1000 = 500, which is less than maxWidth (800)
      expect(result.current.effectiveMaxWidth).toBe(500);
    });

    it('should use maxWidth when ratio calculation exceeds it', () => {
      vi.stubGlobal('innerWidth', 2000);

      const { result } = renderHook(() =>
        useResizablePanel({
          ...defaultOptions,
          maxWidthRatio: 0.8, // 80% of 2000 = 1600
        })
      );

      // maxWidth (800) is less than calculated (1600)
      expect(result.current.effectiveMaxWidth).toBe(800);
    });

    it('should clamp default width to effective max', () => {
      vi.stubGlobal('innerWidth', 600);

      const { result } = renderHook(() =>
        useResizablePanel({
          ...defaultOptions,
          defaultWidth: 400,
          maxWidthRatio: 0.5, // 50% of 600 = 300
        })
      );

      // Default (400) clamped to effective max (300)
      expect(result.current.width).toBe(300);
    });
  });

  describe('startResize', () => {
    it('should set isResizing to true', () => {
      const { result } = renderHook(() => useResizablePanel(defaultOptions));

      const mockEvent = {
        preventDefault: vi.fn(),
        clientX: 500,
      } as unknown as React.MouseEvent;

      act(() => {
        result.current.startResize(mockEvent);
      });

      expect(result.current.isResizing).toBe(true);
      expect(mockEvent.preventDefault).toHaveBeenCalled();
    });
  });

  describe('resize behavior', () => {
    it('should update width on mouse move', () => {
      const { result } = renderHook(() => useResizablePanel(defaultOptions));

      // Start resize
      const startEvent = {
        preventDefault: vi.fn(),
        clientX: 500,
      } as unknown as React.MouseEvent;

      act(() => {
        result.current.startResize(startEvent);
      });

      // Simulate mouse move
      const moveEvent = new MouseEvent('mousemove', { clientX: 550 });
      act(() => {
        document.dispatchEvent(moveEvent);
      });

      // Width should increase by 50
      expect(result.current.width).toBe(450);
    });

    it('should clamp width to minWidth during resize', () => {
      const { result } = renderHook(() => useResizablePanel(defaultOptions));

      const startEvent = {
        preventDefault: vi.fn(),
        clientX: 500,
      } as unknown as React.MouseEvent;

      act(() => {
        result.current.startResize(startEvent);
      });

      // Move far left to try to go below minWidth
      const moveEvent = new MouseEvent('mousemove', { clientX: 100 });
      act(() => {
        document.dispatchEvent(moveEvent);
      });

      expect(result.current.width).toBe(200); // minWidth
    });

    it('should clamp width to maxWidth during resize', () => {
      const { result } = renderHook(() => useResizablePanel(defaultOptions));

      const startEvent = {
        preventDefault: vi.fn(),
        clientX: 500,
      } as unknown as React.MouseEvent;

      act(() => {
        result.current.startResize(startEvent);
      });

      // Move far right to try to exceed maxWidth
      const moveEvent = new MouseEvent('mousemove', { clientX: 1500 });
      act(() => {
        document.dispatchEvent(moveEvent);
      });

      expect(result.current.width).toBe(800); // maxWidth
    });

    it('should stop resizing on mouse up', () => {
      const { result } = renderHook(() => useResizablePanel(defaultOptions));

      const startEvent = {
        preventDefault: vi.fn(),
        clientX: 500,
      } as unknown as React.MouseEvent;

      act(() => {
        result.current.startResize(startEvent);
      });

      expect(result.current.isResizing).toBe(true);

      act(() => {
        document.dispatchEvent(new MouseEvent('mouseup'));
      });

      expect(result.current.isResizing).toBe(false);
    });
  });

  describe('localStorage persistence', () => {
    it('should save width to localStorage on change', () => {
      const { result } = renderHook(() => useResizablePanel(defaultOptions));

      // Start resize
      const startEvent = {
        preventDefault: vi.fn(),
        clientX: 500,
      } as unknown as React.MouseEvent;

      act(() => {
        result.current.startResize(startEvent);
      });

      // Move to change width
      const moveEvent = new MouseEvent('mousemove', { clientX: 550 });
      act(() => {
        document.dispatchEvent(moveEvent);
      });

      expect(localStorage.setItem).toHaveBeenCalledWith('test-panel-width', '450');
    });

    it('should handle localStorage errors gracefully', () => {
      vi.mocked(localStorage.setItem).mockImplementation(() => {
        throw new Error('QuotaExceeded');
      });

      const { result } = renderHook(() => useResizablePanel(defaultOptions));

      // Should not throw
      expect(result.current.width).toBe(400);
    });
  });

  describe('resetWidth', () => {
    it('should reset width to default', () => {
      vi.mocked(localStorage.getItem).mockReturnValue('600');

      const { result } = renderHook(() => useResizablePanel(defaultOptions));

      expect(result.current.width).toBe(600);

      act(() => {
        result.current.resetWidth();
      });

      expect(result.current.width).toBe(400);
    });
  });

  describe('window resize', () => {
    it('should recalculate effective max on window resize', () => {
      vi.stubGlobal('innerWidth', 1000);

      const { result } = renderHook(() =>
        useResizablePanel({
          ...defaultOptions,
          maxWidthRatio: 0.5,
        })
      );

      expect(result.current.effectiveMaxWidth).toBe(500);

      // Simulate window resize
      vi.stubGlobal('innerWidth', 800);
      act(() => {
        window.dispatchEvent(new Event('resize'));
      });

      expect(result.current.effectiveMaxWidth).toBe(400);
    });

    it('should clamp current width when window shrinks', () => {
      vi.stubGlobal('innerWidth', 1000);
      vi.mocked(localStorage.getItem).mockReturnValue('450');

      const { result } = renderHook(() =>
        useResizablePanel({
          ...defaultOptions,
          maxWidthRatio: 0.5, // 500 effective max
        })
      );

      expect(result.current.width).toBe(450);

      // Shrink window
      vi.stubGlobal('innerWidth', 600);
      act(() => {
        window.dispatchEvent(new Event('resize'));
      });

      // Effective max now 300, width clamped
      expect(result.current.width).toBe(300);
    });
  });

  describe('cleanup', () => {
    it('should remove event listeners on unmount', () => {
      const removeEventListenerSpy = vi.spyOn(document, 'removeEventListener');

      const { result, unmount } = renderHook(() => useResizablePanel(defaultOptions));

      // Start resizing
      const startEvent = {
        preventDefault: vi.fn(),
        clientX: 500,
      } as unknown as React.MouseEvent;

      act(() => {
        result.current.startResize(startEvent);
      });

      unmount();

      expect(removeEventListenerSpy).toHaveBeenCalledWith('mousemove', expect.any(Function));
      expect(removeEventListenerSpy).toHaveBeenCalledWith('mouseup', expect.any(Function));
    });

    it('should remove window resize listener on unmount', () => {
      const removeEventListenerSpy = vi.spyOn(window, 'removeEventListener');

      const { unmount } = renderHook(() => useResizablePanel(defaultOptions));

      unmount();

      expect(removeEventListenerSpy).toHaveBeenCalledWith('resize', expect.any(Function));
    });
  });
});
