
import React, { useState, useRef, useEffect, useCallback } from 'react';
import { IconStopwatch } from '@/components/Icons';
import { useUITranslation } from '@/contexts/LanguageContext';
import { storage } from '@/lib/storage';

type Mode = 'idle' | 'stopwatch' | 'timer';

interface TimerState {
  mode: Mode;
  isRunning: boolean;
  startedAt: number | null; // timestamp when started/resumed
  pausedTime: number; // time value when paused
  timerDuration: number;
}

const saveState = (state: TimerState) => {
  storage.setTimerState(state);
};

const loadState = (): TimerState | null => {
  return storage.getTimerState();
};

const clearState = () => {
  storage.removeTimerState();
};

export const TimerStopwatch = () => {
  const { tUI, formatTimeLocalized } = useUITranslation();
  const [isExpanded, setIsExpanded] = useState(false);
  const [showActivePanel, setShowActivePanel] = useState(false); // Controls visibility of active timer panel
  const [mode, setMode] = useState<Mode>('idle');
  const [isRunning, setIsRunning] = useState(false);
  const [time, setTime] = useState(0); // in seconds
  const [timerDuration, setTimerDuration] = useState(25 * 60); // 25 minutes default (pomodoro)
  const [customMinutes, setCustomMinutes] = useState(25);
  const [customSeconds, setCustomSeconds] = useState(0);
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Restore state from localStorage on mount
  useEffect(() => {
    const saved = loadState();
    if (saved && saved.mode !== 'idle') {
      setMode(saved.mode);
      setTimerDuration(saved.timerDuration);

      if (saved.isRunning && saved.startedAt) {
        // Calculate elapsed time since start
        const now = Date.now();
        const elapsedSeconds = Math.floor((now - saved.startedAt) / 1000);

        if (saved.mode === 'stopwatch') {
          setTime(saved.pausedTime + elapsedSeconds);
        } else {
          // Timer counts down
          const newTime = Math.max(0, saved.pausedTime - elapsedSeconds);
          if (newTime <= 0) {
            // Timer already finished
            setTime(0);
            setIsRunning(false);
            setMode('idle');
            clearState();
            return;
          }
          setTime(newTime);
        }
        setStartedAt(saved.startedAt);
        setIsRunning(true);
      } else {
        // Paused state
        setTime(saved.pausedTime);
        setIsRunning(false);
      }
    }
  }, []);

  // Preset timer durations with localized labels
  const presets = [
    { label: formatTimeLocalized('5m'), seconds: 5 * 60 },
    { label: formatTimeLocalized('15m'), seconds: 15 * 60 },
    { label: formatTimeLocalized('25m'), seconds: 25 * 60 },
    { label: formatTimeLocalized('45m'), seconds: 45 * 60 },
  ];

  // Close dropdown when clicking outside - always close any open panel
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsExpanded(false);
        setShowActivePanel(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Timer/Stopwatch logic
  useEffect(() => {
    if (isRunning) {
      intervalRef.current = setInterval(() => {
        setTime((prev) => {
          if (mode === 'timer') {
            if (prev <= 1) {
              // Timer finished
              setIsRunning(false);
              setStartedAt(null);
              setMode('idle');
              clearState();
              playSound();
              return 0;
            }
            return prev - 1;
          } else {
            // Stopwatch - count up
            return prev + 1;
          }
        });
      }, 1000);
    } else if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isRunning, mode]);

  const playSound = () => {
    // Simple beep using Web Audio API
    try {
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      const oscillator = audioContext.createOscillator();
      const gainNode = audioContext.createGain();
      oscillator.connect(gainNode);
      gainNode.connect(audioContext.destination);
      oscillator.frequency.value = 800;
      oscillator.type = 'sine';
      gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
      oscillator.start(audioContext.currentTime);
      oscillator.stop(audioContext.currentTime + 0.5);
    } catch (e) {
      // Fallback - do nothing
    }
  };

  const formatTime = (seconds: number) => {
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    if (hrs > 0) {
      return `${hrs}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const startStopwatch = () => {
    const now = Date.now();
    setMode('stopwatch');
    setTime(0);
    setIsRunning(true);
    setStartedAt(now);
    setIsExpanded(false);
    saveState({
      mode: 'stopwatch',
      isRunning: true,
      startedAt: now,
      pausedTime: 0,
      timerDuration: 0,
    });
  };

  const startTimer = (duration: number) => {
    const now = Date.now();
    setMode('timer');
    setTimerDuration(duration);
    setTime(duration);
    setIsRunning(true);
    setStartedAt(now);
    setIsExpanded(false);
    saveState({
      mode: 'timer',
      isRunning: true,
      startedAt: now,
      pausedTime: duration,
      timerDuration: duration,
    });
  };

  const startCustomTimer = () => {
    const totalSeconds = customMinutes * 60 + customSeconds;
    if (totalSeconds > 0) {
      startTimer(totalSeconds);
    }
  };

  const adjustCustomTime = (field: 'minutes' | 'seconds', delta: number) => {
    if (field === 'minutes') {
      setCustomMinutes(prev => Math.max(0, Math.min(99, prev + delta)));
    } else {
      setCustomSeconds(prev => {
        const newVal = prev + delta;
        if (newVal < 0) return 59;
        if (newVal > 59) return 0;
        return newVal;
      });
    }
  };

  const togglePause = () => {
    if (isRunning) {
      // Pausing - save current time
      setIsRunning(false);
      setStartedAt(null);
      saveState({
        mode,
        isRunning: false,
        startedAt: null,
        pausedTime: time,
        timerDuration,
      });
    } else {
      // Resuming - start new timer from current time
      const now = Date.now();
      setIsRunning(true);
      setStartedAt(now);
      saveState({
        mode,
        isRunning: true,
        startedAt: now,
        pausedTime: time,
        timerDuration,
      });
    }
  };

  const reset = () => {
    setIsRunning(false);
    setStartedAt(null);
    if (mode === 'timer') {
      setTime(timerDuration);
      saveState({
        mode,
        isRunning: false,
        startedAt: null,
        pausedTime: timerDuration,
        timerDuration,
      });
    } else {
      setTime(0);
      saveState({
        mode,
        isRunning: false,
        startedAt: null,
        pausedTime: 0,
        timerDuration: 0,
      });
    }
  };

  const stop = () => {
    setIsRunning(false);
    setMode('idle');
    setTime(0);
    setStartedAt(null);
    setIsExpanded(false);
    setShowActivePanel(false);
    clearState();
  };

  // Determine button appearance
  const isActive = mode !== 'idle';
  const displayTime = formatTime(time);

  // Progress percentage for timer
  const progress = mode === 'timer' ? ((timerDuration - time) / timerDuration) * 100 : 0;

  return (
    <div className="relative" ref={dropdownRef}>
      {/* Main Button */}
      <button
        onClick={() => {
          if (isActive) {
            // Toggle active panel visibility on click
            setShowActivePanel(!showActivePanel);
          } else {
            setIsExpanded(!isExpanded);
          }
        }}
        className={`h-8 flex items-center gap-2 px-2 rounded-md transition-all ${
          isActive
            ? isRunning
              ? 'bg-brand-600/20 text-brand-400 border border-brand-500/30'
              : 'bg-yellow-600/20 text-yellow-400 border border-yellow-500/30'
            : isExpanded
            ? 'bg-gray-200 dark:bg-[#3d3d3d] text-gray-900 dark:text-white'
            : 'text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-[#333]'
        }`}
        title={isActive ? tUI('timer.openTimer') : tUI('timer.openTimer')}
      >
        <IconStopwatch className="w-4 h-4" />
        {isActive && (
          <span className="text-xs font-mono font-bold tabular-nums">
            {displayTime}
          </span>
        )}
      </button>

      {/* Expanded Panel - when idle */}
      {isExpanded && !isActive && (
        <div className="absolute right-0 top-full mt-2 w-64 bg-white dark:bg-[#2d2d2d] rounded-xl shadow-xl border border-gray-200 dark:border-[#3d3d3d] z-50 overflow-hidden">
          {/* Header */}
          <div className="px-4 py-3 border-b border-gray-200 dark:border-[#3d3d3d]">
            <h4 className="text-sm font-bold text-gray-900 dark:text-white">{tUI('timer.timeTools')}</h4>
          </div>

          <div className="p-3 space-y-3">
            {/* Stopwatch Section */}
            <div>
              <div className="text-[10px] font-bold text-gray-500 uppercase mb-2">{tUI('timer.stopwatch')}</div>
              <button
                onClick={startStopwatch}
                className="w-full px-3 py-2 bg-gray-50 dark:bg-[#1e1e1e] hover:bg-gray-200 dark:hover:bg-[#333] border border-gray-200 dark:border-[#3d3d3d] rounded-lg text-xs font-medium text-gray-700 dark:text-gray-300 flex items-center justify-center gap-2 transition-colors"
              >
                <span className="text-green-400">&#9654;</span>
                {tUI('timer.startStopwatch')}
              </button>
            </div>

            {/* Divider */}
            <div className="border-t border-gray-200 dark:border-[#3d3d3d]"></div>

            {/* Timer Section */}
            <div>
              <div className="text-[10px] font-bold text-gray-500 uppercase mb-2">{tUI('timer.timer')}</div>

              {/* Quick Presets */}
              <div className="grid grid-cols-4 gap-1.5 mb-3">
                {presets.map((preset) => (
                  <button
                    key={preset.label}
                    onClick={() => startTimer(preset.seconds)}
                    className="px-2 py-2 bg-gray-50 dark:bg-[#1e1e1e] hover:bg-brand-600/20 hover:border-brand-500/30 border border-gray-200 dark:border-[#3d3d3d] rounded-lg text-xs font-bold text-gray-700 dark:text-gray-300 hover:text-brand-400 transition-all"
                  >
                    {preset.label}
                  </button>
                ))}
              </div>

              {/* Custom Timer Input */}
              <div className="bg-gray-50 dark:bg-[#1e1e1e] rounded-lg p-2 border border-gray-200 dark:border-[#3d3d3d]">
                <div className="text-[9px] font-bold text-gray-500 uppercase mb-2 text-center">{tUI('timer.customTimer')}</div>
                <div className="flex items-center justify-center gap-2">
                  {/* Minutes */}
                  <div className="flex flex-col items-center">
                    <button
                      onClick={() => adjustCustomTime('minutes', 1)}
                      className="w-8 h-6 flex items-center justify-center text-gray-500 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white hover:bg-gray-200 dark:hover:bg-[#333] rounded transition-colors"
                    >
                      <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
                      </svg>
                    </button>
                    <input
                      type="number"
                      value={customMinutes}
                      onChange={(e) => setCustomMinutes(Math.max(0, Math.min(99, parseInt(e.target.value) || 0)))}
                      className="w-10 h-8 bg-white dark:bg-[#2d2d2d] border border-gray-200 dark:border-[#3d3d3d] rounded text-center text-sm font-mono font-bold text-gray-900 dark:text-white focus:outline-none focus:border-brand-500 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                    />
                    <button
                      onClick={() => adjustCustomTime('minutes', -1)}
                      className="w-8 h-6 flex items-center justify-center text-gray-500 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white hover:bg-gray-200 dark:hover:bg-[#333] rounded transition-colors"
                    >
                      <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                      </svg>
                    </button>
                    <span className="text-[9px] text-gray-500 mt-1">{tUI('timer.min')}</span>
                  </div>

                  <span className="text-xl font-bold text-gray-500 pb-4">:</span>

                  {/* Seconds */}
                  <div className="flex flex-col items-center">
                    <button
                      onClick={() => adjustCustomTime('seconds', 5)}
                      className="w-8 h-6 flex items-center justify-center text-gray-500 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white hover:bg-gray-200 dark:hover:bg-[#333] rounded transition-colors"
                    >
                      <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
                      </svg>
                    </button>
                    <input
                      type="number"
                      value={customSeconds.toString().padStart(2, '0')}
                      onChange={(e) => setCustomSeconds(Math.max(0, Math.min(59, parseInt(e.target.value) || 0)))}
                      className="w-10 h-8 bg-white dark:bg-[#2d2d2d] border border-gray-200 dark:border-[#3d3d3d] rounded text-center text-sm font-mono font-bold text-gray-900 dark:text-white focus:outline-none focus:border-brand-500 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                    />
                    <button
                      onClick={() => adjustCustomTime('seconds', -5)}
                      className="w-8 h-6 flex items-center justify-center text-gray-500 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white hover:bg-gray-200 dark:hover:bg-[#333] rounded transition-colors"
                    >
                      <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                      </svg>
                    </button>
                    <span className="text-[9px] text-gray-500 mt-1">{tUI('timer.sec')}</span>
                  </div>

                  {/* Start Button - aligned with input row */}
                  <div className="flex flex-col items-center">
                    <div className="h-6"></div>
                    <button
                      onClick={startCustomTimer}
                      disabled={customMinutes === 0 && customSeconds === 0}
                      className="w-8 h-8 flex items-center justify-center bg-gradient-to-r from-brand-600 to-purple-600 hover:from-brand-500 hover:to-purple-500 disabled:from-gray-600 disabled:to-gray-600 disabled:cursor-not-allowed rounded-lg text-white transition-all"
                    >
                      <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M8 5v14l11-7z" />
                      </svg>
                    </button>
                    <div className="h-6"></div>
                    <span className="text-[9px] text-transparent mt-1">.</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Active Timer/Stopwatch Controls - shown when timer is active AND panel is toggled open */}
      {isActive && showActivePanel && (
        <div className="absolute right-0 top-full mt-2 w-52 bg-white dark:bg-[#2d2d2d] rounded-xl shadow-xl border border-gray-200 dark:border-[#3d3d3d] z-50 overflow-hidden">
          {/* Progress bar for timer */}
          {mode === 'timer' && (
            <div className="h-1 bg-gray-100 dark:bg-[#1e1e1e]">
              <div
                className="h-full bg-brand-500 transition-all duration-1000"
                style={{ width: `${progress}%` }}
              />
            </div>
          )}

          <div className="p-3">
            {/* Mode indicator */}
            <div className="text-[10px] font-bold text-gray-500 uppercase mb-2 text-center">
              {mode === 'timer' ? tUI('timer.timer') : tUI('timer.stopwatch')}
            </div>

            {/* Time Display */}
            <div className="text-center mb-3">
              <span className={`text-2xl font-mono font-bold ${isRunning ? 'text-gray-900 dark:text-white' : 'text-yellow-500 dark:text-yellow-400'}`}>
                {displayTime}
              </span>
            </div>

            {/* Control Buttons */}
            <div className="flex gap-1.5">
              <button
                onClick={togglePause}
                className={`flex-1 px-2 py-2 rounded-lg text-xs font-bold transition-colors whitespace-nowrap ${
                  isRunning
                    ? 'bg-yellow-600/20 text-yellow-600 dark:text-yellow-400 hover:bg-yellow-600/30'
                    : 'bg-green-600/20 text-green-600 dark:text-green-400 hover:bg-green-600/30'
                }`}
              >
                {isRunning ? tUI('timer.pause') : tUI('timer.resume')}
              </button>
              <button
                onClick={reset}
                className="flex-1 px-2 py-2 bg-gray-50 dark:bg-[#1e1e1e] hover:bg-gray-200 dark:hover:bg-[#333] rounded-lg text-xs font-medium text-gray-600 dark:text-gray-400 transition-colors whitespace-nowrap"
              >
                {tUI('timer.reset')}
              </button>
              <button
                onClick={stop}
                className="w-8 flex-shrink-0 py-2 bg-red-600/20 hover:bg-red-600/30 rounded-lg text-xs font-medium text-red-600 dark:text-red-400 transition-colors"
              >
                &times;
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
