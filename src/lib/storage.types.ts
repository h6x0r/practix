/**
 * Type definitions for localStorage storage layer
 */

import { User } from '@/types';

/**
 * Roadmap preferences stored in localStorage
 */
export interface RoadmapPreferences {
  role: string;
  level: string;
  goal: string;
}

/**
 * Timer/Stopwatch state stored in localStorage
 */
export interface TimerState {
  mode: 'idle' | 'stopwatch' | 'timer';
  isRunning: boolean;
  startedAt: number | null; // timestamp when started/resumed
  pausedTime: number; // time value when paused
  timerDuration: number;
}

/**
 * Mock user data stored in localStorage (for development)
 */
export type MockUser = User;
