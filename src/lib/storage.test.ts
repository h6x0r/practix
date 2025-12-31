import { describe, it, expect, beforeEach, vi } from 'vitest';

// Mock localStorage before importing storage
const mockStorage: Record<string, string> = {};

vi.stubGlobal('localStorage', {
  getItem: vi.fn((key: string) => mockStorage[key] || null),
  setItem: vi.fn((key: string, value: string) => {
    mockStorage[key] = value;
  }),
  removeItem: vi.fn((key: string) => {
    delete mockStorage[key];
  }),
  clear: vi.fn(() => {
    Object.keys(mockStorage).forEach((key) => delete mockStorage[key]);
  }),
});

import { storage } from './storage';

describe('storage utility', () => {
  beforeEach(() => {
    // Clear mock storage before each test
    Object.keys(mockStorage).forEach((key) => delete mockStorage[key]);
    vi.clearAllMocks();
  });

  describe('getTaskCode / setTaskCode', () => {
    it('should save and retrieve task code', () => {
      storage.setTaskCode('hello-world', 'package main');

      expect(localStorage.setItem).toHaveBeenCalledWith(
        'kodla_task_hello-world',
        'package main',
      );

      const result = storage.getTaskCode('hello-world');
      expect(result).toBe('package main');
    });

    it('should return null for non-existent task code', () => {
      const result = storage.getTaskCode('non-existent');
      expect(result).toBeNull();
    });
  });

  describe('getCompletedTasks / addCompletedTask', () => {
    it('should add and retrieve completed tasks', () => {
      storage.addCompletedTask('task-1');
      storage.addCompletedTask('task-2');

      const result = storage.getCompletedTasks();
      expect(result).toContain('task-1');
      expect(result).toContain('task-2');
    });

    it('should not add duplicate completed tasks', () => {
      storage.addCompletedTask('task-1');
      storage.addCompletedTask('task-1');

      const result = storage.getCompletedTasks();
      expect(result.filter((t) => t === 'task-1')).toHaveLength(1);
    });

    it('should return empty array when no completed tasks', () => {
      const result = storage.getCompletedTasks();
      expect(result).toEqual([]);
    });
  });

  describe('getLanguage / setLanguage', () => {
    it('should save and retrieve language preference', () => {
      storage.setLanguage('ru');

      expect(localStorage.setItem).toHaveBeenCalledWith('kodla_language', 'ru');

      const result = storage.getLanguage();
      expect(result).toBe('ru');
    });

    it('should return null when no language set', () => {
      const result = storage.getLanguage();
      expect(result).toBeNull();
    });
  });

  describe('getTheme / setTheme', () => {
    it('should save and retrieve theme preference', () => {
      storage.setTheme('dark');

      expect(localStorage.setItem).toHaveBeenCalledWith('kodla_theme', 'dark');

      const result = storage.getTheme();
      expect(result).toBe('dark');
    });

    it('should return null when no theme set', () => {
      const result = storage.getTheme();
      expect(result).toBeNull();
    });
  });
});
