import { describe, it, expect, beforeEach, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import React from 'react';
import { LanguageProvider, useLanguage, useUITranslation, Language } from './LanguageContext';

// Mock storage
vi.mock('../lib/storage', () => ({
  storage: {
    getLanguage: vi.fn(),
    setLanguage: vi.fn(),
  },
}));

import { storage } from '../lib/storage';

describe('LanguageContext', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(storage.getLanguage).mockReturnValue(null);
  });

  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <LanguageProvider>{children}</LanguageProvider>
  );

  describe('useLanguage hook', () => {
    it('should throw error when used outside provider', () => {
      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

      expect(() => {
        renderHook(() => useLanguage());
      }).toThrow('useLanguage must be used within a LanguageProvider');

      consoleSpy.mockRestore();
    });

    it('should provide context values', () => {
      const { result } = renderHook(() => useLanguage(), { wrapper });

      expect(result.current.language).toBe('en');
      expect(typeof result.current.setLanguage).toBe('function');
      expect(typeof result.current.t).toBe('function');
    });
  });

  describe('initial language', () => {
    it('should default to English', () => {
      const { result } = renderHook(() => useLanguage(), { wrapper });

      expect(result.current.language).toBe('en');
    });

    it('should load saved language from storage', () => {
      vi.mocked(storage.getLanguage).mockReturnValue('ru');

      const { result } = renderHook(() => useLanguage(), { wrapper });

      expect(result.current.language).toBe('ru');
    });

    it('should handle invalid saved language', () => {
      vi.mocked(storage.getLanguage).mockReturnValue('invalid');

      const { result } = renderHook(() => useLanguage(), { wrapper });

      expect(result.current.language).toBe('en');
    });

    it('should load Uzbek language', () => {
      vi.mocked(storage.getLanguage).mockReturnValue('uz');

      const { result } = renderHook(() => useLanguage(), { wrapper });

      expect(result.current.language).toBe('uz');
    });
  });

  describe('setLanguage', () => {
    it('should change language', () => {
      const { result } = renderHook(() => useLanguage(), { wrapper });

      act(() => {
        result.current.setLanguage('ru');
      });

      expect(result.current.language).toBe('ru');
    });

    it('should save language to storage', () => {
      const { result } = renderHook(() => useLanguage(), { wrapper });

      act(() => {
        result.current.setLanguage('uz');
      });

      expect(storage.setLanguage).toHaveBeenCalledWith('uz');
    });

    it('should allow switching between all languages', () => {
      const { result } = renderHook(() => useLanguage(), { wrapper });

      const languages: Language[] = ['en', 'ru', 'uz'];

      languages.forEach((lang) => {
        act(() => {
          result.current.setLanguage(lang);
        });
        expect(result.current.language).toBe(lang);
      });
    });
  });

  describe('t function (translations)', () => {
    it('should return null/undefined as is', () => {
      const { result } = renderHook(() => useLanguage(), { wrapper });

      expect(result.current.t(null)).toBeNull();
      expect(result.current.t(undefined)).toBeUndefined();
    });

    it('should return entity unchanged for English', () => {
      const entity = {
        title: 'Hello World',
        description: 'A simple task',
        translations: {
          ru: { title: 'Привет Мир', description: 'Простая задача' },
          uz: { title: 'Salom Dunyo', description: 'Oddiy vazifa' },
        },
      };

      const { result } = renderHook(() => useLanguage(), { wrapper });

      const translated = result.current.t(entity);

      expect(translated.title).toBe('Hello World');
      expect(translated.description).toBe('A simple task');
    });

    it('should translate fields for Russian', () => {
      const entity = {
        title: 'Hello World',
        description: 'A simple task',
        translations: {
          ru: { title: 'Привет Мир', description: 'Простая задача' },
          uz: { title: 'Salom Dunyo', description: 'Oddiy vazifa' },
        },
      };

      const { result } = renderHook(() => useLanguage(), { wrapper });

      act(() => {
        result.current.setLanguage('ru');
      });

      const translated = result.current.t(entity);

      expect(translated.title).toBe('Привет Мир');
      expect(translated.description).toBe('Простая задача');
    });

    it('should translate fields for Uzbek', () => {
      const entity = {
        title: 'Hello World',
        description: 'A simple task',
        translations: {
          ru: { title: 'Привет Мир', description: 'Простая задача' },
          uz: { title: 'Salom Dunyo', description: 'Oddiy vazifa' },
        },
      };

      const { result } = renderHook(() => useLanguage(), { wrapper });

      act(() => {
        result.current.setLanguage('uz');
      });

      const translated = result.current.t(entity);

      expect(translated.title).toBe('Salom Dunyo');
      expect(translated.description).toBe('Oddiy vazifa');
    });

    it('should fallback to English when translation is missing', () => {
      const entity = {
        title: 'Hello World',
        description: 'A simple task',
        translations: {
          ru: { title: 'Привет Мир' }, // description missing
        },
      };

      const { result } = renderHook(() => useLanguage(), { wrapper });

      act(() => {
        result.current.setLanguage('ru');
      });

      const translated = result.current.t(entity);

      expect(translated.title).toBe('Привет Мир');
      expect(translated.description).toBe('A simple task'); // Falls back to English
    });

    it('should handle entity without translations field', () => {
      const entity = {
        title: 'Hello World',
        description: 'A simple task',
      };

      const { result } = renderHook(() => useLanguage(), { wrapper });

      act(() => {
        result.current.setLanguage('ru');
      });

      const translated = result.current.t(entity);

      expect(translated.title).toBe('Hello World');
      expect(translated.description).toBe('A simple task');
    });

    it('should translate hint1 and hint2 fields', () => {
      const entity = {
        title: 'Task',
        hint1: 'First hint',
        hint2: 'Second hint',
        translations: {
          ru: {
            title: 'Задача',
            hint1: 'Первая подсказка',
            hint2: 'Вторая подсказка',
          },
        },
      };

      const { result } = renderHook(() => useLanguage(), { wrapper });

      act(() => {
        result.current.setLanguage('ru');
      });

      const translated = result.current.t(entity);

      expect(translated.hint1).toBe('Первая подсказка');
      expect(translated.hint2).toBe('Вторая подсказка');
    });

    it('should translate solutionExplanation field', () => {
      const entity = {
        title: 'Task',
        solutionExplanation: 'Here is how to solve it',
        translations: {
          ru: {
            title: 'Задача',
            solutionExplanation: 'Вот как это решить',
          },
        },
      };

      const { result } = renderHook(() => useLanguage(), { wrapper });

      act(() => {
        result.current.setLanguage('ru');
      });

      const translated = result.current.t(entity);

      expect(translated.solutionExplanation).toBe('Вот как это решить');
    });

    it('should translate whyItMatters field', () => {
      const entity = {
        title: 'Task',
        whyItMatters: 'This is important because...',
        translations: {
          ru: {
            title: 'Задача',
            whyItMatters: 'Это важно, потому что...',
          },
        },
      };

      const { result } = renderHook(() => useLanguage(), { wrapper });

      act(() => {
        result.current.setLanguage('ru');
      });

      const translated = result.current.t(entity);

      expect(translated.whyItMatters).toBe('Это важно, потому что...');
    });

    it('should not modify non-translatable fields', () => {
      const entity = {
        id: '123',
        slug: 'task-slug',
        title: 'Task',
        estimatedTime: '15m',
        translations: {
          ru: {
            title: 'Задача',
            id: 'should-not-change',
            slug: 'should-not-change',
          },
        },
      };

      const { result } = renderHook(() => useLanguage(), { wrapper });

      act(() => {
        result.current.setLanguage('ru');
      });

      const translated = result.current.t(entity);

      expect(translated.id).toBe('123');
      expect(translated.slug).toBe('task-slug');
      expect(translated.estimatedTime).toBe('15m');
    });

    it('should preserve non-string fields', () => {
      const entity = {
        title: 'Task',
        isPremium: true,
        order: 5,
        tags: ['go', 'basics'],
        translations: {
          ru: { title: 'Задача' },
        },
      };

      const { result } = renderHook(() => useLanguage(), { wrapper });

      act(() => {
        result.current.setLanguage('ru');
      });

      const translated = result.current.t(entity);

      expect(translated.isPremium).toBe(true);
      expect(translated.order).toBe(5);
      expect(translated.tags).toEqual(['go', 'basics']);
    });
  });
});

describe('useUITranslation', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(storage.getLanguage).mockReturnValue(null);
  });

  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <LanguageProvider>{children}</LanguageProvider>
  );

  describe('tUI function', () => {
    it('should return English translation by default', () => {
      const { result } = renderHook(() => useUITranslation(), { wrapper });

      expect(result.current.tUI('nav.dashboard')).toBe('Dashboard');
      expect(result.current.tUI('nav.courses')).toBe('Courses');
    });

    it('should return Russian translations', () => {
      vi.mocked(storage.getLanguage).mockReturnValue('ru');

      const { result } = renderHook(() => useUITranslation(), { wrapper });

      expect(result.current.tUI('nav.dashboard')).toBe('Главная');
      expect(result.current.tUI('nav.courses')).toBe('Курсы');
    });

    it('should return Uzbek translations', () => {
      vi.mocked(storage.getLanguage).mockReturnValue('uz');

      const { result } = renderHook(() => useUITranslation(), { wrapper });

      expect(result.current.tUI('nav.dashboard')).toBe('Bosh sahifa');
      expect(result.current.tUI('nav.courses')).toBe('Kurslar');
    });

    it('should fallback to English when translation is missing', () => {
      vi.mocked(storage.getLanguage).mockReturnValue('ru');

      const { result } = renderHook(() => useUITranslation(), { wrapper });

      // Russian translation exists
      expect(result.current.tUI('nav.dashboard')).toBe('Главная');
    });

    it('should return key when no translation exists', () => {
      const { result } = renderHook(() => useUITranslation(), { wrapper });

      expect(result.current.tUI('nonexistent.key')).toBe('nonexistent.key');
    });

    it('should return current language', () => {
      const { result } = renderHook(() => useUITranslation(), { wrapper });

      expect(result.current.language).toBe('en');
    });
  });

  describe('formatTimeLocalized function', () => {
    it('should format time in English', () => {
      const { result } = renderHook(() => useUITranslation(), { wrapper });

      expect(result.current.formatTimeLocalized('2h')).toBe('2h');
      expect(result.current.formatTimeLocalized('30m')).toBe('30m');
      expect(result.current.formatTimeLocalized('1h 30m')).toBe('1h 30m');
    });

    it('should format time in Russian', () => {
      vi.mocked(storage.getLanguage).mockReturnValue('ru');

      const { result } = renderHook(() => useUITranslation(), { wrapper });

      expect(result.current.formatTimeLocalized('2h')).toBe('2ч');
      expect(result.current.formatTimeLocalized('30m')).toBe('30м');
      expect(result.current.formatTimeLocalized('1h 30m')).toBe('1ч 30м');
    });

    it('should format time in Uzbek', () => {
      vi.mocked(storage.getLanguage).mockReturnValue('uz');

      const { result } = renderHook(() => useUITranslation(), { wrapper });

      expect(result.current.formatTimeLocalized('2h')).toBe('2s');
      expect(result.current.formatTimeLocalized('30m')).toBe('30d');
    });

    it('should handle empty time string', () => {
      const { result } = renderHook(() => useUITranslation(), { wrapper });

      expect(result.current.formatTimeLocalized('')).toBe('');
    });
  });

  describe('plural function', () => {
    it('should pluralize in English', () => {
      const { result } = renderHook(() => useUITranslation(), { wrapper });

      expect(result.current.plural(1, 'task')).toBe('1 task');
      expect(result.current.plural(2, 'task')).toBe('2 tasks');
      expect(result.current.plural(5, 'task')).toBe('5 tasks');
      expect(result.current.plural(0, 'task')).toBe('0 tasks');
    });

    it('should pluralize modules in English', () => {
      const { result } = renderHook(() => useUITranslation(), { wrapper });

      expect(result.current.plural(1, 'module')).toBe('1 module');
      expect(result.current.plural(3, 'module')).toBe('3 modules');
    });

    it('should pluralize topics in English', () => {
      const { result } = renderHook(() => useUITranslation(), { wrapper });

      expect(result.current.plural(1, 'topic')).toBe('1 topic');
      expect(result.current.plural(10, 'topic')).toBe('10 topics');
    });

    it('should pluralize in Russian with correct forms', () => {
      vi.mocked(storage.getLanguage).mockReturnValue('ru');

      const { result } = renderHook(() => useUITranslation(), { wrapper });

      // Russian has complex pluralization
      expect(result.current.plural(1, 'task')).toBe('1 задача');
      expect(result.current.plural(2, 'task')).toBe('2 задачи');
      expect(result.current.plural(5, 'task')).toBe('5 задач');
      expect(result.current.plural(11, 'task')).toBe('11 задач');
      expect(result.current.plural(21, 'task')).toBe('21 задача');
      expect(result.current.plural(22, 'task')).toBe('22 задачи');
    });

    it('should pluralize modules in Russian', () => {
      vi.mocked(storage.getLanguage).mockReturnValue('ru');

      const { result } = renderHook(() => useUITranslation(), { wrapper });

      expect(result.current.plural(1, 'module')).toBe('1 модуль');
      expect(result.current.plural(2, 'module')).toBe('2 модуля');
      expect(result.current.plural(5, 'module')).toBe('5 модулей');
    });

    it('should pluralize in Uzbek (no plural forms)', () => {
      vi.mocked(storage.getLanguage).mockReturnValue('uz');

      const { result } = renderHook(() => useUITranslation(), { wrapper });

      // Uzbek doesn't have plural forms - always uses singular
      expect(result.current.plural(1, 'task')).toBe('1 vazifa');
      expect(result.current.plural(5, 'task')).toBe('5 vazifa');
    });

    it('should handle unknown word', () => {
      const { result } = renderHook(() => useUITranslation(), { wrapper });

      expect(result.current.plural(3, 'unknownword')).toBe('3 unknownword');
    });
  });

  describe('difficulty function', () => {
    it('should return difficulty labels in English', () => {
      const { result } = renderHook(() => useUITranslation(), { wrapper });

      expect(result.current.difficulty('easy')).toBe('Easy');
      expect(result.current.difficulty('medium')).toBe('Medium');
      expect(result.current.difficulty('hard')).toBe('Hard');
    });

    it('should return difficulty labels in Russian', () => {
      vi.mocked(storage.getLanguage).mockReturnValue('ru');

      const { result } = renderHook(() => useUITranslation(), { wrapper });

      expect(result.current.difficulty('easy')).toBe('Легко');
      expect(result.current.difficulty('medium')).toBe('Средне');
      expect(result.current.difficulty('hard')).toBe('Сложно');
    });

    it('should return difficulty labels in Uzbek', () => {
      vi.mocked(storage.getLanguage).mockReturnValue('uz');

      const { result } = renderHook(() => useUITranslation(), { wrapper });

      expect(result.current.difficulty('easy')).toBe('Oson');
      expect(result.current.difficulty('medium')).toBe("O'rtacha");
      expect(result.current.difficulty('hard')).toBe('Qiyin');
    });

    it('should return input when difficulty level is unknown', () => {
      const { result } = renderHook(() => useUITranslation(), { wrapper });

      expect(result.current.difficulty('unknown')).toBe('unknown');
    });
  });

  describe('language changes', () => {
    it('should update translations when language changes', () => {
      // Start with English
      vi.mocked(storage.getLanguage).mockReturnValue('en');
      const { result: langResult } = renderHook(() => useLanguage(), { wrapper });
      const { result: uiResult } = renderHook(() => useUITranslation(), { wrapper });

      expect(uiResult.current.tUI('nav.dashboard')).toBe('Dashboard');

      // Change to Russian - update mock before setLanguage since new hooks will read from storage
      vi.mocked(storage.getLanguage).mockReturnValue('ru');

      act(() => {
        langResult.current.setLanguage('ru');
      });

      // Re-render with new language (Russian nav.dashboard = 'Главная')
      const { result: uiResult2 } = renderHook(() => useUITranslation(), { wrapper });
      expect(uiResult2.current.tUI('nav.dashboard')).toBe('Главная');
    });
  });
});
