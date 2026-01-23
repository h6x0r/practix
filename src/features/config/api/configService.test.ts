import { describe, it, expect, beforeEach, vi } from 'vitest';
import { configService } from './configService';

vi.mock('../data/repository', () => ({
  configRepository: {
    getNavItems: vi.fn(),
    getPrompt: vi.fn(),
  },
  MOCK_PROMPTS: {
    systemPrompt: 'system',
    hintPrompt: 'hint',
  },
}));

import { configRepository } from '../data/repository';

describe('configService', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('getNavigation', () => {
    it('should fetch navigation items', async () => {
      const mockNavItems = [
        { path: '/dashboard', label: 'Dashboard', icon: 'home' },
        { path: '/courses', label: 'Courses', icon: 'book' },
        { path: '/playground', label: 'Playground', icon: 'code' },
      ];

      vi.mocked(configRepository.getNavItems).mockResolvedValueOnce(mockNavItems);

      const result = await configService.getNavigation();

      expect(configRepository.getNavItems).toHaveBeenCalled();
      expect(result).toHaveLength(3);
      expect(result[0].path).toBe('/dashboard');
    });

    it('should return empty array when no nav items', async () => {
      vi.mocked(configRepository.getNavItems).mockResolvedValueOnce([]);

      const result = await configService.getNavigation();

      expect(result).toHaveLength(0);
    });
  });

  describe('getPromptTemplate', () => {
    it('should fetch prompt template', async () => {
      vi.mocked(configRepository.getPrompt).mockResolvedValueOnce('You are a helpful coding assistant.');

      const result = await configService.getPromptTemplate('systemPrompt');

      expect(configRepository.getPrompt).toHaveBeenCalledWith('systemPrompt');
      expect(result).toBe('You are a helpful coding assistant.');
    });

    it('should return empty string when prompt not found', async () => {
      vi.mocked(configRepository.getPrompt).mockResolvedValueOnce(null);

      const result = await configService.getPromptTemplate('hintPrompt');

      expect(result).toBe('');
    });

    it('should return empty string for undefined prompt', async () => {
      vi.mocked(configRepository.getPrompt).mockResolvedValueOnce(undefined);

      const result = await configService.getPromptTemplate('systemPrompt');

      expect(result).toBe('');
    });
  });
});
