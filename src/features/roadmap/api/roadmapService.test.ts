import { describe, it, expect, beforeEach, vi } from 'vitest';
import { roadmapService } from './roadmapService';

vi.mock('@/lib/api', () => ({
  api: {
    get: vi.fn(),
    post: vi.fn(),
    delete: vi.fn(),
  },
}));

import { api } from '@/lib/api';

describe('roadmapService', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('getTemplates', () => {
    it('should fetch roadmap templates', async () => {
      const mockTemplates = [
        { id: '1', title: 'Backend Developer', description: 'Server-side development', icon: '🔧', levels: ['junior', 'middle', 'senior'] },
        { id: '2', title: 'Frontend Developer', description: 'Client-side development', icon: '🎨', levels: ['junior', 'middle'] },
      ];

      vi.mocked(api.get).mockResolvedValueOnce(mockTemplates);

      const result = await roadmapService.getTemplates();

      expect(api.get).toHaveBeenCalledWith('/roadmaps/templates');
      expect(result).toHaveLength(2);
    });

    it('should throw on API error', async () => {
      vi.mocked(api.get).mockRejectedValueOnce(new Error('Service unavailable'));

      await expect(roadmapService.getTemplates()).rejects.toThrow('Service unavailable');
    });
  });

  describe('getUserRoadmap', () => {
    it('should fetch user roadmap', async () => {
      const mockRoadmap = {
        id: 'roadmap-1',
        name: 'My Backend Path',
        phases: [{ id: 'phase-1', name: 'Basics', tasks: [] }],
      };

      vi.mocked(api.get).mockResolvedValueOnce(mockRoadmap);

      const result = await roadmapService.getUserRoadmap();

      expect(api.get).toHaveBeenCalledWith('/roadmaps/me');
      expect(result).toEqual(mockRoadmap);
    });

    it('should return null when no roadmap exists (404)', async () => {
      vi.mocked(api.get).mockRejectedValueOnce({ status: 404 });

      const result = await roadmapService.getUserRoadmap();

      expect(result).toBeNull();
    });

    it('should throw on other errors', async () => {
      vi.mocked(api.get).mockRejectedValueOnce(new Error('Server error'));

      await expect(roadmapService.getUserRoadmap()).rejects.toThrow('Server error');
    });
  });

  describe('deleteRoadmap', () => {
    it('should delete user roadmap', async () => {
      vi.mocked(api.delete).mockResolvedValueOnce(undefined);

      await roadmapService.deleteRoadmap();

      expect(api.delete).toHaveBeenCalledWith('/roadmaps/me');
    });

    it('should throw on delete error', async () => {
      vi.mocked(api.delete).mockRejectedValueOnce(new Error('Not found'));

      await expect(roadmapService.deleteRoadmap()).rejects.toThrow('Not found');
    });
  });

  describe('generateVariants', () => {
    it('should generate roadmap variants', async () => {
      const mockResponse = {
        variants: [
          { variantId: 'v1', name: 'Quick Path', estimatedMonths: 3 },
          { variantId: 'v2', name: 'Deep Dive', estimatedMonths: 6 },
        ],
        expiresAt: '2025-01-16T00:00:00Z',
      };

      vi.mocked(api.post).mockResolvedValueOnce(mockResponse);

      const result = await roadmapService.generateVariants({
        role: 'backend',
        level: 'junior',
      });

      expect(api.post).toHaveBeenCalledWith('/roadmaps/generate-variants', {
        role: 'backend',
        level: 'junior',
      });
      expect(result.variants).toHaveLength(2);
    });

    it('should throw on premium-required error', async () => {
      vi.mocked(api.post).mockRejectedValueOnce(new Error('Premium required'));

      await expect(roadmapService.generateVariants({ role: 'backend', level: 'junior' })).rejects.toThrow('Premium required');
    });
  });

  describe('getUserVariants', () => {
    it('should fetch user variants', async () => {
      const mockResponse = {
        variants: [{ variantId: 'v1', name: 'Saved Variant' }],
        expiresAt: '2025-01-16T00:00:00Z',
      };

      vi.mocked(api.get).mockResolvedValueOnce(mockResponse);

      const result = await roadmapService.getUserVariants();

      expect(api.get).toHaveBeenCalledWith('/roadmaps/variants');
      expect(result.variants).toHaveLength(1);
    });

    it('should throw on API error', async () => {
      vi.mocked(api.get).mockRejectedValueOnce(new Error('Unauthorized'));

      await expect(roadmapService.getUserVariants()).rejects.toThrow('Unauthorized');
    });
  });

  describe('selectVariant', () => {
    it('should select variant and create roadmap', async () => {
      const mockRoadmap = {
        id: 'roadmap-from-variant',
        name: 'Selected Path',
        phases: [],
      };

      vi.mocked(api.post).mockResolvedValueOnce(mockRoadmap);

      const params = {
        variantId: 'v1',
        name: 'My Path',
        description: 'Custom description',
        totalTasks: 50,
        estimatedHours: 100,
        estimatedMonths: 3,
        targetRole: 'backend',
        difficulty: 'medium' as const,
        phases: [],
      };

      const result = await roadmapService.selectVariant(params);

      expect(api.post).toHaveBeenCalledWith('/roadmaps/select-variant', params);
      expect(result).toEqual(mockRoadmap);
    });

    it('should throw on expired variant', async () => {
      vi.mocked(api.post).mockRejectedValueOnce(new Error('Variant expired'));

      await expect(roadmapService.selectVariant({
        variantId: 'expired',
        name: 'Test',
        description: '',
        totalTasks: 0,
        estimatedHours: 0,
        estimatedMonths: 0,
        targetRole: '',
        difficulty: 'easy',
        phases: [],
      })).rejects.toThrow('Variant expired');
    });
  });

  describe('canGenerate', () => {
    it('should check generation permission for free user', async () => {
      const mockResponse = {
        canGenerate: true,
        isPremium: false,
        generationCount: 0,
      };

      vi.mocked(api.get).mockResolvedValueOnce(mockResponse);

      const result = await roadmapService.canGenerate();

      expect(api.get).toHaveBeenCalledWith('/roadmaps/can-generate');
      expect(result.canGenerate).toBe(true);
      expect(result.isPremium).toBe(false);
    });

    it('should return cannot generate with reason', async () => {
      const mockResponse = {
        canGenerate: false,
        reason: 'Free generation limit reached',
        isPremium: false,
        generationCount: 1,
      };

      vi.mocked(api.get).mockResolvedValueOnce(mockResponse);

      const result = await roadmapService.canGenerate();

      expect(result.canGenerate).toBe(false);
      expect(result.reason).toBe('Free generation limit reached');
    });

    it('should throw on API error', async () => {
      vi.mocked(api.get).mockRejectedValueOnce(new Error('Network error'));

      await expect(roadmapService.canGenerate()).rejects.toThrow('Network error');
    });
  });
});
