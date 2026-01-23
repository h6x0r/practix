
import { api } from '@/lib/api';
import { RoadmapUI, RoadmapVariantsResponse, RoadmapGenerationInput } from '../model/types';

export interface RoadmapTemplate {
  id: string;
  title: string;
  description: string;
  icon: string;
  levels: string[];
}

// ============================================================================
// NEW: Variant-based generation (v2)
// ============================================================================

export interface GenerateVariantsParams extends RoadmapGenerationInput {}

export interface SelectVariantParams {
  variantId: string;
  name: string;
  description: string;
  totalTasks: number;
  estimatedHours: number;
  estimatedMonths: number;
  targetRole: string;
  difficulty: 'easy' | 'medium' | 'hard';
  phases: any[];
}

/**
 * Roadmap Service - Connected to Real Backend API
 *
 * Endpoints:
 * - GET /roadmaps/templates - Available roadmap templates
 * - GET /roadmaps/me - User's current roadmap
 * - POST /roadmaps/generate-variants - Generate roadmap variants (v2)
 * - POST /roadmaps/select-variant - Select and create roadmap from variant
 * - GET /roadmaps/can-generate - Check generation limits
 * - DELETE /roadmaps/me - Delete/reset roadmap
 */
export const roadmapService = {
  /**
   * Get available roadmap templates
   */
  getTemplates: async (): Promise<RoadmapTemplate[]> => {
    return api.get<RoadmapTemplate[]>('/roadmaps/templates');
  },

  /**
   * Get user's current roadmap
   * Returns null if no roadmap exists
   */
  getUserRoadmap: async (): Promise<RoadmapUI | null> => {
    try {
      return await api.get<RoadmapUI>('/roadmaps/me');
    } catch (err: unknown) {
      if (err && typeof err === 'object' && 'status' in err && err.status === 404) {
        return null;
      }
      throw err;
    }
  },

  /**
   * Delete/reset user's roadmap
   */
  deleteRoadmap: async (): Promise<void> => {
    await api.delete('/roadmaps/me');
  },

  // ============================================================================
  // NEW: Variant-based generation (v2)
  // ============================================================================

  /**
   * Generate 3-5 roadmap variants based on user input
   * First generation is FREE, regeneration requires Premium
   */
  generateVariants: async (params: GenerateVariantsParams): Promise<RoadmapVariantsResponse> => {
    return api.post<RoadmapVariantsResponse>('/roadmaps/generate-variants', params);
  },

  /**
   * Get user's saved variants (if any were generated but not yet selected)
   */
  getUserVariants: async (): Promise<RoadmapVariantsResponse> => {
    return api.get<RoadmapVariantsResponse>('/roadmaps/variants');
  },

  /**
   * Select a variant and create the roadmap from it
   * Saves the selected variant as the user's active roadmap
   */
  selectVariant: async (params: SelectVariantParams): Promise<RoadmapUI> => {
    return api.post<RoadmapUI>('/roadmaps/select-variant', params);
  },

  /**
   * Check if user can generate a roadmap
   * Returns generation status and limits
   */
  canGenerate: async (): Promise<{
    canGenerate: boolean;
    reason?: string;
    isPremium: boolean;
    generationCount: number;
  }> => {
    return api.get('/roadmaps/can-generate');
  },
};
