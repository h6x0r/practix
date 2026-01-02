import { NavItemConfig } from '@/types';
import { configRepository, MOCK_PROMPTS } from '../data/repository';

export const configService = {
  getNavigation: async (): Promise<NavItemConfig[]> => {
    return configRepository.getNavItems();
  },

  getPromptTemplate: async (key: keyof typeof MOCK_PROMPTS): Promise<string> => {
    const prompt = await configRepository.getPrompt(key);
    return prompt || '';
  }
};
