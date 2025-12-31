
import { NavItemConfig } from '@/types';

export const MOCK_NAV_ITEMS: NavItemConfig[] = [
  { label: 'Dashboard', path: '/', iconKey: 'dashboard', translationKey: 'nav.dashboard' },
  { label: 'Courses', path: '/courses', iconKey: 'book', translationKey: 'nav.courses' },
  { label: 'Playground', path: '/playground', iconKey: 'terminal', translationKey: 'nav.playground' },
  { label: 'Roadmap', path: '/roadmap', iconKey: 'map', translationKey: 'nav.roadmap' },
  { label: 'Leaderboard', path: '/leaderboard', iconKey: 'trophy', translationKey: 'nav.leaderboard' },
  { label: 'My Tasks', path: '/my-tasks', iconKey: 'code', translationKey: 'nav.myTasks' },
  { label: 'Analytics', path: '/analytics', iconKey: 'chart', translationKey: 'nav.analytics' },
  { label: 'Payments', path: '/premium', iconKey: 'creditCard', translationKey: 'nav.payments', authRequired: true },
  { label: 'Settings', path: '/settings', iconKey: 'settings', translationKey: 'nav.settings', authRequired: true },
  { label: 'Admin', path: '/admin', iconKey: 'chart', translationKey: 'nav.admin', adminOnly: true },
];

export const MOCK_PROMPTS = {
  tutor: `
      You are an expert programming tutor for \${language}.
      The student is working on the task: "\${taskTitle}".
      
      Here is their current code:
      \`\`\`\${language}
      \${userCode}
      \`\`\`
      
      The student asks: "\${question}"
      
      Provide a helpful, concise hint or explanation. Do not give the full solution code directly. 
      Focus on guiding them to the answer. Use Markdown formatting.
  `,
};

export const configRepository = {
  getNavItems: async () => MOCK_NAV_ITEMS,
  getPrompt: async (key: keyof typeof MOCK_PROMPTS) => MOCK_PROMPTS[key]
};
