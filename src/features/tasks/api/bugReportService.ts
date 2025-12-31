import { api } from '@/lib/api';
import { createLogger } from '@/lib/logger';

const log = createLogger('BugReportService');

export type BugCategory = 'description' | 'solution' | 'editor' | 'hints' | 'ai-tutor' | 'other';
export type BugSeverity = 'low' | 'medium' | 'high';

export interface BugReportData {
  title: string;
  description: string;
  category: BugCategory;
  severity: BugSeverity;
  taskId?: string;
  metadata?: {
    userCode?: string;
    browserInfo?: string;
    url?: string;
  };
}

export interface BugReport extends BugReportData {
  id: string;
  userId: string;
  status: string;
  createdAt: string;
  updatedAt: string;
  user?: { name: string; email: string };
  task?: { title: string; slug: string };
}

export const bugReportService = {
  /**
   * Submit a new bug report
   */
  submit: async (data: BugReportData): Promise<BugReport> => {
    log.info('Submitting bug report', data.category, data.title);
    return api.post<BugReport>('/bugreports', data);
  },

  /**
   * Get user's own bug reports
   */
  getMyReports: async (): Promise<BugReport[]> => {
    return api.get<BugReport[]>('/bugreports/my');
  },
};
