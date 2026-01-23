import { contextManagementModule } from './module';
import { contextManagementTopics } from './topics';

export const contextManagement = {
  ...contextManagementModule,
  topics: contextManagementTopics,
};
