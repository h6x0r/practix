import { contextTopic } from './context/topic';
import { contextTasks } from './context/tasks';

export const contextManagementTopics = [
  {
    ...contextTopic,
    tasks: contextTasks,
  },
];
