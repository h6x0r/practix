import { improvementTopic } from './improvement/topic';
import { improvementTasks } from './improvement/tasks';

export const iterationRefinementTopics = [
  {
    ...improvementTopic,
    tasks: improvementTasks,
  },
];
