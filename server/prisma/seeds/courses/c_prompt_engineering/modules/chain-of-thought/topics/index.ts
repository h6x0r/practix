import { reasoningTopic } from './reasoning/topic';
import { reasoningTasks } from './reasoning/tasks';

export const chainOfThoughtTopics = [
  {
    ...reasoningTopic,
    tasks: reasoningTasks,
  },
];
