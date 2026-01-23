import { formattingTopic } from './formatting/topic';
import { formattingTasks } from './formatting/tasks';

export const outputStructuringTopics = [
  {
    ...formattingTopic,
    tasks: formattingTasks,
  },
];
