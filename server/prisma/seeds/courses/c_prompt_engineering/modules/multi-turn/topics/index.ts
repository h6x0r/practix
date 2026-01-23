import { dialogueTopic } from './dialogue/topic';
import { dialogueTasks } from './dialogue/tasks';

export const multiTurnTopics = [
  {
    ...dialogueTopic,
    tasks: dialogueTasks,
  },
];
