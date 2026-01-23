import { personasTopic } from './personas/topic';
import { personasTasks } from './personas/tasks';

export const roleDesignTopics = [
  {
    ...personasTopic,
    tasks: personasTasks,
  },
];
