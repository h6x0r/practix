import { multiTurnModule } from './module';
import { multiTurnTopics } from './topics';

export const multiTurn = {
  ...multiTurnModule,
  topics: multiTurnTopics,
};
