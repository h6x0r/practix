import contextRecall from './01-context-recall';
import conversationState from './02-conversation-state';
import topicTransitions from './03-topic-transitions';
import conversationSummary from './04-conversation-summary';
import conversationReset from './05-conversation-reset';

export const dialogueTasks = [
  contextRecall,
  conversationState,
  topicTransitions,
  conversationSummary,
  conversationReset,
];
