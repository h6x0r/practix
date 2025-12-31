import { Topic } from '../../../../types';
import { topicMeta } from './topic';
import { tasks } from './tasks';

export const topic: Topic = {
	...topicMeta,
	tasks,
};
