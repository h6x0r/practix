import topicMeta from './topic';
import tasks from './tasks';
import { Topic } from '../../../../../../types';

const topic: Topic = {
	...topicMeta,
	tasks,
};

export default topic;
