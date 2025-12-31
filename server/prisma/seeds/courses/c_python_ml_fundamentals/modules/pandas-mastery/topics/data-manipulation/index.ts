import topicMeta from './topic';
import { Topic } from '../../../../../../types';
import tasks from './tasks';

const topic: Topic = {
	...topicMeta,
	tasks,
};

export default topic;
