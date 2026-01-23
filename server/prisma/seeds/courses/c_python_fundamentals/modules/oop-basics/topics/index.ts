import topic from './classes/topic';
import { classesTasks } from './classes/tasks';

export const topics = [
	{
		...topic,
		tasks: classesTasks,
	},
];

export default topics;
