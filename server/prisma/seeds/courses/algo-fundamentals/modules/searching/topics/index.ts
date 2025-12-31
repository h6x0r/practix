import { Topic } from '../../../../../types';
import { tasks } from './basics/tasks';

export const basicsTopic: Topic = {
	slug: 'searching-basics',
	title: 'Searching Techniques',
	description: 'Master binary search and its variations: finding elements, bounds, peaks, and searching in rotated arrays',
	order: 1,
	tasks,
	translations: {
		ru: {
			title: 'Техники поиска',
			description: 'Освойте бинарный поиск и его вариации: поиск элементов, границ, пиков и поиск в повёрнутых массивах',
		},
		uz: {
			title: 'Qidiruv texnikalari',
			description: 'Binar qidiruv va uning variantlarini o\'rganing: elementlarni, chegaralarni, cho\'qqilarni topish va aylantirilgan massivlarda qidiruv',
		},
	},
};

export const topics = [basicsTopic];

export default topics;
