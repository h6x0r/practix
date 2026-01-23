import { Topic } from '../../../../../../types';
import tasks from './tasks';

export const topic: Topic = {
	slug: 'lists',
	title: 'Lists and Dictionaries',
	description: 'Working with lists, list comprehensions, and dictionaries.',
	order: 1,
	tasks: tasks,
	translations: {
		ru: {
			title: 'Списки и словари',
			description: 'Работа со списками, list comprehensions и словарями.',
		},
		uz: {
			title: "Ro'yxatlar va lug'atlar",
			description: "Ro'yxatlar, list comprehensions va lug'atlar bilan ishlash.",
		},
	},
};
