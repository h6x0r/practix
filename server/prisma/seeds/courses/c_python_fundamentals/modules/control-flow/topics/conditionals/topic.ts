import { Topic } from '../../../../../../types';
import tasks from './tasks';

export const topic: Topic = {
	slug: 'conditionals',
	title: 'Conditionals and Loops',
	description: 'if/elif/else statements, for and while loops, break and continue.',
	order: 1,
	tasks: tasks,
	translations: {
		ru: {
			title: 'Условия и циклы',
			description: 'Операторы if/elif/else, циклы for и while, break и continue.',
		},
		uz: {
			title: 'Shartlar va sikllar',
			description: 'if/elif/else operatorlari, for va while sikllari, break va continue.',
		},
	},
};
