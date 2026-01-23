import { Topic } from '../../../../../../types';
import tasks from './tasks';

export const topic: Topic = {
	slug: 'parameters',
	title: 'Function Parameters',
	description: 'Default arguments, *args, **kwargs, and lambda functions.',
	order: 1,
	tasks: tasks,
	translations: {
		ru: {
			title: 'Параметры функций',
			description: 'Аргументы по умолчанию, *args, **kwargs и лямбда-функции.',
		},
		uz: {
			title: 'Funksiya parametrlari',
			description: "Standart argumentlar, *args, **kwargs va lambda funksiyalari.",
		},
	},
};
