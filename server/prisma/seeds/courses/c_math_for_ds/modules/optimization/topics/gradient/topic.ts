import { Topic } from '../../../../../../types';
import tasks from './tasks';

export const topic: Topic = {
	slug: 'gradient',
	title: 'Gradient Descent',
	description: 'Learn optimization algorithms: gradient descent, learning rate, and convergence.',
	order: 1,
	tasks: tasks,
	translations: {
		ru: {
			title: 'Градиентный спуск',
			description: 'Изучите алгоритмы оптимизации: градиентный спуск, скорость обучения и сходимость.',
		},
		uz: {
			title: 'Gradient tushishi',
			description: 'Optimizatsiya algoritmlarini o\'rganing: gradient tushishi, o\'rganish tezligi va yaqinlashish.',
		},
	},
};
