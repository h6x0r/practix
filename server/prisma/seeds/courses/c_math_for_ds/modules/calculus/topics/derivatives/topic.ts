import { Topic } from '../../../../../../types';
import tasks from './tasks';

export const topic: Topic = {
	slug: 'derivatives',
	title: 'Derivatives & Gradients',
	description: 'Learn derivatives, partial derivatives, and gradient computation for optimization.',
	order: 1,
	tasks: tasks,
	translations: {
		ru: {
			title: 'Производные и градиенты',
			description: 'Изучите производные, частные производные и вычисление градиентов для оптимизации.',
		},
		uz: {
			title: 'Hosilalar va gradientlar',
			description: 'Optimizatsiya uchun hosilalar, qisman hosilalar va gradient hisoblashni o\'rganing.',
		},
	},
};
