import { Topic } from '../../../../../types';
import { tasks } from './basics/tasks';

export const basicsTopic: Topic = {
	slug: 'algo-trees-basics',
	title: 'Binary Tree Operations',
	description: 'Essential tree algorithms: traversals, depth calculation, validation, and path problems.',
	difficulty: 'medium',
	estimatedTime: '6h',
	order: 1,
	tasks,
	translations: {
		ru: {
			title: 'Операции с бинарными деревьями',
			description: 'Основные алгоритмы деревьев: обходы, вычисление глубины, валидация и задачи о путях.',
		},
		uz: {
			title: 'Binar daraxt operatsiyalari',
			description: 'Asosiy daraxt algoritmlari: o\'tishlar, chuqurlik hisoblash, validatsiya va yo\'l masalalari.',
		},
	},
};
