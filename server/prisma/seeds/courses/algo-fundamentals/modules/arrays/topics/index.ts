import { Topic } from '../../../../../types';
import { tasks } from './basics/tasks';

export const basicsTopic: Topic = {
	slug: 'algo-arrays-basics',
	title: 'Array Operations',
	description: 'Essential array algorithms: two pointers, sliding window, prefix sums, and common patterns.',
	difficulty: 'easy',
	estimatedTime: '5h',
	order: 1,
	tasks,
	translations: {
		ru: {
			title: 'Операции с массивами',
			description: 'Основные алгоритмы массивов: два указателя, скользящее окно, префиксные суммы и общие паттерны.',
		},
		uz: {
			title: 'Massiv operatsiyalari',
			description: 'Asosiy massiv algoritmlari: ikki ko\'rsatkich, sirg\'anuvchi oyna, prefiks yig\'indilari va umumiy patternlar.',
		},
	},
};
