import { Topic } from '../../../../../types';
import { tasks } from './basics/tasks';

export const basicsTopic: Topic = {
	slug: 'algo-sorting-basics',
	title: 'Sorting Fundamentals',
	description: 'Learn essential sorting algorithms from simple to efficient: bubble, selection, insertion, merge, and quick sort.',
	difficulty: 'medium',
	estimatedTime: '8h',
	order: 1,
	tasks,
	translations: {
		ru: {
			title: 'Основы сортировки',
			description: 'Изучите основные алгоритмы сортировки от простых до эффективных: пузырьковая, выбором, вставками, слиянием и быстрая.',
		},
		uz: {
			title: 'Saralash asoslari',
			description: 'Oddiydan samaradorga asosiy saralash algoritmlarini o\'rganing: pufakchali, tanlash, qo\'yish, birlashtirish va tez saralash.',
		},
	},
};
