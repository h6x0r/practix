import { Module } from '../../../../types';
import { basicsTopic } from './topics';

export const sortingModule: Module = {
	slug: 'algo-sorting',
	title: 'Sorting Algorithms',
	description: 'Master fundamental sorting algorithms: bubble, selection, insertion, merge, and quick sort.',
	difficulty: 'medium',
	estimatedTime: '8h',
	order: 6,
	topics: [basicsTopic],
	translations: {
		ru: {
			title: 'Алгоритмы сортировки',
			description: 'Освойте фундаментальные алгоритмы сортировки: пузырьковая, выбором, вставками, слиянием и быстрая.',
		},
		uz: {
			title: 'Saralash algoritmlari',
			description: 'Asosiy saralash algoritmlarini o\'rganing: pufakchali, tanlash, qo\'yish, birlashtirish va tez saralash.',
		},
	},
};
