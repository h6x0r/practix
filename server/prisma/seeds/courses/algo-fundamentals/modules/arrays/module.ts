import { Module } from '../../../../types';
import { basicsTopic } from './topics';

export const arraysModule: Module = {
	slug: 'algo-arrays',
	title: 'Arrays',
	description: 'Master array manipulation techniques: traversal, two pointers, sliding window, and prefix sums.',
	section: 'algorithms',
	order: 1,
	difficulty: 'easy',
	estimatedTime: '5h',
	topics: [basicsTopic],
	translations: {
		ru: {
			title: 'Массивы',
			description: 'Освойте техники работы с массивами: обход, два указателя, скользящее окно и префиксные суммы.',
		},
		uz: {
			title: 'Massivlar',
			description: 'Massivlar bilan ishlash texnikalarini o\'rganing: o\'tish, ikki ko\'rsatkich, sirg\'anuvchi oyna va prefiks yig\'indilari.',
		},
	},
};
