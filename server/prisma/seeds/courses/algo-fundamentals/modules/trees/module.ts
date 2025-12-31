import { Module } from '../../../../types';
import { basicsTopic } from './topics';

export const treesModule: Module = {
	slug: 'algo-trees',
	title: 'Trees',
	description: 'Master binary tree operations: traversals, depth-first and breadth-first search, and common tree problems.',
	section: 'algorithms',
	order: 5,
	difficulty: 'medium',
	estimatedTime: '6h',
	topics: [basicsTopic],
	translations: {
		ru: {
			title: 'Деревья',
			description: 'Освойте операции с бинарными деревьями: обходы, поиск в глубину и ширину, и распространённые задачи.',
		},
		uz: {
			title: 'Daraxtlar',
			description: 'Binar daraxt operatsiyalarini o\'rganing: o\'tishlar, chuqurlik va kenglik bo\'yicha qidirish va keng tarqalgan masalalar.',
		},
	},
};
