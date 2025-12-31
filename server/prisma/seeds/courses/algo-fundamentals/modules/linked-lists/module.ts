import { Module } from '../../../../types';
import { basicsTopic } from './topics';

export const linkedListsModule: Module = {
	slug: 'algo-linked-lists',
	title: 'Linked Lists',
	description: 'Master linked list operations: traversal, reversal, cycle detection, and merging techniques.',
	section: 'algorithms',
	order: 3,
	difficulty: 'medium',
	estimatedTime: '6h',
	topics: [basicsTopic],
	translations: {
		ru: {
			title: 'Связные списки',
			description: 'Освойте операции со связными списками: обход, разворот, обнаружение циклов и техники слияния.',
		},
		uz: {
			title: 'Bog\'langan ro\'yxatlar',
			description: 'Bog\'langan ro\'yxat operatsiyalarini o\'rganing: o\'tish, teskari aylantirish, tsikl aniqlash va birlashtirish texnikalari.',
		},
	},
};
