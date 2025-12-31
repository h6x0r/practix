import { Topic } from '../../../../../types';
import { tasks } from './basics/tasks';

export const basicsTopic: Topic = {
	slug: 'algo-linked-lists-basics',
	title: 'Linked List Operations',
	description: 'Essential linked list algorithms: reversal, cycle detection, merging, and common patterns.',
	difficulty: 'medium',
	estimatedTime: '6h',
	order: 1,
	tasks,
	translations: {
		ru: {
			title: 'Операции со связными списками',
			description: 'Основные алгоритмы связных списков: разворот, обнаружение циклов, слияние и распространённые паттерны.',
		},
		uz: {
			title: 'Bog\'langan ro\'yxat operatsiyalari',
			description: 'Asosiy bog\'langan ro\'yxat algoritmlari: teskari aylantirish, tsikl aniqlash, birlashtirish va keng tarqalgan patternlar.',
		},
	},
};
