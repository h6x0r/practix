import { Module } from '../../../../types';
import { basicsTopic } from './topics';

export const stacksQueuesModule: Module = {
	slug: 'algo-stacks-queues',
	title: 'Stacks & Queues',
	description: 'Master stack and queue operations: LIFO/FIFO patterns, monotonic stacks, and classic problems.',
	section: 'algorithms',
	order: 4,
	difficulty: 'medium',
	estimatedTime: '5h',
	topics: [basicsTopic],
	translations: {
		ru: {
			title: 'Стеки и очереди',
			description: 'Освойте операции со стеками и очередями: паттерны LIFO/FIFO, монотонные стеки и классические задачи.',
		},
		uz: {
			title: 'Steklar va navbatlar',
			description: 'Stek va navbat operatsiyalarini o\'rganing: LIFO/FIFO patternlari, monoton steklar va klassik masalalar.',
		},
	},
};
