import { Topic } from '../../../../../types';
import { tasks } from './basics/tasks';

export const basicsTopic: Topic = {
	slug: 'algo-stacks-queues-basics',
	title: 'Stack & Queue Operations',
	description: 'Essential stack and queue algorithms: valid parentheses, monotonic stack, and queue implementation.',
	difficulty: 'medium',
	estimatedTime: '5h',
	order: 1,
	tasks,
	translations: {
		ru: {
			title: 'Операции со стеками и очередями',
			description: 'Основные алгоритмы стеков и очередей: проверка скобок, монотонный стек и реализация очереди.',
		},
		uz: {
			title: 'Stek va navbat operatsiyalari',
			description: 'Asosiy stek va navbat algoritmlari: qavslarni tekshirish, monoton stek va navbat amalga oshirish.',
		},
	},
};
