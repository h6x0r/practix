/**
 * gRPC Unary Server Interceptors Topic
 * Learn to build interceptors for logging, timeouts, retries, and composition
 */

import { Topic } from '../../../../types';
import { tasks } from './tasks';

export const topic: Topic = {
	title: 'Unary Server Interceptors',
	description: 'Build gRPC unary server interceptors for cross-cutting concerns like logging, timeouts, retries, and context propagation.',
	difficulty: 'medium',
	estimatedTime: '2h',	order: 0,
	tasks: tasks,
	translations: {
		ru: {
			title: 'Унарные серверные перехватчики',
			description: 'Создайте унарные серверные перехватчики gRPC для сквозных задач, таких как логирование, таймауты, повторные попытки и передача контекста.',
		},
		uz: {
			title: 'Unary server interceptorlar',
			description: 'Logga yozish, vaqt tugashi, qayta urinishlar va kontekstni uzatish kabi kesishuvchi vazifalar uchun gRPC unary server interceptorlarini yarating.',
		},
	},
};
