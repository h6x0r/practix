import { Topic } from '../../../../types';
import * as taskImports from './tasks';

const tasks = Object.values(taskImports);

export const topic: Topic = {
	title: 'Context-Aware Logging Implementation',
	description: 'Implement production-ready logging with request tracing and structured fields for observability.',
	difficulty: 'medium',
	estimatedTime: '45m',
	order: 1,
	tasks,
	translations: {
		ru: {
			title: 'Реализация контекстно-зависимого логирования',
			description: 'Реализация готового к продакшену логирования с трассировкой запросов и структурированными полями для наблюдаемости.'
		},
		uz: {
			title: 'Kontekstga bog\'liq loglashtirish amalga oshirish',
			description: 'Ishlab chiqarishga tayyor loglashtirish, so\'rovlarni kuzatish va kuzatish uchun tuzilgan maydonlar bilan amalga oshirish.'
		}
	}
};
