import { Module } from '../../../../types';
import { topics } from './topics';

export const loggingModule: Module = {
	title: 'Structured Logging',
	description: 'Master context-aware logging with request tracing and structured fields for production observability.',
	section: 'production-patterns',
	order: 11,
	topics,
	translations: {
		ru: {
			title: 'Структурированное логирование',
			description: 'Освойте контекстно-зависимое логирование с трассировкой запросов и структурированными полями для наблюдаемости в production.',
		},
		uz: {
			title: 'Strukturalashtirilgan logging',
			description: 'Ishlab chiqarishda kuzatish uchun so\'rov kuzatuvi va strukturalashtirilgan maydonlar bilan kontekstga bog\'liq loggingni o\'rganing.',
		},
	},
};
