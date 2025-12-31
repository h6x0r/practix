import { Module } from '../../../../types';
import { topics } from './topics';

export const metricsModule: Module = {
	title: 'Metrics Collection',
	description: 'Master Prometheus metrics with thread-safe counters and HTTP endpoints for production observability.',
	section: 'production-patterns',
	order: 12,
	topics,
	translations: {
		ru: {
			title: 'Сбор метрик',
			description: 'Освойте метрики Prometheus с потокобезопасными счетчиками и HTTP-эндпоинтами для наблюдаемости в продакшене.'
		},
		uz: {
			title: 'Metrikalarni yig\'ish',
			description: 'Prometheus metrikalari, thread-safe hisoblagichlar va HTTP endpointlar yordamida ishlab chiqarish muhitida kuzatish imkoniyatlarini o\'rganing.'
		}
	}
};
