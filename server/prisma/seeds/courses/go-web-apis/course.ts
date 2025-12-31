import { CourseMeta } from '../../types';

export const courseMeta: CourseMeta = {
	slug: 'go-web-apis',
	title: 'Go Web & APIs',
	description: 'Build production-ready web services: HTTP middleware, gRPC, rate limiting, circuit breakers, caching, and retry patterns.',
	category: 'language',
	icon: '🌐',
	estimatedTime: '14h',
	order: 3,
	translations: {
		ru: {
			title: 'Go Web и API',
			description: 'Создавайте производственные веб-сервисы: HTTP middleware, gRPC, ограничение частоты запросов, circuit breakers, кэширование и паттерны повторных попыток.'
		},
		uz: {
			title: 'Go Web va API',
			description: 'Ishlab chiqarishga tayyor veb-xizmatlar yarating: HTTP middleware, gRPC, so\'rovlar chastotasini cheklash, circuit breakers, keshlash va qayta urinish naqshlari.'
		}
	}
};
