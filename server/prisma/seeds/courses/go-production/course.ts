import { CourseMeta } from '../../types';

export const courseMeta: CourseMeta = {
	slug: 'go-production',
	title: 'Go Production',
	description: 'Production-ready Go: testing, logging, metrics, profiling, database operations, and configuration management.',
	category: 'language',
	icon: '🚀',
	estimatedTime: '12h',
	order: 4,
	translations: {
		ru: {
			title: 'Go в продакшене',
			description: 'Go для производства: тестирование, логирование, метрики, профилирование, работа с базами данных и управление конфигурацией.'
		},
		uz: {
			title: 'Go ishlab chiqarishda',
			description: 'Ishlab chiqarishga tayyor Go: testlash, loglash, metrikalar, profillash, ma\'lumotlar bazasi bilan ishlash va konfiguratsiyani boshqarish.'
		}
	}
};
