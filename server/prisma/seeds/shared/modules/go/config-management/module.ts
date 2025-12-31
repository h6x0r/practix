import { Module } from '../../../../types';
import { topics } from './topics';

export const configManagementModule: Module = {
	title: 'Configuration Management',
	description: 'Master environment variable parsing, default values, validation, and configuration patterns for production-ready Go applications.',
	section: 'configuration-safety',
	order: 17,
	topics,
	translations: {
		ru: {
			title: 'Управление конфигурацией',
			description: 'Освойте парсинг переменных окружения, значения по умолчанию, валидацию и паттерны конфигурации для промышленных Go-приложений.',
		},
		uz: {
			title: 'Konfiguratsiyani boshqarish',
			description: 'Ishlab chiqarishga tayyor Go ilovalar uchun muhit o\'zgaruvchilarini tahlil qilish, standart qiymatlar, validatsiya va konfiguratsiya patternlarini o\'rganing.',
		},
	},
};
