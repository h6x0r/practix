import { Module } from '../../../../types';
import { topics } from './topics';

export const cachingModule: Module = {
	title: 'Caching Strategies',
	description: 'Implement caching patterns for performance optimization.',
	section: 'production-patterns',
	order: 10,
	topics,
	translations: {
		ru: {
			title: 'Стратегии кеширования',
			description: 'Реализуйте паттерны кеширования для оптимизации производительности.'
		},
		uz: {
			title: 'Keshlash strategiyalari',
			description: 'Ishlash unumdorligini oshirish uchun keshlash patternlarini amalga oshiring.'
		}
	}
};
