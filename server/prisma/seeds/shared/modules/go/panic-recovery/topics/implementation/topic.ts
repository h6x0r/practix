import { Topic } from '../../../../types';
import { tasks } from './tasks';

export const topic: Topic = {
	title: 'Panic Recovery Patterns Implementation',
	description: 'Implement safe panic recovery patterns and convert panics to errors for production-grade Go applications.',
	difficulty: 'medium',
	estimatedTime: '50m',
	order: 1,
	translations: {
		ru: {
			title: 'Реализация паттернов восстановления после паники',
			description: 'Реализация безопасных паттернов восстановления после паники и преобразование паник в ошибки для промышленных Go-приложений.'
		},
		uz: {
			title: 'Panikdan tiklash patternlarini amalga oshirish',
			description: 'Ishlab chiqarish darajasidagi Go ilovalar uchun xavfsiz panikdan tiklash patternlarini amalga oshirish va paniklarni xatolarga aylantirish.'
		}
	},
	tasks
};
