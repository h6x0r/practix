import { CourseMeta } from '../../types';

export const courseMeta: CourseMeta = {
	slug: 'go-basics',
	title: 'Go Basics',
	description: 'Master Go fundamentals: syntax, types, error handling, pointers, data structures, and essential patterns.',
	category: 'language',
	icon: '🐹',
	estimatedTime: '12h',
	order: 1,
	translations: {
		ru: {
			title: 'Основы Go',
			description: 'Изучите основы Go: синтаксис, типы данных, обработку ошибок, указатели, структуры данных и важные паттерны.'
		},
		uz: {
			title: 'Go asoslari',
			description: 'Go asoslarini o\'rganing: sintaksis, ma\'lumot turlari, xatolarni boshqarish, ko\'rsatkichlar, ma\'lumot tuzilmalari va muhim naqshlar.'
		}
	}
};
