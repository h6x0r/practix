import { CourseMeta } from '../../types';

export const courseMeta: CourseMeta = {
	slug: 'java-core',
	title: 'Java Core',
	description: 'Master Java fundamentals: syntax, OOP, interfaces, exception handling, and the Collections Framework.',
	category: 'language',
	icon: '☕',
	estimatedTime: '14h',
	order: 5,
	translations: {
		ru: {
			title: 'Java Основы',
			description: 'Освойте основы Java: синтаксис, ООП, интерфейсы, обработку исключений и Collections Framework.'
		},
		uz: {
			title: 'Java Asoslari',
			description: 'Java asoslarini o\'rganing: sintaksis, OOP, interfeyslar, xatolarni boshqarish va Collections Framework.'
		}
	}
};
