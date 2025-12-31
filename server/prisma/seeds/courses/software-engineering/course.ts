import { CourseMeta } from '../../types';

export const courseMeta: CourseMeta = {
	slug: 'software-engineering',
	title: 'Software Engineering Principles',
	description: 'Master fundamental software engineering principles: SOLID, Clean Code, Refactoring, GRASP, Anti-patterns, and API Design. Language-agnostic concepts with Go and Java examples.',
	category: 'cs',
	icon: '🏛️',
	estimatedTime: '40h',
	order: 10,
	translations: {
		ru: {
			title: 'Принципы разработки ПО',
			description: 'Освойте фундаментальные принципы разработки: SOLID, Чистый код, Рефакторинг, GRASP, Анти-паттерны и Проектирование API. Языко-независимые концепции с примерами на Go и Java.'
		},
		uz: {
			title: 'Dasturiy ta\'minot muhandisligi',
			description: 'Dasturiy ta\'minot muhandisligining asosiy tamoyillarini o\'rganing: SOLID, Toza kod, Refaktoring, GRASP, Anti-patternlar va API dizayni. Go va Java misollari bilan til-mustaqil tushunchalar.'
		}
	}
};
