import { CourseMeta } from '../../types';

export const courseMeta: CourseMeta = {
	slug: 'java-advanced',
	title: 'Java Advanced',
	description: 'Advanced Java: concurrency, I/O, database connectivity, virtual threads, and production-ready patterns.',
	category: 'language',
	icon: '🔧',
	estimatedTime: '16h',
	order: 7,
	translations: {
		ru: {
			title: 'Продвинутая Java',
			description: 'Продвинутая Java: многопоточность, ввод-вывод, подключение к базам данных, виртуальные потоки и промышленные паттерны.'
		},
		uz: {
			title: 'Ilg\'or Java',
			description: 'Ilg\'or Java: parallellik, I/O, ma\'lumotlar bazasiga ulanish, virtual threadlar va ishlab chiqarish uchun tayyor patternlar.'
		}
	}
};
