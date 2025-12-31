import { CourseMeta } from '../../types';

export const courseMeta: CourseMeta = {
	slug: 'go-concurrency',
	title: 'Go Concurrency',
	description: 'Master Go concurrency: goroutines, channels, synchronization primitives, and advanced concurrency patterns.',
	category: 'language',
	icon: '⚡',
	estimatedTime: '15h',
	order: 2,
	translations: {
		ru: {
			title: 'Параллелизм в Go',
			description: 'Освойте параллелизм в Go: горутины, каналы, примитивы синхронизации и продвинутые паттерны параллельного программирования.'
		},
		uz: {
			title: 'Go da parallellik',
			description: 'Go da parallellikni o\'zlashtiring: gorutinalar, kanallar, sinxronizatsiya primitivlari va ilg\'or parallel dasturlash naqshlari.'
		}
	}
};
