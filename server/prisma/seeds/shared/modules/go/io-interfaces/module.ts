import { Module } from '../../../../types';
import { topics } from './topics';

export const module: Module = {
	slug: 'io-interfaces',
	title: 'IO Interfaces',
	description: 'Master Go io.Reader and io.Writer interfaces for efficient file and stream operations in production systems.',
	difficulty: 'easy',
	estimatedTime: '20m',
	order: 18,
	isPremium: false,
	section: 'core',
	topics,
	translations: {
		ru: {
			title: 'IO Интерфейсы',
			description: 'Освойте интерфейсы io.Reader и io.Writer в Go для эффективной работы с файлами и потоками в production системах.'
		},
		uz: {
			title: 'IO Interfeyslari',
			description: 'Ishlab chiqarish tizimlarida fayllar va oqimlar bilan samarali ishlash uchun Go io.Reader va io.Writer interfeyslarini o\'rganing.'
		}
	}
};
