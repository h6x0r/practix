import { Module } from '../../../../types';
import { topics } from './topics';

export const module: Module = {
	slug: 'pointers',
	title: 'Pointers',
	description: 'Master Go pointer fundamentals including nil safety, pointer receivers, double pointers, and linked list operations for production-grade code.',
	difficulty: 'easy',
	estimatedTime: '45m',
	order: 19,
	isPremium: false,
	section: 'core',
	topics,
	translations: {
		ru: {
			title: 'Указатели',
			description: 'Освойте основы указателей в Go, включая безопасность nil, pointer receivers, двойные указатели и операции со связанными списками для продакшен-кода.'
		},
		uz: {
			title: 'Pointerlar',
			description: 'Ishlab chiqarish darajasidagi kod uchun nil xavfsizligi, pointer receiverlar, qo\'sh pointerlar va bog\'langan ro\'yxat operatsiyalarini o\'z ichiga olgan Go pointer asoslarini o\'rganing.'
		}
	}
};
