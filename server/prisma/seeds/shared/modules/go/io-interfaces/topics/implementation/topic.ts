import { Topic } from '../../../../types';
import { tasks } from './tasks';

export const topic: Topic = {
	title: 'IO Interface Implementation',
	description: 'Implement efficient IO operations using Go Reader and Writer interfaces for real-world file and stream handling.',
	difficulty: 'easy',
	estimatedTime: '20m',
	order: 0,
	tasks,
	translations: {
		ru: {
			title: 'Реализация интерфейсов ввода-вывода',
			description: 'Реализация эффективных операций ввода-вывода с использованием интерфейсов Reader и Writer в Go для работы с файлами и потоками в реальных условиях.'
		},
		uz: {
			title: 'Kiritish-chiqarish interfeyslarini joriy qilish',
			description: 'Go dasturlash tilida Reader va Writer interfeyslaridan foydalanib, haqiqiy fayl va oqim bilan ishlash uchun samarali kiritish-chiqarish operatsiyalarini amalga oshirish.'
		}
	}
};
