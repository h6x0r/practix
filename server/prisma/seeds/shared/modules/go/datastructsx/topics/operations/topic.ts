import { Topic } from '../../../../types';
import { tasks } from './tasks';

export const topic: Topic = {
	title: 'Data Structure Operations',
	description: 'Implement essential slice and map operations using Go generics for type-safe, high-performance data manipulation.',
	difficulty: 'easy',
	estimatedTime: '40m',
	order: 0,
	tasks,
	translations: {
		ru: {
			title: 'Операции со структурами данных',
			description: 'Реализация основных операций со срезами и картами с использованием дженериков Go для типобезопасной и высокопроизводительной обработки данных.'
		},
		uz: {
			title: 'Ma\'lumot tuzilmalari operatsiyalari',
			description: 'Tipdan xavfsiz va yuqori samaradorlikda ma\'lumotlarni qayta ishlash uchun Go generiklari yordamida slice va map asosiy operatsiyalarini amalga oshirish.'
		}
	}
};
