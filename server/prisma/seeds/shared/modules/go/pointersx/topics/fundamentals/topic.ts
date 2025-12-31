import { Topic } from '../../../../types';
import { tasks } from './tasks';

export const topic: Topic = {
	title: 'Pointer Fundamentals',
	description: 'Master Go pointer operations from basic dereferencing to advanced patterns like double pointers and linked list traversal with nil-safe implementations.',
	difficulty: 'easy',
	estimatedTime: '45m',
	order: 0,
	tasks,
	translations: {
		ru: {
			title: 'Основы указателей',
			description: 'Освоение операций с указателями в Go от базового разыменования до продвинутых паттернов, таких как двойные указатели и обход связанных списков с безопасной обработкой nil.'
		},
		uz: {
			title: 'Ko\'rsatkichlar asoslari',
			description: 'Go dasturlash tilida ko\'rsatkichlar bilan ishlashni asosiy dereference\'dan tortib qo\'sh ko\'rsatkichlar va bog\'langan ro\'yxatlarni nil-xavfsiz implementatsiya bilan o\'rganish.'
		}
	}
};
