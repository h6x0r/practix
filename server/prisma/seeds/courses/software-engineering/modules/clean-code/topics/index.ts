import { Topic } from '../../../../../types';
import { tasks } from './principles/tasks';

export const principlesTopic: Topic = {
	slug: 'se-clean-code-principles',
	title: 'Clean Code Principles',
	description: 'Meaningful names, small functions, comments, error handling, formatting, and testing.',
	difficulty: 'medium',
	estimatedTime: '8h',
	order: 1,
	tasks,
	translations: {
		ru: {
			title: 'Принципы чистого кода',
			description: 'Осмысленные имена, малые функции, комментарии, обработка ошибок, форматирование и тестирование.',
		},
		uz: {
			title: 'Toza kod tamoyillari',
			description: 'Mazmunli nomlar, kichik funksiyalar, izohlar, xatolarni boshqarish, formatlash va test yozish.',
		},
	},
};
