import { Topic } from '../../../../../types';
import { tasks } from './fundamentals/tasks';

export const fundamentalsTopic: Topic = {
	slug: 'sec-coding-fundamentals',
	title: 'Secure Coding Fundamentals',
	description: 'Input validation, output encoding, error handling, secure logging, and API security.',
	difficulty: 'medium',
	estimatedTime: '6h',
	order: 1,
	tasks,
	translations: {
		ru: {
			title: 'Основы безопасного кодирования',
			description: 'Валидация ввода, кодирование вывода, обработка ошибок, безопасное логирование и безопасность API.',
		},
		uz: {
			title: 'Xavfsiz kodlash asoslari',
			description: 'Kirishni tekshirish, chiqishni kodlash, xatolarni qayta ishlash, xavfsiz loglash va API xavfsizligi.',
		},
	},
};
