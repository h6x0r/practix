import { Module } from '../../../../types';
import { fundamentalsTopic } from './topics';

export const secureCodingModule: Module = {
	slug: 'sec-secure-coding',
	title: 'Secure Coding Practices',
	description: 'Learn secure coding: input validation, output encoding, error handling, logging, and secure API design.',
	section: 'security',
	order: 5,
	difficulty: 'medium',
	estimatedTime: '6h',
	topics: [fundamentalsTopic],
	translations: {
		ru: {
			title: 'Практики безопасного кодирования',
			description: 'Изучите безопасное кодирование: валидация ввода, кодирование вывода, обработка ошибок, логирование и безопасный дизайн API.'
		},
		uz: {
			title: 'Xavfsiz kodlash amaliyotlari',
			description: 'Xavfsiz kodlashni o\'rganing: kirishni tekshirish, chiqishni kodlash, xatolarni qayta ishlash, loglash va xavfsiz API dizayni.'
		}
	}
};
