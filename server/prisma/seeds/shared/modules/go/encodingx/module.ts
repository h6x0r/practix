import { Module } from '../../../../types';
import { topics } from './topics';

export const module: Module = {
	slug: 'json-encoding',
	title: 'JSON Encoding',
	description: 'Implement production-grade JSON validation with strict parsing, unknown field rejection, and security-hardened encoding patterns.',
	difficulty: 'medium',
	estimatedTime: '35m',
	order: 21,
	isPremium: false,
	section: 'core',
	topics,
	translations: {
		ru: {
			title: 'Кодирование JSON',
			description: 'Реализуйте валидацию JSON продакшен-уровня со строгим парсингом, отклонением неизвестных полей и защищенными паттернами кодирования.'
		},
		uz: {
			title: 'JSON kodlash',
			description: 'Qat\'iy parsing, noma\'lum maydonlarni rad etish va xavfsizlik bilan mustahkamlangan kodlash patternlari bilan ishlab chiqarish darajasidagi JSON validatsiyani amalga oshiring.'
		}
	}
};
