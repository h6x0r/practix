import { Module } from '../../../../types';
import { topics } from './topics';

export const panicRecoveryModule: Module = {
	title: 'Panic Recovery',
	description: 'Master panic recovery patterns, safe goroutine execution, and robust error handling to build resilient production systems.',
	section: 'configuration-safety',
	order: 18,
	topics,
	translations: {
		ru: {
			title: 'Восстановление после паники',
			description: 'Освойте паттерны восстановления после паники, безопасное выполнение горутин и надежную обработку ошибок для создания устойчивых продакшен-систем.'
		},
		uz: {
			title: 'Panikdan tiklanish',
			description: 'Barqaror ishlab chiqarish tizimlarini yaratish uchun panikdan tiklanish patternlari, xavfsiz goroutine bajarish va ishonchli xatolarni qayta ishlashni o\'rganing.'
		}
	}
};
