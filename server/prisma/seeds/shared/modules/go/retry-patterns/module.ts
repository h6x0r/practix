import { Module } from '../../../../types';
import { topics } from './topics';

export const retryPatternsModule: Module = {
	title: 'Retry Patterns',
	description: 'Master retry patterns with exponential backoff for reliable systems.',
	section: 'production-patterns',
	order: 9,
	topics,
	translations: {
		ru: {
			title: 'Паттерны повторных попыток',
			description: 'Освойте паттерны повторных попыток с экспоненциальной задержкой для создания надежных систем.'
		},
		uz: {
			title: 'Qayta urinish patternlari',
			description: 'Ishonchli tizimlar uchun eksponensial kechikish bilan qayta urinish patternlarini o\'rganing.'
		}
	}
};
