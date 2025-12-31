import { Topic } from '../../../../types';
import { tasks } from './tasks';

export const topic: Topic = {
	title: 'Retry Patterns Implementation',
	description: 'Build robust retry mechanisms with exponential backoff.',
	difficulty: 'medium',
	estimatedTime: '1.5h',	order: 1,
	translations: {
		ru: {
			title: 'Реализация паттернов повторных попыток',
			description: 'Создание надежных механизмов повторных попыток с экспоненциальной задержкой.'
		},
		uz: {
			title: 'Qayta urinish patternlarini amalga oshirish',
			description: 'Eksponensial kechikish bilan ishonchli qayta urinish mexanizmlarini qurish.'
		}
	},
	tasks
};
