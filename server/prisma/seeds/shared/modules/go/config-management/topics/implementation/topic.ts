import { Topic } from '../../../../types';
import { tasks } from './tasks';

export const topic: Topic = {
	title: 'Configuration Patterns Implementation',
	description: 'Implement production-ready configuration parsing, validation, and error handling patterns.',
	difficulty: 'medium',
	estimatedTime: '50m',
	order: 1,
	tasks,
	translations: {
		ru: {
			title: 'Реализация паттернов конфигурации',
			description: 'Реализация готовых к продакшену паттернов парсинга конфигурации, валидации и обработки ошибок.'
		},
		uz: {
			title: 'Konfiguratsiya namunalarini amalga oshirish',
			description: 'Ishlab chiqarishga tayyor konfiguratsiya tahlili, tekshiruvi va xatolarni boshqarish namunalarini amalga oshirish.'
		}
	}
};
