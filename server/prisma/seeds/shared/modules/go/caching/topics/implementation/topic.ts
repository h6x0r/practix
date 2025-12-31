import { Topic } from '../../../../types';
import { tasks } from './tasks';

export const topic: Topic = {
	title: 'Caching Implementation',
	description: 'Build efficient TTL-based caching systems.',
	difficulty: 'medium',
	estimatedTime: '2h',	order: 1,
	translations: {
		ru: {
			title: 'Реализация кэширования',
			description: 'Создание эффективных систем кэширования на основе TTL.'
		},
		uz: {
			title: 'Keshlashni amalga oshirish',
			description: 'TTL asosida samarali keshlash tizimlarini qurish.'
		}
	},
	tasks
};
