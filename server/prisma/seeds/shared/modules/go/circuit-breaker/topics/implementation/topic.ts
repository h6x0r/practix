import { Topic } from '../../../../types';
import { tasks } from './tasks';

export const topic: Topic = {
	title: 'Circuit Breaker Implementation',
	description: 'Implement the circuit breaker pattern for fault tolerance.',
	difficulty: 'hard',
	estimatedTime: '2.5h',	order: 1,
	translations: {
		ru: {
			title: 'Реализация Circuit Breaker',
			description: 'Реализация паттерна автоматического выключателя для отказоустойчивости.'
		},
		uz: {
			title: 'Circuit Breaker ni amalga oshirish',
			description: 'Xatolarga chidamlilik uchun avtomatik uzgich patternini amalga oshirish.'
		}
	},
	tasks
};
