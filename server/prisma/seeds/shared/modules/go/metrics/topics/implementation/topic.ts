import { Topic } from '../../../../types';
import * as taskImports from './tasks';

const tasks = Object.values(taskImports);

export const topic: Topic = {
	title: 'Prometheus Metrics Implementation',
	description: 'Implement production-ready metrics collection with Prometheus-compatible endpoints and thread-safe counters.',
	difficulty: 'medium',
	estimatedTime: '45m',	order: 1,
	tasks,
	translations: {
		ru: {
			title: 'Реализация метрик Prometheus',
			description: 'Реализация сбора метрик для продакшена с эндпоинтами, совместимыми с Prometheus, и потокобезопасными счетчиками.'
		},
		uz: {
			title: 'Prometheus metrikalarini joriy qilish',
			description: 'Prometheus bilan mos keluvchi endpoint\'lar va potokdan xavfsiz hisoblagichlar bilan ishlab chiqarish uchun tayyor metrikalar yig\'ishni amalga oshirish.'
		}
	}
};
