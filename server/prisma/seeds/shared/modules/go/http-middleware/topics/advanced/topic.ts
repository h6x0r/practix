/**
 * HTTP Middleware Advanced Topic
 * Advanced body manipulation, concurrency control, and middleware composition
 */

import { Topic } from '../../../../types';
import { tasks } from './tasks';

export const advancedTopic: Topic = {
	title: 'Advanced Middleware Patterns',
	description: 'Master advanced HTTP middleware techniques including body manipulation, concurrency control, and middleware composition.',
	difficulty: 'medium',
	estimatedTime: '3h',	order: 1,
	tasks: tasks,
	translations: {
		ru: {
			title: 'Продвинутые паттерны промежуточного ПО',
			description: 'Освойте продвинутые техники HTTP промежуточного ПО, включая манипуляцию телом запроса, контроль параллелизма и композицию промежуточного ПО.',
		},
		uz: {
			title: 'Ilg\'or middleware patternlari',
			description: 'HTTP middleware ning ilg\'or usullarini o\'rganing, jumladan so\'rov tanasini boshqarish, parallellikni nazorat qilish va middleware kompozitsiyasi.',
		},
	},
};
