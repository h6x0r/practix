import { Topic } from '../../../../../types';
import { tasks } from './principles/tasks';

export const principlesTopic: Topic = {
	slug: 'se-api-design-principles',
	title: 'API Design Principles',
	description: 'RESTful resource naming, HTTP methods, request/response design, error handling, and versioning strategies.',
	difficulty: 'medium',
	estimatedTime: '5h',
	order: 1,
	tasks,
	translations: {
		ru: {
			title: 'Принципы проектирования API',
			description: 'RESTful именование ресурсов, HTTP методы, дизайн запросов/ответов, обработка ошибок и стратегии версионирования.',
		},
		uz: {
			title: 'API dizayn tamoyillari',
			description: 'RESTful resurs nomlash, HTTP metodlar, so\'rov/javob dizayni, xatolarni qayta ishlash va versiyalash strategiyalari.',
		},
	},
};
