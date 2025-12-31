import { Topic } from '../../../../../types';
import { tasks } from './patterns/tasks';

export const patternsTopic: Topic = {
	slug: 'go-dp-structural-patterns',
	title: 'Structural Patterns',
	description: 'Adapter, Bridge, Composite, Decorator, Facade, Flyweight, and Proxy patterns.',
	difficulty: 'medium',
	estimatedTime: '7h',
	order: 1,
	tasks,
	translations: {
		ru: {
			title: 'Структурные паттерны',
			description: 'Паттерны Adapter, Bridge, Composite, Decorator, Facade, Flyweight и Proxy.',
		},
		uz: {
			title: 'Strukturaviy patternlar',
			description: 'Adapter, Bridge, Composite, Decorator, Facade, Flyweight va Proxy patternlari.',
		},
	},
};
