import { Topic } from '../../../../../types';
import { tasks } from './patterns/tasks';

export const patternsTopic: Topic = {
	slug: 'go-dp-creational-patterns',
	title: 'Creational Patterns',
	description: 'Singleton, Factory Method, Abstract Factory, Builder, and Prototype patterns.',
	difficulty: 'medium',
	estimatedTime: '6h',
	order: 1,
	tasks,
	translations: {
		ru: {
			title: 'Порождающие паттерны',
			description: 'Паттерны Singleton, Factory Method, Abstract Factory, Builder и Prototype.',
		},
		uz: {
			title: 'Yaratuvchi patternlar',
			description: 'Singleton, Factory Method, Abstract Factory, Builder va Prototype patternlari.',
		},
	},
};
