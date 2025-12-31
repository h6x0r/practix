import { Topic } from '../../../../../types';
import { tasks } from './patterns/tasks';

export const patternsTopic: Topic = {
	slug: 'java-dp-creational-patterns',
	title: 'Creational Patterns',
	description: 'Singleton, Factory Method, Abstract Factory, Builder, and Prototype patterns.',
	difficulty: 'medium',
	estimatedTime: '5h',
	order: 0,
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
