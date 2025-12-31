import { Topic } from '../../../../../types';
import { tasks } from './patterns/tasks';

export const patternsTopic: Topic = {
	slug: 'java-dp-behavioral-patterns',
	title: 'Behavioral Patterns',
	description: 'Chain of Responsibility, Command, Iterator, Mediator, Memento, Observer, State, Strategy, Template Method, Visitor, and Interpreter patterns.',
	difficulty: 'hard',
	estimatedTime: '9h',
	order: 0,
	tasks,
	translations: {
		ru: {
			title: 'Поведенческие паттерны',
			description: 'Паттерны Chain of Responsibility, Command, Iterator, Mediator, Memento, Observer, State, Strategy, Template Method, Visitor и Interpreter.',
		},
		uz: {
			title: 'Xulq-atvor patternlari',
			description: 'Chain of Responsibility, Command, Iterator, Mediator, Memento, Observer, State, Strategy, Template Method, Visitor va Interpreter patternlari.',
		},
	},
};
