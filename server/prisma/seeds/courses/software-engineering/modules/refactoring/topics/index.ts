import { Topic } from '../../../../../types';
import { tasks } from './principles/tasks';

export const principlesTopic: Topic = {
	slug: 'se-refactoring-principles',
	title: 'Refactoring Techniques',
	description: 'Learn essential refactoring techniques: Extract Method, Extract Variable, Rename, Move Method, Replace Conditional with Polymorphism, and Introduce Parameter Object.',
	difficulty: 'medium',
	estimatedTime: '5h',
	order: 1,
	tasks,
	translations: {
		ru: {
			title: 'Техники рефакторинга',
			description: 'Изучите основные техники рефакторинга: Extract Method, Extract Variable, Rename, Move Method, Replace Conditional with Polymorphism и Introduce Parameter Object.',
		},
		uz: {
			title: 'Refaktoring texnikalari',
			description: 'Asosiy refaktoring texnikalarini o\'rganing: Extract Method, Extract Variable, Rename, Move Method, Replace Conditional with Polymorphism va Introduce Parameter Object.',
		},
	},
};
