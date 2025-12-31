import { Module } from '../../../../types';
import { principlesTopic } from './topics';

export const refactoringModule: Module = {
	slug: 'se-refactoring',
	title: 'Refactoring',
	description: 'Master essential refactoring techniques to improve code quality and maintainability.',
	section: 'software-engineering',
	order: 3,
	difficulty: 'medium',
	estimatedTime: '5h',
	topics: [principlesTopic],
};
